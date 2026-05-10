// std
#include <string>

// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex07:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif
#define TINYOBJLOADER_IMPLEMENTATION
#include "importers.h" // loadNvdb, loadTetMesh, loadObj

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex07_render_graph::LaunchParams)

struct {
  std::vector<std::string> objFiles;
  std::vector<std::string> nvdbFiles;
  std::vector<std::string> tetMeshFiles;
  std::vector<ex07_render_graph::TriangleMesh> triangleMeshes;
  std::vector<ex07_render_graph::Volume> volumes;
  std::vector<Buffer<uint8_t>> deviceGrids;
  std::vector<Buffer<ex07_render_graph::Tet>> deviceTets;
  std::vector<std::pair<Buffer<vec3f>,Buffer<vec3i>>> deviceMeshes;
  std::vector<Transfunc> transfuncs;
  float unitDistance{1.f};
#ifdef RTCORE
  OWLGroup triangleTLAS;
#endif
} g_appState;

namespace ex07_render_graph {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingAE();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex07_render_graph [file.obj|file.nvdb|file.bin]\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-') {
      if (endsWith(arg,".obj")) {
        g_appState.objFiles.push_back(arg);
      } else if (endsWith(arg,".nvdb")) {
        g_appState.nvdbFiles.push_back(arg);
      } else if (endsWith(arg,".bin")) {
        g_appState.tetMeshFiles.push_back(arg);
      }
    }
  }
}

extern "C" int main(int argc, char *argv[]) {

  if (argc < 2) {
    printUsage();
    exit(-1);
  }

  parseCommandLine(argc, argv);

  Pipeline pl(argc, argv, "ex07_render_graph");

  box3f worldBounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  for (auto f: g_appState.objFiles) {
    auto obj = loadObj(f);
    if (!obj.isValid) {
      printUsage();
      exit(-1);
    }
    g_appState.triangleMeshes.push_back(obj.mesh);
    g_appState.deviceMeshes.push_back(obj.onDevice);
    worldBounds.extend(obj.mesh.bounds);
  }

  for (auto f: g_appState.nvdbFiles) {
    auto nvdb = loadNvdb(f);
    if (!nvdb.isValid) {
      printUsage();
      exit(-1);
    }
    g_appState.volumes.push_back(nvdb.volume);
    g_appState.deviceGrids.push_back(nvdb.onDevice);
    worldBounds.extend(nvdb.volume.bounds);
  }

  for (auto f: g_appState.tetMeshFiles) {
    auto tetMesh = loadTetMesh(f);
    if (!tetMesh.isValid) {
      printUsage();
      exit(-1);
    }
    g_appState.volumes.push_back(tetMesh.volume);
    g_appState.deviceTets.push_back(tetMesh.onDevice);
    worldBounds.extend(tetMesh.volume.bounds);
  }

  // TODO!!!!
  // TODO: build BLASes here??
  // assign volume handles:
  for (int i=0; i<g_appState.deviceGrids.size(); ++i) {
    Volume &volume = g_appState.volumes[i];
    volume.asNvdb.handle = (nanovdb::NanoGrid<float> *)g_appState.deviceGrids[i].data();
  }

  // construct transfuncs:
  for (int i=0; i<g_appState.volumes.size(); ++i) {
    if (!pl.transfuncValid(i)) {
      const Volume &volume = g_appState.volumes[i];
      dvr_course::Transfunc tf;
      tf.valueRange = volume.dataRange;

      if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
      tf.setLUT(std::vector<vec4f>({
        {0.f,0.f,1.f,0.1f },
        {0.f,1.f,0.f,0.1f }
      }));
      g_appState.transfuncs.push_back(tf);
    }
  }

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(worldBounds);
  pl.setCamera(&cam);

  Buffer meshBuffer(g_appState.triangleMeshes.size(), g_appState.triangleMeshes.data());
  Buffer volumeBuffer(g_appState.volumes.size(), g_appState.volumes.data());

  std::vector<ex07_render_graph::Transfunc> deviceTransfuncs(g_appState.transfuncs.size());
  for (int i=0; i<g_appState.transfuncs.size(); ++i) {
    pl.setTransfunc(&g_appState.transfuncs[i],i);
  }

  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.01f, 5.f);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "woodcockTrackingAE");
  pl.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  pl.setRayGen(woodcockTrackingAE);
#endif

  LaunchParams parms;

#ifdef RTCORE
  // build mesh BVHs
  OWLVarDecl triangleGeomVars[]
    = {
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh,vertices)},
      { "indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleMesh,indices)},
      { nullptr /* sentinel to mark end of list */ }
    };
  OWLGeomType triangleGeomType = owlGeomTypeCreate(pl.owlContext(),
                                                   OWL_TRIANGLES,
                                                   sizeof(TriangleMesh),
                                                   triangleGeomVars, -1);
  owlGeomTypeSetClosestHit(triangleGeomType, 0, pl.owlModule(), "TriangleMeshClosestHit");

  owlBuildPrograms(pl.owlContext());

  g_appState.triangleTLAS = owlInstanceGroupCreate(pl.owlContext(),
                                                   g_appState.triangleMeshes.size());

  for (int i=0; i<g_appState.triangleMeshes.size(); ++i) {
    auto &mesh = g_appState.triangleMeshes[i];
    mesh.meshID = i; // TODO (assign in importer?)
    auto &onDevice = g_appState.deviceMeshes[mesh.meshID];
    OWLGeom geom = owlGeomCreate(pl.owlContext(), triangleGeomType);
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                                   OWL_FLOAT3,
                                                   onDevice.first.size(),
                                                   onDevice.first.data());

    OWLBuffer indexBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                                  OWL_INT3,
                                                  onDevice.second.size(),
                                                  onDevice.second.data());

    owlTrianglesSetVertices(geom,vertexBuffer,onDevice.first.size(),sizeof(vec3f),0);
    owlTrianglesSetIndices(geom,indexBuffer,onDevice.second.size(),sizeof(vec3i),0);
    OWLGroup group = owlTrianglesGeomGroupCreate(pl.owlContext(), 1, &geom);
    owlGroupBuildAccel(group);
    owlGeomSetBuffer(geom,"vertices",vertexBuffer);
    owlGeomSetBuffer(geom,"indices",indexBuffer);
    owlInstanceGroupSetChild(g_appState.triangleTLAS, i, group);
  }

  owlGroupBuildAccel(g_appState.triangleTLAS);
  //owlParamsSetGroup(lp, "world", asTriMesh.tlasGroup);
#endif

#if 0//def RTCORE
  // ######################################################
  // variant with user geometry
  // ######################################################

  OWLVarDecl tetsGeomVars[]
  = {
     { "tets",  OWL_BUFPTR, OWL_OFFSETOF(TetMesh,tets)},
     { "numTets",  OWL_INT, OWL_OFFSETOF(TetMesh,numTets)},
     { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType userGeomType = owlGeomTypeCreate(pl.owlContext(),
                                               OWL_GEOM_USER,
                                               sizeof(TetMesh),
                                               tetsGeomVars, -1);
  owlGeomTypeSetBoundsProg(userGeomType, pl.owlModule(), "TetBounds");
  owlGeomTypeSetIntersectProg(userGeomType, 0, pl.owlModule(), "TetIntersect");
  owlGeomTypeSetClosestHit(userGeomType, 0, pl.owlModule(), "TetClosestHit");

  OWLGeom userGeom = owlGeomCreate(pl.owlContext(), userGeomType);
  //owlGeomSetPrimCount(userGeom, tets.size());

  //OWLBuffer tetBuffer = owlDeviceBufferCreate(pl.owlContext(),
  //                                            OWL_USER_TYPE(Tet{}),
  //                                            tets.size(),
  //                                            tets.data());
  //owlGeomSetBuffer(userGeom, "tets", tetBuffer);
  //owlGeomSet1i(userGeom, "numTets", (int)tets.size());

  owlBuildPrograms(pl.owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pl.owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  g_appState.userGeomTLAS = owlInstanceGroupCreate(pl.owlContext(), 1);
  owlInstanceGroupSetChild(g_appState.userGeomTLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(g_appState.userGeomTLAS);
#endif

  // volumes
  pl.launchParam("volumes", (RawPointer &)parms.volumes) = (Volume *)volumeBuffer.data();
  pl.launchParam("numVolumes", parms.numVolumes) = volumeBuffer.size();
  // meshes
#ifdef RTCORE
  owlParamsSetGroup(pl.owlLaunchParams(), "triangleTLAS", g_appState.triangleTLAS);
#endif
  // lighting
  pl.launchParam("ambientColor", parms.ambientColor) = vec3f(1.f);
  pl.launchParam("ambientRadiance", parms.ambientRadiance) = 1.f;

  // Render and present...
  // For default (PNG image) pipeline this
  // loop returns immediately
  do {
    struct {
      vec3f lower_left, horizontal, vertical;
    } screen;
    cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

    // transfer functions on device:
    for (int i=0; i<g_appState.transfuncs.size(); ++i) {
      deviceTransfuncs[i].valueRange = pl.getTransfunc(i)->valueRange;
      deviceTransfuncs[i].size = pl.getTransfunc(i)->size;
      deviceTransfuncs[i].values = pl.getTransfunc(i)->rgbaLUT;
    }
    Buffer transfuncBuffer(deviceTransfuncs.size(), deviceTransfuncs.data());

    // update camera:
    pl.launchParam("camera.org", parms.camera.org) = cam.getPosition();
    pl.launchParam("camera.dir_00", parms.camera.dir_00) = screen.lower_left;
    pl.launchParam("camera.dir_du", parms.camera.dir_du) = screen.horizontal / imgWidth;
    pl.launchParam("camera.dir_dv", parms.camera.dir_dv) = screen.vertical / imgHeight;
    // update transfuncs:
    pl.launchParam("transfuncs", (RawPointer &)parms.transfuncs)
        = (ex07_render_graph::Transfunc *)transfuncBuffer.data();
    // update framebuffer:
    pl.launchParam("fbPointer", (RawPointer &)parms.fbPointer) = fb.fbPointer;
    pl.launchParam("fbDepth", (RawPointer &)parms.fbDepth) = fb.fbDepth;
    pl.launchParam("accumBuffer", (RawPointer &)parms.accumBuffer) = fb.accumBuffer;
    // update DVR params:
    pl.launchParam("unitDistance", parms.unitDistance) = g_appState.unitDistance;
    // update accum:
    pl.launchParam("accumID", parms.accumID) = pl.frameID;

    // set params:
    SET_LAUNCH_PARAMS(parms);

    pl.launch();
    pl.present();
  } while (pl.isRunning());

  return 0;
}

} // namespace ex07_render_graph



