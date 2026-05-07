// std
#include <fstream>
#include <string>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex05:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex05_tets_n_friends::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance{1.f};
  bool useOptixTriangles;
#ifdef RTCORE
  OWLGroup trianglesTLAS, userGeomTLAS;
#endif
} g_appState;

namespace ex05_tets_n_friends {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingAE();
extern void woodcockTrackingWithAccel();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex05_tets_n_friends\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".ic"))
      g_appState.filepath = arg;
  }
}

static std::vector<Tet> loadTets(const std::ifstream &in) {

}

extern "C" int main(int argc, char *argv[]) {

  if (argc < 2) {
    printUsage();
    exit(-1);
  }

  parseCommandLine(argc, argv);

  if (g_appState.filepath.empty()) {
    printUsage();
    exit(-1);
  }

  std::ifstream in(g_appState.filepath);
  if (!in.good()) {
    printUsage();
    exit(-1);
  }

  std::vector<Tet> tets = loadTets(in);

  box3f volbounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  box1f dataRange(INFINITY, -INFINITY);

  Buffer deviceTets(tets.size(), tets.data());

  Pipeline pl(argc, argv, "ex05_tets_n_friends");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = dataRange;

    if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
    tf.setLUT(std::vector<vec4f>({
      {0.149f, 0.015f, 0.705f, 1.0f},
      {0.486f, 0.603f, 0.956f, 0.75f},
      {0.866f, 0.866f, 0.866f, 0.5f},
      {0.996f, 0.690f, 0.552f, 0.25f},
      {0.752f, 0.298f, 0.231f, 0.0f}
    }));
    pl.setTransfunc(&tf);
  }

  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.01f, 5.f);

  g_appState.useOptixTriangles = true;
  pl.uiParam("Use OptiX triangle sampler", &g_appState.useOptixTriangles);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "woodcockTrackingWithAccel");
  pl.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  pl.setRayGen(woodcockTrackingWithAccel);
#endif

  LaunchParams parms;

#ifdef RTCORE
  // ######################################################
  // variant with triangle geometry
  // ######################################################

  std::vector<vec3f> vertex;
  std::vector<vec3i> index;
  for (size_t i=0; i<cells.size(); ++i) {
    const ICONCell &cell = cells[i];
    vec3f v1 = toCartesian({cell.height[0],cell.lat.x,cell.lon.x});
    vec3f v2 = toCartesian({cell.height[0],cell.lat.y,cell.lon.y});
    vec3f v3 = toCartesian({cell.height[0],cell.lat.z,cell.lon.z});
    vertex.push_back(v1);
    vertex.push_back(v2);
    vertex.push_back(v3);
    index.push_back({int(i)*3,int(i)*3+1,int(i)*3+2});
  }

  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(ICONTriangleGeom,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(ICONTriangleGeom,vertex)},
    { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType trianglesGeomType = owlGeomTypeCreate(pl.owlContext(),
                                                    OWL_TRIANGLES,
                                                    sizeof(ICONTriangleGeom),
                                                    trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType, 0, pl.owlModule(), "TetTrianglesClosestHit");
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(pl.owlContext(),OWL_FLOAT3,vertex.size(),vertex.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(pl.owlContext(),OWL_INT3,index.size(),index.data());

  OWLGeom trianglesGeom = owlGeomCreate(pl.owlContext(),trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,vertex.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,index.size(),sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLGroup trianglesBLAS = owlTrianglesGeomGroupCreate(pl.owlContext(),1,&trianglesGeom);
  owlGroupBuildAccel(trianglesBLAS);

  g_appState.trianglesTLAS = owlInstanceGroupCreate(pl.owlContext(),1,&trianglesBLAS);
  owlGroupBuildAccel(g_appState.trianglesTLAS);


  // ######################################################
  // variant with user geometry
  // ######################################################

  OWLVarDecl iconGeomVars[]
  = {
     { "cells",  OWL_BUFPTR, OWL_OFFSETOF(ICONGrid,cells)},
     { "numCells",  OWL_UINT, OWL_OFFSETOF(ICONGrid,numCells)},
     { nullptr /* sentinel to mark end of list */ }
  };
  OWLGeomType userGeomType = owlGeomTypeCreate(pl.owlContext(),
                                               OWL_GEOM_USER,
                                               sizeof(ICONGrid),
                                               iconGeomVars, -1);
  owlGeomTypeSetBoundsProg(userGeomType, pl.owlModule(), "TetBounds");
  owlGeomTypeSetIntersectProg(userGeomType, 0, pl.owlModule(), "TetIntersect");
  owlGeomTypeSetClosestHit(userGeomType, 0, pl.owlModule(), "TetClosestHit");

  OWLGeom userGeom = owlGeomCreate(pl.owlContext(), userGeomType);
  owlGeomSetPrimCount(userGeom, cells.size());

  OWLBuffer cellBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                               OWL_USER_TYPE(ICONCell{}),
                                               cells.size(),
                                               cells.data());
  owlGeomSetBuffer(userGeom, "tets", cellBuffer);
  owlGeomSet1ui(userGeom, "numTets", (unsigned)cells.size());

  owlBuildPrograms(pl.owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pl.owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  g_appState.userGeomTLAS = owlInstanceGroupCreate(pl.owlContext(), 1);
  owlInstanceGroupSetChild(g_appState.userGeomTLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(g_appState.userGeomTLAS);
#endif

  // volume
#ifdef RTCORE
  owlParamsSetGroup(pl.owlLaunchParams(), "volume.handle", g_appState.trianglesTLAS);
  pl.launchParam("volume.useTriangles", parms.volume.useTriangles) = true;
#endif
  pl.launchParam("volume.tets", (RawPointer &)parms.volume.tets) = deviceTets.data();
  pl.launchParam("volume.numTets", parms.volume.numTets) = (int)deviceTets.size();
  pl.launchParam("volume.bounds", parms.volume.bounds) = volbounds;
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

    // update camera:
    pl.launchParam("camera.org", parms.camera.org) = cam.getPosition();
    pl.launchParam("camera.dir_00", parms.camera.dir_00) = screen.lower_left;
    pl.launchParam("camera.dir_du", parms.camera.dir_du) = screen.horizontal / imgWidth;
    pl.launchParam("camera.dir_dv", parms.camera.dir_dv) = screen.vertical / imgHeight;
    // update transfunc:
    pl.launchParam("transfunc.valueRange", parms.transfunc.valueRange) = pl.getTransfunc()->valueRange;
    pl.launchParam("transfunc.size", parms.transfunc.size) = pl.getTransfunc()->size;
    pl.launchParam("transfunc.values", (RawPointer &)parms.transfunc.values) = pl.getTransfunc()->rgbaLUT;
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

} // namespace ex05_tets_n_friends



