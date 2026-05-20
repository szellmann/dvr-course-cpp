// std
#include <fstream>
#include <string>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex06:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex06_need_for_speed::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance{1.f};
#ifdef RTCORE
  OWLGroup userGeomTLAS;
#endif
} g_appState;

namespace ex06_need_for_speed {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingAE();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex06_need_for_speed file.bin\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".bin"))
      g_appState.filepath = arg;
  }
}

static std::vector<Tet> loadTets(std::ifstream &in) {
  // TODO: error checking for data loader we assume the data to be in the exact
  // format below, but the file format has builtin support for an arbitrary
  // number of data arrays. We currently just assume there are at least two
  // scalar data arrays, and that dataArray[0] is per-cell and dataArra[1] is
  // per-vertex

  auto loadVector = [](std::ifstream &in, auto &vec) {
    uint64_t size;
    in.read((char *)&size,sizeof(size));
    vec.resize(size);
    in.read((char *)vec.data(),vec.size()*sizeof(vec[0]));
  };

  uint64_t numDataArrays=0;
  std::vector<vec3f> vertices;
  std::vector<int> cellTypes;
  std::vector<int> cellIndices;
  std::vector<int> connectivity;
  std::vector<float> cellValues, vertexValues;

  // vertex positions:
  loadVector(in,vertices);
  // topology:
  loadVector(in,cellTypes);
  loadVector(in,cellIndices);
  loadVector(in,connectivity);
  // data arrays:
  in.read((char *)&numDataArrays,sizeof(numDataArrays));
  loadVector(in,cellValues);
  loadVector(in,vertexValues);

  // assemble tets:
  std::vector<Tet> tets;
  for (size_t i=0; i<cellTypes.size(); ++i) {
    #define VTK_TET_ 10
    if (cellTypes[i] != VTK_TET_) continue;
    int i0 = connectivity[cellIndices[i]];
    int i1 = connectivity[cellIndices[i]+1];
    int i2 = connectivity[cellIndices[i]+2];
    int i3 = connectivity[cellIndices[i]+3];
    vec3f v0 = vertices[i0];
    vec3f v1 = vertices[i1];
    vec3f v2 = vertices[i2];
    vec3f v3 = vertices[i3];
    float s0, s1, s2, s3;
    if (!vertexValues.empty()) {
      s0 = vertexValues[i0];
      s1 = vertexValues[i1];
      s2 = vertexValues[i2];
      s3 = vertexValues[i3];
    } else {
      s0 = cellValues[cellIndices[i]/4];
      s1 = cellValues[cellIndices[i]/4];
      s2 = cellValues[cellIndices[i]/4];
      s3 = cellValues[cellIndices[i]/4];
    }

    // Store tets in our simple, flattened format, i.e., values are encoded in
    // the 'w' coordinate of the positional vectors
    Tet tet;
    tet.v0 = vec4f(v0,s0);
    tet.v1 = vec4f(v1,s1);
    tet.v2 = vec4f(v2,s2);
    tet.v3 = vec4f(v3,s3);
    tets.push_back(tet);
  }
  return tets;
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

  Buffer deviceTets(tets.size(), tets.data());

  box3f volbounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  box1f dataRange(INFINITY, -INFINITY);

  for (size_t i=0; i<tets.size(); ++i) {
    volbounds.extend(tets[i].v0.xyz); dataRange.extend(tets[i].v0.w);
    volbounds.extend(tets[i].v1.xyz); dataRange.extend(tets[i].v1.w);
    volbounds.extend(tets[i].v2.xyz); dataRange.extend(tets[i].v2.w);
    volbounds.extend(tets[i].v3.xyz); dataRange.extend(tets[i].v3.w);
  }

  Pipeline pl(argc, argv, "ex06_need_for_speed");

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
      {0.149f, 0.015f, 0.705f, 0.0f},
      {0.486f, 0.603f, 0.956f, 0.25f},
      {0.866f, 0.866f, 0.866f, 0.5f},
      {0.996f, 0.690f, 0.552f, 0.75f},
      {0.752f, 0.298f, 0.231f, 1.0f}
    }));
    pl.setTransfunc(&tf);
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
  owlGeomSetPrimCount(userGeom, tets.size());

  OWLBuffer tetBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                              OWL_USER_TYPE(Tet{}),
                                              tets.size(),
                                              tets.data());
  owlGeomSetBuffer(userGeom, "tets", tetBuffer);
  owlGeomSet1i(userGeom, "numTets", (int)tets.size());

  owlBuildPrograms(pl.owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pl.owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  g_appState.userGeomTLAS = owlInstanceGroupCreate(pl.owlContext(), 1);
  owlInstanceGroupSetChild(g_appState.userGeomTLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(g_appState.userGeomTLAS);
#endif

  // volume
#ifdef RTCORE
  owlParamsSetGroup(pl.owlLaunchParams(), "volume.handle", g_appState.userGeomTLAS);
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
    pl.launchParam("camera.dir_du", parms.camera.dir_du) = screen.horizontal / fb.width;
    pl.launchParam("camera.dir_dv", parms.camera.dir_dv) = screen.vertical / fb.height;
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

} // namespace ex06_need_for_speed



