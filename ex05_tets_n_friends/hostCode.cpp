// std
#include <string>

// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex05:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif
#include "importers.h" // loadTetMesh

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex05_tets_n_friends::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance{1.f};
#ifdef RTCORE
  OWLGroup userGeomTLAS;
#endif
} g_appState;

namespace ex05_tets_n_friends {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingAE();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex05_tets_n_friends file.bin\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".bin"))
      g_appState.filepath = arg;
  }
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

  auto tetMesh = loadTetMesh(g_appState.filepath);
  if (!tetMesh.isValid) {
    printUsage();
    exit(-1);
  }

  if (tetMesh.volume.asTetMesh.numTets <= 0) {
    printUsage();
    exit(-1);
  }

  const Volume &volume = tetMesh.volume;

  Pipeline pl(argc, argv, "ex05_tets_n_friends");

  Pipeline::RTConfig conf;
#ifdef RTCORE
  conf.ptxCode = ptxCode;
  conf.launchParamsDecl = launchParams_owl;
  conf.sizeOfLaunchParamsStruct = sizeof(LaunchParams);
#else
  conf.rayGens.push_back({"woodcockTrackingAE",woodcockTrackingAE});
#endif
  pl.initRT(conf);
  pl.setRayGen("woodcockTrackingAE");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(volume.bounds);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = volume.dataRange;

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
  owlGeomSetPrimCount(userGeom, volume.asTetMesh.numTets);

  OWLBuffer tetBuffer = owlDeviceBufferCreate(pl.owlContext(),
                                              OWL_USER_TYPE(Tet{}),
                                              volume.asTetMesh.numTets,
                                              volume.asTetMesh.tets);
  owlGeomSetBuffer(userGeom, "tets", tetBuffer);
  owlGeomSet1i(userGeom, "numTets", volume.asTetMesh.numTets);

  owlBuildPrograms(pl.owlContext());

  OWLGroup userGeomBLAS = owlUserGeomGroupCreate(pl.owlContext(), 1, &userGeom);
  owlGroupBuildAccel(userGeomBLAS);

  g_appState.userGeomTLAS = owlInstanceGroupCreate(pl.owlContext(), 1);
  owlInstanceGroupSetChild(g_appState.userGeomTLAS, 0, userGeomBLAS);

  owlGroupBuildAccel(g_appState.userGeomTLAS);

  owlParamsSetGroup(pl.owlLaunchParams(), "volume.asTetMesh.handle", g_appState.userGeomTLAS);
#endif

  // volume
  pl.launchParam("volume.asTetMesh.tets", (RawPointer &)parms.volume.asTetMesh.tets) = volume.asTetMesh.tets;
  pl.launchParam("volume.asTetMesh.numTets", parms.volume.asTetMesh.numTets) = volume.asTetMesh.numTets;
  pl.launchParam("volume.bounds", parms.volume.bounds) = volume.bounds;
  pl.launchParam("volume.dataRange", parms.volume.dataRange) = volume.dataRange;
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

} // namespace ex05_tets_n_friends



