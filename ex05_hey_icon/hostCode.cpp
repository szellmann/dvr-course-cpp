// std
#include <fstream>
#include <string>

// nanovdb
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>

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

DECL_LAUNCH_PARAMS(ex05_hey_icon::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance;
} g_appState;

namespace ex05_hey_icon {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodockTrackingAE();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex05_hey_icon\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".nvdb"))
      g_appState.filepath = arg;
  }
}

extern "C" int main(int argc, char *argv[]) {

  // if (argc < 2) {
  //   printUsage();
  //   exit(-1);
  // }

  parseCommandLine(argc, argv);

  // if (g_appState.filepath.empty()) {
  //   printUsage();
  //   exit(-1);
  // }

  box3f volbounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  Random rnd;
  std::vector<ICONCell> cells;
  for (int lon=0; lon<90; lon+=30) {
    for (int lat=0; lat<90.f; lat+=90) {
      ICONCell cell;
      cell.lon.x = deg2rad(-lon);
      cell.lat.x = deg2rad(lat);

      cell.lon.y = deg2rad(-lon-15);
      cell.lat.y = deg2rad(lat+90.f);

      cell.lon.z = deg2rad(-lon-30.f);
      cell.lat.z = deg2rad(lat);

      cell.numLayers = 2;
      //cell.height[0] = 6371.f;
      cell.height[0] = 100.f;
      cell.height[1] = 120.f + 50*rnd();

      float r = cell.height[cell.numLayers-1]-cell.height[0];

      // bottom triangle vertices
      vec3f bv1 = toCartesian({cell.height[0],cell.lat.x,cell.lon.x});
      vec3f bv2 = toCartesian({cell.height[0],cell.lat.y,cell.lon.y});
      vec3f bv3 = toCartesian({cell.height[0],cell.lat.z,cell.lon.z});

      // top triangle vertices
      vec3f tv1 = toCartesian({cell.height[cell.numLayers-1],cell.lat.x,cell.lon.x});
      vec3f tv2 = toCartesian({cell.height[cell.numLayers-1],cell.lat.y,cell.lon.y});
      vec3f tv3 = toCartesian({cell.height[cell.numLayers-1],cell.lat.z,cell.lon.z});

      volbounds.extend(bv1-r); volbounds.extend(bv1+r);
      volbounds.extend(bv2-r); volbounds.extend(bv2+r);
      volbounds.extend(bv3-r); volbounds.extend(bv3+r);
      volbounds.extend(tv1-r); volbounds.extend(tv1+r);
      volbounds.extend(tv2-r); volbounds.extend(tv2+r);
      volbounds.extend(tv3-r); volbounds.extend(tv3+r);

      cells.push_back(cell);
    }
  }

  Buffer deviceCells(cells.size(), cells.data());
  ICONGrid deviceGrid;
  deviceGrid.cells = deviceCells.data();
  deviceGrid.numCells = deviceCells.size();

  Pipeline pl(argc, argv, "ex05_hey_icon");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = {0.f,1.f};//{gridHandle.grid<float>()->tree().root().minimum(),
                     //gridHandle.grid<float>()->tree().root().maximum()};

    if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
    tf.setLUT(std::vector<vec4f>({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    }));
    pl.setTransfunc(&tf);
  }

  g_appState.unitDistance = 1.0f;
  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.001f, 5.f);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "woodockTrackingAE");
  pl.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  pl.setRayGen(woodockTrackingAE);
#endif

  LaunchParams parms;

  // volume
  pl.launchParam("volume.handle", (RawPointer &)parms.volume.handle) = &deviceGrid;
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

} // namespace ex05_hey_icon



