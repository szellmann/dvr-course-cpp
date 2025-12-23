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

// ex01:
#include "Params.h"

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex03_multi_volume::LaunchParams)

struct {
  std::vector<std::string> filepaths;
  std::vector<ex03_multi_volume::Volume> volumes;
  std::vector<Buffer<uint8_t>> deviceGrids;
  std::vector<Transfunc> transfuncs;
  float unitDistance;
} g_appState;

namespace ex03_multi_volume {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void multiVolumeWoodcock();
extern void blendingWoodcock();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex03_multi_volume file.nvdb\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".nvdb"))
      g_appState.filepaths.push_back(arg);
  }
}

extern "C" int main(int argc, char *argv[]) {

  if (argc < 2) {
    printUsage();
    exit(-1);
  }

  parseCommandLine(argc, argv);

  if (g_appState.filepaths.empty()) {
    printUsage();
    exit(-1);
  }

  Pipeline pl(argc, argv, "ex03_multi_volume");

  box3f worldBounds(
    {INFINITY,INFINITY,INFINITY},
    {-INFINITY,-INFINITY,-INFINITY}
  );

  // construct NVDB volumes:
  for (int i=0; i<g_appState.filepaths.size(); ++i) {
    nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;

    try {
      auto grid = nanovdb::io::readGrid(g_appState.filepaths[i]);
      auto hostbuffer = nanovdb::HostBuffer::create(grid.bufferSize());
      std::memcpy(hostbuffer.data(), grid.data(), grid.bufferSize());
      gridHandle = std::move(hostbuffer);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      printUsage();
      exit(-1);
    }

    Buffer deviceGrid(gridHandle.bufferSize(), (uint8_t *)gridHandle.data());
    g_appState.deviceGrids.push_back(deviceGrid);

    // construct device-side volumes:
    auto boundsMin = gridHandle.gridMetaData()->worldBBox().min();
    auto boundsMax = gridHandle.gridMetaData()->worldBBox().max()-nanovdb::Vec3d(1.f);
    box3f volbounds({(float)boundsMin[0], (float)boundsMin[1], (float)boundsMin[2]},
                    {(float)boundsMax[0], (float)boundsMax[1], (float)boundsMax[2]});

    Volume volume;
    volume.handle = (nanovdb::NanoGrid<float> *)deviceGrid.data();
    volume.filterLinear = true;
    volume.bounds = volbounds;
    g_appState.volumes.push_back(volume);

    worldBounds.extend(volbounds);

    // construct transfuncs:

    // TODO: this won't allow us to load TFs from file anymore!!
    // ...is this even a to-do?!
    if (!pl.transfuncValid(i)) {
      dvr_course::Transfunc tf;
      tf.valueRange = {gridHandle.grid<float>()->tree().root().minimum(),
                       gridHandle.grid<float>()->tree().root().maximum()};

      tf.valueRange.lower
        = fminf(tf.valueRange.lower, gridHandle.grid<float>()->tree().root().background());
      tf.valueRange.upper
        = fmaxf(tf.valueRange.upper, gridHandle.grid<float>()->tree().root().background());

      vec3f rgb = 0.f;
      rgb[i%3] = 1.f;
      if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
      tf.setLUT(std::vector<vec4f>({
        {rgb.r,rgb.g,rgb.b,0.f },
        {rgb.r,rgb.g,rgb.b,1.f }
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

  Buffer volumeBuffer(g_appState.volumes.size(), g_appState.volumes.data());

  std::vector<ex03_multi_volume::Transfunc> deviceTransfuncs(g_appState.transfuncs.size());
  for (int i=0; i<g_appState.transfuncs.size(); ++i) {
    pl.setTransfunc(&g_appState.transfuncs[i],i);
  }

  g_appState.unitDistance = 1.0f;
  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.001f, 5.f);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "multiVolumeWoodcock");
#else
  pl.setRayGen(multiVolumeWoodcock);
#endif

  LaunchParams parms;

  // volumes
  pl.launchParam("volumes", (RawPointer &)parms.volumes) = (Volume *)volumeBuffer.data();
  pl.launchParam("numVolumes", parms.numVolumes) = volumeBuffer.size();
  // lighting
  pl.launchParam("ambientColor", parms.ambientColor) = vec3f(1.f);
  pl.launchParam("ambientRadiance", parms.ambientRadiance) = 1.f;
  // blending
  pl.launchParam("blendMode", parms.blendMode) = BLEND_MODE_MIX;

  pl.setKeyDownHandler([&](char key) {
    if (key == '1') {
#ifdef RTCORE
      pl.setRayGen("multiVolumeWoodcock");
#else
      pl.setRayGen(multiVolumeWoodcock);
#endif
      pl.resetAccumulation();
    }
    if (key == '2') {
#ifdef RTCORE
      pl.setRayGen("blendingWoodcock");
#else
      pl.setRayGen(blendingWoodcock);
#endif
      pl.launchParam("blendMode", parms.blendMode) = BLEND_MODE_MIX;
      pl.resetAccumulation();
    }
    if (key == '3') {
#ifdef RTCORE
      pl.setRayGen("blendingWoodcock");
#else
      pl.setRayGen(blendingWoodcock);
#endif
      pl.launchParam("blendMode", parms.blendMode) = BLEND_MODE_MAX_ALPHA;
      pl.resetAccumulation();
    }
  });

  // Render and present...
  // For default (PNG image) pipeline this
  // loop returns immediately
  do {
    // camera:
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
        = (ex03_multi_volume::Transfunc *)transfuncBuffer.data();
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

} // namespace ex03_multi_volume



