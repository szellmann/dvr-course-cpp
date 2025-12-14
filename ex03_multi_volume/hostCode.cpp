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

struct NVDB {
  uint8_t *gridData{nullptr};
};

struct {
  std::vector<std::string> filepaths;
  std::vector<NVDB> nvdbVolumes;
  std::vector<ex03_multi_volume::Volume> volumes;
  std::vector<Transfunc> transfuncs;
  float unitDistance;
} g_appState;

namespace ex03_multi_volume {
#ifndef RTCORE
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

  try {
    // construct NVDB volumes:
#ifdef RTCORE
#else
    for (int i=0; i<g_appState.filepaths.size(); ++i) {
      NVDB nvdb;
      // TODO: not sure about the lifetime of those handles.. need to check?!
      nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;
      auto grid = nanovdb::io::readGrid(g_appState.filepaths[i]);
      nvdb.gridData = (uint8_t *)std::malloc(grid.bufferSize() + NANOVDB_DATA_ALIGNMENT);
      void *dataPtr = nanovdb::alignPtr(nvdb.gridData);
      std::memcpy(nvdb.gridData, grid.data(), grid.bufferSize());
      auto buffer = nanovdb::HostBuffer::createFull(grid.bufferSize(), dataPtr);
      gridHandle = std::move(buffer);
      g_appState.nvdbVolumes.push_back(nvdb);
#endif
      // construct device-side volumes:
      auto boundsMin = gridHandle.gridMetaData()->worldBBox().min();
      auto boundsMax = gridHandle.gridMetaData()->worldBBox().max();
      box3f volbounds({(float)boundsMin[0], (float)boundsMin[1], (float)boundsMin[2]},
                      {(float)boundsMax[0], (float)boundsMax[1], (float)boundsMax[2]});

      Volume volume;
      volume.handle = gridHandle.grid<float>();
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
        tf.rgbaLUT = std::vector<vec4f>({
          {rgb.r,rgb.g,rgb.b,0.f },
          {rgb.r,rgb.g,rgb.b,1.f }
        });
        g_appState.transfuncs.push_back(tf);
      }
    }
  } catch (...) {
    printUsage();
    exit(-1);
  }

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(worldBounds);
  pl.setCamera(&cam);

  Buffer volumeBuffer(
      g_appState.volumes.size(), OWL_USER_TYPE(Volume), g_appState.volumes.data());

  std::vector<ex03_multi_volume::Transfunc> deviceTransfuncs(g_appState.transfuncs.size());
  for (int i=0; i<g_appState.transfuncs.size(); ++i) {
    pl.setTransfunc(&g_appState.transfuncs[i],i);
  }

  g_appState.unitDistance = 1.0f;
  pl.addParam("Unit distance", &g_appState.unitDistance, 0.001f, 5.f);

#ifdef RTCORE
  pl.setRayGen("multiVolumeWoodcock");
  OWLParams lp = pl.createLaunchParams({
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOFF(LaunchParams,camera.dir_00) }
  });
  owlParamsSet3fv(lp,"camera.dir_00",(const float *)&camera.dir_00);
  // ... more owl setup
#else
  pl.setRayGen(multiVolumeWoodcock);
  LaunchParams parms;
  // volumes
  parms.volumes = (Volume *)volumeBuffer.getPointer();
  parms.numVolumes = volumeBuffer.getSize();
  // lighting
  parms.ambientColor = vec3f(1.f);
  parms.ambientRadiance = 1.f;
  // blending
  parms.blendMode = BLEND_MODE_MIX;
#endif

  pl.setKeyDownHandler([&](char key) {
    if (key == '1') {
      pl.setRayGen(multiVolumeWoodcock);
      pl.resetAccumulation();
    }
    if (key == '2') {
      pl.setRayGen(blendingWoodcock);
      parms.blendMode = BLEND_MODE_MIX;
      pl.resetAccumulation();
    }
    if (key == '3') {
      pl.setRayGen(blendingWoodcock);
      parms.blendMode = BLEND_MODE_MAX_ALPHA;
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
      deviceTransfuncs[i].size = (int)pl.getTransfunc(i)->rgbaLUT.size();
      deviceTransfuncs[i].values = pl.getTransfunc(i)->rgbaLUT.data();
    }
    Buffer transfuncBuffer(deviceTransfuncs.size(),
                           OWL_USER_TYPE(ex03_multi_volume::Transfunc),
                           deviceTransfuncs.data());
#ifdef RTCORE
    owlParamsSet3fv(lp,"camera.dir_00",(const float *)&camera.dir_00);
    // ...
#else
    // update camera:
    parms.camera.org = cam.getPosition();
    parms.camera.dir_00 = screen.lower_left;
    parms.camera.dir_du = screen.horizontal / imgWidth;
    parms.camera.dir_dv = screen.vertical / imgHeight;
    // update transfuncs:
    parms.transfuncs = (ex03_multi_volume::Transfunc *)transfuncBuffer.getPointer();
    // update framebuffer:
    parms.fbPointer   = fb.fbPointer;
    parms.fbDepth     = fb.fbDepth;
    parms.accumBuffer = fb.accumBuffer;
    // update DVR params:
    parms.unitDistance = g_appState.unitDistance;
    // update accum:
    parms.accumID = pl.frameID;
#endif

    // set params:
    SET_LAUNCH_PARAMS(parms);

    pl.launch();
    pl.present();
  } while (pl.isRunning());

  return 0;
}

} // namespace ex03_multi_volume



