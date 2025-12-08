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

DECL_LAUNCH_PARAMS(ex01_let_there_be_voxels::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
} g_appState;

namespace ex01_let_there_be_voxels {
#ifndef RTCORE
extern void simpleRayMarcher();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex01_let_there_be_voxels file.nvdb\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-' && endsWith(arg,".nvdb"))
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

  uint8_t *gridData{nullptr};
  nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;

  try {
#ifdef RTCORE

#else
    auto grid = nanovdb::io::readGrid(g_appState.filepath);
    gridData = (uint8_t *)std::malloc(grid.bufferSize() + NANOVDB_DATA_ALIGNMENT);
    void *dataPtr = nanovdb::alignPtr(gridData);
    std::memcpy(gridData, grid.data(), grid.bufferSize());
    auto buffer = nanovdb::HostBuffer::createFull(grid.bufferSize(), dataPtr);
    gridHandle = std::move(buffer);
#endif
  } catch (...) {
    printUsage();
    exit(-1);
  }

  auto boundsMin = gridHandle.gridMetaData()->worldBBox().min();
  auto boundsMax = gridHandle.gridMetaData()->worldBBox().max();
  box3f volbounds({(float)boundsMin[0], (float)boundsMin[1], (float)boundsMin[2]},
                  {(float)boundsMax[0], (float)boundsMax[1], (float)boundsMax[2]});

  Pipeline pl(argc, argv, "ex01_let_there_be_voxels");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(cam);

  if (pl.transfunc == nullptr) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = {gridHandle.grid<float>()->tree().root().minimum(),
                     gridHandle.grid<float>()->tree().root().maximum()};

    tf.valueRange.lower
      = fminf(tf.valueRange.lower, gridHandle.grid<float>()->tree().root().background());
    tf.valueRange.upper
      = fmaxf(tf.valueRange.upper, gridHandle.grid<float>()->tree().root().background());

    if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
    tf.rgbaLUT = std::vector<vec4f>({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    });
    pl.setTransfunc(tf);
  }

#ifdef RTCORE
  pl.setRayGen("simpleRayMarcher");
  OWLParams lp = pl.createLaunchParams({
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOFF(LaunchParams,camera.dir_00) }
  });
  owlParamsSet3fv(lp,"camera.dir_00",(const float *)&camera.dir_00);
  // ... more owl setup
#else
  pl.setRayGen(simpleRayMarcher);
  LaunchParams parms;
  // volume
  parms.volume.handle = gridHandle.grid<float>();
  parms.volume.filterLinear = true;
  parms.volume.bounds = volbounds;
  // framebuffer
  parms.fbPointer   = fb.fbPointer;
  parms.fbDepth     = fb.fbDepth;
  parms.accumBuffer = fb.accumBuffer;
  // lighting
  parms.ambientColor = vec3f(1.f);
  parms.ambientRadiance = 1.f;
  // DVR
  parms.samplingRate = 2.f;
  parms.unitDistance = 1.0f;
#endif

  // Render and present...
  // For default (PNG image) pipeline this
  // loop returns immediately
  do {
    struct {
      vec3f lower_left, horizontal, vertical;
    } screen;
    cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);
#ifdef RTCORE
    owlParamsSet3fv(lp,"camera.dir_00",(const float *)&camera.dir_00);
    // ...
#else
    // update camera:
    parms.camera.org = cam.getPosition();
    parms.camera.dir_00 = screen.lower_left;
    parms.camera.dir_du = screen.horizontal / imgWidth;
    parms.camera.dir_dv = screen.vertical / imgHeight;
    // update transfunc:
    parms.transfunc.valueRange = pl.transfunc->valueRange;
    parms.transfunc.size = (int)pl.transfunc->rgbaLUT.size();
    parms.transfunc.values = pl.transfunc->rgbaLUT.data();
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

} // namespace ex01_let_there_be_voxels



