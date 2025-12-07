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

DECL_LAUNCH_PARAMS(ex02_woodcock::LaunchParams)

struct {
  std::string filepath;
  std::string xfFile;
} g_appState;

namespace ex02_woodcock {
#ifndef RTCORE
extern void woodockTrackingAE();
#endif

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex02_woodcock file.nvdb\n");
}

static void parseCommandLine(int argc, char *argv[]) {

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg[0] != '-')
      g_appState.filepath = arg;
    else if (arg == "--xf")
      g_appState.xfFile = argv[++i];
  }
}

static bool loadXF(dvr_course::Transfunc &tf) {
  std::ifstream in(g_appState.xfFile);

  if (!in.good()) {
    return false;
  }

  in.read((char *)&tf.opacity, sizeof(tf.opacity));
  in.read((char *)&tf.valueRange, sizeof(tf.valueRange));
  in.read((char *)&tf.relRange, sizeof(tf.relRange));

  int numValues;
  in.read((char *)&numValues, sizeof(numValues));

  if (numValues <= 0) {
    return false;
  }

  tf.rgbaLUT.resize(numValues);
  in.read((char *)tf.rgbaLUT.data(), sizeof(tf.rgbaLUT[0]) * tf.rgbaLUT.size());

  return true;
}

extern "C" int main(int argc, char *argv[]) {

  // common namespace for helper classes:
  // Camera, FB, wrappers for RTX execution model, etc. etc.
  using namespace dvr_course;

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

  Pipeline pl("ex02_woodcock");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(cam);

  dvr_course::Transfunc tf;
  if (g_appState.xfFile.empty()) {
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
  } else {
    loadXF(tf);
  }
  pl.setTransfunc(tf);

#ifdef RTCORE
  pl.setRayGen("woodockTrackingAE");
  OWLParams lp = pl.createLaunchParams({
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOFF(LaunchParams,camera.dir_00) }
  });
  owlParamsSet3fv(lp,"camera.dir_00",(const float *)&camera.dir_00);
  // ... more owl setup
#else
  pl.setRayGen(woodockTrackingAE);
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
  // DRV
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
    parms.transfunc.valueRange = tf.valueRange;
    parms.transfunc.size = (int)tf.rgbaLUT.size();
    parms.transfunc.values = tf.rgbaLUT.data();
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

} // namespace ex02_woodcock



