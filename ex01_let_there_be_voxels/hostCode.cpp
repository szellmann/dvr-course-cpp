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

struct {
  std::string filepath;
  std::string xfFile;
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
    if (arg[0] != '-')
      g_appState.filepath = arg;
    else if (arg == "--xf")
      g_appState.xfFile = argv[++i];
  }
}

static bool loadXF(std::vector<vec4f> &xf) {
  float opacity;
  box1f valueRange, relRange;

  std::ifstream in(g_appState.xfFile);

  if (!in.good()) {
    return false;
  }

  in.read((char *)&opacity, sizeof(opacity));
  in.read((char *)&valueRange, sizeof(valueRange));
  in.read((char *)&relRange, sizeof(relRange));

  int numValues;
  in.read((char *)&numValues, sizeof(numValues));

  if (numValues <= 0) {
    return false;
  }

  xf.resize(numValues);
  in.read((char *)xf.data(), sizeof(xf[0]) * xf.size());

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

  auto bbox = gridHandle.gridMetaData()->indexBBox();
  box3f volbounds({(float)bbox.min()[0], (float)bbox.min()[1], (float)bbox.min()[2]},
                  {(float)bbox.max()[0], (float)bbox.max()[1], (float)bbox.max()[2]});

  Pipeline pl("ex01_let_there_be_voxels");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(fb);

  Camera cam;
  cam.viewAll(volbounds);

  struct {
    vec3f lower_left, horizontal, vertical;
  } screen;
  cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

  std::vector<vec4f> tfValues;

  if (g_appState.xfFile.empty()) {
    tfValues = std::vector<vec4f>({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    });
  } else {
    loadXF(tfValues);
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
  // transfunc
  parms.transfunc.valueRange = {0,1};
  parms.transfunc.size = (int)tfValues.size();
  parms.transfunc.values = tfValues.data();
  // camera
  parms.camera.org = cam.getPosition();
  parms.camera.dir_00 = screen.lower_left;
  parms.camera.dir_du = screen.horizontal / imgWidth;
  parms.camera.dir_dv = screen.vertical / imgHeight;
  // framebuffer
  parms.fbPointer   = fb.fbPointer;
  parms.fbDepth     = fb.fbDepth;
  parms.accumBuffer = fb.accumBuffer;
  // lighting
  parms.ambientColor = vec3f(1.f);
  parms.ambientRadiance = 1.f;
  // DRV
  parms.samplingRate = 2.f;
  parms.unitDistance = 1.0f;
  // set params:
  pl.setLaunchParams(&parms,sizeof(parms), std::alignment_of<LaunchParams>());
#endif

  // Render and present...
  // For default (PNG image) pipeline this
  // loop returns immediately
  do {
    pl.launch();
    pl.present();
  } while (pl.isRunning());

  return 0;
}

} // namespace ex01_let_there_be_voxels



