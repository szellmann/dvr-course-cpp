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

// ex04:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex04_lights_on::LaunchParams)

struct {
  std::string filepath;
  Transfunc transfunc;
  float unitDistance{1.f};
  // AO:
  int ambientSamples{1};
  float occlusionDistance{2.f};
  // setup
  bool ratioTrackingForShadows{true};
  bool ratioTrackingForAO{true};
} g_appState;

namespace ex04_lights_on {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void woodcockTrackingSS();
#endif

inline float deg2rad(float d) {
  return d*float(M_PI)/180.f;
}

inline vec3f toCartesian(const vec3f spherical) {
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
  return {x,y,z};
}

void printUsage() {
  fprintf(stderr, "%s", "Usage: ex04_lights_on file.nvdb\n");
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

  nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle;

  try {
    auto grid = nanovdb::io::readGrid(g_appState.filepath);
    auto hostbuffer = nanovdb::HostBuffer::create(grid.bufferSize());
    std::memcpy(hostbuffer.data(), grid.data(), grid.bufferSize());
    gridHandle = std::move(hostbuffer);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    printUsage();
    exit(-1);
  }

  Buffer deviceGrid(gridHandle.bufferSize(), (uint8_t *)gridHandle.data());

  auto boundsMin = gridHandle.gridMetaData()->worldBBox().min();
  auto boundsMax = gridHandle.gridMetaData()->worldBBox().max();
  box3f volbounds({(float)boundsMin[0], (float)boundsMin[1], (float)boundsMin[2]},
                  {(float)boundsMax[0], (float)boundsMax[1], (float)boundsMax[2]});

  Pipeline pl(argc, argv, "ex04_lights_on");

  Pipeline::RTConfig conf;
#ifdef RTCORE
  conf.ptxCode = ptxCode;
  conf.launchParamsDecl = launchParams_owl;
  conf.sizeOfLaunchParamsStruct = sizeof(LaunchParams);
#else
  conf.rayGens.push_back({"woodcockTrackingSS",woodcockTrackingSS});
#endif
  pl.initRT(conf);
  pl.setRayGen("woodcockTrackingSS");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.viewAll(volbounds);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    tf.valueRange = {gridHandle.grid<float>()->tree().root().minimum(),
                     gridHandle.grid<float>()->tree().root().maximum()};

    tf.valueRange.lower
      = fminf(tf.valueRange.lower, gridHandle.grid<float>()->tree().root().background());
    tf.valueRange.upper
      = fmaxf(tf.valueRange.upper, gridHandle.grid<float>()->tree().root().background());

    if (tf.valueRange.empty()) tf.valueRange = {0.f,1.f};
    tf.setLUT(std::vector<vec4f>({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    }));
    pl.setTransfunc(&tf);
  }

  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.001f, 5.f);
  pl.uiParam("Ambient samples", &g_appState.ambientSamples, 0, 1024);
  pl.uiParam("Occlusion distance", &g_appState.occlusionDistance, 0.1f, 64.f);

  vec2f lightDir(0,240);
  pl.uiParam("Light dir", &lightDir, vec2f(0.f), vec2f(360.f));

  float lightIntensity{1.f};
  pl.uiParam("Light intensity", &lightIntensity, 0.f, 32.f);

  pl.uiParam("Use ratio tracking for shadows", &g_appState.ratioTrackingForShadows);
  pl.uiParam("Use ratio tracking for ambient occlusion", &g_appState.ratioTrackingForAO);

  LaunchParams parms;

  // volume
  pl.launchParam("volume.handle", (RawPointer &)parms.volume.handle) = (nanovdb::NanoGrid<float> *)deviceGrid.data();
  pl.launchParam("volume.filterLinear", parms.volume.filterLinear) = true;
  pl.launchParam("volume.bounds", parms.volume.bounds) = volbounds;

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
    // update lighting params:
    pl.launchParam("ambientColor", parms.ambientColor) = vec3f(1.f);
    pl.launchParam("ambientRadiance", parms.ambientRadiance) = 1.f;
    pl.launchParam("ambientSamples", parms.ambientSamples) = g_appState.ambientSamples;
    pl.launchParam("occlusionDistance", parms.occlusionDistance) = g_appState.occlusionDistance;
    pl.launchParam("directionalLight.dir", parms.directionalLight.dir)
        = toCartesian({1.f,deg2rad(lightDir.x),deg2rad(lightDir.y)});
    pl.launchParam("directionalLight.intensity", parms.directionalLight.intensity) = lightIntensity;
    pl.launchParam("ratioTrackingForShadows", parms.ratioTrackingForShadows)
        = g_appState.ratioTrackingForShadows;
    pl.launchParam("ratioTrackingForAO", parms.ratioTrackingForAO)
        = g_appState.ratioTrackingForAO;
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

} // namespace ex04_lights_on



