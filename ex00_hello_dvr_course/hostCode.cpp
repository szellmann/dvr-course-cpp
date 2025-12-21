// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex00:
#include "Params.h"
#ifdef RTCORE
#include "Params-owl.h"
#endif

// common namespace for helper classes:
// Camera, FB, wrappers for RTX execution model, etc. etc.
using namespace dvr_course;

DECL_LAUNCH_PARAMS(ex00_hello_dvr_course::LaunchParams)

struct {
  Transfunc transfunc;
  float samplingRate;
  float unitDistance;
} g_appState;

namespace ex00_hello_dvr_course {
#ifdef RTCORE
extern "C" char ptxCode[];
#else
extern void simpleRayMarcher();
#endif

extern "C" int main(int argc, char *argv[]) {

  Pipeline pl(argc, argv, "ex00_hello_dvr_course");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(&fb);

  Camera cam;
  cam.setOrientation(vec3f(0,0,-2.5),
                     vec3f(0,0,0),
                     vec3f(0,1,0),
                     90.f*M_PI/180.f);
  pl.setCamera(&cam);

  if (!pl.transfuncValid()) {
    auto &tf = g_appState.transfunc;
    std::vector<vec4f> tfValues({
      {0.f,0.f,1.f,0.1f },
      {0.f,1.f,0.f,0.1f }
    });
    tf.valueRange = {0.f,1.f};
    tf.setLUT(tfValues);
    pl.setTransfunc(&tf);
  }

  g_appState.samplingRate = 2.f;
  pl.uiParam("Sampling rate", &g_appState.samplingRate, 0.001f, 5.f);

  g_appState.unitDistance = 0.1f;
  pl.uiParam("Unit distance", &g_appState.unitDistance, 0.001f, 5.f);

#ifdef RTCORE
  pl.setRayGen(ptxCode, "simpleRayMarcher");
  pl.setLaunchParamsDecl(launchParams_owl, sizeof(LaunchParams));
#else
  pl.setRayGen(simpleRayMarcher);
#endif

  LaunchParams parms;

  // volume
  pl.launchParam("volume.bounds", parms.volume.bounds) = box3f({-1,-1,-1},{1,1,1});
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
    pl.launchParam("samplingRate", parms.samplingRate) = g_appState.samplingRate;
    pl.launchParam("unitDistance", parms.unitDistance) = g_appState.unitDistance;
    // update accum:
    pl.launchParam("accumID", parms.accumID) = pl.frameID;

    // set params:
    SET_LAUNCH_PARAMS(parms);

    // render:
    pl.launch();
    pl.present();
  } while (pl.isRunning());

  return 0;
}

} // namespace ex00_hello_dvr_course



