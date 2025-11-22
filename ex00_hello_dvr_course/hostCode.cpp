// Header with common resources; .h: host, .cuh: device
#include <dvr_course-common.h>

// ex00:
#include "Params.h"

DECL_LAUNCH_PARAMS(ex00_hello_dvr_course::LaunchParams)

namespace ex00_hello_dvr_course {
#ifndef RTCORE
extern void simpleRayMarcher();
#endif

extern "C" int main(int argc, char *argv[]) {

  // common namespace for helper classes:
  // Camera, FB, wrappers for RTX execution model, etc. etc.
  using namespace dvr_course;

  Pipeline pl("ex00_hello_dvr_course");

  int imgWidth=512, imgHeight=512;
  Frame fb(imgWidth, imgHeight);
  pl.setFrame(fb);

  Camera cam;
  cam.setOrientation(vec3f(0,0,-2.5),
                     vec3f(0,0,0),
                     vec3f(0,1,0),
                     90.f*M_PI/180.f);

  struct {
    vec3f lower_left, horizontal, vertical;
  } screen;
  cam.getScreen(screen.lower_left,screen.horizontal,screen.vertical);

  std::vector<vec4f> tfValues({
    {0.f,0.f,1.f,0.1f },
    {0.f,1.f,0.f,0.1f }
  });

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
  parms.volume.bounds = box3f({-1,-1,-1},{1,1,1});
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
  parms.unitDistance = 0.1f;
  // set params:
  SET_LAUNCH_PARAMS(parms);
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

} // namespace ex00_hello_dvr_course



