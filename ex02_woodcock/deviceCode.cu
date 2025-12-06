// ======================================================================== //
// Copyright 2025-2025 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// nanovdb
#include <nanovdb/math/SampleFromVoxels.h>
// common
#include <dvr_course-common.cuh>
// ex01
#include "Params.h"

using namespace dvr_course;

// ========================================================
// device code for example 00: hello_dvr_course
// ========================================================
namespace ex02_woodcock {

extern "C" __constant__ LaunchParams optixLaunchParams;

// ========================================================
// Helpers
// ========================================================
inline  __device__ Ray generateRay(const vec2f screen, Random &rnd)
{
  auto &lp = optixLaunchParams;
  vec3f org = lp.camera.org;
  vec3f dir
    = lp.camera.dir_00
    + (screen.u+rnd()) * lp.camera.dir_du
    + (screen.v+rnd()) * lp.camera.dir_dv;
  dir = normalize(dir);
  if (fabs(dir.x) < 1e-5f) dir.x = 1e-5f;
  if (fabs(dir.y) < 1e-5f) dir.y = 1e-5f;
  if (fabs(dir.z) < 1e-5f) dir.z = 1e-5f;
  return Ray(org,dir,0.f,1e10f);
}

inline __device__ bool sampleVolume(const Volume &vol, vec3f pos, float &value)
{
  // simple homogeneous volume:
  auto acc = vol.handle->getAccessor();
  if (!vol.filterLinear) {
    auto smp = nanovdb::math::createSampler<0>(acc);
    value = smp(nanovdb::math::Vec3<float>(pos.x,pos.y,pos.z));
    return true;
  } else {
    auto smp = nanovdb::math::createSampler<1>(acc);
    value = smp(nanovdb::math::Vec3<float>(pos.x,pos.y,pos.z));
    return true;
  }
  return true;
}

inline __device__ vec4f postClassify(Transfunc tf, float v)
{
  v = (v - tf.valueRange.lower) / (tf.valueRange.upper - tf.valueRange.lower);
  int idx = v*tf.size;
  float frac = (v*tf.size)-idx;
  vec4f v1 = tf.values[idx];
  vec4f v2 = tf.values[idx+1];
  return v1*frac+v2*(1.f-frac);
}

// ========================================================
// Main ray gen prog (simple ray marcher)
// ========================================================
RAYGEN_PROGRAM(simpleRayMarcher)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  float t0, t1;
  if (!boxTest(ray, lp.volume.bounds, t0, t1))
    return;

  ray.tmin = t0, ray.tmax = t1;

  vec3f albedo = 0.f;
  float extinction = 0.f;

  const float majorant = 1.f;

  float t=ray.tmin;

  while (1) {
    // In later chapters majorants will vary in space:
    if (majorant <= 0.f)
      break;

    t -= (logf(1.f - rnd()) / (majorant / lp.unitDistance));

    if (t >= ray.tmax)
      break;

    vec3f P = ray.org+ray.dir*t;

    float value{0.f};
    if (!sampleVolume(lp.volume, P, value))
      continue;

    vec4f sample = postClassify(lp.transfunc, value);
    float u = rnd();
    if (sample.w >= u * majorant) {
      albedo = vec3f(sample.x,sample.y,sample.z);
      extinction = sample.w;
      break;
    }
  }

  vec3f color = albedo;
  float alpha = extinction;

  color.x = linear_to_srgb(color.x);
  color.y = linear_to_srgb(color.y);
  color.z = linear_to_srgb(color.z);
  lp.accumBuffer[pixelID] = vec4f(color,alpha);
  lp.fbPointer[pixelID] = make_rgba(vec4f(color,alpha));
}

} // namespace ex02_woodcock



