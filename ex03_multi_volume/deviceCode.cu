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
// device code for example 03: multi_volume
// ========================================================
namespace ex03_multi_volume {

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
  if (fabsf(dir.x) < 1e-5f) dir.x = 1e-5f;
  if (fabsf(dir.y) < 1e-5f) dir.y = 1e-5f;
  if (fabsf(dir.z) < 1e-5f) dir.z = 1e-5f;
  return Ray(org,dir,0.f,1e10f);
}

inline __device__ bool sampleVolume(const Volume &vol, vec3f pos, float &value)
{
  // sample nvdb volume:
  auto acc = vol.handle->getAccessor();
  nanovdb::math::Vec3<float> nvdbPos(pos.x,pos.y,pos.z);
  if (!vol.filterLinear) {
    auto smp = nanovdb::math::createSampler<0>(acc);
    value = smp(vol.handle->worldToIndexF(nvdbPos));
  } else {
    auto smp = nanovdb::math::createSampler<1>(acc);
    value = smp(vol.handle->worldToIndexF(nvdbPos));
  }
  return true;
}

inline __device__ vec4f postClassify(Transfunc tf, float v)
{
  v = (v - tf.valueRange.lower) / (tf.valueRange.upper - tf.valueRange.lower);
  int idx = v*(tf.size);
  float frac = (v*tf.size)-idx;
  vec4f v1 = tf.values[clamp(idx,0,tf.size-1)];
  vec4f v2 = tf.values[clamp(idx+1,0,tf.size-1)];
  return v1*frac+v2*(1.f-frac);
}

inline __device__ float woodcockTracking(const Ray &ray,
                                         Random &rnd,
                                         float majorant,
                                         int volumeID,
                                         //output:
                                         vec3f &albedo,
                                         float &extinction)
{
  auto &lp = optixLaunchParams;

  float t=ray.tmin;

  while (1) {
    // In later chapters majorants will vary in space:
    if (majorant <= 0.f)
      break;

    t -= (logf(1.f - rnd()) / (majorant / lp.unitDistance));

    if (t > ray.tmax)
      break;

    vec3f P = ray.org+ray.dir*t;

    float value{0.f};
    if (!sampleVolume(lp.volumes[volumeID], P, value))
      continue;

    vec4f sample = postClassify(lp.transfuncs[volumeID], value);
    float u = rnd();
    if (sample.w >= u * majorant) {
      albedo = vec3f(sample.x,sample.y,sample.z);
      extinction = sample.w;
      break;
    }
  }

  return fminf(t,ray.tmax);
}

// ========================================================
// Ray gen prog using Woodcock as volume "depth test"
// ========================================================
RAYGEN_PROGRAM(multiVolumeWoodcock)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  float hitT = INFINITY;
  vec3f color = 0.f;
  float alpha = 0.f;
  for (int i=0; i<lp.numVolumes; ++i) {
    float t0, t1;
    if (!boxTest(ray, lp.volumes[i].bounds, t0, t1))
      return;

    ray.tmin = t0, ray.tmax = t1;

    const float majorant = 1.f;

    vec3f albedo = 0.f;
    float extinction = 0.f;

    float t = woodcockTracking(ray, rnd, majorant, i, albedo, extinction);

    if (t < hitT) {
      color = albedo * lp.ambientColor * lp.ambientRadiance;
      alpha = extinction > 0.f ? 1.f : 0.f;
      hitT = t;
    }
  }

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}



// ========================================================
// Ray gen prog blending volume samples together
// ========================================================
RAYGEN_PROGRAM(blendingWoodcock)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  float t0, t1;
  // TODO: here all boxes must be the same..
  if (!boxTest(ray, lp.volumes[0].bounds, t0, t1))
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

    if (t > ray.tmax)
      break;

    vec3f P = ray.org+ray.dir*t;

    vec4f sample = 0.f;

    if (lp.blendMode == BLEND_MODE_MIX) {
      vec3f blendColor = 0.f;
      float maxAlpha = 0.f;
      for (int i=0; i<lp.numVolumes; ++i) {
        float value{0.f};
        if (!sampleVolume(lp.volumes[i], P, value))
          continue;

        vec4f c = postClassify(lp.transfuncs[i], value);
        blendColor += vec3f(c) * c.a;
        maxAlpha = fmaxf(maxAlpha,c.a);
      }
      blendColor /= maxAlpha;
      sample = vec4f(blendColor,maxAlpha);
    } else if (lp.blendMode == BLEND_MODE_MAX_ALPHA) {
      for (int i=0; i<lp.numVolumes; ++i) {
        float value{0.f};
        if (!sampleVolume(lp.volumes[i], P, value))
          continue;

        vec4f c = postClassify(lp.transfuncs[i], value);
        if (c.a > sample.a) {
          sample = c;
        }
      }
    }

    float u = rnd();
    if (sample.w >= u * majorant) {
      albedo = vec3f(sample.x,sample.y,sample.z);
      extinction = sample.w;
      break;
    }
  }

  vec3f color = albedo * lp.ambientColor * lp.ambientRadiance;
  float alpha = extinction > 0.f ? 1.f : 0.f;

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}

} // namespace ex03_multi_volume



