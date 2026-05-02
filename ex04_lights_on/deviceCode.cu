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
// ex04
#include "Params.h"

using namespace dvr_course;

// ========================================================
// device code for example 04: lights_on
// ========================================================
namespace ex04_lights_on {

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

  return fminf(t,ray.tmax);
}

inline __device__ vec3f cosineSampleHemisphere(float u1, float u2)
{
  float r = sqrtf(u1);
  float theta = float(M_PI*2.f) * u2;
  return { r*cosf(theta), r*sinf(theta), sqrtf(1.f-u1) };
}

inline __device__ vec3f uniformSampleSphere(float u1, float u2)
{
  float z = 1.f-2.f*u1;
  float r = sqrtf(fmaxf(0.f,1.f-z*z));
  float phi = float(M_PI*2.f) * u2;
  return { r*cosf(phi), r*sinf(phi), z };
}

inline __device__ float ambientOcclusion(vec3f hitPos, vec3f n, Random &rnd)
{
  auto &lp = optixLaunchParams;
 
  float ao = 0.f;
  float aoWeights = 0.f;
  for (int sample=0; sample<lp.ambientSamples; ++sample) {
    vec3f u, v, w = n;
    make_orthonormal_basis(u,v,w);
    vec3f sp = cosineSampleHemisphere(rnd(),rnd());
    vec3f dir = normalize(sp.x*u + sp.y*v + sp.z*w);

    Ray aoRay;
    aoRay.org = hitPos + n*1e-3f;
    aoRay.dir = dir;
    aoRay.tmin = 0.f;
    aoRay.tmax = lp.occlusionDistance;

    // Clip against bounding box here. While during tracking, sampleVolume()
    // may return false, we can't be sure the ray definitely left the volume;
    // only that at the sample position the volume is not defined, so clipping
    // has to happen here:
    float t0, t1;
    boxTest(aoRay, lp.volume.bounds, t0, t1);
    aoRay.tmax = fminf(aoRay.tmax, t1);

    vec3f albedo = 0.f;
    float extinction = 0.f;

    const float majorant = 1.f;

    float t = woodcockTracking(aoRay, rnd, majorant, albedo, extinction);

    float weight = fmaxf(0.f, dot(aoRay.dir,n));
    if (t < aoRay.tmax)
      ao += weight;
    aoWeights += weight;
  }

  if (aoWeights > 0.f)
    return ao/aoWeights;
  else
    return 0.f;
}

// ========================================================
// Main ray gen prog (woodcock tracking with single
// scattering)
// ========================================================
RAYGEN_PROGRAM(woodcockTrackingSS)()
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

  float t = woodcockTracking(ray, rnd, majorant, albedo, extinction);
  // use random sample as normal vector:
  vec3f N = uniformSampleSphere(rnd(),rnd());
  float aoV = 1.f-ambientOcclusion(ray.eval(t), N, rnd);
  vec3f color = albedo * lp.ambientColor * lp.ambientRadiance * aoV;
  float alpha = extinction > 0.f ? 1.f : 0.f;

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}

} // namespace ex04_lights_on



