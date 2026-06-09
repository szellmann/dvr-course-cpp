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

// common
#include <dvr_course-common.cuh>
// ex07
#include "Params.h"

using namespace dvr_course;

// ========================================================
// device code for example 09: ANARI
// ========================================================
namespace ex09_anari {

extern "C" __constant__ LaunchParams optixLaunchParams;

// ========================================================
// evalTet() implementation using four plane tests
// ========================================================
using Plane = vec4f;

inline __device__ Plane makePlane(vec3f a, vec3f b, vec3f c)
{
  vec3f N = cross(b-a,c-a);
  return { N,dot(a,N) };
}

inline __device__ float evalPlane(Plane p, vec3f v)
{ return dot(v,p.xyz)-p.w; }

inline __device__ bool evalTet(float &value, vec3f P, const Tet &tet)
{
  vec3f va = vec3f(tet.v0)-P;
  vec3f vb = vec3f(tet.v1)-P;
  vec3f vc = vec3f(tet.v2)-P;
  vec3f vd = vec3f(tet.v3)-P;

  Plane pa = makePlane(vb,vd,vc);
  Plane pb = makePlane(va,vc,vd);
  Plane pc = makePlane(va,vd,vb);
  Plane pd = makePlane(va,vb,vc);

  float fa = evalPlane(pa,vec3f(0.f))/evalPlane(pa,va);
  if (fa < 0.f || fa > 1.f) return false;
  
  float fb = evalPlane(pb,vec3f(0.f))/evalPlane(pb,vb);
  if (fb < 0.f || fa > 1.f) return false;
  
  float fc = evalPlane(pc,vec3f(0.f))/evalPlane(pc,vc);
  if (fc < 0.f || fa > 1.f) return false;
  
  float fd = evalPlane(pd,vec3f(0.f))/evalPlane(pd,vd);
  if (fd < 0.f || fa > 1.f) return false;

  value = fa*tet.v0.w + fb*tet.v1.w + fc*tet.v2.w + fd*tet.v3.w;
  return true;
}

#ifdef RTCORE
struct VolumePRD {
  float value;
  unsigned primID;
};
#endif

// ========================================================
// sampleVolume function used in isect prog
// ========================================================
inline __device__ bool sampleVolume(const Volume &vol, vec3f pos, float &value)
{
  if (vol.type == Volume::TET) {
#ifdef RTCORE
    VolumePRD prd;
    prd.value = 0.f;
    prd.primID = ~0u;
    owl::Ray ray;
    ray.origin = owl::vec3f(pos.x,pos.y,pos.z);
    ray.direction = owl::vec3f(1.f);
    ray.tmin = ray.tmax = 0.f;
    owl::traceRay(vol.asTetMesh.handle,ray,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    if (prd.primID != ~0u) {
      value = prd.value;
      return true;
    }
#else
    // on non-RT hardware we resort to just linearly
    // iterating over all primitives (veeeryy slow...)
    for (unsigned i=0; i<vol.asTetMesh.numTets; ++i) {
      if (evalTet(value,pos,vol.asTetMesh.tets[i]))
        return true;
    }
#endif
  }
  return false;
}

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

inline __device__ vec4f postClassify(Transfunc tf, float v)
{
  v = (v - tf.valueRange.lower) / tf.valueRange.size();
  int idx = v*(tf.size);
  float frac = (v*tf.size)-idx;
  vec4f v1 = tf.values[clamp(idx,0,tf.size-1)];
  vec4f v2 = tf.values[clamp(idx+1,0,tf.size-1)];
  return v1*(1.f-frac)+v2*frac;
}

inline __device__ float woodcockTracking(const Ray &ray,
                                         Random &rnd,
                                         float majorant,
                                         int volumeID,
                                         //output:
                                         vec3f &albedo,
                                         float &transmittance)
{
  auto &lp = optixLaunchParams;

  transmittance = 1.f;

  float t=ray.tmin;

  while (1) {
    // In later chapters majorants will vary in space:
    if (majorant <= 0.f)
      break;

    t -= (logf(1.f - rnd()) / (majorant / lp.unitDistance));

    if (t > ray.tmax)
      break;

    vec3f P = ray.eval(t);

    float value{0.f};
    if (!sampleVolume(lp.volumes[volumeID], P, value))
      continue;

    vec4f sample = postClassify(lp.transfuncs[volumeID], value);
    float u = rnd();
    if (sample.w >= u * majorant) {
      albedo = vec3f(sample.x,sample.y,sample.z);
      transmittance = 0.f;
      return t;
    }
  }

  return INFINITY;
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

inline  __device__ vec4f over(const vec4f &A, const vec4f &B)
{
  return A + (1.f-A.w)*B;
}


// ========================================================
// OptiX Tetty geometry (only when using OWL!)
// ========================================================
#ifdef RTCORE
OPTIX_BOUNDS_PROGRAM(TetBounds)(const void *geomData,
                                owl::box3f &result, // mind the owl:: namespace!
                                int leafID)
{
  const TetMesh &self = *(const TetMesh *)geomData;
  const Tet &tet = self.tets[leafID];
  result = owl::box3f(1e20f,-1e20f);
  result.extend((const owl::vec3f &)tet.v0);
  result.extend((const owl::vec3f &)tet.v1);
  result.extend((const owl::vec3f &)tet.v2);
  result.extend((const owl::vec3f &)tet.v3);
}

OPTIX_INTERSECT_PROGRAM(TetIntersect)()
{
  const TetMesh &self = owl::getProgramData<TetMesh>();
  int leafID = optixGetPrimitiveIndex();
  owl::Ray ray(optixGetObjectRayOrigin(),
               optixGetObjectRayDirection(),
               optixGetRayTmin(),
               optixGetRayTmax());

  vec3f pos(ray.origin.x,ray.origin.y,ray.origin.z);
  float value{0.f};
  if (evalTet(value,pos,self.tets[leafID])) {
    if (optixReportIntersection(ray.tmin, 0)) {
      VolumePRD &prd = owl::getPRD<VolumePRD>();
      prd.value = value;
      prd.primID = leafID;
    }
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(TetClosestHit)()
{
  // empty
}
#endif

struct HitRec {
  enum HitType { Volume, None, };
  HitType hitType;
  float hitT;
  vec4f color;
  vec3f Ng;
};

inline __device__
HitRec worldIntersection(Ray ray, Random &rnd)
{
  auto &lp = optixLaunchParams;

  HitRec hitRec;
  hitRec.hitType = HitRec::None;
  hitRec.hitT = INFINITY;
  hitRec.color = 0.f;

  for (int i=0; i<lp.numVolumes; ++i) {
    float t0, t1;
    if (!boxTest(ray, lp.volumes[i].bounds, t0, t1))
      continue;

    const float majorant = 1.f;

    vec3f albedo = 0.f;
    float transmittance = 1.f;

    Ray traversalRay(ray);
    traversalRay.tmin = t0, traversalRay.tmax = t1;
    float t = woodcockTracking(traversalRay, rnd, majorant, i, albedo, transmittance);

    if (t < hitRec.hitT) {
      hitRec.hitType   = HitRec::Volume;
      hitRec.hitT      = t;
      hitRec.color.xyz = albedo * lp.ambientColor * lp.ambientRadiance;
      hitRec.color.w   = 1.f-transmittance;
      hitRec.Ng        = uniformSampleSphere(rnd(),rnd());
    }
  }

  return hitRec;
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

    HitRec hitRec = worldIntersection(aoRay, rnd);
    float t = hitRec.hitT;

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
// Ray gen prog (direct light path tracer)
// ========================================================
RAYGEN_PROGRAM(directLighting)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();
  const int pixelID = threadIndex.x + getLaunchDims().x * threadIndex.y;

  Random rnd(lp.accumID*launchDim.x*launchDim.y+(unsigned)threadIndex.x,
             (unsigned)threadIndex.y);

  Ray ray = generateRay(vec2f(threadIndex)+vec2f(.5f), rnd);

  HitRec hitRec = worldIntersection(ray,rnd);

  // if (hitRec.hitType != HitRec::None) {
  //   float aoV = 1.f-ambientOcclusion(ray.eval(hitRec.hitT), hitRec.Ng, rnd);
  //   hitRec.color.xyz *= aoV;

  //   Ray shadowRay;
  //   shadowRay.org = ray.eval(hitRec.hitT) + hitRec.Ng*1e-3f;
  //   shadowRay.dir = normalize(lp.directionalLight.dir);
  //   shadowRay.tmin = 0.f;
  //   shadowRay.tmax = INFINITY;
  //   HitRec shadowRec = worldIntersection(shadowRay, rnd);
  //   hitRec.color.xyz *= 1.f-shadowRec.color.w;
  // }

  vec4f finalColor = over(hitRec.color, lp.backgroundColor);

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(finalColor, lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}

} // namespace ex09_anari
