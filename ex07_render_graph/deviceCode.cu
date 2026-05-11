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
// ex07
#include "Params.h"

using namespace dvr_course;

// ========================================================
// device code for example 07: render_graph
// ========================================================
namespace ex07_render_graph {

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
  if (vol.type == Volume::NVDB) {
    // sample nvdb volume:
    auto acc = vol.asNvdb.handle->getAccessor();
    nanovdb::math::Vec3<float> nvdbPos(pos.x,pos.y,pos.z);
    if (!vol.asNvdb.filterLinear) {
      auto smp = nanovdb::math::createSampler<0>(acc);
      value = smp(vol.asNvdb.handle->worldToIndexF(nvdbPos));
    } else {
      auto smp = nanovdb::math::createSampler<1>(acc);
      value = smp(vol.asNvdb.handle->worldToIndexF(nvdbPos));
    }
    return true;
  }

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
                                         float &transmission)
{
  auto &lp = optixLaunchParams;

  transmission = 1.f;

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
      transmission = 0.f;
      return t;
    }
  }

  return INFINITY;
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

// ========================================================
// Triangle geometry
// ========================================================
#ifdef RTCORE
struct TrianglePRD {
  unsigned primID;
  float t;
  vec3f Ng;
};

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMeshClosestHit)()
{
  TrianglePRD& prd = owl::getPRD<TrianglePRD>();
  const TriangleMesh& self = owl::getProgramData<TriangleMesh>();
  prd.primID = optixGetPrimitiveIndex();
  prd.t = optixGetRayTmax();
  const vec3i index  = self.indices[prd.primID];
  const vec3f& v1    = self.vertices[index.x];
  const vec3f& v2    = self.vertices[index.y];
  const vec3f& v3    = self.vertices[index.z];
  prd.Ng = normalize(cross(v2-v1,v3-v1));
}
#endif
// ========================================================
// Sphere intersection, used for the makeshift
// traversal structure
// ========================================================
inline __device__
bool intersectSphere(const Ray &ray, float radius, float &tnear, float &tfar) {
  float A = dot(ray.dir,ray.dir);
  float B = dot(ray.dir,ray.org) * 2.f;
  float C = dot(ray.org,ray.org) - radius*radius;

  float d = B*B - 4.f*A*C;
  if (d < 0.f) return false;

  d = sqrtf(d);

  float q = B < 0.f ? -0.5f * (B-d) : -0.5f * (B+d);

  float t1 = q/A;
  float t2 = C/q;

  tnear = fminf(t1,t2);
  tfar  = fmaxf(t1,t2);
  return true;
}

// ========================================================
// Ray gen prog (woodcock tracking, A+E)
// ========================================================
RAYGEN_PROGRAM(woodcockTrackingAE)()
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

#ifdef RTCORE
  TrianglePRD prd;
  prd.primID = ~0u;
  owl::Ray owlRay;
  owlRay.origin = owl::vec3f(ray.org.x,ray.org.y,ray.org.z);
  owlRay.direction = owl::vec3f(ray.dir.x,ray.dir.y,ray.dir.z);
  owlRay.tmin = 0.f;
  owlRay.tmax = INFINITY;
  owl::traceRay(lp.triangleTLAS,owlRay,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
  if (prd.primID != ~0u) {
    if (prd.t < hitT) {
      color = vec3f(0.8f)*fmaxf(0.f,dot(prd.Ng,-ray.dir));
      alpha = 1.f;
      hitT = prd.t;
    }
  }
#endif

  for (int i=0; i<lp.numVolumes; ++i) {
    float t0, t1;
    if (!boxTest(ray, lp.volumes[i].bounds, t0, t1))
      continue;

    ray.tmin = t0, ray.tmax = t1;

    const float majorant = 1.f;

    vec3f albedo = 0.f;
    float transmission = 1.f;

    float t = woodcockTracking(ray, rnd, majorant, i, albedo, transmission);

    if (t < hitT) {
      color = albedo * lp.ambientColor * lp.ambientRadiance;
      alpha = 1.f-transmission;
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

} // namespace ex07_render_graph



