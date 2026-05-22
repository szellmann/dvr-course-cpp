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
// ex06
#include "Params.h"

using namespace dvr_course;

// ========================================================
// device code for example 06: need_for_speed
// ========================================================
namespace ex06_need_for_speed {

extern "C" __constant__ LaunchParams optixLaunchParams;

// ========================================================
// Grid helpers
// ========================================================
inline __device__ vec3i projectToGrid(
    const vec3f &V, const box3f &worldBounds, const vec3i &dims)
{
  const vec3f V01 = (V-worldBounds.lower)/(worldBounds.upper-worldBounds.lower);
  const vec3f Vscale = V01*vec3f(dims.x,dims.y,dims.z);
  return clamp(vec3i(Vscale.x,Vscale.y,Vscale.z),vec3i(0),dims-vec3i(1));
}

inline __device__ void rasterizeBox(const box4f &box)
{
  auto &lp = optixLaunchParams;
  const box3f worldBounds = lp.volume.grid.worldBounds;
  const vec3i dims = lp.volume.grid.dims;

  vec3i lo = projectToGrid(box.lower.xyz,worldBounds,dims);
  vec3i up = projectToGrid(box.upper.xyz,worldBounds,dims);

  for (int z=lo.z; z<=up.z; ++z) {
    for (int y=lo.y; y<=up.y; ++y) {
      for (int x=lo.x; x<=up.x; ++x) {
        auto index = linearIndex(vec3i(x,y,z),dims);
        box1f &valueRange = lp.volume.grid.valueRanges[index];
        atomicMin(&valueRange.lower,box.lower.w);
        atomicMax(&valueRange.upper,box.upper.w);
      }
    }
  }
}

// ========================================================
// "Ray gen prog" to build grid
// ========================================================
RAYGEN_PROGRAM(buildGridAccel)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();

  size_t tetID = threadIndex.x + size_t(launchDim.x) * threadIndex.y;
  if (tetID >= lp.volume.numTets) return;
  const Tet &tet = lp.volume.tets[tetID];
  box4f tetBounds(INFINITY,-INFINITY);
  tetBounds.extend(tet.v0);
  tetBounds.extend(tet.v1);
  tetBounds.extend(tet.v2);
  tetBounds.extend(tet.v3);
  rasterizeBox(tetBounds);
}

// ========================================================
// "Ray gen prog" to update majorants on-the-fly
// ========================================================
RAYGEN_PROGRAM(updateMajorantDensities)()
{
  auto &lp = optixLaunchParams;
  const vec2i threadIndex = getLaunchIndex();
  const vec2i launchDim = getLaunchDims();

  size_t mcID = threadIndex.x + size_t(launchDim.x) * threadIndex.y;

  box1f &valueRange = lp.volume.grid.valueRanges[mcID];

  if (valueRange.upper < valueRange.lower) { // is cell empty?
    lp.volume.grid.majorants[mcID] = 0.f;
    return;
  }

  auto &tf = lp.transfunc;

  valueRange.lower -= tf.valueRange.lower;
  valueRange.lower /= tf.valueRange.size();
  valueRange.upper -= tf.valueRange.lower;
  valueRange.upper /= tf.valueRange.size();

  int numValues = tf.size;

  // compute min/max indices into the transfer function array wrt.
  // the values this cell contains:
  int lo = clamp(
      int(valueRange.lower * (numValues-1)), 0, numValues-1);
  int hi = clamp(
      int(valueRange.upper * (numValues-1)) + 1, 0, numValues-1);

  // iterate over the tf array, the maximum opacity is our macrocell
  // majorant:
  float maxOpacity = 0.f;
  for (int i=lo; i<=hi; ++i) {
    maxOpacity = fmaxf(maxOpacity,tf.values[i].a);
  }
  lp.volume.grid.majorants[mcID] = maxOpacity;
}

// ========================================================
// DDA3 implementation
// ========================================================
template <typename Ray, typename Func>
inline __device__ void dda3(
    Ray ray, const vec3i &gridDims, const box3f &modelBounds, const Func  &func)
{
  // move ray so tmin becomes 0
  const float ray_tmin = ray.tmin;
  ray.org = ray.org + ray.tmin * ray.dir;
  ray.tmin = 0.f;
  ray.tmax -= ray_tmin;

  const vec3f rcp_dir = 1.f / ray.dir;

  const vec3f lo = (modelBounds.lower - ray.org) * rcp_dir;
  const vec3f hi = (modelBounds.upper - ray.org) * rcp_dir;

  vec3f tnear = min(lo,hi);
  const vec3f tfar = max(lo,hi);

  if (ray.dir.x == 0.f) {
    tnear.x = INFINITY;
  }
  if (ray.dir.y == 0.f) {
    tnear.y = INFINITY;
  }
  if (ray.dir.z == 0.f) {
    tnear.z = INFINITY;
  }

  vec3i cellID = projectToGrid(ray.org,modelBounds,gridDims);

  // Distance in world space to get from cell to cell
  const vec3f dist(max(vec3f(0.f),(tfar-tnear)/vec3f(gridDims)));

  // Cell increment
  const vec3i step = {
    ray.dir.x > 0.f ? 1 : -1,
    ray.dir.y > 0.f ? 1 : -1,
    ray.dir.z > 0.f ? 1 : -1
  };

  // Stop when we reach grid borders
  const vec3i stop = {
    ray.dir.x > 0.f ? gridDims.x : -1,
    ray.dir.y > 0.f ? gridDims.y : -1,
    ray.dir.z > 0.f ? gridDims.z : -1
  };

  // Increment in world space
  vec3f tnext = {
    ray.dir.x > 0.f ? tnear.x + float(cellID.x+1) * dist.x
                    : tnear.x + float(gridDims.x-cellID.x) * dist.x,
    ray.dir.y > 0.f ? tnear.y + float(cellID.y+1) * dist.y
                    : tnear.y + float(gridDims.y-cellID.y) * dist.y,
    ray.dir.z > 0.f ? tnear.z + float(cellID.z+1) * dist.z
                    : tnear.z + float(gridDims.z-cellID.z) * dist.z
  };


  float t0 = 0.f;

  while (1) { // loop over grid cells
    const float t1 = fminf(reduce_min(tnext),ray.tmax);
    if (!func(linearIndex(cellID,gridDims),ray_tmin+t0,ray_tmin+t1))
      return;

    const float t_closest = reduce_min(tnext);
    if (tnext.x == t_closest) {
      tnext.x += dist.x;
      cellID.x += step.x;
      if (cellID.x==stop.x) {
        break;
      }
    }
    if (tnext.y == t_closest) {
      tnext.y += dist.y;
      cellID.y += step.y;
      if (cellID.y==stop.y) {
        break;
      }
    }
    if (tnext.z == t_closest) {
      tnext.z += dist.z;
      cellID.z += step.z;
      if (cellID.z==stop.z) {
        break;
      }
    }
    t0 = t1;
  }
}

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
struct PRD {
  float value;
  unsigned primID;
};
#endif

// ========================================================
// sampleVolume function used in isect prog
// ========================================================
inline __device__ bool sampleVolume(const Volume &vol, vec3f pos, float &value)
{
#ifdef RTCORE
  PRD prd;
  prd.value = 0.f;
  prd.primID = ~0u;
  owl::Ray ray;
  ray.origin = owl::vec3f(pos.x,pos.y,pos.z);
  ray.direction = owl::vec3f(1.f);
  ray.tmin = ray.tmax = 0.f;
  owl::traceRay(vol.handle,ray,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
  if (prd.primID != ~0u) {
    value = prd.value;
    return true;
  }
#else
  // on non-RT hardware we resort to just linearly
  // iterating over all primitives (veeeryy slow...)
  for (unsigned i=0; i<vol.numTets; ++i) {
    if (evalTet(value,pos,vol.tets[i]))
      return true;
  }
#endif
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
    if (!sampleVolume(lp.volume, P, value))
      continue;

    vec4f sample = postClassify(lp.transfunc, value);
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
      PRD &prd = owl::getPRD<PRD>();
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

  vec3f albedo = 0.f;
  float transmission = 1.f;
  auto integrationFunc = [&](int mcID, float t0, float t1) {
    ray.tmin = t0, ray.tmax = t1;
    const float majorant = lp.volume.grid.majorants[mcID];
    float t = woodcockTracking(ray, rnd, majorant, albedo, transmission);
    return t >= ray.tmax; // if true, no "hit" was found -> traverse to next macrocell
  };

  Ray traversalRay(ray);
  dda3(traversalRay,lp.volume.grid.dims,lp.volume.grid.worldBounds,integrationFunc);

  vec3f color = albedo * lp.ambientColor * lp.ambientRadiance;
  float alpha = 1.f-transmission;

  float accum = 1.f/(lp.accumID+1);
  lp.accumBuffer[pixelID] = lerp(vec4f(color,alpha), lp.accumBuffer[pixelID], accum);

  vec4f accumColor = lp.accumBuffer[pixelID];
  accumColor.r = linear_to_srgb(accumColor.r);
  accumColor.g = linear_to_srgb(accumColor.g);
  accumColor.b = linear_to_srgb(accumColor.b);
  lp.fbPointer[pixelID] = make_rgba(accumColor);
}

} // namespace ex06_need_for_speed



