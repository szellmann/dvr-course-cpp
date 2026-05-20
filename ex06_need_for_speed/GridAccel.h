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

#pragma once

#include <buffer.h>
#include <for_each.h>
#include <vecmath.h>

namespace ex06_need_for_speed {

using namespace vecmath;

// ========================================================
// Grid type
// ========================================================

struct GridAccel
{
  box1f *valueRanges()
  { return m_valueRanges.data(); }

  float *majorants()
  { return m_majorants.data(); }

 private:
  // min/max ranges per grid cell
  dvr_course::Buffer<box1f> m_valueRanges;

  // majorants per grid cell
  dvr_course::Buffer<float> m_majorants;

  // number of grid cells in x/y/z
  vec3i m_dims{0};

  // world-space bounding box of grid
  box3f m_worldBounds;
};

// ========================================================
// Grid helpers
// ========================================================
inline __device__ vec3i projectToGrid(
    const vec3f &V, const vec3i &dims, const box3f &worldBounds)
{
  const vec3f V01 = (V-worldBounds.lower)/(worldBounds.upper-worldBounds.lower);
  const vec3f Vscale = V01*vec3f(dims.x,dims.y,dims.z);
  return clamp(vec3i(Vscale.x,Vscale.y,Vscale.z),vec3i(0),dims-vec3i(1));
}

// ========================================================
// DDA3 grid traversal
// ========================================================
typedef vec3i GridIterationState;

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

  vec3i cellID = projectToGrid(ray.org,gridDims,modelBounds);

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
    const float t1 = min(reduce_min(tnext),ray.tmax);
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

} // namespace ex06_need_for_speed



