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

// ========================================================
// Common header for use in *device* code
// ========================================================

#pragma once

#include "vecmath.h"

namespace dvr_course {
using namespace vecmath;
} // dvr_course

#ifndef __CUDACC__
#define __constant__
#define __shared__
#endif


#ifndef RTCORE
namespace dvr_course {
const vec2i getLaunchIndex(void);
const vec2i getLaunchDims(void);
} // dvr_course
#define RAYGEN_PROGRAM(name) void name

inline __device__ float linear_to_srgb(float x) {
  if (x <= 0.0031308f) {
    return 12.92f * x;
  }
  return 1.055f * powf(x, 1.f/2.4f) - 0.055f;
}

inline __device__ uint32_t make_8bit(const float f)
{
  return fminf(255,fmaxf(0,int(f*256.f)));
}

inline __device__ uint32_t make_rgba(const vecmath::vec3f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (0xffU << 24);
}

inline __device__ uint32_t make_rgba(const vecmath::vec4f color)
{
  return
    (make_8bit(color.x) << 0) +
    (make_8bit(color.y) << 8) +
    (make_8bit(color.z) << 16) +
    (make_8bit(color.w) << 24);
}

#endif


