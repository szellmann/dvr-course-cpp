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

#include <cstdint>
#include "vecmath.h"

namespace dvr_course {

// import into "dvr_course":
using namespace vecmath;

struct Frame
{
  Frame() = default;
  Frame(int w, int h);
  ~Frame();

  void clear(const vec4f &rgba = vec4f(0.f), float depth = 1.f);

  uint32_t *fbPointer{nullptr};
  float    *fbDepth{nullptr};
  vec4f    *accumBuffer{nullptr};
  int       width{0}, height{0};
};

} // namespace dvr_course
