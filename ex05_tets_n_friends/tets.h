// ======================================================================== //
// Copyright 2025-2026 Stefan Zellmann                                      //
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

// common
#include <vecmath.h>

using namespace vecmath;

namespace ex05_tets_n_friends {

struct Tet {
  vec4f v0, v1, v2, v3;

  inline __device__ bool sample(const vec3f P, float &value) {

  }

  inline __host__ __device__
  box3f getBounds() const {
    box3f bounds(
      {INFINITY,INFINITY,INFINITY},
      {-INFINITY,-INFINITY,-INFINITY}
    );

    bounds.extend(v0);
    bounds.extend(v1);
    bounds.extend(v2);
    bounds.extend(v3);

    return bounds;
  }
};

} // namespace ex05_tets_n_friends


