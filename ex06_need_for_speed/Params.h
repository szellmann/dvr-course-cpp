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

// common
#include <vecmath.h>

using namespace vecmath;

// ========================================================
// structs with trivial layout, no default init, etc.
// to safely cross host/device borders
// ========================================================
namespace ex06_need_for_speed {

struct Tet { vec4f v0, v1, v2, v3; };
struct TetMesh {
#ifdef RTCORE
  OptixTraversableHandle handle;
#endif
  Tet *tets;
  int numTets;
};

// ========================================================
// The volume now also contains a uniform grid for
// traversal with DDA3. valueRanges (per macrocell) are set
// up once at the beginning; majorants (also per mc) are
// updated whenever the RGBA transfer function changes
// (we interpret alpha as majorant extinction)
// ========================================================
struct Volume {
  // currently our volume only supports one field single type; in later
  // examples we will use a tagged union to distinguish between different
  // spatial fields
  enum { TET, } type;
  TetMesh asTetMesh;
  box3f bounds;
  box1f dataRange;
  struct {
    box1f *valueRanges;
    float *majorants;
    vec3i  dims;
    box3f  worldBounds;
  } grid;
};

struct Transfunc {
  box1f  valueRange;
  vec4f *values;
  int size;
};

struct LaunchParams {
  // volume:
  Volume volume;

  // transfunc:
  Transfunc transfunc;

  // camera:
  struct {
    vec3f org;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;

  // framebuffer:
  uint32_t *fbPointer;
  float    *fbDepth;
  vec4f    *accumBuffer;
  int       accumID;

  // lighting:
  vec3f ambientColor;
  float ambientRadiance;

  // DVR:
  float unitDistance;
};

} // namespace ex06_need_for_speed


