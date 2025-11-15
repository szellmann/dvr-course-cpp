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

#include <dvr_course-common.cuh>

using namespace vecmath;

// ========================================================
// structs with trivial layout, no default init, etc.
// to safely cross host/device borders
// ========================================================
namespace ex00_hello_dvr_course {

struct Transfunc {
  box1f  valueRange;
  vec4f *values;
  int size;
};

struct LaunchParams {
  // volume:
  struct {
    int handle;
    box3f bounds;
  } volume;

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
  float samplingRate;
  float unitDistance;
};

} // namespace ex00_hello_dvr_course


