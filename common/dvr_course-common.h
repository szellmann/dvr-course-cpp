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
// Common header for use in *host* code
// ========================================================

#pragma once

// std
#include <iostream>
#include <string>
// ours
#include "fb.h"
#include "vecmath.h"

namespace dvr_course {
using namespace vecmath;


inline bool endsWith(const std::string &s, const std::string &suffix) {
  if (s.length() < suffix.length())
    return false;

  return s.substr(s.size()-suffix.size(),suffix.size()) == suffix;
}

} // dvr_course

#include "camera.h"
#include "pipeline.h"

