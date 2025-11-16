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

// std
#include <functional>
#include <string>
// ours
#include "fb.h"

// ========================================================
// Common render pipeline class for DVR
// ========================================================
namespace dvr_course {

struct Pipeline {

  Pipeline(std::string name = "dvr-course-cpp") : name(name) {}
  std::string name = "dvr-course-cpp";

#ifdef RTCORE
  // for use with RTCORE (load from module)

  //   ray-gen
  void setRayGen(const char *name);

  //   launch-params
  std::vector<Param> launchParams;
  void setLaunchParams(const std::vector<OWLVarDecl> &params)
  { launchParams = params; }

#else
  // for use with non-RTCORE (set as function pointer)

  //   ray-gen
  void setRayGen(const std::function<void()> &f)
  { func = f; }

  std::function<void()> func;

  //   launch-params
  void setLaunchParams(const void *ptr, size_t len, size_t a);
#endif

  bool isValid() const { return fb != nullptr; }

  // Frame
  void setFrame(Frame &f) { fb = &f; }
  Frame *fb{nullptr};

  // Interface
  bool isRunning() const { return false; }
  void launch() const;
  void present() const;
};

} // dvr_course



