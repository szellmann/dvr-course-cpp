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
#include <memory>
#include <string>
// ours
#include "camera.h"
#include "fb.h"
#include "transfunc.h"

#ifndef RTCORE
# define DECL_LAUNCH_PARAMS(T) extern "C" T optixLaunchParams = {};
# define SET_LAUNCH_PARAMS(p) optixLaunchParams = (p);
#else
# define DECL_LAUNCH_PARAMS(T)
# define SET_LAUNCH_PARAMS(p)
#endif

typedef void *RawPointer;

struct _OWLContext;
struct _OWLModule;
struct _OWLVarDecl;
typedef _OWLContext *OWLContext;
typedef struct _OWLLaunchParams *OWLLaunchParams, *OWLParams, *OWLGlobals;
typedef _OWLModule *OWLModule;
typedef _OWLVarDecl OWLVarDecl;

#ifdef OWL_IS_FAKE
#include <owl/fakeOwl/fake/owl.h>
typedef fake::TraversableHandle OptixTraversableHandle;
#else
typedef unsigned long long OptixTraversableHandle;
#endif

// ========================================================
// Common render pipeline class for DVR
// ========================================================
namespace dvr_course {

struct Pipeline {

  Pipeline(std::string name = "dvr-course-cpp");
  Pipeline(int argc, char *argv[], std::string name = "dvr-course-cpp");
  ~Pipeline();

  // no window and event handling when compiled in interactive mode:
  void setHeadless(bool headless);

  // for use in headless mode; need to manually notify the pipeline
  // that camera or frame were updated:
  void markCameraUpdate();
  void markFrameUpdate();
  void markTransfuncUpdate();

#ifdef RTCORE

  // for use with RTCORE (load from module)

  struct RTConfig {
    const char *ptxCode;
    OWLVarDecl *launchParamsDecl;
    size_t      sizeOfLaunchParamsStruct;
  };

  // get OWL context
  OWLContext owlContext();

  // get OWL module
  OWLModule owlModule();

  // get OWL params
  OWLParams owlLaunchParams();
#else
  // for use with non-RTCORE (set ray-gen as function pointer)

  struct RayGen {
    std::string name;
    std::function<void()> func;
  };

  struct RTConfig {
    std::vector<RayGen> rayGens;
  };
#endif

  void initRT(RTConfig rtConfig);
  void setRayGen(const char *name);

  //   launch-params
#define DECL_LAUNCH_PARM_FUNC(T) T &launchParam(std::string name, T &value);

  DECL_LAUNCH_PARM_FUNC(bool)
  DECL_LAUNCH_PARM_FUNC(int)
  DECL_LAUNCH_PARM_FUNC(vec2i)
  DECL_LAUNCH_PARM_FUNC(vec3i)
  DECL_LAUNCH_PARM_FUNC(vec4i)
  DECL_LAUNCH_PARM_FUNC(float)
  DECL_LAUNCH_PARM_FUNC(vec2f)
  DECL_LAUNCH_PARM_FUNC(vec3f)
  DECL_LAUNCH_PARM_FUNC(vec4f)
  DECL_LAUNCH_PARM_FUNC(box1f)
  DECL_LAUNCH_PARM_FUNC(box3f)
  DECL_LAUNCH_PARM_FUNC(RawPointer)
#ifdef RTCORE
  DECL_LAUNCH_PARM_FUNC(OptixTraversableHandle)
#endif

  // Frame
  void setFrame(Frame *f);
  Frame *fb{nullptr};

  int frameID{0};

  // Camera
  void setCamera(Camera *cam);
  Camera *camera{nullptr};

  // Transfunc
  void setTransfunc(Transfunc *tf, int index=0);
  Transfunc *getTransfunc(int index=0) const;
  bool transfuncValid(int index=0) const;

  // Histogram (displayed as background in the TFE)
  void setHistogram(const std::vector<int> &hist, int index=0);

  // UI params
  struct UIConfig {
    UIConfig() : hint(None) {}
    enum Hint { Color=0x1, None=0x0, };
    Hint hint;
    std::vector<std::string> alternativeNames;
  };

  // Boolean UI param, renders as a checkbox
  void uiParam(std::string name,
               bool *b,
               const UIConfig &config = {});

  // Integer UI param, renders as a slider
  void uiParam(std::string name,
               int *i,
               int mini,
               int maxi,
               const UIConfig &config = {});

  // Float32 UI param, renders as a slider
  void uiParam(std::string name,
               float *f,
               float minf,
               float maxf,
               const UIConfig &config = {});

  // Float32 Vec2 UI param, renders as sliders with labels "_X" and "_Y";
  // if config.alternativeNames is set, these will be used as labels
  void uiParam(std::string name,
               vec2f *v,
               vec2f minv,
               vec2f maxv,
               const UIConfig &config = {});

  // Integer Vec3 UI param, renders as int inputs with labels "_X", "_Y", and "_Z";
  // if config.alternativeNames is set, these will be used as labels
  void uiParam(std::string name,
               vec3i *v,
               vec3i minv,
               vec3i maxv,
               const UIConfig &config = {});

  // Integer Vec3 UI param, renders as sliders with labels "_X", "_Y", and "_Z";
  // if config.alternativeNames is set, these will be used as labels
  void uiParam(std::string name,
               vec3f *v,
               vec3f minv,
               vec3f maxv,
               const UIConfig &config = {});

  // Options UI param, renders as a drop down list
  void uiParam(std::string name,
               const std::vector<std::string> &options,
               int *o,
               const UIConfig &config = {});

  // Functional UI param, renders as a button which upon press executes the functional
  void uiParam(std::string name,
               std::function<void(void)> f,
               const UIConfig &config = {});

  // Interface
  bool isValid() const { return fb != nullptr && camera != nullptr; }
  bool isRunning();
  void launch();
  void present() const;
  void resetAccumulation();
  // frameless launch:
  void launch2D(const vec2i launchDims);

  // Events
  typedef std::function<void(char)> KeyDownHandler;
  void setKeyDownHandler(KeyDownHandler kdh);

  typedef std::function<void(const Transfunc *,int)> TransfuncUpdateHandler;
  void setTransfuncUpdateHandler(TransfuncUpdateHandler tuh);

  // Private impl - the declaration is public for compatibility with CUDA
  // device lambdas; don't create objects outside of this class!
  struct Impl;
 private:

  std::unique_ptr<Impl> impl;

  bool running{false};
};

} // dvr_course



