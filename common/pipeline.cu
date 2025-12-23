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

// std
#include <fstream>
#include <map>
#ifdef INTERACTIVE
# include <SDL3/SDL.h>
# define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
# include "imgui_impl_sdl3.h"
# include "imgui_impl_sdlrenderer3.h"
# include "tfe.h"
#else
// stb_image
# define STB_IMAGE_WRITE_IMPLEMENTATION
# include "stb/stb_image_write.h"
#endif
// ours
#include "pipeline.h"
#include "thread_pool.h"
#include "for_each.h"
#include "dvr_course-common.h"
#include "dvr_course-common.cuh"

#ifndef RTCORE
static thread_local vecmath::vec2i launchIndex;
static thread_local vecmath::vec2i launchDims;
#endif

#ifdef RTCORE
// dummy ray-gen data (we pass all data through launch parms!)
struct RayGenData {};
OWLVarDecl rayGenVars[]
= {
   { nullptr /* sentinel to mark end of list */ }
};

// map C++ to owl types:
template<typename T>
OWLDataType mapOwlType(const T &t) { return OWL_USER_TYPE(t); }
OWLDataType mapOwlType(RawPointer) { return OWL_RAW_POINTER; }
OWLDataType mapOwlType(bool) { return OWL_BOOL; }
OWLDataType mapOwlType(float) { return OWL_FLOAT; }
OWLDataType mapOwlType(vecmath::vec2f) { return OWL_FLOAT2; }
OWLDataType mapOwlType(vecmath::vec3f) { return OWL_FLOAT3; }
OWLDataType mapOwlType(vecmath::vec4f) { return OWL_FLOAT4; }
OWLDataType mapOwlType(int) { return OWL_INT; }
// ... TODO
#endif

namespace dvr_course {

#ifndef RTCORE
const vec2i getLaunchIndex(void)
{ return launchIndex; }

const vec2i getLaunchDims(void)
{ return launchDims; }

const bool debug(void) {
  return launchIndex.x == launchDims.x/2 && launchIndex.y == launchDims.y/2;
}
#endif

static bool loadXF(std::string xfFile, dvr_course::Transfunc &tf) {
  std::ifstream in(xfFile);

  if (!in.good()) {
    return false;
  }

  in.read((char *)&tf.opacity, sizeof(tf.opacity));
  in.read((char *)&tf.valueRange, sizeof(tf.valueRange));
  in.read((char *)&tf.relRange, sizeof(tf.relRange));

  int numValues;
  in.read((char *)&numValues, sizeof(numValues));

  if (numValues <= 0) {
    return false;
  }

  std::vector<vec4f> rgbaLUT(numValues);
  in.read((char *)rgbaLUT.data(), sizeof(rgbaLUT[0]) * rgbaLUT.size());
  tf.setLUT(rgbaLUT);

  return true;
}

void clearFramebuffer(const Frame *fb,
                      thread_pool &pool,
                      const vec4f &rgba = vec4f(0.f),
                      float depth = 0.f)
{
  int width = fb->width; int height = fb->height;
#ifdef RTCORE
  cuda::for_each(/*TODO: stream*/0, 0, width, 0, height,
#else
  parallel::for_each(pool, 0, width, 0, height,
#endif
    [=] __device__ (int x, int y) {
      int pixelID = x+y*width;
      if (fb->fbPointer) {
        fb->fbPointer[pixelID] = make_rgba(rgba);
      }

      if (fb->fbDepth) {
        fb->fbDepth[pixelID] = depth;
      }

      if (fb->accumBuffer) {
        fb->accumBuffer[pixelID] = vec4f(0.f);
      }
    });
}

struct Pipeline::Impl
{
  Impl(Pipeline *parent, std::string name) : parent(parent), name(name) {}
  Impl(int argc, char *argv[], Pipeline *parent, std::string name)
    : parent(parent), name(name)
  {
    parseCommandLine(argc,argv);
    if (!xfFile.empty()) {
      if (loadXF(xfFile,ourTransfunc)) {
        transfuncs.resize(1);
        transfuncs[0] = &ourTransfunc;
      }
    }

#ifdef INTERACTIVE
    if (!transfuncs.empty()) {
      tfe.resize(1);
      tfe[0].setLookupTable(transfuncs[0]->getLUT());
    }
#endif
  }
  ~Impl() = default;

  void parseCommandLine(int argc, char *argv[])
  {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--bgcolor") {
        bgcolor.r = std::stof(argv[++i]);
        bgcolor.g = std::stof(argv[++i]);
        bgcolor.b = std::stof(argv[++i]);
      } else if (arg == "--sample-limit") {
        sampleLimit = atoi(argv[++i]);
      } else if (arg == "--xf") {
        xfFile = argv[++i];
      }
    }
  }

  void init(Frame *frame, Camera *camera)
  {
    if (!frame || !camera) {
      fprintf(stderr,"Pipeline invalid on init, aborting...\n");
      abort();
    }

    fb = frame;
    width = fb->width;
    height = fb->height;
#ifdef INTERACTIVE
    manip = CameraManip(camera, width, height);

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
      throw std::runtime_error("failed to initialize SDL");
  
    Uint32 window_flags =
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN;
    sdl_window = SDL_CreateWindow(name.c_str(), width, height, window_flags);
  
    if (sdl_window == nullptr)
      throw std::runtime_error("failed to create SDL window");
  
    sdl_renderer = SDL_CreateRenderer(sdl_window, nullptr);
  
    SDL_SetWindowPosition(
        sdl_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    if (sdl_renderer == nullptr) {
      SDL_DestroyWindow(sdl_window);
      SDL_Quit();
      throw std::runtime_error("Failed to create SDL renderer");
    }
  
    SDL_ShowWindow(sdl_window);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForSDLRenderer(sdl_window, sdl_renderer);
    ImGui_ImplSDLRenderer3_Init(sdl_renderer);

    for (int i=0; i<tfe.size(); ++i) {
      tfe[i].setSDL3Renderer(sdl_renderer);
    }
#endif

#ifdef RTCORE
    initOWL();
#endif
  }

#ifdef RTCORE
  void initOWL()
  {
    owl.context = owlContextCreate(nullptr,1);
    owl.module = owlModuleCreate(owl.context,owl.ptxCode);
    owl.rayGen = owlRayGenCreate(owl.context,
                                 owl.module,
                                 owl.rayGenName,
                                 sizeof(RayGenData),
                                 rayGenVars,-1);
    owl.launchParams = owlParamsCreate(owl.context,
                                       owl.sizeOfLaunchParamsStruct,
                                       owl.launchParamsDecl,
                                       -1);
    owlBuildPrograms(owl.context);
    owlBuildPipeline(owl.context);
    owlBuildSBT(owl.context);
  }

  void updateLaunchParams()
  {
    for (auto &it : owl.lpMap) {
      std::string name = it.first;
      const LP &lp = it.second;
      if (lp.type == OWL_FLOAT) {
        float f1 = *(float *)lp.value;
        owlParamsSet1f(owl.launchParams, name.c_str(), f1);
      }
      else if (lp.type == OWL_FLOAT2) {
        vec2f f2 = *(vec2f *)lp.value;
        owlParamsSet2f(owl.launchParams, name.c_str(), f2.x, f2.y);
      }
      else if (lp.type == OWL_FLOAT3) {
        vec3f f3 = *(vec3f *)lp.value;
        owlParamsSet3f(owl.launchParams, name.c_str(), f3.x, f3.y, f3.z);
      }
      else if (lp.type == OWL_FLOAT4) {
        vec4f f4 = *(vec4f *)lp.value;
        owlParamsSet4f(owl.launchParams, name.c_str(), f4.x, f4.y, f4.z, f4.w);
      }
      else if (lp.type == OWL_INT) {
        int i1 = *(int *)lp.value;
        owlParamsSet1i(owl.launchParams, name.c_str(), i1);
      }
      else if (lp.type == OWL_BOOL) {
        bool b1 = *(bool *)lp.value;
        owlParamsSet1b(owl.launchParams, name.c_str(), b1);
      }
      else if (lp.type == OWL_RAW_POINTER) {
        char **raw = (char **)lp.value;
        owlParamsSetPointer(owl.launchParams, name.c_str(), *raw);
      }
      else if (lp.type >= OWL_USER_TYPE_BEGIN) {
        owlParamsSetRaw(owl.launchParams, name.c_str(), lp.value);
      }
    }
  }
#endif

  void cleanup()
  {
#ifdef INTERACTIVE
    if (fbTexture)
      SDL_DestroyTexture(fbTexture);
#endif

#ifdef RTCORE
    owlModuleRelease(owl.module);
    owlRayGenRelease(owl.rayGen);
    owlContextDestroy(owl.context);
#endif
  }

  void setTransfunc(Transfunc *tf, int index)
  {
    if (index >= transfuncs.size()) {
      transfuncs.resize(index+1);
#ifdef INTERACTIVE
      tfe.resize(index+1);
#endif
    }
    transfuncs[index] = tf;
    assert(transfuncs[index] != nullptr);
#ifdef INTERACTIVE
    tfe[index].setLookupTable(transfuncs[index]->getLUT());
#else
    if (transfuncs[index]->size < 300) {
      std::vector<vec4f> newLUT(300);
      resampleLUT(newLUT,transfuncs[index]->getLUT());
      transfuncs[index]->setLUT(newLUT);
    }
#endif
  }

  void pollEvents(bool &quit, bool &cameraUpdate, bool &windowResize)
  {
#ifdef INTERACTIVE
    quit = false;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      // imgui:
      ImGui_ImplSDL3_ProcessEvent(&event);
      ImGuiIO& io = ImGui::GetIO();
      // quit:
      if (event.type == SDL_EVENT_QUIT) {
        quit = true;
        return;
      }
      if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED
          && event.window.windowID == SDL_GetWindowID(sdl_window)) {
        quit = true;
        return;
      }
      // resize
      if (event.type == SDL_EVENT_WINDOW_RESIZED) {
        if (fb != nullptr) {
          fb->resize(event.window.data1, event.window.data2);
          windowResize = true;
        }
        return;
      }
      // mouse events
      if (!io.WantCaptureMouse) {
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
          SDL_MouseButtonEvent button = event.button;
          CameraManip::MouseButton ourButton{CameraManip::Left};
          if (button.button == SDL_BUTTON_LEFT) ourButton = CameraManip::Left;
          if (button.button == SDL_BUTTON_MIDDLE) ourButton = CameraManip::Middle;
          if (button.button == SDL_BUTTON_RIGHT) ourButton = CameraManip::Right;
          cameraUpdate = manip.handleMouseDown(button.x,button.y,ourButton);
          return;
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
          SDL_MouseButtonEvent button = event.button;
          CameraManip::MouseButton ourButton{CameraManip::Left};
          if (button.button == SDL_BUTTON_LEFT) ourButton = CameraManip::Left;
          if (button.button == SDL_BUTTON_MIDDLE) ourButton = CameraManip::Middle;
          if (button.button == SDL_BUTTON_RIGHT) ourButton = CameraManip::Right;
          cameraUpdate = manip.handleMouseUp(button.x,button.y,ourButton);
          return;
        }
        if (event.type == SDL_EVENT_MOUSE_MOTION) {
          SDL_MouseMotionEvent motion = event.motion;
          cameraUpdate = manip.handleMouseMove(motion.x,motion.y);
          return;
        }
      }
      // keyboard events
      if (!io.WantCaptureKeyboard) {
        if (event.type == SDL_EVENT_KEY_DOWN) {
          SDL_KeyboardEvent key = event.key;
          if (keyDownHandler) {
            // TODO: check if in ascii range
            keyDownHandler(key.key);
          }
          return;
        }
      }
    }
#endif
  }

  void present(const uint32_t *pixels, int w, int h)
  {
#ifdef INTERACTIVE
    if (!fbTexture || width != w || height != h) {
      if (fbTexture) {
        SDL_DestroyTexture(fbTexture);
      }
      width = w;
      height = h;
      fbTexture = SDL_CreateTexture(sdl_renderer,
          SDL_PIXELFORMAT_RGBA32,
          SDL_TEXTUREACCESS_STREAMING,
          width,
          height);

      manip.vpWidth = width;
      manip.vpHeight = height;
    }

    SDL_UpdateTexture(fbTexture,
        nullptr,
        pixels,
        width * sizeof(uint32_t));

    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();

    ImGui::NewFrame();

    //ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking
    //    | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
    //    | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
    //    | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::Begin("Settings");//, nullptr, window_flags);
    ImGui::LabelText("##TFE", "TFE");
    if (transfuncs.size() == 1) {
      tfe[0].drawImmediate();
    } else {
      if (ImGui::BeginTabBar("Lookup Tables")) {
        for (int i=0; i<transfuncs.size(); ++i) {
          ImGui::PushID(i);
          if (ImGui::BeginTabItem(std::to_string(i).c_str())) {
            tfe[i].drawImmediate();
            ImGui::EndTabItem();
            tfID = i;
          }
          ImGui::PopID();
        }
        ImGui::EndTabBar();
      }
    }

    // App-side params
    if (!paramf.empty()) {
      ImGui::LabelText("##App", "App");
      for (int i=0; i<paramf.size(); ++i) {
        Paramf &p = paramf[i];
        if (ImGui::SliderFloat(p.name.c_str(), p.f, p.minf, p.maxf)) {
          parent->resetAccumulation();
        }
      }
    }
    ImGui::End();

    ImGui::Render();

    SDL_SetRenderDrawColorFloat(sdl_renderer, bgcolor.r, bgcolor.g, bgcolor.b, 1.f);
    SDL_RenderClear(sdl_renderer);
    SDL_RenderTextureRotated(
        sdl_renderer,
        fbTexture,
        nullptr,
        nullptr,
        0.0,
        nullptr,
        SDL_FLIP_VERTICAL);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), sdl_renderer);
    SDL_RenderPresent(sdl_renderer);
#else
    // non-interactive: dump to png
    std::string fileName = name+".png";
    stbi_flip_vertically_on_write(1);
    stbi_write_png(fileName.c_str(), width, height, 4, pixels, 4 * width);
    printf("Output: %s\n", fileName.c_str());
#endif
  }

  Pipeline *parent{nullptr};
#ifdef INTERACTIVE
  SDL_Window *sdl_window{nullptr};
  SDL_Renderer *sdl_renderer{nullptr};
  SDL_Texture *fbTexture{nullptr};
  CameraManip manip;
  Pipeline::KeyDownHandler keyDownHandler = 0;
  std::vector<TFE> tfe;
  int tfID{0};
#endif
  Frame *fb{nullptr};
  std::vector<Transfunc *> transfuncs;
  Transfunc ourTransfunc;
  int width{512};
  int height{512};
  std::string name;
  vec3f bgcolor{0.1f, 0.1f, 0.1f};
#ifdef INTERACTIVE
  int sampleLimit{INT_MAX};
#else
  int sampleLimit{1};
#endif
  std::string xfFile;
  thread_pool pool{std::thread::hardware_concurrency()};

  // app-side params:
  struct Paramf
  {
    std::string name;
    float *f;
    float minf;
    float maxf;
  };
  std::vector<Paramf> paramf;

  void uiParam(Paramf p) { paramf.push_back(p); }

#ifdef RTCORE
  struct LP
  {
    OWLDataType type;
    void *value;
  };

  struct {
    OWLContext  context;
    OWLModule   module;
    OWLRayGen   rayGen;
    OWLParams   launchParams;
    const char *rayGenName{nullptr};
    const char *ptxCode{nullptr};
    OWLVarDecl *launchParamsDecl{nullptr};
    size_t      sizeOfLaunchParamsStruct{0ull};
    std::map<std::string,LP> lpMap;
  } owl;
#endif
};

Pipeline::Pipeline(std::string name) : impl(new Impl(this,name)) {}
Pipeline::Pipeline(int argc, char *argv[], std::string name)
  : impl(new Impl(argc,argv,this,name))
{}

Pipeline::~Pipeline() {
  impl->cleanup();
}

#ifdef RTCORE
void Pipeline::setRayGen(const char *name) {
  impl->owl.rayGenName = name;
}

void Pipeline::setRayGen(const char *ptxCode, const char *name) {
  impl->owl.ptxCode = ptxCode;
  setRayGen(name);
}

void Pipeline::setLaunchParamsDecl(OWLVarDecl *decl, size_t sizeOfStruct) {
  impl->owl.launchParamsDecl = decl;
  impl->owl.sizeOfLaunchParamsStruct = sizeOfStruct;
}
#endif

/*
  launch param interface:
*/
#ifdef RTCORE
#define DEF_LAUNCH_PARM_FUNC(T)                               \
T &Pipeline::launchParam(std::string name, T &value) {        \
  impl->owl.lpMap[name] = {mapOwlType(T{}),(void *)&value};   \
  return value;                                               \
}
#else
#define DEF_LAUNCH_PARM_FUNC(T)                               \
T &Pipeline::launchParam(std::string name, T &value) {        \
  return value;                                               \
}
#endif

DEF_LAUNCH_PARM_FUNC(bool)
DEF_LAUNCH_PARM_FUNC(int)
DEF_LAUNCH_PARM_FUNC(float)
DEF_LAUNCH_PARM_FUNC(vec2f)
DEF_LAUNCH_PARM_FUNC(vec3f)
DEF_LAUNCH_PARM_FUNC(vec4f)
DEF_LAUNCH_PARM_FUNC(box1f)
DEF_LAUNCH_PARM_FUNC(box3f)
DEF_LAUNCH_PARM_FUNC(RawPointer)

/*
  transfuncs:
*/
void Pipeline::setTransfunc(Transfunc *tf, int index) {
  impl->setTransfunc(tf,index);
}

Transfunc *Pipeline::getTransfunc(int index) const {
  return impl->transfuncs[index];
}

bool Pipeline::transfuncValid(int index) const {
  return impl->transfuncs.size() > index && impl->transfuncs[index] != nullptr;
}

// ui params:
void Pipeline::uiParam(std::string name, float *f, float minf, float maxf) {
  impl->uiParam({name,f,minf,maxf});
}

void Pipeline::launch() {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

  if (!running)
    impl->init(fb, camera);

  bool quit = false, cameraUpdate = false, windowResize = false;
  impl->pollEvents(quit,cameraUpdate,windowResize);
  running = !quit;
#ifndef INTERACTIVE
  running = (frameID < impl->sampleLimit-1);
#endif

  bool resetAccum = false;

  if (cameraUpdate || windowResize)
    resetAccum = true;

#ifdef INTERACTIVE
  int tfID = impl->tfID;
  if (transfuncValid(tfID) && impl->tfe[tfID].updated()) {
    impl->transfuncs[tfID]->setLUT(impl->tfe[tfID].getUpdatedLookupTable());
    resetAccum = true;
  }
#endif

#ifdef RTCORE
  impl->updateLaunchParams();
#else
  if (!func)
    return;
#endif

  if (frameID == 0)
    clearFramebuffer(fb,impl->pool);

  if (frameID < impl->sampleLimit) {
#ifdef RTCORE
    owlLaunch2D(impl->owl.rayGen, fb->width, fb->height, impl->owl.launchParams);
#else
    parallel::for_each(impl->pool, 0, fb->width, 0, fb->height,
      [&](int x, int y) {
        launchDims = {fb->width,fb->height};
        launchIndex = {x,y};
        func();
      });
#endif
  }

  if (resetAccum)
    frameID = 0;
  else
    frameID++;
}

void Pipeline::present() const {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

#ifdef RTCORE
  std::vector<uint32_t> hostData(fb->width*fb->height);
  cudaMemcpy(hostData.data(), fb->fbPointer, fb->width*fb->height*sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  impl->present(hostData.data(), fb->width, fb->height);
#else
  impl->present(fb->fbPointer, fb->width, fb->height);
#endif
}

void Pipeline::resetAccumulation() {
  frameID = 0;
}

void Pipeline::setKeyDownHandler(KeyDownHandler kdh) {
#ifdef INTERACTIVE
  impl->keyDownHandler = kdh;
#endif
}

} // namespace dvr_course


