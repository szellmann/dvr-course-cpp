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

static thread_local vecmath::vec2i launchIndex;
static thread_local vecmath::vec2i launchDims;

namespace dvr_course {

const vec2i getLaunchIndex(void)
{ return launchIndex; }

const vec2i getLaunchDims(void)
{ return launchDims; }

const bool debug(void) {
  return launchIndex.x == launchDims.x/2 && launchIndex.y == launchDims.y/2;
}

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

  tf.rgbaLUT.resize(numValues);
  in.read((char *)tf.rgbaLUT.data(), sizeof(tf.rgbaLUT[0]) * tf.rgbaLUT.size());

  return true;
}

struct Pipeline::Impl
{
  Impl(std::string name) : name(name) {}
  Impl(int argc, char *argv[], std::string name) : name(name)
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
      tfe[0].setLookupTable(transfuncs[0]->rgbaLUT);
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
  }
  
  void cleanup()
  {
#ifdef INTERACTIVE
    if (fbTexture)
      SDL_DestroyTexture(fbTexture);
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
    tfe[index].setLookupTable(transfuncs[index]->rgbaLUT);
#else
    if (transfuncs[index]->rgbaLUT.size() < 300) {
      std::vector<vec4f> newLUT(300);
      resampleLUT(newLUT,transfuncs[index]->rgbaLUT);
      transfuncs[index]->rgbaLUT = newLUT;
    }
#endif
  }

  void pollEvents(bool &quit, bool &cameraUpdate)
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
        }
        return;
      }
      // mouse events
      if (!io.WantCaptureMouse) {
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
          SDL_MouseButtonEvent button = event.button;
          cameraUpdate = manip.handleMouseDown(button.x,button.y);
          return;
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
          SDL_MouseButtonEvent button = event.button;
          cameraUpdate = manip.handleMouseUp(button.x,button.y);
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

    ImGui::Begin("TFE");//, nullptr, window_flags);
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

  void clearFramebuffer(const vec4f &rgba = vec4f(0.f), float depth = 0.f)
  {
#ifndef RTCORE
    parallel::for_each(pool, 0, width, 0, height,
      [=](int x, int y) {
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
#endif
  }

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
};

Pipeline::Pipeline(std::string name) : impl(new Impl(name)) {}
Pipeline::Pipeline(int argc, char *argv[], std::string name)
  : impl(new Impl(argc,argv,name))
{}

Pipeline::~Pipeline() {}

void Pipeline::setTransfunc(Transfunc *tf, int index) {
  impl->setTransfunc(tf,index);
}

Transfunc *Pipeline::getTransfunc(int index) const {
  return impl->transfuncs[index];
}

bool Pipeline::transfuncValid(int index) const {
  return impl->transfuncs.size() > index && impl->transfuncs[index] != nullptr;
}

void Pipeline::launch() {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

  if (!running)
    impl->init(fb, camera);

  bool quit = false, cameraUpdate = false;
  impl->pollEvents(quit,cameraUpdate);
  running = !quit;
#ifndef INTERACTIVE
  running = (frameID < impl->sampleLimit-1);
#endif

  bool resetAccum = false;

  if (cameraUpdate)
    resetAccum = true;

#ifdef INTERACTIVE
  int tfID = impl->tfID;
  if (transfuncValid(tfID) && impl->tfe[tfID].updated()) {
    impl->transfuncs[tfID]->rgbaLUT = impl->tfe[tfID].getUpdatedLookupTable();
    resetAccum = true;
  }
#endif

  if (!func)
    return;

  if (frameID == 0)
    impl->clearFramebuffer();

  if (frameID < impl->sampleLimit) {
#ifdef RTCORE

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

  impl->present(fb->fbPointer, fb->width, fb->height);
}

void Pipeline::resetAccumulation() {
  frameID = 0;
}

void Pipeline::setKeyDownHandler(KeyDownHandler kdh) {
  impl->keyDownHandler = kdh;
}

} // namespace dvr_course


