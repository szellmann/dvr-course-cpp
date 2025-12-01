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

#ifdef INTERACTIVE
# include <SDL3/SDL.h>
# define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
# include "imgui_impl_sdl3.h"
# include "imgui_impl_sdlrenderer3.h"
#else
// stb_image
# define STB_IMAGE_WRITE_IMPLEMENTATION
# include "stb_image_write.h"
#endif
// ours
#include "pipeline.h"
#include "thread_pool.h"
#include "for_each.h"
#include "tfe.h"
#include "dvr_course-common.h"
#include "dvr_course-common.cuh"

static thread_local vecmath::vec2i launchIndex;
static thread_local vecmath::vec2i launchDims;

namespace dvr_course {

const vec2i getLaunchIndex(void)
{ return launchIndex; }

const vec2i getLaunchDims(void)
{ return launchDims; }

struct Pipeline::Impl
{
  Impl(std::string name) : name(name) {}
  ~Impl() = default;

  void init(Frame *frame, Camera *camera, Transfunc *tf)
  {
    if (!frame || !camera) {
      fprintf(stderr,"Pipeline invalid on init, aborting...\n");
      abort();
    }

    fb = frame;
    width = fb->width;
    height = fb->height;
    transfunc = tf;
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

    std::vector<vec4f> colors;
    if (transfunc) {
      colors = transfunc->rgbaLUT;
    } else {
      colors = std::vector<vec4f>({{0.f,0.1f,0.9f,0.0f},
                                   {1.f,0.5f,0.3f,1.f}});
    }

    tfe.setLookupTable(colors);
    tfe.setSDL3Renderer(sdl_renderer);
#endif
  }
  
  void cleanup()
  {
#ifdef INTERACTIVE
    if (fbTexture)
      SDL_DestroyTexture(fbTexture);
#endif
  }

  void pollEvents(bool &quit)
  {
#ifdef INTERACTIVE
    quit = false;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      // imgui:
      ImGui_ImplSDL3_ProcessEvent(&event);
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
      // mouse events
      ImGuiIO& io = ImGui::GetIO();
      if (!io.WantCaptureMouse) {
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
          SDL_MouseButtonEvent button = event.button;
          manip.handleMouseDown(button.x,button.y);
          return;
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
          SDL_MouseButtonEvent button = event.button;
          manip.handleMouseUp(button.x,button.y);
          return;
        }
        if (event.type == SDL_EVENT_MOUSE_MOTION) {
          SDL_MouseMotionEvent motion = event.motion;
          manip.handleMouseMove(motion.x,motion.y);
          return;
        }
      }
    }
#else
    // non-interactive pipeline: one shot
    quit = true;
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
    tfe.drawImmediate();
    //ImGui::Image((ImTextureID)fbTexture,
    //    ImGui::GetContentRegionAvail(),
    //    ImVec2(0, 1),
    //    ImVec2(1, 0));

    ImGui::End();

    ImGui::Render();

    SDL_SetRenderDrawColorFloat(sdl_renderer, 0.1f, 0.1f, 0.1f, 1.f);
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
      });
#endif
  }

#ifdef INTERACTIVE
  SDL_Window *sdl_window{nullptr};
  SDL_Renderer *sdl_renderer{nullptr};
  SDL_Texture *fbTexture{nullptr};
  CameraManip manip;
  TFE tfe;
#endif
  Frame *fb{nullptr};
  Transfunc *transfunc{nullptr};
  int width{512};
  int height{512};
  std::string name;
  thread_pool pool{std::thread::hardware_concurrency()};
};

Pipeline::Pipeline(std::string name) : impl(new Impl(name)) {}
Pipeline::~Pipeline() {}

void Pipeline::launch() {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

  if (!running)
    impl->init(fb, camera, transfunc);

  bool quit = false;
  impl->pollEvents(quit);
  running = !quit;

  if (transfunc && impl->tfe.updated()) {
    transfunc->rgbaLUT = impl->tfe.getUpdatedLookupTable();
  }

  if (!func)
    return;

  impl->clearFramebuffer();

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

void Pipeline::present() const {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

  impl->present(fb->fbPointer, fb->width, fb->height);
}

} // namespace dvr_course


