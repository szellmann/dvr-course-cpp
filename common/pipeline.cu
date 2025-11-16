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

// stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// ours
#include "pipeline.h"
#include "dvr_course-common.h"
#include "dvr_course-common.cuh"

// launch params symbol with C-linkage:
void *optixLaunchParams{NULL};

static thread_local vecmath::vec2i launchIndex;
static thread_local vecmath::vec2i launchDims;

namespace dvr_course {

const vec2i getLaunchIndex(void)
{ return launchIndex; }

const vec2i getLaunchDims(void)
{ return launchDims; }

#ifndef RTCORE
void Pipeline::setLaunchParams(const void *ptr, size_t len, size_t a)
{
  std::free(optixLaunchParams);
  optixLaunchParams = std::aligned_alloc(a,len);
  std::memcpy(&optixLaunchParams,ptr,len);
}
#endif
void Pipeline::launch() const {
#ifdef RTCORE

#else
  launchDims = {fb->width,fb->height};
  for (int y=0; y<launchDims.y; ++y) {
    for (int x=0; x<launchDims.x; ++x) {
      launchIndex = {x,y};
      func();
    }
  }
#endif
}

void Pipeline::present() const {
  if (!isValid()) {
    fprintf(stderr,"Pipeline invalid, aborting...\n");
    abort();
  }

  std::string fileName = name+".png";
  stbi_flip_vertically_on_write(1);
  stbi_write_png(
      fileName.c_str(), fb->width, fb->height, 4, fb->fbPointer, 4 * fb->width);
  printf("Output: %s\n", fileName.c_str());
}

} // namespace dvr_course


