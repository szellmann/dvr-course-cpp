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

#include "dvr_course-common.cuh"
#include "fb.h"

namespace dvr_course {

Frame::Frame(int w, int h) : width(w), height(h)
{
#ifndef RTCORE
  fbPointer   = (uint32_t *)std::malloc(w*h*sizeof(uint32_t));
  fbDepth     = (float *)std::malloc(w*h*sizeof(float));
  accumBuffer = (vec4f *)std::malloc(w*h*sizeof(vec4f));
#endif
}

Frame::~Frame()
{
#ifndef RTCORE
  std::free(fbPointer);
  std::free(fbDepth);
  std::free(accumBuffer);
#endif
}

void Frame::clear(const vec4f &rgba, float depth)
{
#ifndef RTCORE
  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      int pixelID = x+y*width;
      if (fbPointer) {
        fbPointer[pixelID] = make_rgba(rgba);
      }

      if (fbDepth) {
        fbDepth[pixelID] = depth;
      }
    }
  }
#endif
}

} // namespace dvr_course
