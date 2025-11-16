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

#include "vecmath.h"

namespace dvr_course {

struct Camera
{
  void setAspect(float a) {
    aspect = a;
  }

  void setOrientation(const vec3f &origin,
                      const vec3f &poi,
                      const vec3f &up,
                      float fovy)
  {
    position = origin;
    upVector = up;
    this->fovy = fovy;
    frame.vz
      = (poi==origin)
      ? vec3f(0,0,1)
      : /* negative because we use NEGATIZE z axis */ - normalize(poi - origin);
    frame.vx = cross(up,frame.vz);
    if (dot(frame.vx,frame.vx) < 1e-8f)
      frame.vx = vec3f(0,1,0);
    else
      frame.vx = normalize(frame.vx);
    frame.vy = normalize(cross(frame.vz,frame.vx));
    distance = length(poi-origin);
    forceUpFrame();
  }

  void forceUpFrame()
  {
    // frame.vz remains unchanged
    if (fabsf(dot(frame.vz,upVector)) < 1e-6f)
      // looking along upvector; not much we can do here ...
      return;
    frame.vx = normalize(cross(upVector,frame.vz));
    frame.vy = normalize(cross(frame.vz,frame.vx));
  }

  vec3f getPosition() const {
    return position;
  }

  void getScreen(vec3f &lower_left, vec3f &horizontal, vec3f &vertical) const {
    float screen_height = 2.f*tanf(0.5f*fovy);
    vertical   = screen_height * frame.vy;
    horizontal = screen_height * aspect * frame.vx;
    lower_left
      =
      /* NEGATIVE z axis! */
      -frame.vz
      - 0.5f * vertical
      - 0.5f * horizontal;
  }

  void viewAll(const box3f &box) {
    vec3f up(0,1,0);
    float diagonal = length(box.size());
    float r = diagonal * 0.5f;
    vec3f eye = box.center() + vec3f(0, 0, r + r / std::atan(fovy));
    setOrientation(eye, box.center(), up, this->fovy);
  }

  vec3f position, upVector;
  float distance;
  float fovy{90.f*M_PI/180.f};
  float aspect{1.f};

  struct {
    vec3f vx, vy, vz;
  } frame;
};

} // namespace dvr_course



