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

// common
#include <vecmath.h>

using namespace vecmath;

namespace ex05_hey_icon {

inline __host__ __device__ float deg2rad(float d)
{
  return d*float(M_PI)/180.f;
}

inline __host__ __device__ vec3f toSpherical(const vec3f cartesian)
{
  float r = length(cartesian);
  float lat = asinf(cartesian.z/r);
  float lon = atan2f(cartesian.y, cartesian.x);
  return {r,lat,lon};
}

inline __host__ __device__ vec3f toCartesian(const vec3f spherical)
{
  const float r = spherical.x;
  const float lat = spherical.y;
  const float lon = spherical.z;

  float x = r * cosf(lat) * cosf(lon);
  float y = r * cosf(lat) * sinf(lon);
  float z = r * sinf(lat);
  return {x,y,z};
}

inline __host__ __device__
vec3f triangleLerp(const vec3f a, const vec3f b, const vec3f c, float u, float v)
{
  const vec3f s2 = c * v;
  const vec3f s3 = b * u;
  const vec3f s1 = a * (1.f-u-v);
  return s1+s2+s3;
}


#define MAX_LAYERS 32

struct ICONCell {
  // Latitude, per triangle corner, in ccw order
  vec3f lat;

  // Longitude, per triangle corner, in ccw order
  vec3f lon;

  // Per-layer values:
  // (if MAX_LAYERS gets exceeded we must create another cell!)

  // Number of layers
  int numLayers;

  // Height per layer
  float height[MAX_LAYERS];

  // Value per layer
  float value[MAX_LAYERS];

  inline __host__ __device__
  box3f getBounds() const {
    // bottom triangle vertices
    vec3f bv1 = toCartesian({height[0],lat.x,lon.x});
    vec3f bv2 = toCartesian({height[0],lat.y,lon.y});
    vec3f bv3 = toCartesian({height[0],lat.z,lon.z});

    // top triangle vertices
    vec3f tv1 = toCartesian({height[numLayers-1],lat.x,lon.x});
    vec3f tv2 = toCartesian({height[numLayers-1],lat.y,lon.y});
    vec3f tv3 = toCartesian({height[numLayers-1],lat.z,lon.z});

    box3f bounds(
      {INFINITY,INFINITY,INFINITY},
      {-INFINITY,-INFINITY,-INFINITY}
    );

    bounds.extend(bv1);
    bounds.extend(bv2);
    bounds.extend(bv3);
    bounds.extend(tv1);
    bounds.extend(tv2);
    bounds.extend(tv3);

    // sphere extrema in cartesian coordinates:
    const vec3f left(-height[numLayers-1],0,0);
    const vec3f right(height[numLayers-1],0,0);
    const vec3f bottom(0,-height[numLayers-1],0);
    const vec3f top(0,height[numLayers-1],0);
    const vec3f back(0,0,-height[numLayers-1]);
    const vec3f front(0,0,height[numLayers-1]);

    // sphere extrema in spherical coordinates:
    const vec2f sleft(0.f,M_PI);
    const vec2f sright(0.f,0.f);
    const vec2f sbottom(0.f,-M_PI*0.5f);
    const vec2f stop(0.f,M_PI*0.5f);
    const vec2f sback(-M_PI*0.5f,0.f);
    const vec2f sfront(M_PI*0.5f,0.f);

    // top triangle edges in spherical coordinates:
    const vec2f se1(lat.y-lat.x,lon.y-lon.x);
    const vec2f se2(lat.z-lat.y,lon.z-lon.y);
    const vec2f se3(lat.x-lat.z,lon.z-lon.z);

    if (dot(se1,sleft) > 0 && dot(se2,sleft) > 0 && dot(se3,sleft) > 0) {
      bounds.extend(left);
    }

    if (dot(se1,sright) > 0 && dot(se2,sright) > 0 && dot(se3,sright) > 0) {
      bounds.extend(right);
    }

    if (dot(se1,sbottom) > 0 && dot(se2,sbottom) > 0 && dot(se3,sbottom) > 0) {
      bounds.extend(bottom);
    }

    if (dot(se1,stop) > 0 && dot(se2,stop) > 0 && dot(se3,stop) > 0) {
      bounds.extend(top);
    }

    if (dot(se1,sback) > 0 && dot(se2,sback) > 0 && dot(se3,sback) > 0) {
      bounds.extend(back);
    }

    if (dot(se1,sfront) > 0 && dot(se2,sfront) > 0 && dot(se3,sfront) > 0) {
      bounds.extend(front);
    }

    return bounds;
  }
};

typedef vec4f Plane;

inline __device__ Plane makePlane(const vec3f a, const vec3f b, const vec3f c)
{
  vec3f N = cross(b-a,c-a);
  return Plane(N,dot(a,N));
}

inline __device__ float evalPlane(const Plane &p, const vec3f pos)
{
  return dot(pos,vec3f(p))-p.w;
}

inline __device__ bool sample(const ICONCell &cell, vec3f pos, float &value)
{
  const vec3f spherical = toSpherical(pos);
  if (spherical.x < cell.height[0] || spherical.x > cell.height[cell.numLayers-1])
    return false;

  // bottom triangle vertices
  vec3f bv1 = toCartesian({cell.height[0],cell.lat.x,cell.lon.x});
  vec3f bv2 = toCartesian({cell.height[0],cell.lat.y,cell.lon.y});
  vec3f bv3 = toCartesian({cell.height[0],cell.lat.z,cell.lon.z});

  // top triangle vertices
  vec3f tv1 = toCartesian({cell.height[cell.numLayers-1],cell.lat.x,cell.lon.x});
  vec3f tv2 = toCartesian({cell.height[cell.numLayers-1],cell.lat.y,cell.lon.y});
  vec3f tv3 = toCartesian({cell.height[cell.numLayers-1],cell.lat.z,cell.lon.z});

  auto p1 = makePlane(bv1,bv2,tv2);
  auto p2 = makePlane(bv2,bv3,tv3);
  auto p3 = makePlane(bv3,bv1,tv1);

  if (evalPlane(p1,pos) > 0.f) return false; /* ccw */
  if (evalPlane(p2,pos) > 0.f) return false; /* ccw */
  if (evalPlane(p3,pos) > 0.f) return false; /* ccw */

  // interpolate value
  float h = spherical.x;
  for (int i=0; i<cell.numLayers-1; ++i) {
    float h0 = cell.height[i];
    float h1 = cell.height[i+1];

    if (h >= h0 && h<= h1) {
      float v0 = cell.value[i];
      float v1 = cell.value[i+1];
      float f = (h-h0)/(h1-h0);
      value = v0*(1.f-f) + v1*f;
      break;
    }
  }

  return true;
}

struct ICONGrid {
  ICONCell *cells;
  unsigned numCells;
};

} // namespace ex05_hey_icon


