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

// nanovdb
#include <nanovdb/GridHandle.h>
// common
#include <vecmath.h>

using namespace vecmath;

// ========================================================
// structs with trivial layout, no default init, etc.
// to safely cross host/device borders
// ========================================================
namespace ex07_render_graph {

struct Tet { vec4f v0, v1, v2, v3; };
struct TetMesh { Tet *tets; int numTets; };

// ========================================================
// tagged union volume type, can be nvdb or tet-mesh
// ========================================================
struct Volume {
  int volID;
  enum { NVDB, TET, } type;
  union {
    struct {
      nanovdb::NanoGrid<float> *handle;
      bool filterLinear;
    } asNvdb;
    struct {
#ifdef RTCORE
      OptixTraversableHandle handle;
#endif
      Tet *tets;
      int numTets;
    } asTetMesh;
  };
  box3f bounds;
  box1f dataRange;
};


struct TriangleMesh {
  int meshID;
#ifdef RTCORE
  OptixTraversableHandle handle;
#endif
  vec3f *vertices;
  vec3i *indices;
  box3f bounds;
};

struct Transfunc {
  box1f  valueRange;
  vec4f *values;
  int size;
};

struct LaunchParams {
  // TODO: put all the objects below into a TLAS

  // N volumes:
  Volume *volumes;

  int numVolumes;

  // N transfuncs (one per volume):
  Transfunc *transfuncs;

#ifdef RTCORE
  OptixTraversableHandle triangleTLAS;
#endif

  // camera:
  struct {
    vec3f org;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;

  // framebuffer:
  uint32_t *fbPointer;
  float    *fbDepth;
  vec4f    *accumBuffer;
  int       accumID;

  // renderer:
  vec4f backgroundColor;

  // lighting:
  vec3f ambientColor;
  float ambientRadiance;
  int   ambientSamples;
  float occlusionDistance;
  struct {
    vec3f dir;
    vec3f color;
    float intensity;
  } directionalLight;

  // DVR:
  float unitDistance;
};

} // namespace ex07_render_graph


