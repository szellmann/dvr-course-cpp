// ======================================================================== //
// Copyright 2025-2026 Stefan Zellmann                                      //
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

#include <owl/owl_host.h>
#include "Params.h"

namespace ex07_render_graph {

/* mapping from our launch params struct to owl's var decl
  that owl can build its shader binding table from: */
OWLVarDecl launchParams_owl[]
= {
   // volumes
   { "volumes",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,volumes) },
   { "numVolumes",  OWL_INT, OWL_OFFSETOF(LaunchParams,numVolumes) },
   // meshes
   { "triangleTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams,triangleTLAS) },
   // xf data
   { "transfuncs", OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,transfuncs) },
   // camera settings
   { "camera.org", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.org) },
   { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_00) },
   { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_du) },
   { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_dv) },
   // framebuffer
   { "fbPointer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbPointer) },
   { "fbDepth",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbDepth) },
   { "accumBuffer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,accumBuffer) },
   { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
   // lighting
   { "ambientColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,ambientColor) },
   { "ambientRadiance", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,ambientRadiance) },
   { "ambientSamples", OWL_INT, OWL_OFFSETOF(LaunchParams,ambientSamples) },
   { "occlusionDistance", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,occlusionDistance) },
   { "directionalLight.dir", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,directionalLight.dir) },
   { "directionalLight.color", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,directionalLight.color) },
   { "directionalLight.intensity", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,directionalLight.intensity) },
   // render settings
   { "unitDistance", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,unitDistance) },
   { nullptr /* sentinel to mark end of list */ }
};

} // namespace ex07_render_graph


