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
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
// ours
#ifdef RTCORE
# include <owl/owl.h>
#else
# include "owl-interop.h"
#endif

namespace dvr_course {

// ========================================================
// Wrapper to pass arrays between host and device code
// TODO: also meant to wrap owl buffers later!
// ========================================================
template <typename T>
struct Buffer {
  Buffer(size_t size, OWLDataType owlType, const T *ptr)
    : size(size), owlType(owlType)
  {
#ifdef RTCORE
    cudaMalloc(&data,size*sizeof(T));
    cudaMemcpy(data,ptr,size*sizeof(T),cudaMemcpyDefault);
#else
    data = (T *)std::malloc(size*sizeof(T));
    std::memcpy(data,ptr,size*sizeof(T));
#endif
  }

  ~Buffer() {
#ifdef RTCORE
    cudaFree(data);
#else
    std::free(data);
#endif
  }

  T *getPointer() const {
    return data;
  }

  size_t getSize() const
  { return size; }

  T *data{nullptr};
  size_t size{0ull};
  /* we want the type as tempalte _and_ as owl type:
    plain memory buffers are too hard to deal with in
    terms of alignment issues, e.g., making sure to store
    a nvdb grid with proper alignment in a byte array isn't
    that simple, at the same time, OWL does just that, so
    this is that... */
  OWLDataType owlType{OWL_INVALID_TYPE};
};

} // dvr_course


