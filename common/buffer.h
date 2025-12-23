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
// cuda
#include <cuda_runtime.h>

namespace dvr_course {

// ========================================================
// Wrapper to pass arrays between host and device code
// ========================================================
template <typename T>
struct Buffer {
  Buffer(size_t size, const T *ptr) : size_(size)
  {
#ifdef RTCORE
    cudaMalloc(&data_,size_*sizeof(T));
    cudaMemcpy(data_,ptr,size_*sizeof(T),cudaMemcpyDefault);
#else
    data_ = (T *)std::malloc(size_*sizeof(T));
    std::memcpy(data_,ptr,size_*sizeof(T));
#endif
  }

  ~Buffer() {
#ifdef RTCORE
    cudaFree(data_);
#else
    std::free(data_);
#endif
  }

  T *data() const {
    return data_;
  }

  size_t size() const
  { return size_; }

 private:
  T *data_{nullptr};
  size_t size_{0ull};
};

} // dvr_course


