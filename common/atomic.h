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

#ifndef __CUDACC__
#include <atomic>
#include <cstdint>
#include <cstring> // memcpy
#endif

// ========================================================
// Implementations of common atomic functions
// ========================================================
namespace dvr_course {
#ifdef __CUDACC__
inline __device__ float atomicMin(float *address, float val) {
  int ret = __float_as_int(*address);
  while (val < __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}

inline __device__ float atomicMax(float *address, float val) {
  int ret = __float_as_int(*address);
  while (val > __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
}
#else
inline __device__ float atomicMin(float *address, float val) {
  auto *atomic_ptr = (std::atomic<uint32_t> *)address;
  uint32_t currentBits = atomic_ptr->load(std::memory_order_relaxed);
  for (;;) {
    float currentVal;
    std::memcpy(&currentVal, &currentBits, sizeof(currentVal));
    if (val >= currentVal) return currentVal;
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(val));
    if (atomic_ptr->compare_exchange_weak(currentBits, bits,
                                          std::memory_order_release,
                                          std::memory_order_relaxed)) {
      return currentVal;
    }
  }
}

inline __device__ float atomicMax(float *address, float val) {
  auto *atomic_ptr = (std::atomic<uint32_t> *)address;
  uint32_t currentBits = atomic_ptr->load(std::memory_order_relaxed);
  for (;;) {
    float currentVal;
    std::memcpy(&currentVal, &currentBits, sizeof(currentVal));
    if (val <= currentVal) return currentVal;
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(val));
    if (atomic_ptr->compare_exchange_weak(currentBits, bits,
                                          std::memory_order_release,
                                          std::memory_order_relaxed)) {
      return currentVal;
    }
  }
}
#endif
} // namespace dvr_course



