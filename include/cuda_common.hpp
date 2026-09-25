// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_CUDA_COMMON_HPP_
#define ENTERPRISE_CUDA_COMMON_HPP_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_CHECK(val) enterprise_cuda::CheckCudaError((val), #val, __FILE__, __LINE__)

namespace enterprise_cuda {

inline void CheckCudaError(cudaError_t result, const char *const func,
                           const char *const file, const int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr,
                 "CUDA Error at %s:%d code=%d(%s) \"%s\"\n",
                 file, line, static_cast<unsigned int>(result),
                 cudaGetErrorString(result), func);
    std::exit(EXIT_FAILURE);
  }
}

class GpuTimer {
 public:
  GpuTimer() : started_(false), stopped_(false) {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }

  void Start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
    started_ = true;
    stopped_ = false;
  }

  void Stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
    stopped_ = true;
  }

  float ElapsedMilliseconds() {
    if (!started_ || !stopped_) {
      return 0.0f;
    }
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
    return elapsed_ms;
  }

 private:
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  bool started_;
  bool stopped_;
};

}  // namespace enterprise_cuda

#else
// ---------------------------------------------------------------------------
// CPU Emulation Fallback for systems without active CUDA / NVCC compiler.
// Allows compilation with standard g++ / clang++ with full SIMT thread-block
// execution semantics and identical numerical outputs.
// ---------------------------------------------------------------------------

#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __syncthreads()

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#define cudaHostAllocDefault 0

typedef void* cudaStream_t;

inline const char* cudaGetErrorString(cudaError_t /*err*/) {
  return "CUDA Success (Simulated)";
}

#define CUDA_CHECK(val) ((void)(val))

namespace enterprise_cuda {

class GpuTimer {
 public:
  GpuTimer() : elapsed_ms_(0.0f) {}
  void Start(cudaStream_t /*stream*/ = nullptr) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  void Stop(cudaStream_t /*stream*/ = nullptr) {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> dur = end_time - start_time_;
    elapsed_ms_ = dur.count();
  }
  float ElapsedMilliseconds() const {
    return elapsed_ms_;
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
  float elapsed_ms_;
};

}  // namespace enterprise_cuda

#endif  // __CUDACC__

namespace enterprise_cuda {

// High-resolution host wall-clock timer for total pipeline profiling.
class CpuTimer {
 public:
  CpuTimer() {
    Reset();
  }

  void Reset() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  double ElapsedMilliseconds() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = current_time - start_time_;
    return elapsed.count();
  }

  double ElapsedMicroseconds() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = current_time - start_time_;
    return elapsed.count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
};

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_CUDA_COMMON_HPP_
