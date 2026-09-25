// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "cpu_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace enterprise_cuda {

namespace {

// Clamps coordinate to valid range [0, max_dim - 1]
inline int Clamp(int val, int max_dim) {
  if (val < 0) return 0;
  if (val >= max_dim) return max_dim - 1;
  return val;
}

}  // namespace

std::vector<float> GenerateGaussianKernel2D(int kernel_size, float sigma) {
  std::vector<float> kernel(kernel_size * kernel_size);
  int radius = kernel_size / 2;
  double sum = 0.0;
  double two_sigma_sq = 2.0 * sigma * sigma;

  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      double dist_sq = static_cast<double>(x * x + y * y);
      double weight = std::exp(-dist_sq / two_sigma_sq) / (M_PI * two_sigma_sq);
      int idx = (y + radius) * kernel_size + (x + radius);
      kernel[idx] = static_cast<float>(weight);
      sum += weight;
    }
  }

  // Normalize so weights sum to exactly 1.0f
  for (size_t i = 0; i < kernel.size(); ++i) {
    kernel[i] = static_cast<float>(kernel[i] / sum);
  }
  return kernel;
}

void CpuGaussianBlurRgb(const ImageRGB& input, ImageRGB* output, int kernel_size, float sigma) {
  int width = input.width();
  int height = input.height();
  output->Allocate(width, height);

  std::vector<float> kernel = GenerateGaussianKernel2D(kernel_size, sigma);
  int radius = kernel_size / 2;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum_r = 0.0f;
      float sum_g = 0.0f;
      float sum_b = 0.0f;

      for (int ky = -radius; ky <= radius; ++ky) {
        int sample_y = Clamp(y + ky, height);
        for (int kx = -radius; kx <= radius; ++kx) {
          int sample_x = Clamp(x + kx, width);
          float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];

          sum_r += weight * input.at(sample_x, sample_y, 0);
          sum_g += weight * input.at(sample_x, sample_y, 1);
          sum_b += weight * input.at(sample_x, sample_y, 2);
        }
      }

      output->at(x, y, 0) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, sum_r + 0.5f)));
      output->at(x, y, 1) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, sum_g + 0.5f)));
      output->at(x, y, 2) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, sum_b + 0.5f)));
    }
  }
}

void CpuSobelFilter(const ImageGray& input, ImageGray* output) {
  int width = input.width();
  int height = input.height();
  output->Allocate(width, height);

  // Sobel 3x3 operator kernels
  const int kGx[3][3] = {
      {-1, 0, 1},
      {-2, 0, 2},
      {-1, 0, 1}
  };

  const int kGy[3][3] = {
      {-1, -2, -1},
      { 0,  0,  0},
      { 1,  2,  1}
  };

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sum_x = 0;
      int sum_y = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        int sample_y = Clamp(y + ky, height);
        for (int kx = -1; kx <= 1; ++kx) {
          int sample_x = Clamp(x + kx, width);
          int val = input.at(sample_x, sample_y);
          sum_x += kGx[ky + 1][kx + 1] * val;
          sum_y += kGy[ky + 1][kx + 1] * val;
        }
      }

      float magnitude = std::sqrt(static_cast<float>(sum_x * sum_x + sum_y * sum_y));
      output->at(x, y) = static_cast<uint8_t>(std::min(255.0f, magnitude));
    }
  }
}

void CpuMedianFilterRgb(const ImageRGB& input, ImageRGB* output, int window_size) {
  int width = input.width();
  int height = input.height();
  output->Allocate(width, height);

  int radius = window_size / 2;
  int num_elements = window_size * window_size;
  int median_idx = num_elements / 2;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      std::vector<uint8_t> win_r(num_elements);
      std::vector<uint8_t> win_g(num_elements);
      std::vector<uint8_t> win_b(num_elements);

      int count = 0;
      for (int ky = -radius; ky <= radius; ++ky) {
        int sample_y = Clamp(y + ky, height);
        for (int kx = -radius; kx <= radius; ++kx) {
          int sample_x = Clamp(x + kx, width);
          win_r[count] = input.at(sample_x, sample_y, 0);
          win_g[count] = input.at(sample_x, sample_y, 1);
          win_b[count] = input.at(sample_x, sample_y, 2);
          count++;
        }
      }

      std::nth_element(win_r.begin(), win_r.begin() + median_idx, win_r.end());
      std::nth_element(win_g.begin(), win_g.begin() + median_idx, win_g.end());
      std::nth_element(win_b.begin(), win_b.begin() + median_idx, win_b.end());

      output->at(x, y, 0) = win_r[median_idx];
      output->at(x, y, 1) = win_g[median_idx];
      output->at(x, y, 2) = win_b[median_idx];
    }
  }
}

void CpuRotateImageRgb(const ImageRGB& input, ImageRGB* output, float angle_degrees) {
  int width = input.width();
  int height = input.height();
  output->Allocate(width, height);

  float rad = -angle_degrees * static_cast<float>(M_PI / 180.0);
  float cos_theta = std::cos(rad);
  float sin_theta = std::sin(rad);

  float cx = width * 0.5f;
  float cy = height * 0.5f;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float dx = x - cx;
      float dy = y - cy;

      // Inverse mapping
      float src_x = cos_theta * dx - sin_theta * dy + cx;
      float src_y = sin_theta * dx + cos_theta * dy + cy;

      if (src_x >= 0.0f && src_x < width - 1 && src_y >= 0.0f && src_y < height - 1) {
        int x0 = static_cast<int>(src_x);
        int y0 = static_cast<int>(src_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = src_x - x0;
        float fy = src_y - y0;
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        for (int c = 0; c < 3; ++c) {
          float p00 = input.at(x0, y0, c);
          float p10 = input.at(x1, y0, c);
          float p01 = input.at(x0, y1, c);
          float p11 = input.at(x1, y1, c);
          float val = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;
          output->at(x, y, c) = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val + 0.5f)));
        }
      } else {
        output->at(x, y, 0) = 0;
        output->at(x, y, 1) = 0;
        output->at(x, y, 2) = 0;
      }
    }
  }
}

void CpuConvolve1D(const Signal1D& input, const Signal1D& filter_kernel, Signal1D* output) {
  size_t in_len = input.size();
  size_t k_len = filter_kernel.size();
  output->Resize(in_len, 0.0f);

  int radius = static_cast<int>(k_len / 2);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < in_len; ++i) {
    double sum = 0.0;
    for (size_t k = 0; k < k_len; ++k) {
      int sample_idx = static_cast<int>(i) + static_cast<int>(k) - radius;
      if (sample_idx >= 0 && sample_idx < static_cast<int>(in_len)) {
        sum += static_cast<double>(input[sample_idx]) * filter_kernel[k];
      }
    }
    (*output)[i] = static_cast<float>(sum);
  }
}

VerificationReport CompareImages(const ImageRGB& ref, const ImageRGB& test, double tolerance) {
  VerificationReport report;
  if (ref.width() != test.width() || ref.height() != test.height()) {
    report.passed = false;
    return report;
  }

  size_t total_bytes = ref.SizeInBytes();
  const uint8_t* p_ref = ref.data();
  const uint8_t* p_test = test.data();

  double sum_sq_err = 0.0;
  double max_diff = 0.0;

  for (size_t i = 0; i < total_bytes; ++i) {
    double diff = std::fabs(static_cast<double>(p_ref[i]) - static_cast<double>(p_test[i]));
    if (diff > max_diff) max_diff = diff;
    sum_sq_err += diff * diff;
  }

  report.max_absolute_error = max_diff;
  report.mean_squared_error = sum_sq_err / total_bytes;
  report.root_mean_squared_error = std::sqrt(report.mean_squared_error);

  if (report.mean_squared_error < 1e-10) {
    report.psnr_db = 99.99; // effectively identical
  } else {
    report.psnr_db = 10.0 * std::log10((255.0 * 255.0) / report.mean_squared_error);
  }

  report.passed = (max_diff <= tolerance);
  return report;
}

VerificationReport CompareImages(const ImageGray& ref, const ImageGray& test, double tolerance) {
  VerificationReport report;
  if (ref.width() != test.width() || ref.height() != test.height()) {
    report.passed = false;
    return report;
  }

  size_t total_bytes = ref.SizeInBytes();
  const uint8_t* p_ref = ref.data();
  const uint8_t* p_test = test.data();

  double sum_sq_err = 0.0;
  double max_diff = 0.0;

  for (size_t i = 0; i < total_bytes; ++i) {
    double diff = std::fabs(static_cast<double>(p_ref[i]) - static_cast<double>(p_test[i]));
    if (diff > max_diff) max_diff = diff;
    sum_sq_err += diff * diff;
  }

  report.max_absolute_error = max_diff;
  report.mean_squared_error = sum_sq_err / total_bytes;
  report.root_mean_squared_error = std::sqrt(report.mean_squared_error);

  if (report.mean_squared_error < 1e-10) {
    report.psnr_db = 99.99;
  } else {
    report.psnr_db = 10.0 * std::log10((255.0 * 255.0) / report.mean_squared_error);
  }

  report.passed = (max_diff <= tolerance);
  return report;
}

VerificationReport CompareSignals(const Signal1D& ref, const Signal1D& test, double tolerance) {
  VerificationReport report;
  if (ref.size() != test.size() || ref.size() == 0) {
    report.passed = false;
    return report;
  }

  size_t n = ref.size();
  double sum_sq_err = 0.0;
  double max_diff = 0.0;
  double max_signal = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double diff = std::fabs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
    if (diff > max_diff) max_diff = diff;
    sum_sq_err += diff * diff;
    if (std::fabs(ref[i]) > max_signal) max_signal = std::fabs(ref[i]);
  }

  report.max_absolute_error = max_diff;
  report.mean_squared_error = sum_sq_err / n;
  report.root_mean_squared_error = std::sqrt(report.mean_squared_error);

  if (report.mean_squared_error < 1e-12) {
    report.psnr_db = 99.99;
  } else {
    report.psnr_db = 20.0 * std::log10(max_signal / report.root_mean_squared_error);
  }

  report.passed = (max_diff <= tolerance);
  return report;
}

}  // namespace enterprise_cuda
