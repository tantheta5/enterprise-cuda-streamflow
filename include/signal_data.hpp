// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_SIGNAL_DATA_HPP_
#define ENTERPRISE_SIGNAL_DATA_HPP_

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace enterprise_cuda {

// Represents a 1D single-channel or multi-channel discrete time signal.
class Signal1D {
 public:
  Signal1D() = default;

  explicit Signal1D(size_t length, float initial_val = 0.0f)
      : samples_(length, initial_val) {}

  explicit Signal1D(std::vector<float> samples)
      : samples_(std::move(samples)) {}

  size_t size() const { return samples_.size(); }
  size_t SizeInBytes() const { return samples_.size() * sizeof(float); }
  float* data() { return samples_.data(); }
  const float* data() const { return samples_.data(); }

  float& operator[](size_t idx) { return samples_[idx]; }
  const float& operator[](size_t idx) const { return samples_[idx]; }

  void Resize(size_t new_size, float val = 0.0f) {
    samples_.resize(new_size, val);
  }

  // Load signal from CSV file (one float per line or comma-separated)
  bool LoadFromCsv(const std::string& filepath) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
      std::fprintf(stderr, "Failed to open signal CSV file: %s\n", filepath.c_str());
      return false;
    }

    samples_.clear();
    std::string line;
    bool is_first_line = true;

    while (std::getline(infile, line)) {
      if (line.empty()) continue;
      // Skip header if non-numeric
      if (is_first_line) {
        is_first_line = false;
        size_t first_char = line.find_first_not_of(" \t\r\n");
        if (first_char != std::string::npos &&
            (std::isalpha(line[first_char]) || line[first_char] == '#')) {
          continue;
        }
      }

      std::stringstream ss(line);
      std::string token;
      while (std::getline(ss, token, ',')) {
        try {
          size_t start = token.find_first_not_of(" \t\r\n");
          size_t end = token.find_last_not_of(" \t\r\n");
          if (start != std::string::npos) {
            float val = std::stof(token.substr(start, end - start + 1));
            samples_.push_back(val);
          }
        } catch (...) {
          // Skip invalid token
        }
      }
    }
    return !samples_.empty();
  }

  // Save signal to CSV file
  bool SaveToCsv(const std::string& filepath, const std::string& header = "signal_amplitude") const {
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
      std::fprintf(stderr, "Failed to create output signal CSV: %s\n", filepath.c_str());
      return false;
    }

    if (!header.empty()) {
      outfile << header << "\n";
    }

    for (size_t i = 0; i < samples_.size(); ++i) {
      outfile << samples_[i] << "\n";
    }
    return true;
  }

  // Calculate signal energy sum(x^2)
  double Energy() const {
    double sum = 0.0;
    for (float v : samples_) {
      sum += static_cast<double>(v) * v;
    }
    return sum;
  }

  // Calculate peak absolute amplitude
  float PeakAmplitude() const {
    float max_val = 0.0f;
    for (float v : samples_) {
      float abs_v = std::fabs(v);
      if (abs_v > max_val) max_val = abs_v;
    }
    return max_val;
  }

 private:
  std::vector<float> samples_;
};

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_SIGNAL_DATA_HPP_
