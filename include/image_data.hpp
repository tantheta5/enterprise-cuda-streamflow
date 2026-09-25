// Copyright 2026 Enterprise CUDA Project Authors. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef ENTERPRISE_IMAGE_DATA_HPP_
#define ENTERPRISE_IMAGE_DATA_HPP_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace enterprise_cuda {

struct RgbPixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

// Represents a 2D RGB image buffer (3 channels, 8-bits per channel).
class ImageRGB {
 public:
  ImageRGB() : width_(0), height_(0), data_(nullptr) {}

  ImageRGB(int width, int height) : width_(width), height_(height) {
    size_t num_bytes = static_cast<size_t>(width_) * height_ * 3;
    data_ = static_cast<uint8_t*>(std::malloc(num_bytes));
    if (data_ != nullptr) {
      std::memset(data_, 0, num_bytes);
    }
  }

  ~ImageRGB() {
    if (data_ != nullptr) {
      std::free(data_);
      data_ = nullptr;
    }
  }

  // Disable copy to avoid accidental double-free; provide explicit clone.
  ImageRGB(const ImageRGB&) = delete;
  ImageRGB& operator=(const ImageRGB&) = delete;

  // Move constructor
  ImageRGB(ImageRGB&& other) noexcept
      : width_(other.width_), height_(other.height_), data_(other.data_) {
    other.width_ = 0;
    other.height_ = 0;
    other.data_ = nullptr;
  }

  // Move assignment
  ImageRGB& operator=(ImageRGB&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        std::free(data_);
      }
      width_ = other.width_;
      height_ = other.height_;
      data_ = other.data_;
      other.width_ = 0;
      other.height_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  void Allocate(int width, int height) {
    if (data_ != nullptr) {
      std::free(data_);
      data_ = nullptr;
    }
    width_ = width;
    height_ = height;
    size_t num_bytes = static_cast<size_t>(width_) * height_ * 3;
    data_ = static_cast<uint8_t*>(std::malloc(num_bytes));
    if (data_ != nullptr) {
      std::memset(data_, 0, num_bytes);
    }
  }

  ImageRGB Clone() const {
    ImageRGB copy(width_, height_);
    if (data_ != nullptr && copy.data_ != nullptr) {
      std::memcpy(copy.data_, data_, SizeInBytes());
    }
    return copy;
  }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return 3; }
  size_t TotalPixels() const { return static_cast<size_t>(width_) * height_; }
  size_t SizeInBytes() const { return TotalPixels() * 3; }
  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }

  inline uint8_t& at(int x, int y, int c) {
    return data_[(static_cast<size_t>(y) * width_ + x) * 3 + c];
  }

  inline const uint8_t& at(int x, int y, int c) const {
    return data_[(static_cast<size_t>(y) * width_ + x) * 3 + c];
  }

 private:
  int width_;
  int height_;
  uint8_t* data_;
};

// Represents a 2D Grayscale image buffer (1 channel, 8-bits or float32).
class ImageGray {
 public:
  ImageGray() : width_(0), height_(0), data_(nullptr) {}

  ImageGray(int width, int height) : width_(width), height_(height) {
    size_t num_bytes = static_cast<size_t>(width_) * height_;
    data_ = static_cast<uint8_t*>(std::malloc(num_bytes));
    if (data_ != nullptr) {
      std::memset(data_, 0, num_bytes);
    }
  }

  ~ImageGray() {
    if (data_ != nullptr) {
      std::free(data_);
      data_ = nullptr;
    }
  }

  ImageGray(const ImageGray&) = delete;
  ImageGray& operator=(const ImageGray&) = delete;

  ImageGray(ImageGray&& other) noexcept
      : width_(other.width_), height_(other.height_), data_(other.data_) {
    other.width_ = 0;
    other.height_ = 0;
    other.data_ = nullptr;
  }

  ImageGray& operator=(ImageGray&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        std::free(data_);
      }
      width_ = other.width_;
      height_ = other.height_;
      data_ = other.data_;
      other.width_ = 0;
      other.height_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  void Allocate(int width, int height) {
    if (data_ != nullptr) {
      std::free(data_);
      data_ = nullptr;
    }
    width_ = width;
    height_ = height;
    size_t num_bytes = static_cast<size_t>(width_) * height_;
    data_ = static_cast<uint8_t*>(std::malloc(num_bytes));
    if (data_ != nullptr) {
      std::memset(data_, 0, num_bytes);
    }
  }

  ImageGray Clone() const {
    ImageGray copy(width_, height_);
    if (data_ != nullptr && copy.data_ != nullptr) {
      std::memcpy(copy.data_, data_, SizeInBytes());
    }
    return copy;
  }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return 1; }
  size_t TotalPixels() const { return static_cast<size_t>(width_) * height_; }
  size_t SizeInBytes() const { return TotalPixels(); }
  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }

  inline uint8_t& at(int x, int y) {
    return data_[static_cast<size_t>(y) * width_ + x];
  }

  inline const uint8_t& at(int x, int y) const {
    return data_[static_cast<size_t>(y) * width_ + x];
  }

 private:
  int width_;
  int height_;
  uint8_t* data_;
};

// Converts RGB image to Grayscale using standard perceptual weights:
// Y = 0.299*R + 0.587*G + 0.114*B
inline ImageGray RgbToGrayscale(const ImageRGB& rgb_img) {
  ImageGray gray_img(rgb_img.width(), rgb_img.height());
  const uint8_t* rgb_data = rgb_img.data();
  uint8_t* gray_data = gray_img.data();
  size_t total = rgb_img.TotalPixels();

  for (size_t i = 0; i < total; ++i) {
    float r = static_cast<float>(rgb_data[i * 3 + 0]);
    float g = static_cast<float>(rgb_data[i * 3 + 1]);
    float b = static_cast<float>(rgb_data[i * 3 + 2]);
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    gray_data[i] = static_cast<uint8_t>(y + 0.5f);
  }
  return gray_img;
}

// Converts Grayscale image back to RGB (replicating gray value to R, G, B)
inline ImageRGB GrayscaleToRgb(const ImageGray& gray_img) {
  ImageRGB rgb_img(gray_img.width(), gray_img.height());
  const uint8_t* gray_data = gray_img.data();
  uint8_t* rgb_data = rgb_img.data();
  size_t total = gray_img.TotalPixels();

  for (size_t i = 0; i < total; ++i) {
    uint8_t val = gray_data[i];
    rgb_data[i * 3 + 0] = val;
    rgb_data[i * 3 + 1] = val;
    rgb_data[i * 3 + 2] = val;
  }
  return rgb_img;
}

}  // namespace enterprise_cuda

#endif  // ENTERPRISE_IMAGE_DATA_HPP_
