# ==============================================================================
# Enterprise CUDA StreamFlow Build System
# Course Project: CUDA at Scale for the Enterprise
# ==============================================================================

CXX ?= g++
NVCC ?= $(shell which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fopenmp -Iinclude
NVCCFLAGS := -std=c++17 -O3 -Xcompiler -Wall,-fopenmp -Iinclude \
             -gencode arch=compute_50,code=sm_50 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

BIN_DIR := bin
SRC_DIR := src
INC_DIR := include

SIM_TARGET := $(BIN_DIR)/enterprise_cuda_sim
CUDA_TARGET := $(BIN_DIR)/enterprise_cuda_engine

COMMON_SRCS := $(SRC_DIR)/image_io.cpp $(SRC_DIR)/cpu_reference.cpp $(SRC_DIR)/stream_pipeline.cpp
SIM_SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/cuda_sim_kernels.cpp $(COMMON_SRCS)
CUDA_SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/cuda_kernels.cu $(COMMON_SRCS)

# Check if NVCC is available
HAS_NVCC := $(shell which $(NVCC) 2>/dev/null)

.PHONY: all cuda sim clean run test benchmark verify help

all:
ifneq ($(HAS_NVCC),)
	@echo "--> NVIDIA CUDA Compiler detected ($(NVCC)). Building both native CUDA and SIM targets..."
	@$(MAKE) cuda
	@$(MAKE) sim
else
	@echo "--> NVCC not detected in PATH. Building SIM target (OpenMP multi-threaded SIMT emulator)..."
	@$(MAKE) sim
endif

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

sim: $(BIN_DIR)
	@echo "--> Compiling Simulator Target [$(SIM_TARGET)]..."
	$(CXX) $(CXXFLAGS) $(SIM_SRCS) -o $(SIM_TARGET)
	@echo "--> Successfully built $(SIM_TARGET)"

cuda: $(BIN_DIR)
	@echo "--> Compiling Native CUDA Target [$(CUDA_TARGET)]..."
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS) -o $(CUDA_TARGET)
	@echo "--> Successfully built $(CUDA_TARGET)"

run: all
	@if [ -f $(CUDA_TARGET) ] && command -v nvidia-smi >/dev/null 2>&1; then \
		echo "Running Native CUDA Engine..."; \
		./$(CUDA_TARGET) --mode all; \
	else \
		echo "Running Simulated Engine..."; \
		./$(SIM_TARGET) --mode all; \
	fi

test: verify

verify: all
	@if [ -f $(CUDA_TARGET) ] && command -v nvidia-smi >/dev/null 2>&1; then \
		./$(CUDA_TARGET) --verify; \
	else \
		./$(SIM_TARGET) --verify; \
	fi

benchmark: all
	@if [ -f $(CUDA_TARGET) ] && command -v nvidia-smi >/dev/null 2>&1; then \
		./$(CUDA_TARGET) --benchmark --iterations 10; \
	else \
		./$(SIM_TARGET) --benchmark --iterations 10; \
	fi

clean:
	@echo "--> Cleaning build artifacts..."
	rm -rf $(BIN_DIR)
	rm -f *.o

help:
	@echo "Enterprise CUDA StreamFlow Build Targets:"
	@echo "  make (or make all) : Auto-detects environment and builds appropriate target(s)"
	@echo "  make cuda          : Builds native CUDA executable with NVCC"
	@echo "  make sim           : Builds universal CPU/OpenMP SIMT emulator with GCC/Clang"
	@echo "  make verify        : Runs numerical verification suite comparing CPU vs GPU"
	@echo "  make benchmark     : Runs performance scalability benchmarks"
	@echo "  make run           : Executes batch pipeline on all input images and signals"
	@echo "  make clean         : Removes compiled binaries"
