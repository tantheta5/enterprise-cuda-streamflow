#!/usr/bin/env bash
# ==============================================================================
# Enterprise CUDA StreamFlow Execution & Demonstration Driver
# Course: CUDA at Scale for the Enterprise (Coursera Specialization Capstone)
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================================${NC}"
echo -e "${GREEN}    ENTERPRISE CUDA STREAMFLOW: CAPSTONE DEMONSTRATION & EXECUTION SUITE    ${NC}"
echo -e "${CYAN}================================================================================${NC}"

# Detect execution platform
echo -e "\n${BLUE}[1/5] Detecting System Compute Environment...${NC}"
HAS_NVCC=false
if command -v nvcc >/dev/null 2>&1; then
    HAS_NVCC=true
    NVCC_VER=$(nvcc --version | grep "release" | sed -E 's/.*release ([^,]+).*/\1/')
    echo -e "  --> NVIDIA CUDA Compiler detected: ${GREEN}NVCC Version ${NVCC_VER}${NC}"
else
    echo -e "  --> NVCC not in PATH. Universal OpenMP SIMT CPU emulator enabled."
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "  --> NVIDIA GPU Device: ${GREEN}${GPU_NAME}${NC}"
else
    echo -e "  --> Host CPU: ${GREEN}$(lscpu | grep 'Model name' | sed -e 's/Model name:[[:space:]]*//')${NC}"
    echo -e "  --> CPU Cores available: ${GREEN}$(nproc)${NC}"
fi

# Build Project
echo -e "\n${BLUE}[2/5] Building Software Targets...${NC}"
make clean
make all

# Select Binary
TARGET_BIN="bin/enterprise_cuda_sim"
if [ -f "bin/enterprise_cuda_engine" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    TARGET_BIN="bin/enterprise_cuda_engine"
    echo -e "  --> Active Engine: ${GREEN}Native CUDA GPU Engine (${TARGET_BIN})${NC}"
else
    echo -e "  --> Active Engine: ${YELLOW}Universal High-Throughput SIM Engine (${TARGET_BIN})${NC}"
fi

# Step 3: Run Verification Suite
echo -e "\n${BLUE}[3/5] Executing Mathematical Parity & Verification Suite...${NC}"
./${TARGET_BIN} --verify

# Step 4: Run Scalability Benchmark Suite
echo -e "\n${BLUE}[4/5] Running Enterprise Scalability & Memory Hierarchy Benchmarks...${NC}"
./${TARGET_BIN} --benchmark --iterations 5

# Step 5: Execute Complete Batch Processing Pipeline
echo -e "\n${BLUE}[5/5] Processing Batch Image & Signal Pipeline...${NC}"
./${TARGET_BIN} --mode all --streams 4

echo -e "\n${GREEN}================================================================================${NC}"
echo -e "${GREEN}    DEMONSTRATION RUN COMPLETE - ALL RUBRIC ARTIFACTS GENERATED SUCCESSFULLY    ${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo -e "Proof of execution artifacts summary:"
echo -e "  - Input Images:           $(ls -1 data/input/images/*.ppm 2>/dev/null | wc -l) files in data/input/images/"
echo -e "  - Processed Output Images: $(ls -1 data/output/images/*.ppm 2>/dev/null | wc -l) files in data/output/images/"
echo -e "  - Input Signal Streams:   $(ls -1 data/input/signals/*.csv 2>/dev/null | wc -l) files in data/input/signals/"
echo -e "  - Filtered Output Signals: $(ls -1 data/output/signals/*.csv 2>/dev/null | wc -l) files in data/output/signals/"
echo -e "  - Verification Report:    artifacts/logs/verification_report.txt"
echo -e "  - Performance Benchmark:  artifacts/logs/benchmark_results.csv"
echo -e "  - Execution Log:          artifacts/logs/execution.log"
echo -e "\nReviewers can inspect all generated output files and logs in the repository."
