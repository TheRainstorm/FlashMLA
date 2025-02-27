MAKEFILE_DIR = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR = $(abspath $(MAKEFILE_DIR)/..)
CONDA_ENV ?= /root/miniconda3/envs/kt
# 编译器和编译选项
NVCC = nvcc
CXXFLAGS = -std=c++17 -O0 -g
INCLUDE_DIRS = \
    -I$(ROOT_DIR)/csrc \
    -I$(ROOT_DIR)/csrc/cutlass/include \
    -I$(CONDA_ENV)/lib/python3.11/site-packages/torch/include/torch/csrc/api/include/ \
    -I$(CONDA_ENV)/lib/python3.11/site-packages/torch/include/ \
    -I$(CONDA_ENV)/include/python3.11/ \
    -I$(CONDA_ENV)/lib/python3.11/site-packages/pybind11/include/  # 新增
CUDA_ARCH = -gencode arch=compute_90a,code=sm_90a
LDFLAGS = \
    -L$(CONDA_ENV)/lib/python3.11/site-packages/torch/lib/ \
    -L$(CONDA_ENV)/lib/ \
    -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda \
    -ltorch_python \
    -lpython3.11

MAKEFILE_DIR = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR = $(abspath $(MAKEFILE_DIR)/..)

# 源文件和目标文件
SRCS = \
    $(ROOT_DIR)/tests/test_flash_mla.cpp \
    $(ROOT_DIR)/csrc/flash_fwd_mla_metadata.cu \
    $(ROOT_DIR)/csrc/flash_fwd_mla_fp16_sm90.cu \
    $(ROOT_DIR)/csrc/flash_fwd_mla_bf16_sm90.cu \
    $(ROOT_DIR)/csrc/flash_mla_wrapper_tensor.cpp
    # $(ROOT_DIR)/csrc/flash_api.cpp \
# OBJS = $(SRCS:.cpp=.o)
# OBJS += $(SRCS:.cu=.o)
# SRCS := $(shell find $(ROOT_DIR) -name '*.cpp' -o -name '*.cu')
OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# 最终可执行文件
TARGET = test_flash_mla

# 规则
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(NVCC) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDE_DIRS) $(CUDA_ARCH) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)