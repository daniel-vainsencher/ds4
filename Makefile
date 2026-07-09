CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_ssd.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_ssd.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only -DDS4_ROCM_BUILD
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Default to ROCm; CUDA targets override CORE_OBJS
CORE_OBJS ?= ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
CUDA_CORE_OBJS = ds4.o ds4_distributed.o ds4_ssd.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_ssd.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
# Default to ROCm toolchain; override with CUDA settings if needed
DS4_LINK ?= $(HIPCC) $(ROCM_CFLAGS)
DS4_LINK_LIBS ?= $(ROCM_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH= \
		CORE_OBJS="$(CUDA_CORE_OBJS)" \
		CFLAGS="$(subst -DDS4_ROCM_BUILD,,$(CFLAGS))" \
		DS4_LINK="$(NVCC) $(NVCCFLAGS)" \
		DS4_LINK_LIBS="$(CUDA_LDLIBS)"

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native \
		CORE_OBJS="$(CUDA_CORE_OBJS)" \
		CFLAGS="$(subst -DDS4_ROCM_BUILD,,$(CFLAGS))" \
		DS4_LINK="$(NVCC) $(NVCCFLAGS)" \
		DS4_LINK_LIBS="$(CUDA_LDLIBS)"

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)" \
		CORE_OBJS="$(CUDA_CORE_OBJS)" \
		CFLAGS="$(subst -DDS4_ROCM_BUILD,,$(CFLAGS))" \
		DS4_LINK="$(NVCC) $(NVCCFLAGS)" \
		DS4_LINK_LIBS="$(CUDA_LDLIBS)"

strix-halo rocm:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

# ============================================================================
# Oracle libraries for ds4rust testing
# -Dstatic= removes static linkage so all functions are exported
# ds4.c references functions from ds4_ssd.c and ds4_distributed.c, so we
# include those in the oracle library too.
# ============================================================================

libds4_oracle.a: ds4_oracle.o ds4_ssd.o ds4_distributed.o
	$(AR) rcs $@ $^

ds4_oracle.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -Dstatic= -c ds4.c -o $@

ifeq ($(UNAME_S),Darwin)
oracle: libds4_oracle.a
oracle-rocm: libds4_oracle.a
	@echo "ROCm oracle not available on macOS"
else
# ROCm oracle uses the regular ds4_rocm.o (no -Dstatic= needed).
# The ds4_gpu_test_* functions are already extern "C" and exported.
libds4_rocm.a: ds4_rocm.o
	$(AR) rcs $@ $^

# Full GPU oracle includes ds4.c with GPU support for prefill testing.
# This provides ds4_test_prefill_gpu which needs the full graph infrastructure.
ds4_gpu_oracle.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Dstatic= -c ds4.c -o $@

libds4_gpu_oracle.a: ds4_gpu_oracle.o ds4_ssd.o ds4_distributed.o ds4_rocm.o
	$(AR) rcs $@ $^

oracle: libds4_oracle.a
oracle-rocm: libds4_oracle.a libds4_rocm.a
oracle-rocm-full: libds4_oracle.a libds4_gpu_oracle.a
endif

.PHONY: oracle oracle-rocm oracle-rocm-full

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test tests/test_q4k_dot *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o libds4_oracle.a libds4_rocm.a
