# CS683 PA-1 Task 2  build file.
#
# The compiler flags are PINNED and identical for every stage so that all speedups are
# attributable to the techniques you implement, not to compiler options:
#
#   -O2                 realistic scalar optimization (inlining, register allocation)
#   -fno-tree-vectorize compiler auto-vectorization is OFF, so the SIMD stage is YOUR job
#   -mavx2 -mfma        enable the AVX2 + FMA intrinsics your simd/prefetch/optimized use
#
# Do NOT add -march=native or -ftree-vectorize: the autograder uses exactly these flags.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall
LDFLAGS  :=

BIN          := bin
MAIN         := src/main.cpp
STUDENT_SRC  := src/matmul_naive.cpp src/matmul_simd.cpp src/matmul_prefetch.cpp \
                src/matmul_optimized.cpp

# Instructor-only: reference implementations used to calibrate/regress the tiers.
SOLUTION_SRC := src/matmul_naive.cpp solution/matmul_simd.cpp solution/matmul_prefetch.cpp \
                solution/matmul_optimized.cpp

.PHONY: all run solution clean llama-demo

all: $(BIN)/matmul

# Student build: edit src/matmul_*.cpp, then `make` and `./bin/matmul`.
$(BIN)/matmul: $(STUDENT_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: $(BIN)/matmul
	./$(BIN)/matmul

# Instructor build against solution/ (kept out of the student handout).
solution: $(BIN)/matmul_solution
$(BIN)/matmul_solution: $(SOLUTION_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN):
	mkdir -p $(BIN)

# Clone + build llama.cpp, inject your matmul_optimized into its CPU matmul, and run a
# tiny inference comparing your kernel against llama.cpp's native matmul (single-thread).
# See llama/run_demo.sh. Override the pinned tag or model with variables if needed.
llama-demo:
	bash llama/run_demo.sh

clean:
	rm -rf $(BIN)
