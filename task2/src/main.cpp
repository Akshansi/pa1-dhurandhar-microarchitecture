// main.cpp  PROVIDED test / grading harness for Task 2.  Do NOT modify.
//
// No arguments -> full graded run (correctness smoke test + graded workload with a score).
// You can also check ONE stage in isolation on a workload of your choice while developing:
//
//     ./bin/matmul simd                   # naive vs simd, default size
//     ./bin/matmul prefetch 1024 1024 1024    # custom M N K
//     ./bin/matmul prefetch 1024 1024 1024 42 # ... and a custom RNG seed
//     ./bin/matmul all 512 512 512        # every stage on a custom workload (no score)
//     ./bin/matmul help                   # usage
//
// Every stage's result is checked against the naive reference (relative tolerance), so a
// "yes" in the correct column means your kernel is numerically right on that input.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

// ------------------------------- configuration -----------------------------
// Relative correctness tolerance: pass if max|C-ref| <= kRelTol * max|ref|.
// (A K-term dot product accumulates rounding, so an absolute 1e-3 would be too strict.)
static constexpr float kRelTol = 1e-4f;

// Timing: warmups (untimed) then timed repetitions; the median is reported. Matmul kernels
// are heavier than Task 1's, so fewer reps.
static constexpr int kWarmup = 1;
static constexpr int kReps = 3;

// Default workload and seed. Square M=N=K=kDefN. (Chosen so the working set exceeds L2, so
// cache tiling matters, while the run stays tolerable.)
static constexpr int kDefN = 1024;
static constexpr int mDefN = 1024;
static constexpr int nDefN = 1024;
static constexpr unsigned kDefSeed = 1234u;

// Scoring: 3 scored stages (simd, prefetch, optimized).
static constexpr double kCorrectPtsPerStage = 10.0;  // -> 3 * 10 = 30 correctness points

struct Tier {
    double speedup;
    double points;
};
// Cumulative tiers for matmul_optimized speedup vs naive (highest reached wins).
// CALIBRATION: tuned so the reference solution reaches the top tier. Re-run
// `make solution && ./bin/matmul_solution` on your lab machines and adjust so a correct
// reference lands at 70.
static constexpr Tier kTiers[] = {
    {5.0, 20.0},
    {10.0, 40.0},
    {18.0, 55.0},
    {25.0, 70.0},
};
static constexpr double kMaxTierPts = 70.0;  // = kTiers[last].points

struct Stage {
    const char* name;
    const char* key;
    MatMulFn fn;
    bool scored;  // false only for the naive reference
};

// Stage order is the standard GEMM optimization sequence: a fast register-blocked SIMD
// micro-kernel first, THEN cache-block it and add software prefetch. (Cache tiling only pays
// off once the kernel is fast enough to be memory-bound  a scalar loop is compute-bound and
// tiling would show nothing  so SIMD comes first.)
static const Stage kStages[] = {
    {"naive (ref)", "naive", matmul_naive, false},
    {"simd", "simd", matmul_simd, true},
    {"prefetch", "prefetch", matmul_prefetch, true},
    {"optimized", "optimized", matmul_optimized, true},
};
static constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);
static constexpr int kNumScored = kNumStages - 1;  // all but naive

// ------------------------------- one workload ------------------------------
// only   : nullptr / "all" -> every stage; otherwise just naive + the named stage.
// show   : print the per-stage timing table.
// scored : also compute and print the autograder score (returned).
// When !scored, returns the number of INCORRECT scored stages that ran.
static double run_config(int M, int N, int K, unsigned seed, const char* only,
                         bool show, bool scored) {
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);

    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);

    const bool all = (only == nullptr) || (std::strcmp(only, "all") == 0);

    // Contiguous microbenchmark strides.
    const int lda = K, ldb = K, ldc = N;

    // Reference output from the naive kernel.
    matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
    const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
    const float tol = kRelTol * (ref_mag + 1e-30f);

    const double flops = pa1::matmul_flops(M, N, K);
    double naive_ms = 0.0;
    double optimized_speedup = 0.0;
    bool optimized_ok = false;
    int correct_scored = 0;
    int incorrect_scored = 0;

    if (show) {
        std::printf("\n=== Workload  M=%d N=%d K=%d seed=%u %s ===\n", M, N, K, seed,
                    scored ? "(GRADED)" : "(not graded)");
        std::printf("%-14s  %-7s  %10s  %10s  %9s\n", "stage", "correct", "time(ms)",
                    "GFLOP/s", "speedup");
        std::printf("--------------------------------------------------------------\n");
    }

    for (int s = 0; s < kNumStages; ++s) {
        const bool is_naive = (kStages[s].fn == matmul_naive);
        if (!all && !is_naive && std::strcmp(kStages[s].key, only) != 0) continue;

        auto run = [&]() { kStages[s].fn(A, B, C, M, N, K, lda, ldb, ldc); };
        run();  // one untimed run to produce the result we check
        const bool ok =
            (pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N) <= tol);
        if (kStages[s].scored) {
            if (ok) ++correct_scored; else ++incorrect_scored;
        }

        const double ms = pa1::time_median_ms(run, kWarmup, kReps);
        const double gflops = flops / (ms * 1e6);  // ms->s and FLOP->GFLOP
        if (is_naive) naive_ms = ms;
        const double speedup = (ms > 0.0) ? naive_ms / ms : 0.0;
        if (kStages[s].fn == matmul_optimized) {
            optimized_speedup = speedup;
            optimized_ok = ok;
        }

        if (show) {
            std::printf("%-14s  %-7s  %10.3f  %10.2f  %8.2fx\n", kStages[s].name,
                        ok ? "yes" : "NO", ms, gflops, speedup);
        }
    }

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    pa1::free_floats(ref);

    if (!scored) return static_cast<double>(incorrect_scored);

    // ---- scoring (full graded run only) ------------------------------------
    const double correct_pts = correct_scored * kCorrectPtsPerStage;

    double tier_pts = 0.0;
    if (optimized_ok) {  // no speedup credit for an incorrect optimized kernel
        for (const Tier& t : kTiers) {
            if (optimized_speedup >= t.speedup) tier_pts = t.points;
        }
    }

    const double max_correct_pts = kNumScored * kCorrectPtsPerStage;
    const double total = correct_pts + tier_pts;
    std::printf("--------------------------------------------------------------\n");
    std::printf("correctness : %.0f / %.0f   (%d of %d stages correct)\n", correct_pts,
                max_correct_pts, correct_scored, kNumScored);
    std::printf("optimized   : %.2fx%s  ->  %.0f / %.0f  speedup points\n",
                optimized_speedup, optimized_ok ? "" : " (INCORRECT)", tier_pts,
                kMaxTierPts);
    std::printf("TOTAL       : %.0f / %.0f\n", total, max_correct_pts + kMaxTierPts);
    return total;
}

static void usage(const char* prog) {
    std::printf("Usage:\n");
    std::printf("  %s                        full graded run (score printed)\n", prog);
    std::printf("  %s <stage> [M N K] [seed]     check one stage on a workload\n", prog);
    std::printf("  %s all [M N K] [seed]         run every stage (no score)\n", prog);
    std::printf("\nstage: naive | simd | prefetch | optimized | all\n");
    std::printf("M, N, K default to %d; seed defaults to %u. All sizes must be positive.\n",
                kDefN, kDefSeed);
    std::printf("\nExamples:\n");
    std::printf("  %s simd            %s prefetch 1024 1024 1024        %s all 512 512 512\n",
                prog, prog, prog);
}

static const char* match_stage(const char* s) {
    if (std::strcmp(s, "all") == 0) return "all";
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(s, kStages[i].key) == 0) return kStages[i].key;
    return nullptr;
}

int main(int argc, char** argv) {
    std::printf("CS683 PA-1 Task 2  matrix multiplication optimization harness\n");

    if (argc == 1) {
        // Fast correctness smoke test on a small workload.
        double bad = run_config(256, 256, 256, kDefSeed, nullptr, /*show=*/false,
                                /*scored=*/false);
        if (bad > 0.0)
            std::printf("[smoke] %d scored stage(s) differ from reference on a small input.\n",
                        static_cast<int>(bad));
        else
            std::printf("[smoke] all stages match the reference on a small input.\n");

        run_config(kDefN, mDefN, nDefN, kDefSeed, nullptr, /*show=*/true, /*scored=*/true);
        std::printf(
            "\nhint: cache blocking's payoff grows with size. Try a bigger matmul, e.g.\n"
            "      ./bin/matmul all 2048 2048 2048   (watch simd vs optimized diverge).\n");
        return 0;
    }

    if (std::strcmp(argv[1], "help") == 0 || std::strcmp(argv[1], "-h") == 0 ||
        std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    const char* stage = match_stage(argv[1]);
    if (!stage) {
        std::printf("error: unknown stage '%s'\n\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    int M = mDefN, N = nDefN, K = kDefN;
    unsigned seed = kDefSeed;
    if (argc >= 5) {
        M = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else if (argc != 2) {
        std::printf("error: give either no dimensions, or all three (M N K).\n\n");
        usage(argv[0]);
        return 1;
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));

    if (M <= 0 || N <= 0 || K <= 0) {
        std::printf("error: M, N, K must be positive.\n");
        return 1;
    }

    run_config(M, N, K, seed, stage, /*show=*/true, /*scored=*/false);
    return 0;
}
