#!/usr/bin/env python3
"""
Anthology Task Evaluator: Automated Verification & Grading Harness
Compiles and verifies candidate CUDA implementations against correctness criteria,
edge cases, and performance thresholds.
"""

import sys
import os
import subprocess
import json
import tempfile
import time

HARNESS_TEMPLATE = r"""
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

extern "C" void launch_fused_gemm_bias_relu(
    const float* d_A,
    const float* d_B,
    const float* d_bias,
    float* d_C,
    int M, int N, int K,
    cudaStream_t stream
);

void cpu_reference(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (bias) sum += bias[j];
            C[i * N + j] = (sum > 0.0f) ? sum : 0.0f;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: harness M N K [is_perf_test]\n";
        return 1;
    }
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    bool is_perf = (argc >= 5 && std::atoi(argv[4]) == 1);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_bias(N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = ((i % 23) - 11) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = ((i % 19) - 9) * 0.1f;
    for (int i = 0; i < N; ++i) h_bias[i] = 0.05f;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_bias = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up
    launch_fused_gemm_bias_relu(d_A, d_B, d_bias, d_C, M, N, K, 0);
    cudaDeviceSynchronize();

    float elapsed_ms = 0.0f;
    if (is_perf) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iters = 20;
        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; ++i) {
            launch_fused_gemm_bias_relu(d_A, d_B, d_bias, d_C, M, N, K, 0);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        elapsed_ms /= iters;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else {
        launch_fused_gemm_bias_relu(d_A, d_B, d_bias, d_C, M, N, K, 0);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU validation (validate subset for perf benchmark to ensure rapid feedback)
    int check_M = is_perf ? std::min(M, 32) : M;
    cpu_reference(h_A.data(), h_B.data(), h_bias.data(), h_C_cpu.data(), check_M, N, K);

    float max_abs_diff = 0.0f;
    for (int i = 0; i < check_M * N; ++i) {
        float diff = std::abs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    std::cout << max_abs_diff << " " << elapsed_ms << "\n";
    return 0;
}
"""

def evaluate(solution_path):
    print("=" * 80)
    print(f"Anthology Task Evaluator: Evaluating '{solution_path}'")
    print("=" * 80)

    spec_path = os.path.join(os.path.dirname(__file__), "spec.json")
    with open(spec_path, "r") as f:
        spec = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        harness_src = os.path.join(tmpdir, "harness.cu")
        with open(harness_src, "w") as f:
            f.write(HARNESS_TEMPLATE)

        exe_path = os.path.join(tmpdir, "eval_bin")
        nvcc_cmd = [
            "/opt/cuda/bin/nvcc",
            "-ccbin", "g++-13",
            "-arch=sm_61",
            "-O3",
            harness_src,
            solution_path,
            "-lstdc++",
            "-o", exe_path
        ]

        print(f"[1/3] Compiling solution with nvcc...")
        t0 = time.time()
        res = subprocess.run(nvcc_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("\033[31m[COMPILATION FAILED]\033[0m")
            print(res.stderr)
            return {"score": 0.0, "status": "COMPILATION_ERROR", "error": res.stderr}
        print(f"  Compilation succeeded in {time.time() - t0:.2f}s.\n")

        print("[2/3] Running correctness test suites...")
        total_score = 0.0
        test_details = []

        max_tol = spec["constraints"]["max_abs_tolerance"]
        perf_target_ms = spec["constraints"]["perf_target_1024_max_ms"]

        for tc in spec["test_cases"]:
            name = tc["name"]
            M, N, K = tc["M"], tc["N"], tc["K"]
            weight = tc["weight"]
            is_perf = 1 if name.startswith("perf") else 0

            cmd = [exe_path, str(M), str(N), str(K), str(is_perf)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if proc.returncode != 0:
                    print(f"  [{name}] \033[31mCRASH / NON-ZERO EXIT ({proc.returncode})\033[0m")
                    test_details.append({"name": name, "passed": False, "error": proc.stderr})
                    continue

                parts = proc.stdout.strip().split()
                if len(parts) < 2:
                    print(f"  [{name}] \033[31mINVALID OUTPUT FORMAT\033[0m")
                    test_details.append({"name": name, "passed": False, "error": proc.stdout})
                    continue

                max_diff = float(parts[0])
                time_ms = float(parts[1])

                passed = (max_diff <= max_tol)
                perf_passed = True
                if is_perf and time_ms > perf_target_ms:
                    perf_passed = False

                if passed and perf_passed:
                    total_score += weight
                    status_str = "\033[32mPASS\033[0m"
                    if is_perf:
                        status_str += f" ({time_ms:.2f} ms <= {perf_target_ms:.1f} ms)"
                    print(f"  [{name:18}] {status_str:30} Max Diff: {max_diff:.2e}")
                else:
                    fail_msg = f"Diff: {max_diff:.2e} > {max_tol}" if not passed else f"Time: {time_ms:.2f}ms > {perf_target_ms}ms"
                    print(f"  [{name:18}] \033[31mFAIL\033[0m ({fail_msg})")

                test_details.append({
                    "name": name,
                    "passed": passed and perf_passed,
                    "max_diff": max_diff,
                    "time_ms": time_ms
                })

            except subprocess.TimeoutExpired:
                print(f"  [{name:18}] \033[31mTIMEOUT (exceeded 10s)\033[0m")
                test_details.append({"name": name, "passed": False, "error": "timeout"})

        print("\n" + "=" * 80)
        final_status = "PASSED" if total_score >= 0.99 else "FAILED"
        print(f"Final Score: {total_score * 100:.1f} / 100.0  [{final_status}]")
        print("=" * 80)

        return {
            "score": total_score,
            "status": final_status,
            "tests": test_details
        }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "reference_solution.cu")
    result = evaluate(target)
    sys.exit(0 if result["status"] == "PASSED" else 1)
