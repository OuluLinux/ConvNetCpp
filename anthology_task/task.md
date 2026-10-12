# Anthology Task 101: High-Performance Fused Bias-ReLU GEMM in CUDA

## Objective
Implement an optimized CUDA kernel that performs a fused matrix multiplication with bias addition and rectified linear unit (ReLU) activation:

$$C_{i, j} = \max\left(0.0,\; \sum_{k=0}^{K-1} (A_{i, k} \cdot B_{k, j}) + \text{bias}_j\right)$$

where:
- Matrix $A$ has shape $M \times K$ (row-major)
- Matrix $B$ has shape $K \times N$ (row-major)
- Vector $\text{bias}$ has shape $N$
- Matrix $C$ has shape $M \times N$ (row-major)

The bias addition and activation must be **fused** into the kernel epilogue rather than launched as a separate memory-bound post-processing pass.

---

## Constraints & Requirements

1. **Precision**: Standard IEEE 754 single precision (`float`, FP32).
2. **Dimension Support**: $M, N, K \in [1, 2048]$. The kernel must support arbitrary dimensions that are **not** multiples of the thread block or tile size (e.g. $17 \times 31 \times 23$).
3. **No Closed-Source BLAS**: You cannot use `cublasSgemm` or `CUTLASS`. The solution must be implemented as a custom CUDA kernel demonstrating GPU memory hierarchy awareness (Shared Memory tiling, register caching, coalesced loads).
4. **Memory Safety**: No out-of-bounds global or shared memory access under any matrix configuration.

---

## Evaluation Criteria

### 1. Correctness Criteria
- **Absolute Tolerance**: $\max_{i, j} |C_{\text{pred}}[i, j] - C_{\text{ref}}[i, j]| \le 1.0 \times 10^{-4}$
- **Relative Tolerance**: $\frac{|C_{\text{pred}} - C_{\text{ref}}|}{|C_{\text{ref}}| + 10^{-6}} \le 1.0 \times 10^{-4}$
- **Edge Cases**:
  - Minimum shapes ($1 \times 1 \times 1$)
  - Prime and odd dimensions ($17 \times 31$, $127 \times 255$)
  - Zero matrix inputs ($A = 0, B = 0$)
  - Pure negative inputs (verifying that $\text{ReLU}(x) = 0$ for all $x < 0$)

### 2. Performance Target
- **Benchmark Shape**: $M = 1024, N = 1024, K = 1024$
- **Latency Requirement**: $\le 5.0\text{ ms}$ steady-state kernel execution on NVIDIA Pascal/Turing/Ampere GPUs.
- **Compute Throughput**: $\ge 400\text{ GFLOPS}$.
- **Speedup over CPU**: $\ge 50\times$ speedup relative to single-threaded CPU reference.

---

## Common Suboptimal / Buggy Solutions Tested by Evaluator

The evaluator tests against three canonical errors frequently produced by language models and novice engineers:
1. **Race Condition (`buggy_1_race_condition.cu`)**: Omitting `__syncthreads()` before writing to or after reading from shared memory, resulting in non-deterministic corrupted values.
2. **Uncoalesced Memory Access (`buggy_2_uncoalesced_access.cu`)**: Mapping threads with column-major strides in global memory, degrading memory bus efficiency and failing the latency target.
3. **Missing Boundary Guard (`buggy_3_missing_boundary_check.cu`)**: Assuming matrix dimensions are exact multiples of 16 or 32, leading to CUDA memory exceptions / out-of-bounds writes on odd shapes.

---

## File Structure
- `starter_code.cu`: Starter template for the candidate/agent.
- `reference_solution.cu`: Verified golden reference solution.
- `buggy_solutions/`: Reference buggy solutions for verifier testbench validation.
- `evaluator.py`: Automated grading harness that compiles, verifies correctness, and benchmarks performance.
- `spec.json`: Machine-readable specification.
