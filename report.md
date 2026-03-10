# Problem (benchmarking_script): 4 points
- (a) ``./cs336_systems/benchmarking_model.py``
- (b) Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps.

The results are shown below with context length 128. The deviation is small at the level of >0.x ms

| mode             | model_size   |   d_model |   d_ff |   num_layers |   num_heads |   time_ms |   stddev_ms |
|:-----------------|:-------------|----------:|-------:|-------------:|------------:|----------:|------------:|
| forward_only     | small        |       768 |   3072 |           12 |          12 |   16.4352 |   0.0548096 |
| forward_only     | medium       |      1024 |   4096 |           24 |          16 |   32.3051 |   0.0871129 |
| forward_only     | large        |      1280 |   5120 |           36 |          20 |   67.3233 |   0.035957  |
| forward_only     | xlarge       |      1600 |   6400 |           48 |          25 |  128.521  |   0.0233106 |
| forward_backward | small        |       768 |   3072 |           12 |          12 |   41.7323 |   0.210667  |
| forward_backward | medium       |      1024 |   4096 |           24 |          16 |   99.9967 |   0.100434  |
| forward_backward | large        |      1280 |   5120 |           36 |          20 |  200.448  |   0.0489757 |
| forward_backward | xlarge       |      1600 |   6400 |           48 |          25 |  379.693  |   0.0780556 |
| backward_only    | small        |       768 |   3072 |           12 |          12 |   22.8031 |   0.0102173 |
| backward_only    | medium       |      1024 |   4096 |           24 |          16 |   63.029  |   0.0139085 |
| backward_only    | large        |      1280 |   5120 |           36 |          20 |  133.631  |   0.0495699 |
| backward_only    | xlarge       |      1600 |   6400 |           48 |          25 |  251.914  |   0.0323422 |

- (c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

Now the results have high deviations. The initialization takes time, for example, the gpu can be at the idle state in the beginning. With more warmups, the deviation decreases.

# Problem (nsys_profile): 5 points
Profile your forward pass, backward pass, and optimizer step using nsys with each of the model sizes described in Table 1 and context lengths of 128, 256, 512 and 1024 (you may run out of memory with some of these context lengths for the larger models, in which case just note it in your report).

- (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

The total is about 15.9ms, which matches the time with Python standard library.

- (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats Systems View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

``sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas`` takes the most cumulative GPU time during the forward pass, it is invoked 24 times. No, ``cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)`` takes the most cumulative time during the backward pass.

- (c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

``elementwise_kernel`` and ``vectorized_elementwise_kernel``, which are for element-wise operations such as ReLU activation.

- (d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

The forward has more fraction of time spent on matrix multiplication, while the complete has less with an increase of fraction of element-wise kernel usage.
Only forward:
| Time  | Total Time | Instances | Avg      | Med      | Min      | Max      | StdDev   | Name |
|:------|:-----------|----------:|:---------|:---------|:---------|:---------|:---------|:-----|
| 39.2% | 6.275 ms   |        24 | 261.468 us | 261.327 us | 256.192 us | 267.775 us | 4.758 us | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 20.5% | 3.290 ms   |        48 | 68.543 us  | 68.512 us  | 68.320 us  | 68.832 us  | 137 ns   | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 16.9% | 2.705 ms   |        12 | 225.394 us | 225.456 us | 224.799 us | 225.695 us | 294 ns   | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x128x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 4.8%  | 772.029 us |       146 | 5.287 us   | 5.536 us   | 4.416 us   | 6.272 us   | 587 ns   | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` |

Complete: forward, loss, backward, optimizer step
| Time  | Total Time | Instances | Avg      | Med      | Min      | Max      | StdDev   | Name |
|:------|:-----------|----------:|:---------|:---------|:---------|:---------|:---------|:-----|
| 22.3% | 12.547 ms  |        48 | 261.388 us | 259.791 us | 255.648 us | 267.775 us | 4.553 us  | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 13.0% | 7.314 ms   |        28 | 261.209 us | 245.984 us | 241.503 us | 685.055 us | 83.089 us | `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)` |
| 11.7% | 6.578 ms   |        96 | 68.518 us  | 68.512 us  | 68.255 us  | 69.024 us  | 142 ns    | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 9.6%  | 5.410 ms   |        24 | 225.431 us | 225.488 us | 224.575 us | 226.143 us | 415 ns    | `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x128x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 6.4%  | 3.596 ms   |        17 | 211.504 us | 211.456 us | 210.911 us | 212.288 us | 340 ns    | `sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x128x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` |
| 5.4%  | 3.032 ms   |       251 | 12.080 us  | 5.152 us   | 3.935 us   | 182.752 us | 18.325 us | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` |

- (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

Run time, softmax: 79.267 μs, matrix multiplication: 102.673 + 74.926 = 177.599 μs
FLOPs: softmax: $\approx \mathbf{5N^2}$, matrix multiplication: \mathbf{2Nd^2 + 4N^2d}$$ 



