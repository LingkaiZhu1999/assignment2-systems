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


