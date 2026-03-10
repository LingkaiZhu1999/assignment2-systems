import torch
import timeit
import pandas as pd
import math
from typing import Callable
import torch.cuda.nvtx as nvtx
from torch import Tensor
from jaxtyping import Bool, Float
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient, softmax

# hyperparameters
vocab_size = 10000
batch_size =  4
size = ["small"]
d_model = {"small": 768, "medium": 1024, "large": 1280, "xlarge": 1600, "2.7B": 2560}
d_ff = {"small": 3072, "medium": 4096, "large": 5120, "xlarge": 6400, "2.7B": 102400}
num_layers = {"small": 12, "medium": 24, "large": 36, "xlarge": 48, "2.7B": 32}
num_heads = {"small": 12, "medium": 16, "large": 20, "xlarge": 25, "2.7B": 32}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
use_bf16 = False 
model_dtype = torch.bfloat16 if use_bf16 else torch.float32
num_warmups = 5
num_trials = 1

def mean(lst: list[float]) -> float:
    return sum(lst) / len(lst)

def stddev(lst: list[float]) -> float:
    m = mean(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return output

basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def get_model(size, context_length, rope_theta=10000.):
    return BasicsTransformerLM(vocab_size,
        context_length,
        d_model[size],
        num_layers[size],
        num_heads[size],
        d_ff[size],
        rope_theta,)
        
def profiling(
    model,
    data,
    num_warmups: int = 1,
    num_trials: int = 10,
    optimizer: torch.optim.Optimizer = None,
):
    """Benchmark `run` for `num_trials` and return (mean_ms, stddev_ms)."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.

    for i in range(num_warmups+num_trials):  # Do it multiple times to capture variance
        if i >= num_warmups:
            torch.cuda.cudart().cudaProfilerStart()  # Start CUDA profiler after warmup iterations
        
        nvtx.range_push(f"step {i}")

        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        
        with nvtx.range("forward"):
            y = model(data).mean()
        
        with nvtx.range("loss"):
            loss = cross_entropy(model(data), data[:, 1:])
        with nvtx.range("backward"):
            y.backward()

        if optimizer:
            with nvtx.range("optimizer_step"):
                optimizer.step()
        
        # torch.cuda.synchronize()

        nvtx.range_pop()  # End of step

def main():
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    print(f"device={device}, use_bf16={use_bf16}")
    with nvtx.range("define_input"):
        data = torch.randint(0, vocab_size, (batch_size, 128), device=device) 
    for s in size:
        with nvtx.range(f"define_model_{s}"):
            model = get_model(s, context_length=128).to(device=device, dtype=model_dtype) 
        with nvtx.range(f"define_optimizer_{s}"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        print(f"Benchmarking size={s}...")
        profiling(model, data, num_warmups=num_warmups, num_trials=num_trials, optimizer=optimizer)

if __name__ == "__main__":
    main()