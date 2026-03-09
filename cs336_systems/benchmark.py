import torch
import timeit
import pandas as pd
from typing import Callable
import torch.cuda.nvtx as nvtx

from cs336_basics.model import BasicsTransformerLM

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
num_trials = 10
benchmark_modes = ["forward_only", "forward_backward", "backward_only"]

def mean(lst: list[float]) -> float:
    return sum(lst) / len(lst)

def stddev(lst: list[float]) -> float:
    m = mean(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

def get_model(size, context_length, rope_theta=10000.):
    return BasicsTransformerLM(vocab_size,
        context_length,
        d_model[size],
        num_layers[size],
        num_heads[size],
        d_ff[size],
        rope_theta,)

def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    description: str,
    run: Callable,
    num_warmups: int = 1,
    num_trials: int = 10,
    run_returns_time_ms: bool = False,
):
    """Benchmark `run` for `num_trials` and return (mean_ms, stddev_ms)."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    with nvtx.range(f"{description} | warmup"):
        for _ in range(num_warmups):
            run()
        synchronize_if_cuda()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for i in range(num_trials):  # Do it multiple times to capture variance
        with nvtx.range(f"{description} | measured {i}"):
            if run_returns_time_ms:
                times.append(float(run()))
                continue

            start_time = timeit.default_timer()
            run()  # Actually perform computation
            synchronize_if_cuda()  # Wait for CUDA threads to finish (important!)
            end_time = timeit.default_timer()
            times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    stddev_time = stddev(times) # @inspect stddev_time
    return mean_time, stddev_time

def run_forward_only(model, data):
    with torch.no_grad():
        out = model(data)
        _ = out.mean()


def run_forward_backward(model, data, optimizer):
    optimizer.zero_grad(set_to_none=True)
    out = model(data)
    loss = out.mean()
    loss.backward()


def run_backward_only(model, data, optimizer) -> float:
    # Build graph first, then time backward only.
    optimizer.zero_grad(set_to_none=True)
    out = model(data)
    loss = out.mean()

    synchronize_if_cuda()
    start_time = timeit.default_timer()
    loss.backward()
    synchronize_if_cuda()
    end_time = timeit.default_timer()
    return (end_time - start_time) * 1000

print(f"device={device}, use_bf16={use_bf16}")
data = torch.randint(0, vocab_size, (batch_size, 128), device=device) # @inspect data
time_list = []
for mode in benchmark_modes:
    print(f"\nRunning mode: {mode}")
    for s in size:
        model = get_model(s, context_length=128).to(device=device, dtype=model_dtype) # @inspect
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if mode == "forward_only":
            model.eval()
            run_fn = lambda model=model, data=data: run_forward_only(model, data)
            returns_time_ms = False
        elif mode == "forward_backward":
            model.train()
            run_fn = lambda model=model, data=data, optimizer=optimizer: run_forward_backward(model, data, optimizer)
            returns_time_ms = False
        else:
            model.train()
            run_fn = lambda model=model, data=data, optimizer=optimizer: run_backward_only(model, data, optimizer)
            returns_time_ms = True

        elapsed_ms = benchmark(
            f"{mode} | Model size {s}",
            run_fn,
            num_warmups=num_warmups,
            num_trials=num_trials,
            run_returns_time_ms=returns_time_ms,
        )
        time_list.append((mode, s, d_model[s], d_ff[s], num_layers[s], num_heads[s], elapsed_ms[0], elapsed_ms[1]))
        print(f"Mode {mode}, model size {s}: {elapsed_ms[0]:.2f} ms (± {elapsed_ms[1]:.2f} ms)")

print("Benchmarking complete. Results:")
results_df = pd.DataFrame(
    time_list,
    columns=["mode", "model_size", "d_model", "d_ff", "num_layers", "num_heads", "time_ms", "stddev_ms"],
)
markdown_path = f"./benchmark_results_{model_dtype}_all_modes_warmup_{num_warmups}.md"
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write(results_df.to_markdown(index=False))
print(f"Results saved to {markdown_path}")