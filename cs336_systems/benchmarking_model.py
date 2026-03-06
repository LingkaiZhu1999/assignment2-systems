import torch
import time
import pandas as pd
from typing import Callable

from cs336_basics.model import BasicsTransformerLM

# hyperparameters
vocab_size = 10000
batch_size =  4
size = ["small", "medium", "large", "xlarge", "2.7B"]
d_model = {"small": 768, "medium": 1024, "large": 1280, "xlarge": 1600, "2.7B": 2560}
d_ff = {"small": 3072, "medium": 4096, "large": 5120, "xlarge": 6400, "2.7B": 102400}
num_layers = {"small": 12, "medium": 24, "large": 36, "xlarge": 48, "2.7B": 32}
num_heads = {"small": 12, "medium": 16, "large": 20, "xlarge": 25, "2.7B": 32}

def mean(lst: list[float]) -> float:
    return sum(lst) / len(lst)

def get_model(size, context_length, rope_theta=10000.):
    return BasicsTransformerLM(vocab_size,
        context_length,
        d_model[size],
        num_layers[size],
        num_heads[size],
        d_ff[size],
        rope_theta,)

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times
    mean_time = mean(times) # @inspect mean_time
    return mean_time

def run(model, data, steps, forward_only=False):
    for _ in range(steps):
        out = model(data)
        loss = out.mean()
        if not forward_only:
            loss.backward()

data = torch.randint(0, vocab_size, (batch_size, 128)).cuda() # @inspect data
time_list = []
for s in size:
    model = get_model(s, context_length=128).cuda() # @inspect
    model.train()
    elapsed_ms = benchmark(f"Model size {s}", lambda: run(model, data, steps=5), num_warmups=3)
    time_list.append((s, elapsed_ms))
    print(f"Model size {s}: {elapsed_ms:.2f} ms")

results_df = pd.DataFrame(time_list, columns=["model_size", "time_ms"])
with open("benchmark_results.md", "w", encoding="utf-8") as f:
    f.write(results_df.to_markdown(index=False) + "\n")
print("Saved benchmark table to benchmark_results.md")
