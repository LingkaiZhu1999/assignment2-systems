import os
import torch
import time
import triton
import triton.language as tl

from typing import Callable
from torch.utils.cpp_extension import load_inline
from torch.profiler import ProfilerActivity


def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def pytorch_gelu(x: torch.Tensor):
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

def create_cuda_gelu():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    cuda_gelu_src = open("gelu.cu").read()

    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    build_dir = "var/cuda_gelu"
    os.makedirs(build_dir, exist_ok=True)

    if not torch.cuda.is_available():
        return None

    module = load_inline(
        name="inline_gelu",
        cpp_sources=[cpp_gelu_src],
        cuda_sources=[cuda_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        build_directory=build_dir,
    )

    return module.gelu

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

   # Handle boundary
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    # Store
    tl.store(y_ptr + offsets, y, mask=mask)



def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table

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

if __name__ == "__main__":
    x = torch.tensor([1.0])

    # default Pytorch implementation (fused)
    y1 = pytorch_gelu(x)

    # manual implementation (unfused)
    y2 = manual_gelu(x)

    assert torch.allclose(y1, y2), f"Expected {y1} but got {y2}"

    
    manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))

    cuda_gelu = create_cuda_gelu()
    if cuda_gelu is not None:
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))  
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))

    x = torch.randn(8192, device=get_device())
    y1 = triton_gelu(x)

    #print_ptx_main()  # Look at the generated instructions


    #Let's now benchmark it compared to the PyTorch and CUDA implementations.
    #Remember to set TRITON_INTERPRET=0 for good performance.
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time

    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))

    print(f"Manual GELU time: {manual_time:.2f} ms")
    print(f"PyTorch GELU time: {pytorch_time:.2f} ms")
    print(f"CUDA GELU time: {cuda_time:.2f} ms")
    print(f"Triton GELU time: {triton_time:.2f} ms")
