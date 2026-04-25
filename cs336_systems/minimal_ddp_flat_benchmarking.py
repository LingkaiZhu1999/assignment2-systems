import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from time import perf_counter

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    # if torch.cuda.is_available():
    #     return torch.device(f"cuda:{index}")
    # else:
    #     return torch.device("cpu")
    return torch.device("cpu")
    
def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b

def summarize_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().float().reshape(-1)
    return f"{tuple(tensor.shape)}[mean={t.mean().item():.4f}, std={t.std(unbiased=False).item():.4f}, first={t[0].item():.4f}]"


def summarize_transformer_params(model: nn.Module) -> list[str]:
    """
    Summarize major Transformer parameter groups for compact logging.
    """
    groups = {
        "token_embeddings": None,
        "attn_qkv": None,
        "attn_out": None,
        "mlp": None,
        "norms": None,
        "lm_head": None,
    }

    for name, param in model.named_parameters():
        if "token_embeddings.weight" in name:
            groups["token_embeddings"] = param
        elif ".attn.q_proj.weight" in name and groups["attn_qkv"] is None:
            groups["attn_qkv"] = param
        elif ".attn.output_proj.weight" in name and groups["attn_out"] is None:
            groups["attn_out"] = param
        elif ".ffn." in name and groups["mlp"] is None:
            groups["mlp"] = param
        elif "ln" in name and groups["norms"] is None:
            groups["norms"] = param
        elif "lm_head.weight" in name:
            groups["lm_head"] = param

    summaries: list[str] = []
    for key, tensor in groups.items():
        if tensor is not None:
            summaries.append(f"{key}:{summarize_tensor(tensor)}")
    return summaries 


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_steps: int):
    setup(rank, world_size)
    device = get_device(rank)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(device)

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    model = BasicsTransformerLM(10000, context_length=128, d_model=768, num_layers=12, num_heads=12, d_ff=3072, rope_theta=10000.).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Each rank has own optimizer state
    step_total_times: list[float] = []
    grad_comm_times: list[float] = []

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = perf_counter()

        # Forward pass
        x = data
        x = model(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        flat_grads = parameters_to_vector([p.grad for p in params_with_grad])

        comm_start = perf_counter()
        dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)
        comm_elapsed = perf_counter() - comm_start
        
        vector_to_parameters(flat_grads, [p.grad for p in params_with_grad])

  

        # Update parameters
        optimizer.step()
        step_elapsed = perf_counter() - step_start

        step_total_times.append(step_elapsed)
        grad_comm_times.append(comm_elapsed)
        comm_pct = 100.0 * comm_elapsed / max(step_elapsed, 1e-12)

        print(
            f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item():.6f}, "
            f"step_time = {step_elapsed:.4f}s, grad_comm = {comm_elapsed:.4f}s ({comm_pct:.2f}%), "
            f"params = {summarize_transformer_params(model)}",
            flush=True,
        )

    total_metrics = torch.tensor(
        [sum(step_total_times), sum(grad_comm_times), float(len(step_total_times))], device=device
    )
    dist.all_reduce(total_metrics, op=dist.ReduceOp.SUM)
    global_avg_step_time = (total_metrics[0] / total_metrics[2]).item()
    global_avg_comm_time = (total_metrics[1] / total_metrics[2]).item()
    global_comm_pct = 100.0 * global_avg_comm_time / max(global_avg_step_time, 1e-12)

    if rank == 0:
        print(
            f"[timing_summary] global_avg_step_time = {global_avg_step_time:.4f}s, "
            f"global_avg_grad_comm_time = {global_avg_comm_time:.4f}s, "
            f"global_grad_comm_pct = {global_comm_pct:.2f}%",
            flush=True,
        )

    dist.destroy_process_group()


def generate_sample_data():
    batch_size = 32
    context_length = 128
    # data = torch.randn(batch_size, num_dim)
    data = torch.randint(0, 10000, (batch_size, context_length))
    return data

if __name__ == "__main__":
    world_size = 4
    data = generate_sample_data()
    mp.spawn(fn=data_parallelism_main, args=(world_size, data, 2), nprocs=world_size, join=True)

    # [timing_summary world size 4] global_avg_step_time = 30.7594s, global_avg_grad_comm_time = 0.5853s, global_grad_comm_pct = 1.90%
