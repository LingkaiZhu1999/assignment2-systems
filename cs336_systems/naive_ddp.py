import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import math

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
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"

def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)
    device = get_device(rank)
    full_data = data.to(device)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(device)

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
    if rank == 0:
        single_params = [nn.Parameter(p.detach().clone()) for p in params]
        single_optimizer = torch.optim.AdamW(single_params, lr=1e-3)

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    if rank == 0:
        for _ in range(num_steps):
            x = full_data
            for param in single_params:
                x = x @ param
                x = F.gelu(x)
            single_loss = x.square().mean()
            single_loss.backward()
            single_optimizer.step()

        print("[compare] DDP(rank0) vs single-process parameter deltas:")
        for layer_idx, (ddp_param, single_param) in enumerate(zip(params, single_params)):
            max_abs_diff = (ddp_param.detach() - single_param.detach()).abs().max().item()
            are_close = torch.allclose(ddp_param.detach(), single_param.detach(), atol=1e-6, rtol=1e-5)
            print(f"[compare] layer={layer_idx}, max_abs_diff={max_abs_diff:.6e}, allclose={are_close}")

    dist.destroy_process_group()


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data

if __name__ == "__main__":
    world_size = 4
    data = generate_sample_data()
    mp.spawn(fn=data_parallelism_main, args=(world_size, data, 2, 2), nprocs=world_size, join=True)
