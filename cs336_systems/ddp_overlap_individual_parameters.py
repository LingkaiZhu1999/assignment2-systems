import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.handles = []
        self.module = module
        for param in self.module.parameters():
            with torch.no_grad():
                dist.broadcast(tensor=param, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(lambda p: self.handles.append(
                    dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.AVG, async_op=True))
                    if p.grad is not None else None)

        

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
