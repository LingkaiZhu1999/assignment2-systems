import torch
from einops import rearrange, einsum
import math

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: [Batch, N_q, d], K, V: [Batch, N_k, d]
        batch, N_q, d = Q.shape
        batch, N_k, d = K.shape
        B_q = 16
        B_k = 16
        tiled_Q = rearrange(Q, "... (num_tile B_q) d -> ... num_tile B_q d", B_q=B_q)
        tiled_K = rearrange(K, "... (num_tile B_k) d -> ... num_tile B_k d", B_k=B_k)
        tiled_V = rearrange(V, "... (num_tile B_k) d -> ... num_tile B_k d", B_k=B_k)
        # split
        T_q = torch.ceil(torch.div(N_q, B_q)).int()
        T_k = torch.ceil(torch.div(N_k, B_k)).int()

        O_full = torch.empty((batch, N_q, d), device=Q.device)
        L_full = torch.empty((batch, N_q), device=Q.device)
        for i in range(T_q):
            Q_block = tiled_Q[:, i]
            O = torch.zeros((batch, B_q, d), device=Q.device)
            l = torch.zeros((batch, B_q, ), device=Q.device)
            m = torch.full((batch, B_q, ), -torch.inf, device=Q.device)

            for j in range(T_k):
                K_block, V_block = tiled_K[:, j], tiled_V[:, j]
                S = einsum(Q_block, K_block, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)
                m_prev = m.clone()
                m = torch.maximum(m, torch.max(S, dim=-1).values)
                P = torch.exp(S - rearrange(m, "... B_q -> ... B_q 1"))
                l = torch.exp(m_prev - m) * l + torch.sum(P, dim=-1)
                O = rearrange(torch.exp(m_prev - m),"... -> ... 1") * O + \
                einsum(P, V_block, "... B_q B_k, ... B_k d -> ... B_q d")

            # O = einsum(torch.diag_embed(1./l), O, "... B_q B_q, ... B_q d -> ... B_q d")
            O = O / rearrange(l, "... -> ... 1")
            L = m + torch.log(l)
            O_full[:, i*B_q: B_q*(i+1), :] = O
            L_full[:, i*B_q: B_q*(i+1)] = L
        
        ctx.save_for_backward(O_full, L_full)
        
        return O_full
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
