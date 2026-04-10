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
        # tiled_Q = rearrange(Q, "... (num_tile B_q) d -> ... num_tile B_q d", B_q=B_q)
        # tiled_K = rearrange(K, "... (num_tile B_k) d -> ... num_tile B_k d", B_k=B_k)
        # tiled_V = rearrange(V, "... (num_tile B_k) d -> ... num_tile B_k d", B_k=B_k)
        # split
        T_q = torch.ceil(torch.div(N_q, B_q)).int()
        T_k = torch.ceil(torch.div(N_k, B_k)).int()

        O_full = torch.empty((batch, N_q, d), device=Q.device)
        L_full = torch.empty((batch, N_q), device=Q.device)
        for i in range(T_q):
            # Q_block = tiled_Q[:, i]
            Q_block = Q[:, i*B_q: (i+1)*B_q, :]
            O = torch.zeros((batch, B_q, d), device=Q.device)
            l = torch.zeros((batch, B_q, ), device=Q.device)
            m = torch.full((batch, B_q, ), -torch.inf, device=Q.device)

            for j in range(T_k):
                # K_block, V_block = tiled_K[:, j], tiled_V[:, j]
                K_block, V_block = K[:, j*B_k:(j+1)*B_k, :], V[:, j*B_k:(j+1)*B_k, :]
                S = einsum(Q_block, K_block, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)
                if is_causal:
                    q_start = i * B_q
                    k_start = j * B_k
                    q_idx = q_start + torch.arange(S.shape[-2], device=Q.device)
                    k_idx = k_start + torch.arange(S.shape[-1], device=Q.device)
                    causal_mask = k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)
                    S = torch.where(causal_mask.unsqueeze(0), S, torch.full_like(S, float("-inf")))
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
        
        ctx.save_for_backward(O_full, L_full, Q, K, V)
        ctx.is_causal = is_causal
        
        return O_full
    
    @staticmethod
    def backward(ctx, dO):
        O, L, Q, K, V = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch, N_q, d = Q.shape
        batch, N_k, d = K.shape
        D = einsum(O, dO, "... N_q d, ... N_q d -> ... N_q d")
        D = torch.sum(D, dim=-1)
        S = einsum(Q, K, "... N_q d, ... N_k d -> ... N_q N_k") / math.sqrt(d)
        if is_causal:
            q_idx = torch.arange(N_q, device=Q.device)
            k_idx = torch.arange(N_k, device=Q.device)
            mask = (k_idx.unsqueeze(0) <= q_idx.unsqueeze(1)).unsqueeze(0)
            S = torch.where(mask, S, torch.full_like(S, float("-inf")))
        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, "... N_q N_k, ... N_q d -> ... N_k d")
        dP = einsum(dO, V, "... N_q d, ... N_k d -> ... N_q N_k")
        dS = einsum(P, dP - D.unsqueeze(-1), "... N_q N_k, ... N_q N_k -> ... N_q N_k")
        if is_causal:
            dS = torch.where(mask, dS, torch.zeros_like(dS))
        dQ = einsum(dS, K, "... N_q N_k, ... N_k d -> ... N_q d") / math.sqrt(d)
        dK = einsum(dS, Q, "... N_q N_k, ... N_q d -> ... N_k d") / math.sqrt(d)


        return dQ, dK, dV, None        
    


    # @staticmethod
    # def backward(ctx, dO):
    #     O, L, Q, K, V = ctx.saved_tensors
    #     batch, N_q, d = Q.shape
    #     batch, N_k, d = K.shape
    #     B_q = 16
    #     B_k = 16
    #     T_q = torch.ceil(torch.div(N_q, B_q)).int()
    #     T_k = torch.ceil(torch.div(N_k, B_k)).int()

    #     dQ = torch.zeros((N_q, d), device=Q.device)

    #     D = einsum(O, dO, "... N_q d, ... N_q d -> ... N_q d")
    #     D = torch.sum(D, dim=-1)

    #     for j in range(T_k):
    #         K_block = K[:, j * B_k: (j + 1) * B_k, :]
    #         V_block = V[:, j * B_k: (j + 1) * B_k, :]
    #         dK_block = torch.zeros((B_k, d), device=Q.device)
    #         dV_block = torch.zeros((B_k, d), device=Q.device)
    #         for i in range(T_q):
    #             Q_block = Q[:, i * B_q: (i + 1)* B_q, :]
    #             O_block = O[:, i * B_q: (i + 1)* B_q, :]
    #             dO_block = dO[:, i * B_q: (i + 1)* B_q, :]
    #             dQ_block = dQ[:, i * B_q: (i + 1)* B_q, :]
    #             L_block = L[:, i * B_q: (i + 1)* B_q]
    #             D_block = D[:, i * B_q: (i + 1)* B_q, :]

    #             S_block = einsum(Q_block, K_block, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)
    #             P_block = torch.exp(S_block - L_block.unsqueeze(-1))
    #             dV_block = einsum(P_block, dO_block, "... B_q B_k, ... B_q d -> ... B_k d")
    #             dP_block = einsum(dO_block, V_block, "... B_q d, ... B_k d -> ... B_q B_k")
    #             dS_block = einsum(P_block, dP_block - D_block.unsqueeze(-1), "... B_q B_k, ... B_q B_k -> ... B_q B_k")
    #             dQ_block = einsum(dS_block, K_block, "... B_q B_k, ... B_k d -> ... B_q d") / math.sqrt(d)
    #             dK_block = einsum(dS_block, Q_block, "... B_q B_k, ... B_q d -> ... B_k d") / math.sqrt(d)

    #             dQ[:, j*B_q: (j+1):B_q, :] = dQ_block 
    #         dK[:, ]
    #     return dQ, dK, dV 
