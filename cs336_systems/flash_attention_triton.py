import triton
import triton.language as tl
import torch
import math
from einops import einsum

@triton.jit 
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, 
    O_ptr, L_ptr, 
    stride_qb, stride_qq, stride_qd, 
    stride_kb, stride_kk, stride_kd, 
    stride_vb, stride_vk, stride_vd, 
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, 
    scale, 
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr
    ):

    # Program indices, grid (query_tile_index, batch_index)
    query_tile_index = tl.program_id(0) 
    batch_index = tl.program_id(1)  
    # Offset each pointer with the corresponding batch index 
    # multiplied with the batch stride for each tensor 

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),  
        )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
        )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D), 
        strides=(stride_vk, stride_vd), 
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),  
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES,), 
        strides=(stride_lq,), 
        offsets=(query_tile_index * Q_TILE_SIZE,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,),  
    )

    # load Q, O, L from global memory, initialize m
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    O_block = tl.load(O_block_ptr, boundary_check=(0,), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    m = tl.full((Q_TILE_SIZE, ), -float('inf'), dtype=tl.float32)
    if IS_CAUSAL:
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load K, V from
        K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        if IS_CAUSAL:
            k_idx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = k_idx[None, :] <= q_idx[:, None]
            # attention scores
            S = tl.dot(Q_block, tl.trans(K_block)) * scale - ~mask * 1e6
        else:
            S = tl.dot(Q_block, tl.trans(K_block)) * scale
        m_prev = m
        m = tl.maximum(m, tl.max(S, axis=-1))
        P = tl.exp(S - m[:, None])
        L_block = tl.exp(m_prev - m) * L_block + tl.sum(P, axis=-1)
        O_block = tl.exp(m_prev - m)[:, None] * O_block + tl.dot(tl.cast(P, V_block.dtype), V_block)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_block = O_block / L_block[:, None]
    L_block = m + tl.log(L_block)

    tl.store(O_block_ptr, O_block, boundary_check=(0,))
    tl.store(L_block_ptr, L_block, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, N_q, d = Q.shape
        batch, N_k, d = K.shape
        B_q = 16
        B_k = 16
        scale = 1./ math.sqrt(d)

        O = torch.zeros((batch, N_q, d), device=Q.device)
        L = torch.zeros((batch, N_q), device=Q.device)
        grid = (triton.cdiv(N_q, B_q), batch)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            D=d, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k,
            IS_CAUSAL=is_causal
        )

        O = O.to(Q.dtype)
        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        O, L, Q, K, V = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch, N_q, d = Q.shape
        batch, N_k, d = K.shape
        
        if is_causal:
            mask = torch.tril(torch.ones((N_q, N_k), dtype=torch.bool, device=Q.device), diagonal=0)
            mask = mask[None, :]
        D = einsum(O, dO, "... N_q d, ... N_q d -> ... N_q d")
        D = torch.sum(D, dim=-1)
        S = einsum(Q, K, "... N_q d, ... N_k d -> ... N_q N_k") / math.sqrt(d)
        if is_causal:
            S = torch.where(mask, S, torch.full_like(S, float("-inf")))
        P = torch.exp(S - L.unsqueeze(-1)).to(V.dtype)
        dV = einsum(P, dO, "... N_q N_k, ... N_q d -> ... N_k d")
        dP = einsum(dO, V, "... N_q d, ... N_k d -> ... N_q N_k")
        dS = einsum(P, dP - D.unsqueeze(-1), "... N_q N_k, ... N_q N_k -> ... N_q N_k")
        if is_causal:
            dS = torch.where(mask, dS, torch.zeros_like(dS))
        dQ = einsum(dS, K, "... N_q N_k, ... N_k d -> ... N_q d") / math.sqrt(d)
        dK = einsum(dS, Q, "... N_q N_k, ... N_q d -> ... N_k d") / math.sqrt(d)


        return dQ, dK, dV, None  



        

