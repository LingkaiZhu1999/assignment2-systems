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


@triton.jit
def flash_bwd_kernel(Q_ptr, dQ_ptr, K_ptr, dK_ptr, V_ptr, dV_ptr,
    dO_ptr, L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd, 
    stride_dqb, stride_dqq, stride_dqd,
    stride_kb, stride_kk, stride_kd, 
    stride_dkb, stride_dkk, stride_dkd, 
    stride_vb, stride_vk, stride_vd, 
    stride_dvb, stride_dvk, stride_dvd, 
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS, 
    scale, 
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr):

    # Program indices, grid (query_tile_index, batch_index)
    key_tile_index = tl.program_id(0) 
    batch_index = tl.program_id(1)  
    # Offset each pointer with the corresponding batch index 
    # multiplied with the batch stride for each tensor 

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),  
        )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
        )
    
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb, 
        shape=(N_KEYS, D), 
        strides=(stride_dkk, stride_dkd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
        )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D), 
        strides=(stride_vk, stride_vd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
    )
    
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb, 
        shape=(N_KEYS, D), 
        strides=(stride_dvk, stride_dvd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0),  
    )


    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob, 
        shape=(N_QUERIES, D), 
        strides=(stride_doq, stride_dod), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),  
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES,), 
        strides=(stride_lq,), 
        offsets=(0,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,),  
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db, 
        shape=(N_QUERIES,), 
        strides=(stride_dq,), 
        offsets=(0,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,),  
    )


    # load K, V from global memory, initialize dK, dV=0
    K_block = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    
    dK_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    if IS_CAUSAL:
        k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    d_offsets = tl.arange(0, D)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # load Q, O, dO, dQ from global memory
        Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_block = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")
        L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        if IS_CAUSAL:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = k_idx[None, :] <= q_idx[:, None]
            # attention scores
            S = tl.dot(Q_block, tl.trans(K_block)) * scale - ~mask * 1e6
        else:
            S = tl.dot(Q_block, tl.trans(K_block)) * scale
        P = tl.exp(S - L_block[:, None])
        dV_block += tl.dot(tl.trans(P), dO_block)
        dP = tl.dot(dO_block, tl.trans(V_block))
        dS = P * (dP - D_block[:, None]) * scale
        if IS_CAUSAL:
            dS = tl.where(mask, dS, 0.0)
        
        dQ_block = tl.dot(dS, K_block)
        q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        dQ_ptrs = (
            dQ_ptr
            + batch_index * stride_dqb
            + q_offsets[:, None] * stride_dqq
            + d_offsets[None, :] * stride_dqd
        )
        dQ_mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)
        tl.atomic_add(dQ_ptrs, dQ_block, mask=dQ_mask)

        dK_block += tl.dot(tl.trans(dS), Q_block)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_block, boundary_check=(0,))
    tl.store(dV_block_ptr, dV_block, boundary_check=(0,))


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
        B_q = 16
        B_k = 16
        scale = 1./ math.sqrt(d)
        grid = (triton.cdiv(N_k, B_k), batch)

        dQ = torch.zeros((batch, N_q, d), device=Q.device)
        dK = torch.zeros((batch, N_k, d), device=Q.device)
        dV = torch.zeros((batch, N_k, d), device=Q.device)
        D = torch.sum(dO * O, dim=-1)
        
        flash_bwd_kernel[grid](
            Q, dQ, K, dK, V, dV, dO, L, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            N_q, N_k, scale,
            D=d, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k,
            IS_CAUSAL=is_causal
        )



        return dQ, dK, dV, None



        
