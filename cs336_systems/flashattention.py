from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... q d"],
        K: Float[Tensor, "... k d"],
        V: Float[Tensor, "... k d_v"],
        is_causal: bool = False,
    ) -> Float[Tensor, "... q d_v"]:
        if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
            raise ValueError("Q, K, and V must each have at least 2 dimensions.")
        if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:
            raise ValueError("Q, K, and V must share the same batch dimensions.")
        if Q.shape[-1] != K.shape[-1]:
            raise ValueError("Q and K must have the same head dimension.")
        if K.shape[-2] != V.shape[-2]:
            raise ValueError("K and V must have the same sequence length.")

        batch_shape = Q.shape[:-2]
        q_len = Q.shape[-2]
        k_len = K.shape[-2]
        d = Q.shape[-1]
        d_v = V.shape[-1]
        batch_size = math.prod(batch_shape) if batch_shape else 1

        q_flat = Q.reshape(batch_size, q_len, d)
        k_flat = K.reshape(batch_size, k_len, d)
        v_flat = V.reshape(batch_size, k_len, d_v)

        acc_dtype = torch.float32
        scale = 1.0 / math.sqrt(d)
        row_block = 16
        col_block = 16

        q_acc = q_flat.to(acc_dtype)
        k_acc = k_flat.to(acc_dtype)
        v_acc = v_flat.to(acc_dtype)

        o_acc = torch.empty((batch_size, q_len, d_v), device=Q.device, dtype=acc_dtype)
        lse = torch.empty((batch_size, q_len), device=Q.device, dtype=acc_dtype)

        for i in range(0, q_len, row_block):
            i_end = min(i + row_block, q_len)
            q_i = q_acc[:, i:i_end, :]
            block_rows = q_i.shape[1]

            m_i = torch.full(
                (batch_size, block_rows),
                -torch.inf,
                device=Q.device,
                dtype=acc_dtype,
            )
            l_i = torch.zeros(
                (batch_size, block_rows),
                device=Q.device,
                dtype=acc_dtype,
            )
            o_i = torch.zeros(
                (batch_size, block_rows, d_v),
                device=Q.device,
                dtype=acc_dtype,
            )

            for j in range(0, k_len, col_block):
                j_end = min(j + col_block, k_len)
                k_j = k_acc[:, j:j_end, :]
                v_j = v_acc[:, j:j_end, :]

                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2)) * scale
                if is_causal:
                    q_idx = torch.arange(i, i_end, device=Q.device)[:, None]
                    k_idx = torch.arange(j, j_end, device=Q.device)[None, :]
                    causal_mask = q_idx >= k_idx
                    s_ij = torch.where(causal_mask.unsqueeze(0), s_ij, -1e6)

                m_tilde = s_ij.max(dim=-1).values
                m_new = torch.maximum(m_i, m_tilde)

                p_tilde = torch.exp(s_ij - m_new.unsqueeze(-1))
                alpha = torch.exp(m_i - m_new)
                l_new = alpha * l_i + p_tilde.sum(dim=-1)
                o_new = alpha.unsqueeze(-1) * o_i + torch.matmul(p_tilde, v_j)

                m_i = m_new
                l_i = l_new
                o_i = o_new

            o_acc[:, i:i_end, :] = o_i / l_i.unsqueeze(-1)
            lse[:, i:i_end] = m_i + torch.log(l_i)

        output = o_acc.reshape(*batch_shape, q_len, d_v).to(Q.dtype)
        lse = lse.reshape(*batch_shape, q_len)

        ctx.save_for_backward(lse, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx,
        dO: Float[Tensor, "... q d_v"],
    ):
        raise NotImplementedError("Backward is intentionally not implemented yet.")


FlashAttention2AutogradFunctionPyTorch = FlashAttention2PyTorch


def get_flashattention_autograd_function_pytorch():
    return FlashAttention2PyTorch


def flashattention2_pytorch_reference(
    Q: Float[Tensor, "... q d"],
    K: Float[Tensor, "... k d"],
    V: Float[Tensor, "... k d_v"],
    is_causal: bool = False,
) -> Float[Tensor, "... q d_v"]:
    return FlashAttention2PyTorch.apply(Q, K, V, is_causal)
