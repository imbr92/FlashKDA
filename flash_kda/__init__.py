import torch

from . import _C  # noqa: F401


def get_workspace_size(T_total, H, N=1):
    return torch.ops.flash_kda.get_workspace_size(T_total, H, N)


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    workspace=None,
    beta_transposed=None,
    state_slot_ids=None,
):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int32
            or int64, shape ``[N+1]``. When provided, ``B`` must be 1.
        workspace (torch.Tensor, optional): Reusable uint8 workspace. Allocated
            automatically when omitted.
        beta_transposed (torch.Tensor, optional): Caller-owned contiguous beta
            transpose with shape ``[H, B*T]``. Supplying it avoids an internal
            allocation and is required for allocation-free graph capture.
        state_slot_ids (torch.Tensor, optional): Int64 row ids with shape
            ``[N]`` mapping packed sequences into state pools. Negative ids
            read zero state and skip the final-state write.
    Notes:
        * Currently requires ``K = V = 128``.
        * Q, K, V, g, and beta may use aligned token strides. Their head and
          feature dimensions must remain dense. Output stays contiguous.
        * With ``state_slot_ids``, state tensors are pools shaped
          ``[pool_rows, H, V, K]`` instead of compact ``[N, H, V, K]`` tensors.
    """
    B, T_seq, H = q.shape[:3]
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    if workspace is None:
        workspace = torch.empty(
            get_workspace_size(B * T_seq, H, N),
            dtype=torch.uint8,
            device=q.device,
        )

    torch.ops.flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        beta_transposed=beta_transposed,
        state_slot_ids=state_slot_ids,
    )
