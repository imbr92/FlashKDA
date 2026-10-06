import pytest
import torch

import flash_kda


@torch.inference_mode()
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_strided_inputs_indexed_state_pool_and_cuda_graph(state_dtype):
    torch.manual_seed(19)
    tokens, heads, dim, pool_rows = 48, 2, 128, 5
    packed = torch.randn(tokens, 4 * heads * dim + 64, device="cuda", dtype=torch.bfloat16)
    q = packed[:, : heads * dim].view(1, tokens, heads, dim)
    k = packed[:, heads * dim : 2 * heads * dim].view(1, tokens, heads, dim)
    v = packed[:, 2 * heads * dim : 3 * heads * dim].view(1, tokens, heads, dim)
    g = packed[:, 3 * heads * dim : 4 * heads * dim].view(1, tokens, heads, dim)
    beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 31, 41, tokens], device="cuda", dtype=torch.int32)
    state_slot_ids = torch.tensor([3, -1, 1], device="cuda", dtype=torch.int64)
    original = torch.randn(pool_rows, heads, dim, dim, device="cuda", dtype=state_dtype)

    expected_out = torch.empty_like(q)
    expected_state = torch.stack((original[3], torch.zeros_like(original[0]), original[1]))
    flash_kda.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        dim**-0.5,
        expected_out,
        a_log,
        dt_bias,
        -5.0,
        initial_state=expected_state,
        final_state=expected_state,
        cu_seqlens=cu_seqlens,
    )

    state = original.clone()
    out = torch.empty_like(q)
    workspace = torch.empty(
        flash_kda.get_workspace_size(tokens, heads, cu_seqlens.numel() - 1),
        device="cuda",
        dtype=torch.uint8,
    )
    beta_transposed = beta.permute(0, 2, 1).reshape(heads, tokens).contiguous()

    def run():
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            dim**-0.5,
            out,
            a_log,
            dt_bias,
            -5.0,
            initial_state=state,
            final_state=state,
            cu_seqlens=cu_seqlens,
            workspace=workspace,
            beta_transposed=beta_transposed,
            state_slot_ids=state_slot_ids,
        )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(state[3], expected_state[0], rtol=0, atol=0)
    torch.testing.assert_close(state[1], expected_state[2], rtol=0, atol=0)
    torch.testing.assert_close(state[[0, 2, 4]], original[[0, 2, 4]], rtol=0, atol=0)

    expected_graph_out = out.clone()
    expected_graph_state = state[[3, 1]].clone()
    state.copy_(original)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    state.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    state.copy_(original)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out, expected_graph_out, rtol=0, atol=0)
    torch.testing.assert_close(state[[3, 1]], expected_graph_state, rtol=0, atol=0)
    torch.testing.assert_close(state[[0, 2, 4]], original[[0, 2, 4]], rtol=0, atol=0)
