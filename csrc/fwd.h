#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <
    int D,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool HasCheckpoint = false,
    bool IsVarlen = true,
    typename SeqlenT = int64_t>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    int64_t g_token_stride,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int state_rows,
    int64_t const* state_slot_ids_ptr,
    SeqlenT const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int num_sms,
    cudaStream_t stream
);
