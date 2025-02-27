#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.hpp>
// #include <cutlass/fast_math.h>

// #include "flash_mla.h"
// #include "static_switch.h"

#include "flash_mla_wrapper.hpp"

// void flash_mla_page_kvcache_fwd(
//     void *q_ptr, // (batch_size, seqlen_q, num_heads, head_size), kBFloat16
//     const int32_t batch_size,
//     const int32_t seqlen_q_ori,
//     const int32_t num_heads_ori,
//     const int32_t head_size,
//     void *kcache_ptr, // (num_blocks, page_block_size, num_heads_k), kBFloat16
//     const int num_blocks,
//     const int page_block_size,
//     const int num_heads_k,
//     void *vcache_ptr, // optionnal

//     void *block_table_ptr, //  ( , max_num_blocks_per_seq), kInt32
//     const int max_num_blocks_per_seq,
//     void *cache_seqlens_k_ptr, // (batch_size, ), int32

//     int head_size_v,

//     const float softmax_scale,
//     bool is_causal,

//     void *o_ptr
// );

std::vector<at::Tensor>
mha_fwd_kvcache_mla2(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    std::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal
) {
    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;
    auto opts = q.options();

    void* q_ptr = q.data_ptr();            // (batch_size, seqlen_q, num_heads, head_size), kBFloat16
    const int32_t batch_size = q.size(0);
    const int32_t seqlen_q_ori = q.size(1);
    const int32_t num_heads_ori = q.size(2);
    const int32_t head_size = q.size(3);
    void* kcache_ptr = kcache.data_ptr();      // (num_blocks, page_block_size, num_heads_k), kBFloat16
    const int num_blocks = kcache.size(0);
    const int page_block_size = kcache.size(1);
    const int num_heads_k = kcache.size(2);
    void* vcache_ptr = vcache.data_ptr();    // optionnal

    void* block_table_ptr = block_table.data_ptr();      //  ( , max_num_blocks_per_seq), kInt32
    const int max_num_blocks_per_seq = block_table.size(1);
    void* cache_seqlens_k_ptr = seqlens_k.data_ptr();   // (batch_size, ), int32

    at::Tensor out = torch::empty({batch_size, seqlen_q_ori, num_heads_ori, head_size_v}, opts);
    void* o_ptr = out.data_ptr();

    onnxinfer::contrib::cuda::flash_mla_page_kvcache_fwd<cutlass::half_t>(
        q_ptr,
        batch_size,
        seqlen_q_ori,
        num_heads_ori,
        head_size,

        kcache_ptr,
        num_blocks,
        page_block_size,
        num_heads_k,
        vcache_ptr,
        block_table_ptr,
        max_num_blocks_per_seq,
        cache_seqlens_k_ptr,
        head_size_v,
        softmax_scale,
        is_causal,
        o_ptr
    );
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads_ori, seqlen_q_ori}, opts.dtype(at::kFloat));
    return {out, softmax_lse};
}