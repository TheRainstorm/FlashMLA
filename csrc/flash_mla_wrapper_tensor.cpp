#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.hpp>
#include <cutlass/fast_math.h>

#include "flash_mla_wrapper.h"

std::vector<at::Tensor>
get_mla_metadata_wrapper(
    at::Tensor &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k
) {
    int batch_size = seqlens_k.size(0);
    auto opts = seqlens_k.options();

    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int num_sm_parts, TileSchedulerMetaDataSize, *tile_scheduler_metadata_ptr, *num_splits_ptr;
    // <int> 仅用于占位
    std::tie(num_sm_parts, TileSchedulerMetaDataSize, tile_scheduler_metadata_ptr, num_splits_ptr) = onnxinfer::contrib::cuda::get_mla_metadata(
        seqlens_k.data_ptr(),
        batch_size,
        num_heads_per_head_k,
        num_heads_k,
        dprops,
        stream
    );
    // TODO: from_blob 可能需要手动释放显存
    torch::Tensor tile_scheduler_metadata = torch::from_blob(tile_scheduler_metadata_ptr, {num_sm_parts, TileSchedulerMetaDataSize}, {TileSchedulerMetaDataSize, 1}, opts.dtype(torch::kInt32));
    torch::Tensor num_splits = torch::from_blob(num_splits_ptr, {batch_size + 1}, {1}, opts.dtype(torch::kInt32));
    return {tile_scheduler_metadata, num_splits};
}

std::vector<at::Tensor>
mha_fwd_kvcache_mla_wrapper(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    std::optional<const at::Tensor> &vcache_,    // num_blocks x page_block_size x num_heads_k x head_size_v
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits                 // batch_size + 1
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;

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

    auto opts = q.options();
    at::Tensor out = torch::empty({batch_size, seqlen_q_ori, num_heads_ori, head_size_v}, opts);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads_ori, seqlen_q_ori}, opts.dtype(at::kFloat));

    void *o_ptr = out.data_ptr();
    void *softmax_lse_ptr = softmax_lse.data_ptr();

    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();
    int num_sm_parts = tile_scheduler_metadata.size(0);

    // 这个如果在 flash_mla_page_kvcache_fwd 手动 cudaMalloc，则性能直接相差数量级
    at::Tensor softmax_lse_accum = torch::empty({batch_size + num_sm_parts, num_heads_ori, seqlen_q_ori}, opts.dtype(at::kFloat));
    at::Tensor out_accum = torch::empty({batch_size + num_sm_parts, num_heads_ori, seqlen_q_ori, head_size_v}, opts.dtype(at::kFloat));
    void *softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    void *oaccum_ptr = out_accum.data_ptr();

    auto q_dtype = q.dtype();
    if (q_dtype == torch::kBFloat16) {
        onnxinfer::contrib::cuda::flash_mla_page_kvcache_fwd<cutlass::bfloat16_t>(
            q_ptr,batch_size,seqlen_q_ori,num_heads_ori,head_size,
            kcache_ptr,num_blocks,page_block_size,num_heads_k,vcache_ptr,block_table_ptr,max_num_blocks_per_seq,cache_seqlens_k_ptr,head_size_v,softmax_scale,is_causal,
            tile_scheduler_metadata_ptr,num_sm_parts,num_splits_ptr,
            dprops,stream,
            o_ptr,softmax_lse_ptr,oaccum_ptr,softmax_lseaccum_ptr
        );
    }
    #ifndef FLASH_MLA_DISABLE_FP16
    else if (q_dtype == torch::kHalf) {
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

            // get_mla_metadata
            tile_scheduler_metadata_ptr,
            num_sm_parts,
            num_splits_ptr,

            dprops,
            stream,

            // output
            o_ptr,
            softmax_lse_ptr,
            oaccum_ptr,
            softmax_lseaccum_ptr
        );
    }
    #endif
    else {
        TORCH_CHECK(false, "Unsupported tensor dtype for query");
    }
    
    // flash_mla_page_kvcache_fwd 没有实现转置，在外面临时实现转置
    const int ngroups = num_heads_ori / num_heads_k;
    out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3)
            .reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
            .reshape({batch_size, num_heads_ori, seqlen_q_ori});
    
    return {out, softmax_lse};
}