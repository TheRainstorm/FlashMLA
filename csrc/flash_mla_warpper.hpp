#include <iostream>
// #include <torch/python.h>
// #include <torch/nn/functional.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cutlass/fast_math.h>

#include "flash_mla.h"
#include "static_switch.h"

namespace onnxinfer {
namespace contrib {
namespace cuda {

using namespace std;

template <typename T>
int get_mla_metadata_mod(
    // at::Tensor &seqlens_k,
    void *seqlens_k_ptr, // kInt32
    int batch_size,
    const int num_heads_per_head_k,
    const int num_heads_k,
    int *tile_scheduler_metadata_ptr,
    int *num_splits_ptr,
    cudaDeviceProp *dprops,
    cudaStream_t *stream
) {
    // This should match the logic in the MLA kernel.
    static constexpr int block_size_m = 64;
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;

    // CHECK_DEVICE(seqlens_k);
    // TORCH_CHECK(seqlens_k.is_contiguous());
    // TORCH_CHECK(seqlens_k.dtype() == torch::kInt32);

    // int batch_size = seqlens_k.size(0);
    // int *seqlens_k_ptr = seqlens_k.data_ptr<int>();
    // auto options = seqlens_k.options();

    // auto dprops = at::cuda::getCurrentDeviceProperties();
    int sm_count = dprops->multiProcessorCount;
    int num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, block_size_m);

    // auto tile_scheduler_metadata = torch::empty({num_sm_parts, TileSchedulerMetaDataSize}, options);
    // auto num_splits = torch::empty({batch_size + 1}, options);
    // int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    // int *num_splits_ptr = num_splits.data_ptr<int>();
    cudaMalloc((void **)&tile_scheduler_metadata_ptr, num_sm_parts * TileSchedulerMetaDataSize * sizeof(int));
    cudaMalloc((void **)&num_splits_ptr, (batch_size + 1) * sizeof(int));

    // at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    // auto stream = at::cuda::getCurrentCUDAStream().stream();

    Mla_metadata_params params = {};
    params.seqlens_k_ptr = (int *)seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, *stream);

    // return {tile_scheduler_metadata, num_splits};
    return num_sm_parts;
}

template <typename T>
void flash_mla_page_kvcache_fwd(
    // q: torch.Tensor,
    // k_cache: torch.Tensor,
    void *q_ptr, // (batch_size, seqlen_q, num_heads, head_size), kT
    const int32_t batch_size,
    const int32_t seqlen_q_ori,
    const int32_t num_heads_ori,
    const int32_t head_size,
    void *kcache_ptr, // (num_blocks, page_block_size, num_heads_k), kT
    const int num_blocks,
    const int page_block_size,
    const int num_heads_k,
    void *vcache_ptr, // optionnal

    // block_table: torch.Tensor,
    // cache_seqlens: torch.Tensor,
    void *block_table_ptr, //  ( , max_num_blocks_per_seq), kInt32
    const int max_num_blocks_per_seq,
    void *cache_seqlens_k_ptr, // (batch_size, ), int32

    int head_size_v,

    const float softmax_scale,
    bool is_causal,

    // output
    void *o_ptr
    // void *softmax_lse_ptr = nullptr

    // cudaDeviceProp* dprops,
    // cudaStream_t stream,

    // compare to fmha_page_kvcache_fwd, not used params
    // void* k_ptr,
    // void* v_ptr,
    // const int32_t max_cache_seq_k,
    // const int32_t seqlen_k,
    
    // AllocatorPtr alloc,
    // int window_size_left,
    // int window_size_right,
    // const int32_t num_splits,
    // void* cache_batch_idx_ptr,
    // void* rotary_cos_ptr,
    // void* rotary_sin_ptr,
    // bool is_rotary_interleaved,
    // bool is_bf16
) {
    // auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    // TORCH_CHECK(is_sm90);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties(&dprops, 0); // 0 表示当前设备
    assert(dprops.major == 9 && dprops.minor == 0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // at::Tensor vcache = vcache_.has_value() ? vcache_.value() : kcache;
    // auto q_dtype = q.dtype();
    // TORCH_CHECK(q_dtype == torch::kT);
    // TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");

    // CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);
    
    // TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // CHECK_DEVICE(block_table);
    // TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    // TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    // TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    // TORCH_CHECK(head_size_v % 32 == 0, "head_size_v should be a multiple of 32");
    assert(head_size % 8 == 0);
    assert(head_size_v % 32 == 0);

    // TORCH_CHECK(batch_size > 0, "batch size must be postive");
    // TORCH_CHECK(num_heads_ori % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    assert(num_heads_k > 0);
    assert(num_heads_ori % num_heads_k == 0);

    if (seqlen_q_ori == 1) { is_causal = false; }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    // ???
    // q = q.view({batch_size, seqlen_q_ori, num_heads_k, ngroups, head_size}).transpose(2, 3)
    //         .reshape({batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;
    // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_k);
    // if (vcache_.has_value()) { CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_v); }
    // CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);

    // TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    // CHECK_DEVICE(seqlens_k);
    // CHECK_CONTIGUOUS(seqlens_k);
    // CHECK_SHAPE(seqlens_k, batch_size);

    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // auto opts = q.options();
    // at::Tensor out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    // at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    size_t out_size = batch_size * seqlen_q * num_heads * head_size_v * sizeof(T);
    float *softmax_lse_ptr;
    size_t softmax_lse_size = batch_size * num_heads * seqlen_q * sizeof(float);
    // cudaMalloc((void **)&out_ptr, out_size);
    cudaMalloc((void **)&softmax_lse_ptr, softmax_lse_size);
    // if (softmax_lse_ptr == nullptr) {
    //     cudaMalloc((void **)&softmax_lse_ptr, softmax_lse_size);
    // }
    
    Flash_fwd_mla_params params = {};
    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = (int *)cache_seqlens_k_ptr;
    params.h = num_heads;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
    // Set the pointers and strides.
    params.q_ptr = (void *)q_ptr; // ???
    params.k_ptr = (void *)kcache_ptr;
    params.v_ptr = (void *)vcache_ptr;
    params.o_ptr = (void *)o_ptr;
    params.softmax_lse_ptr = (void *)softmax_lse_ptr;
    // All stride are in elements, not bytes.
    auto get_stride = [](std::vector<int> dims) -> std::vector<int> {
        int size = dims.size();
        std::vector<int> strides(size, 1);
        // 计算stride
        for (int i = size - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    };
    auto q_stride = get_stride({batch_size, seqlen_q, num_heads, head_size});
    auto kcache_stride = get_stride({num_blocks, page_block_size, num_heads_k});
    auto out_stride = get_stride({batch_size, seqlen_q, num_heads, head_size_v});
    auto vcache_stride = kcache_stride;

    params.q_batch_stride = q_stride.at(0);
    params.k_batch_stride = kcache_stride.at(0);
    params.v_batch_stride = vcache_stride.at(0);
    params.o_batch_stride = out_stride.at(0);
    params.q_row_stride = q_stride.at(-3);
    params.k_row_stride = kcache_stride.at(-3);
    params.v_row_stride = vcache_stride.at(-3);
    params.o_row_stride = out_stride.at(-3);
    params.q_head_stride = q_stride.at(-2);
    params.k_head_stride = kcache_stride.at(-2);
    params.v_head_stride = vcache_stride.at(-2);
    params.o_head_stride = out_stride.at(-2);

    params.block_table = (int32_t *)block_table_ptr;
    // params.block_table_batch_stride = block_table.stride(0);
    params.block_table_batch_stride = max_num_blocks_per_seq;
    params.page_block_size = page_block_size;
    
    int *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;
    int num_heads_per_head_k = seqlen_q_ori * num_heads_ori / num_heads_k;
    int num_sm_parts = get_mla_metadata_mod<int>(cache_seqlens_k_ptr, batch_size, num_heads_per_head_k, num_heads_k,
                                    tile_scheduler_metadata_ptr, num_splits_ptr, &dprops, &stream);
    // TORCH_CHECK(tile_scheduler_metadata.dtype() == torch::kInt32, "tile_scheduler_metadata must have dtype int32");
    // TORCH_CHECK(tile_scheduler_metadata.size(1) == TileSchedulerMetaDataSize);
    // CHECK_DEVICE(tile_scheduler_metadata);
    // CHECK_CONTIGUOUS(tile_scheduler_metadata);
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_sm_parts = num_sm_parts;
    // TORCH_CHECK(num_splits.dtype() == torch::kInt32, "num_splits must have dtype int32");
    // CHECK_DEVICE(num_splits);
    // CHECK_CONTIGUOUS(num_splits);
    params.num_splits_ptr = num_splits_ptr;

    float *softmax_lseaccum_ptr;
    float *oaccum_ptr;
    cudaMalloc((void **)&softmax_lseaccum_ptr, (batch_size + params.num_sm_parts) * seqlen_q * num_heads * sizeof(float));
    cudaMalloc((void **)&oaccum_ptr, (batch_size + params.num_sm_parts) * seqlen_q * num_heads * head_size_v * sizeof(float));

    // at::Tensor softmax_lse_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    // at::Tensor out_accum = torch::empty({batch_size + params.num_sm_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(at::kFloat));
    params.softmax_lseaccum_ptr = softmax_lseaccum_ptr;
    params.oaccum_ptr = oaccum_ptr;

    // auto stream = at::cuda::getCurrentCUDAStream().stream();

    // TORCH_CHECK(head_size == 576);
    assert(head_size == 576);
    // run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(params, stream);
    run_mha_fwd_splitkv_mla<T, 576>(params, stream);

    // out = out.view({batch_size, seqlen_q_ori, ngroups, num_heads_k, head_size_v}).transpose(2, 3).reshape({batch_size, seqlen_q_ori, num_heads_ori, head_size_v});
    // softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(2, 3)
    //         .reshape({batch_size, num_heads_ori, seqlen_q_ori});
    // return {out, softmax_lse};

    cudaStreamDestroy(stream);
}

} // namespace cuda
} // namespace contrib
} // namespace onnxinfer