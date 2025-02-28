#include "flash_mla_wrapper.h"

namespace onnxinfer {
namespace contrib {
namespace cuda {

/**
 * @brief Get the mla metadata object
 * 
 * @return num_sm_parts, TileSchedulerMetaDataSize, 
 *  tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), int
 *  num_splits_ptr: (batch_size + 1, )
 */
std::tuple<int, int, int*, int*> get_mla_metadata(
    void *seqlens_k_ptr, // (batch_size, ), int32
    int batch_size,
    const int num_heads_per_head_k,
    const int num_heads_k,
    cudaDeviceProp *dprops,
    cudaStream_t stream
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
    int *tile_scheduler_metadata_ptr, *num_splits_ptr;
    int size1 = num_sm_parts * TileSchedulerMetaDataSize * sizeof(int);
    int size2 = (batch_size + 1) * sizeof(int);
    cudaMalloc((void **)&tile_scheduler_metadata_ptr, size1);
    cudaMalloc((void **)&num_splits_ptr, size2);
    // cudaMemset(tile_scheduler_metadata_ptr, 0, size1);   // 对一致性时发现原版[:, -3:] 很多 -2147483648（32位全1） 这样的数字，而我的则均为 0。无法对上一致性，但是不影响后续计算
    // cudaMemset(num_splits_ptr, 0, size2);

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
    get_mla_metadata_func(params, stream);
    // cudaStreamSynchronize(stream);   // 不加也是正确的

    // return GPU address
    // return {tile_scheduler_metadata, num_splits};
    return std::make_tuple(num_sm_parts, TileSchedulerMetaDataSize, tile_scheduler_metadata_ptr, num_splits_ptr);
}

} // namespace cuda
} // namespace contrib
} // namespace onnxinfer