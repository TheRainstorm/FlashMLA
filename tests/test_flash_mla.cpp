#include <torch/script.h>
#include <iostream>
#include <vector>

// 声明自定义函数
std::vector<at::Tensor>
get_mla_metadata_wrapper(
    at::Tensor &seqlens_k,
    const int num_heads_per_head_k,
    const int num_heads_k
);

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
);

int main() {
    // 设置 CUDA 设备
    torch::Device device(torch::kCUDA, 0);
    auto dtype = torch::kBFloat16;
    torch::manual_seed(0);
    
    // 测试参数配置
    const int b = 128;         // batch_size
    const int s_q = 1;         // 查询序列长度
    const int mean_sk = 4096;  // 平均序列长度
    const int h_q = 16;        // 查询头数
    const int h_kv = 1;        // K/V 头数
    const int d = 576;         // 键维度
    const int dv = 512;        // 值维度
    const int block_size = 64; // 分块大小
    const bool causal = true;  // 是否因果注意力

    // 生成序列长度张量
    torch::Tensor cache_seqlens = torch::full({b}, mean_sk, torch::kInt32).to(device);

    // 计算最大分块数
    const int max_seqlen_pad = ((mean_sk + 256 - 1) / 256) * 256; // 对齐到 256
    const int max_num_blocks = max_seqlen_pad / block_size;
    
    // 生成块表 [b, max_blocks_per_seq]
    torch::Tensor block_table = torch::arange(b * max_num_blocks, torch::kInt32)
                                    .view({b, max_num_blocks}).to(device);

    // 生成 K/V 缓存
    const int num_blocks = b * max_num_blocks;
    torch::Tensor kcache = torch::randn(
        {num_blocks, block_size, h_kv, d}, dtype).to(device);
    // torch::Tensor vcache = torch::randn(
    //     {num_blocks, block_size, h_kv, dv}, dtype).to(device);

    // 生成查询张量 [b, s_q, h_q, d]
    torch::Tensor q = torch::randn({b, s_q, h_q, d}, dtype).to(device);

    // 调用目标函数
    std::optional<const torch::Tensor> vcache_opt;
    const float softmax_scale = 1.0f / std::sqrt(d);
    
    int num_heads_per_head_k = s_q * h_q / h_kv;
    auto res = get_mla_metadata_wrapper(cache_seqlens, num_heads_per_head_k, h_kv);
    auto tile_scheduler_metadata = res[0];
    auto num_splits = res[1];

    auto outputs = mha_fwd_kvcache_mla_wrapper(
        q, kcache, vcache_opt, dv, cache_seqlens,
        block_table, softmax_scale, causal,
        tile_scheduler_metadata, num_splits
    );

    // 验证输出形状
    torch::Tensor out = outputs[0];
    torch::Tensor lse = outputs[1];
    
    bool shape_check = true;
    shape_check &= (out.sizes() == std::vector<int64_t>{b, s_q, h_q, dv});
    shape_check &= (lse.sizes() == std::vector<int64_t>{b, h_q, s_q});

    if (shape_check) {
        std::cout << "Shape check passed!\n";
        // 可添加数值检查（此处示例仅检查是否存在 NaN）
        if (torch::isnan(out).any().item().toBool()) {
            std::cerr << "Error: Output contains NaN values!\n";
            return 1;
        }
        std::cout << "Basic functionality test passed!\n";
    } else {
        std::cerr << "Shape check failed!\n";
        std::cerr << "Out shape: [";
        for (auto dim : out.sizes()) std::cerr << dim << " ";
        std::cerr << "]\nLSE shape: [";
        for (auto dim : lse.sizes()) std::cerr << dim << " ";
        std::cerr << "]\n";
        return 1;
    }
    
    return 0;
}