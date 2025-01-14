#ifndef _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H

#define QKV_WEIGHT_NUM 3
#define KV_WEIGHT_NUM 2

#include "flexflow/batch_config.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"
#include "flexflow/ops/inc_multihead_self_attention.h"

namespace FlexFlow {
namespace Kernels {
namespace IncMultiHeadAttention {

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_k_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
  
//   int page_idx = token_idx / kPagesize;
  
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize +
          token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}

// kv layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
__device__ __forceinline__ size_t get_v_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const num_heads,
                                                     int const head_dim) {
//   return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
//           kPagesize + token_idx % kPagesize) * /* page slot index */
//          num_heads *
//          head_dim;
return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize  +
          token_idx % kPagesize) * /* page slot index */
         num_heads *
         head_dim;
}

template <typename DT>
void compute_attention_kernel_prompt(IncMultiHeadSelfAttentionMeta *m,
                                     BatchConfig const *bc,
                                     int shard_id,
                                     ffStream_t stream);
template <typename DT>
void compute_attention_kernel_generation(IncMultiHeadSelfAttentionMeta const *m,
                                         BatchConfig const *bc,
                                         DT *output_ptr,
                                         ffStream_t stream);

template <typename DT>
void compute_qkv_kernel(IncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        int shard_id,
                        DT *output_ptr,
                        ffStream_t stream);

// [For the tokens in batch]
// Update the kv cache, and compact the q array.
// Source: qkv projeciton array of tokens in the batch.
// Destination: q&kv ptr took by the attention kernel.
// Note that the q&k here are the value after applying with position encoding.
template <typename DT>
void update_qkv_in_batch(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         cudaStream_t stream);
                         
template <typename DT>
void produce_output(IncMultiHeadSelfAttentionMeta const *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    ffStream_t stream);

template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id);

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
template <typename DT>
__global__ void
    apply_rotary_embedding(DT *input_ptr,
                           cuFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
                           int qProjSize,
                           int kProjSize,
                           int num_heads,
                           int num_tokens,
                           int num_kv_heads,
                           int q_block_size,
                           int k_block_size,
                           int q_array_size,
                           bool q_tensor);
#elif defined(FF_USE_HIP_ROCM)
template <typename DT>
__global__ void
    apply_rotary_embedding(DT *input_ptr,
                           hipFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
                           int qProjSize,
                           int kProjSize,
                           int num_heads,
                           int num_tokens,
                           int num_kv_heads,
                           int q_block_size,
                           int k_block_size,
                           int q_array_size,
                           bool q_tensor);
#endif

template <typename DT>
void pre_build_weight_kernel(IncMultiHeadSelfAttentionMeta const *m,
                             GenericTensorAccessorR const weight,
                             DataType data_type,
                             ffStream_t stream);
} // namespace IncMultiHeadAttention
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_INC_MULTIHEAD_SELF_ATTENTION_KERNELS_H
