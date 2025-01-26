/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flashinfer/decode_attention_decl.cuh"
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/page_manager.h"
#include "flexflow/utils/cuda_helper.h"
#include <math_constants.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

using flashinfer::BatchDecodeHandler;
using flashinfer::BatchDecodeWithPagedKVCacheWrapperDispatched;
using flashinfer::BatchPrefillHandler;
using flashinfer::BatchPrefillWithPagedKVCacheWrapperDispatched;
using flashinfer::LogitsPostHook;
using flashinfer::MaskMode;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

template <typename DT>
void incr_attention(IncMultiHeadSelfAttentionMeta *m,
                    BatchConfig const *bc,
                    DT *output_ptr,
                    cudaStream_t stream) {
  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // global constant parameters
  uint32_t const num_q_heads = m->num_q_heads;
  uint32_t const num_kv_heads = m->num_kv_heads;
  uint32_t const head_dim = m->qk_dim;
  uint32_t const batch_size = bc->num_active_requests();
  float const sm_scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->qk_dim) : 1.0f;

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Update custom mask time: " << elapsed << " ms\n";
  //   }

  half *q = static_cast<half *>(m->queryTmp),
       *kv = static_cast<half *>(m->kvCache),
       *o = static_cast<half *>(m->outputTmp);
  paged_kv_t<PageStorage::kIndices, half, int32_t> paged_kv(
      num_kv_heads,
      kPagesize,
      head_dim,
      batch_size,
      QKVLayout::kNHD,
      kv,
      m->handle.incr_attention_metadata->kv_indices,
      m->handle.incr_attention_metadata->kv_indptr,
      m->handle.incr_attention_metadata->kv_last_page_len);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    attn prep time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  void *handler = nullptr;

  if (!bc->prompt_phase) {
    assert(m->handle.incr_attention_metadata->decode_handler_collections.count(
               batch_size) != 0 &&
           "Handler is not initialized");
    handler = m->handle.incr_attention_metadata
                  ->decode_handler_collections[batch_size];
  } else {
    assert(m->handle.incr_attention_metadata->prompt_handler_collections.count(
               batch_size) != 0 &&
           "Handler is not initialized");
    handler = m->handle.incr_attention_metadata
                  ->prompt_handler_collections[batch_size];
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    BeginForward time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  DISPATCH_HEADDIM(head_dim, HEAD_DIM, {
    cudaError_t result;
    if (bc->prompt_phase) {
      result =
          BatchPrefillWithPagedKVCacheWrapperDispatched<PageStorage::kIndices,
                                                        HEAD_DIM,
                                                        LogitsPostHook::kNone,
                                                        PosEncodingMode::kNone,
                                                        false,
                                                        MaskMode::kCausal,
                                                        half,
                                                        half,
                                                        half,
                                                        int32_t>(
              static_cast<BatchPrefillHandler *>(handler),
              q,
              m->handle.incr_attention_metadata->q_indptr,
              /*q_offset=*/nullptr,
              paged_kv,
              /*custom_mask=*/nullptr,
              /*qk_indptr=*/nullptr,
              o,
              /*lse=*/nullptr,
              num_q_heads,
              /*window_left=*/-1,
              /*logits_soft_cap=*/0.f,
              sm_scale,
              /*rope_scale=*/1.f,
              /*rope_theta=*/static_cast<float>(1e4),
              stream);
    } else {
      result =
          BatchDecodeWithPagedKVCacheWrapperDispatched<PageStorage::kIndices,
                                                       HEAD_DIM,
                                                       LogitsPostHook::kNone,
                                                       PosEncodingMode::kNone,
                                                       half,
                                                       half,
                                                       half,
                                                       int32_t>(
              static_cast<BatchDecodeHandler *>(handler),
              q,
              /*q_offset=*/nullptr,
              paged_kv,
              o,
              /*lse=*/nullptr,
              num_q_heads,
              /*window_left=*/-1,
              /*logits_soft_cap=*/0.f,
              sm_scale,
              /*rope_scale=*/1.f,
              /*rope_theta=*/static_cast<float>(1e4),
              stream);
    }
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to run "
                               "IncrementalDecodingAttentionForwardKernel: " +
                               std::string(cudaGetErrorString(result)));
    }
  });

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    actual attn time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  produce_output(m, bc, output_ptr, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    produce_output_kernel time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {

  // cudaEvent_t t_start, t_end;
  // cudaEventCreate(&t_start);
  // cudaEventCreate(&t_end);
  // cudaEventRecord(t_start, stream);

  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }

  // phase 1: Compute QKV Projections of the batch
  compute_qkv(m,
              bc,
              shard_id,
              input_ptr,
              weight_ptr,
              static_cast<DT *>(m->devQKVProjArray),
              bias_ptr,
              stream);

  // phase 2: First maintain the streaming cache, because it need
  // pre-pos-encoding values
  if (m->streaming_cache) {
    // Move pre-pos-encoding cache to where took by attention
    update_kv_in_streaming_cache<DT>(m, bc, stream);
    // Apply pos-encoding to those k values
    apply_pos_encoding_to_streaming_proj<DT>(m, bc, stream);
    // Commit to the streaming cache
    commit_kv<DT>(m, bc, stream);
  }

  // phase 3: Take care of the batch
  {
    // Apply pos-encoding to the batch
    apply_pos_encoding_to_tokens_in_batch(
        m, bc, static_cast<DT *>(m->devQKVProjArray), stream);
    // Move the batch qkv values to where took by attention
    update_qkv_in_batch_paged<DT>(m, bc, stream, false);
  }

  // phase 4: Attention computation
  incr_attention<DT>(m, bc, static_cast<DT *>(m->attn_heads), stream);

  // phase 5: Compute output production and bias together for all tokens
  int num_tokens = bc->num_active_tokens();
  compute_o_prod_bias(
      m, bc, shard_id, output_ptr, weight_ptr, bias_ptr, num_tokens, stream);
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  // cudaEvent_t t_start, t_end;
  // cudaEventCreate(&t_start);
  // cudaEventCreate(&t_end);
  // cudaEventRecord(t_start, stream);

  // assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    if (m->offload) {
      pre_build_weight<half>(m, weight, input.data_type, stream);
    }
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel<half>(
        m,
        bc,
        shard_id,
        input.get_half_ptr(),
        m->offload ? static_cast<half *>(m->weight_ptr) : weight.get_half_ptr(),
        output.get_half_ptr(),
        bias_ptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    if (m->offload) {
      pre_build_weight<float>(m, weight, input.data_type, stream);
    }
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel<float>(
        m,
        bc,
        shard_id,
        input.get_float_ptr(),
        m->offload ? static_cast<float *>(m->weight_ptr)
                   : weight.get_float_ptr(),
        output.get_float_ptr(),
        bias_ptr,
        stream);
  } else {
    assert(false && "Unspported data type");
  }

  // cudaEventRecord(t_end, stream);
  // checkCUDA(cudaEventSynchronize(t_end));
  // float elapsed = 0;
  // checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  // cudaEventDestroy(t_start);
  // cudaEventDestroy(t_end);
  // printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    INC_DECODING_MODE,
                                    attn,
                                    attn->hidden_size,
                                    attn->qk_dim,
                                    attn->v_dim,
                                    attn->o_dim,
                                    attn->rotary_embedding_meta,
                                    attn->qkv_bias,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->final_bias,
                                    attn->scaling_factor,
                                    weight,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    attn->quantization_type,
                                    attn->offload,
                                    attn->streaming_cache) {}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    InferenceMode infer_mode,
    Op const *attn,
    int _hidden_size,
    int _qk_dim,
    int _v_dim,
    int _o_dim,
    RotaryEmbeddingMeta _rotary_embedding_meta,
    bool _qkv_bias,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    bool _final_bias,
    float _scaling_factor,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    DataType _quantization_type,
    bool _offload,
    bool _streaming_cache)
    : OpMeta(handler, attn), weight_ptr(nullptr), bias_ptr(nullptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
  hidden_size = _hidden_size;
  qk_dim = _qk_dim;
  v_dim = _v_dim;
  o_dim = _o_dim;
  size_t size_of_dt = data_type_size(attn->data_type);
  quantization_type = _quantization_type;
  offload = _offload;
  streaming_cache = _streaming_cache;

  global_num_q_heads = _global_num_q_heads;
  global_num_kv_heads = _global_num_kv_heads;
  num_q_heads = _num_q_heads;
  num_kv_heads = _num_kv_heads;
  local_hidden_size = num_q_heads * qk_dim;

  weightSize =
      ((hidden_size * qk_dim + o_dim * (v_dim > 0 ? v_dim : hidden_size)) *
           num_q_heads +
       (hidden_size * qk_dim + hidden_size * v_dim) * num_q_heads) *
      size_of_dt;
  if (quantization_type != DT_NONE) {
    quantized_weightSize = get_quantization_to_byte_size(
        attn->data_type, quantization_type, weightSize);
  }
  // biasSize = _bias ? o_dim * size_of_dt * 4 : 0;

  int qkv_bias_size = qk_dim * num_q_heads + (qk_dim + v_dim) * num_q_heads;
  int final_bias_size = o_dim;
  biasSize =
      (_qkv_bias ? qkv_bias_size : 0) + (final_bias ? final_bias_size : 0);

  // has_load_weights = (bool *)calloc(1, sizeof(bool));
  //*has_load_weights = false;
  rotary_embedding_meta =
      (RotaryEmbeddingMeta *)calloc(1, sizeof(RotaryEmbeddingMeta));
  *rotary_embedding_meta = _rotary_embedding_meta;
  qkv_bias = (bool *)calloc(1, sizeof(bool));
  *qkv_bias = _qkv_bias;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;
  final_bias = (bool *)calloc(1, sizeof(bool));
  *final_bias = _final_bias;

  // allocate weight and bias in the reserve space for cpu offloading
  if (offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(weightSize);
    bias_ptr = gpu_mem_allocator.allocate_reserved_untyped(biasSize);
  }

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = std::max(
        infer_mode == TREE_SEARCH_MODE ? BatchConfig::max_tokens_per_ssm_batch()
                                       : BatchConfig::max_tokens_per_batch(),
        BatchConfig::max_tokens_per_prefilling_batch());
    size_t qkv_max_proj_size =
        max_tokens_per_batch *
        (qk_dim * num_q_heads + qk_dim * num_q_heads + v_dim * num_q_heads);
    size_t query_tmp_size = 0, key_cache_size = 0, value_cache_size = 0;
    size_t streaming_pre_pos_enc_size = 0;
    // assert((BatchConfig::max_sequence_length() +
    //         BatchConfig::max_spec_tree_token_num()) %
    //            kPagesize ==
    //        0);
    size_t max_num_pages =
        round_up_pages(BatchConfig::max_sequence_length() +
                       BatchConfig::max_spec_tree_token_num());
    PageManager *pm = PageManager::get_page_manager();
    size_t total_kv_cache_size_per_layer = pm->get_kv_cache_size_per_layer();
    switch (infer_mode) {
      case INC_DECODING_MODE:
      case TREE_VERIFY_MODE: {
        query_tmp_size = num_q_heads * qk_dim * max_tokens_per_batch;
        // a K-ary tree max node is (k^n - 1) / 2
        if (total_kv_cache_size_per_layer == 0) {
          key_cache_size = num_kv_heads * qk_dim *
                           BatchConfig::max_requests_per_batch() *
                           max_num_pages * kPagesize;
          value_cache_size = num_kv_heads * v_dim *
                             BatchConfig::max_requests_per_batch() *
                             max_num_pages * kPagesize;
        } else {
          key_cache_size = total_kv_cache_size_per_layer / 2 / size_of_dt;
          value_cache_size = total_kv_cache_size_per_layer / 2 / size_of_dt;
        }
        break;
      }
      case TREE_SEARCH_MODE: {
        query_tmp_size = num_q_heads * qk_dim * max_tokens_per_batch;
        // a K-ary tree max node is (k^n - 1) / 2
        key_cache_size = num_kv_heads * qk_dim *
                         BatchConfig::max_requests_per_batch() * max_num_pages *
                         kPagesize;
        value_cache_size = num_kv_heads * v_dim *
                           BatchConfig::max_requests_per_batch() *
                           max_num_pages * kPagesize;
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    if (streaming_cache) {
      size_t max_post_pos_enc_pages = round_up_pages(
          BatchConfig::MAX_STREAMING_POS - BatchConfig::get_max_tree_depth() +
          max(max_tokens_per_batch, BatchConfig::max_spec_tree_token_num()));
      key_cache_size = num_kv_heads * qk_dim *
                       BatchConfig::max_requests_per_batch() *
                       max_post_pos_enc_pages * kPagesize;
      value_cache_size = num_kv_heads * v_dim *
                         BatchConfig::max_requests_per_batch() *
                         max_post_pos_enc_pages * kPagesize;
      streaming_pre_pos_enc_size =
          num_kv_heads * (qk_dim + v_dim) *
          BatchConfig::max_requests_per_batch() *
          round_up_pages(BatchConfig::MAX_STREAMING_POS -
                         BatchConfig::get_max_tree_depth()) *
          kPagesize;
    }
    size_t attn_heads_size = max_tokens_per_batch * num_q_heads * v_dim;
    size_t output_tmp_size = max_tokens_per_batch * num_q_heads * v_dim;
    size_t complex_size =
        (max_tokens_per_batch * (qk_dim * num_q_heads + qk_dim * num_q_heads)) /
        2;
    size_t totalSize =
        (qkv_max_proj_size + query_tmp_size + key_cache_size +
         value_cache_size + streaming_pre_pos_enc_size + attn_heads_size) *
            size_of_dt +
        output_tmp_size * data_type_size(DT_HALF) +
        complex_size * sizeof(cuFloatComplex); // more components will
                                               // be added here later
    if (offload) {
      // assert that we have enough reserved work space left
      size_t totalSharedSize =
          infer_mode == TREE_VERIFY_MODE
              ? totalSize -
                    (query_tmp_size + key_cache_size + value_cache_size +
                     streaming_pre_pos_enc_size + qkv_max_proj_size) *
                        size_of_dt
              : totalSize - (query_tmp_size + key_cache_size +
                             value_cache_size + streaming_pre_pos_enc_size) *
                                size_of_dt;

      size_t instance_size =
          size_of_dt *
          (infer_mode == TREE_VERIFY_MODE
               ? query_tmp_size + key_cache_size + value_cache_size +
                     streaming_pre_pos_enc_size + qkv_max_proj_size
               : query_tmp_size + key_cache_size + value_cache_size +
                     streaming_pre_pos_enc_size);

      if (quantization_type != DT_NONE) {
        totalSharedSize += quantized_weightSize;
      }
      assert(gpu_mem_allocator.reserved_total_size -
                 gpu_mem_allocator.reserved_allocated_size >=
             totalSharedSize);
      gpu_mem_allocator.create_legion_instance(
          reserveInst, instance_size, "IncMultiHeadSelfAttentionMeta");
    } else {
      gpu_mem_allocator.create_legion_instance(
          reserveInst, totalSize, "IncMultiHeadSelfAttentionMeta");
    }

    // in tree_verify, enable devQKVProjArray;
    if (offload) {
      devQKVProjArray = gpu_mem_allocator.allocate_reserved_untyped(
          qkv_max_proj_size * size_of_dt);
    } else {
      devQKVProjArray = gpu_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size * size_of_dt);
    }

    // use key value cache in all mode.
    if (query_tmp_size > 0) {
      queryTmp = gpu_mem_allocator.allocate_instance_untyped(query_tmp_size *
                                                             size_of_dt);
    }
    kvCache = gpu_mem_allocator.allocate_instance_untyped(
        (key_cache_size + value_cache_size) * size_of_dt);
    if (streaming_pre_pos_enc_size > 0) {
      streamingPrePosEncBuf = gpu_mem_allocator.allocate_instance_untyped(
          streaming_pre_pos_enc_size * size_of_dt);
    }
    outputTmp = gpu_mem_allocator.allocate_instance<half>(output_tmp_size);

    token_infos =
        static_cast<BatchConfig::PerTokenInfo *>(handler.batch_config_metadata);
    request_infos = reinterpret_cast<BatchConfig::PerRequestInfo *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo));
    request_available = reinterpret_cast<bool *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo));
    streaming_cache_infos = reinterpret_cast<StreamingCacheInfo *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BatchConfig::request_available) +
        sizeof(BatchConfig::causalMask));

    if (offload) {
      // token_infos =
      //     gpu_mem_allocator.allocate_reserved<BatchConfig::PerTokenInfo>(
      //         tokeninfo_size);
      // offset += sizeof(BatchConfig::PerTokenInfo) * tokeninfo_size;
      attn_heads = gpu_mem_allocator.allocate_reserved_untyped(attn_heads_size *
                                                               size_of_dt);
      // offset += attn_heads_size * size_of_dt;
      complex_input =
          gpu_mem_allocator.allocate_reserved<cuFloatComplex>(complex_size);
      // offset += complex_size * sizeof(cuFloatComplex);
      // request_infos =
      //     gpu_mem_allocator.allocate_reserved<BatchConfig::PerRequestInfo>(
      //         requestinfo_size);
    } else {
      // token_infos =
      //     gpu_mem_allocator.allocate_instance<BatchConfig::PerTokenInfo>(
      //         tokeninfo_size);
      attn_heads = gpu_mem_allocator.allocate_instance_untyped(attn_heads_size *
                                                               size_of_dt);
      complex_input =
          gpu_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
      // request_infos =
      //     gpu_mem_allocator.allocate_instance<BatchConfig::PerRequestInfo>(
      //         requestinfo_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }
    if (!offload) {
      assert(gpu_mem_allocator.reserved_total_size ==
             gpu_mem_allocator.reserved_allocated_size);
    }
  }

  // set attention constants
  handler.incr_attention_metadata->set_enabled(true);
  handler.incr_attention_metadata->set_num_q_heads(num_q_heads);
  handler.incr_attention_metadata->set_num_kv_heads(num_kv_heads);
  handler.incr_attention_metadata->set_head_dim(qk_dim);

  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

}; // namespace FlexFlow
