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
#include "cuComplex.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/cuda_helper.h"
#include <math_constants.h>

// flashinfer & paged attention
#include "flashinfer/decode_attention_decl.cuh"
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/page_manager.h"

#include "flexflow/flash_api.h"
// only for debugging
#include <torch/nn/functional.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

// flashinfer & paged attention
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

std::string get_fwd_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                               int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("fwd", m->decoding_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

std::string get_peft_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                                int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("bwd", m->bwd_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

template <typename DT>
__global__ void store_kv_cache(DT const *devQKVProjArray,
                               DT *kCache_ptr,
                               DT *vCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int num_tokens,
                               int max_seq_len,
                               int head_dim,
                               int num_q_heads,
                               int num_kv_heads) {
  CUDA_KERNEL_LOOP(i, num_tokens * head_dim * num_kv_heads) {
    // devQKVProjArray: [head_dim, tot_num_heads, num_tokens]
    // kCache_ptr: [head_dim, num_kv_heads, max_seq_len, 1]
    // vCache_ptr: [head_dim, num_kv_heads, max_seq_len, 1]

    // i is iterating over one set of key/val projections from the input
    int token_idx = i / (head_dim * num_kv_heads);
    int head_idx = (i / head_dim) % num_kv_heads;
    int offset = i % head_dim;

    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int key_src_idx = token_idx * head_dim * tot_num_heads +
                      head_dim * num_q_heads + head_dim * head_idx + offset;
    int val_src_idx = key_src_idx + head_dim * num_kv_heads;

    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    int dst_idx =
        tok_id * head_dim * num_kv_heads + head_idx * head_dim + offset;

    kCache_ptr[dst_idx] = devQKVProjArray[key_src_idx];
    vCache_ptr[dst_idx] = devQKVProjArray[val_src_idx];
  }
}

template <typename DT>
__global__ void store_query_cache(DT const *devQKVProjArray,
                                  DT *qCache_ptr,
                                  int num_tokens_in_batch,
                                  int first_token_offset_in_batch,
                                  int first_token_depth_in_request,
                                  int head_dim,
                                  int num_q_heads,
                                  int num_kv_heads) {
  CUDA_KERNEL_LOOP(i, num_tokens_in_batch * head_dim * num_q_heads) {
    int hidden_size = head_dim * num_q_heads;
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;

    int token_idx = i / hidden_size;
    int offset = i % hidden_size;
    int src_idx =
        (first_token_offset_in_batch + token_idx) * (head_dim * tot_num_heads) +
        offset;

    qCache_ptr[first_token_depth_in_request * hidden_size + i] =
        devQKVProjArray[src_idx];
  }
}

template <typename DT>
__global__ void fill_entries_above_diagonal(DT *matrix,
                                            size_t num_rows,
                                            size_t num_cols,
                                            size_t num_q_heads,
                                            size_t entries_above_diagonal,
                                            DT value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_q_heads) {
    size_t head_idx = i / entries_above_diagonal;
    size_t entry_idx = i % entries_above_diagonal;
    size_t y = (-1 + sqrt(8 * (float)entry_idx + 1)) / 2;
    size_t x = entry_idx - y * (y + 1) / 2;
    y += (num_cols - num_rows) + 1;
    matrix[head_idx * num_rows * num_cols + num_cols * y + x] = value;
  }
}

bool is_finetuning_bwd_request(BatchConfig const *bc, int request_id) {
  return bc->requestsInfo[request_id].finetuning_request &&
         bc->requestsInfo[request_id].finetuning_backward_phase;
}

bool is_decoding_request(BatchConfig const *bc, int request_id) {
  return !bc->requestsInfo[request_id].finetuning_request &&
         !bc->requestsInfo[request_id].prompt_phase;
}

template <typename DT>
void run_batched_matmul(IncMultiHeadSelfAttentionMeta const *meta,
                        cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        int m,
                        int n,
                        int k,
                        void const *alpha,
                        const DT *A,
                        cudaDataType Atype,
                        int lda,
                        long long int strideA,
                        const DT *B,
                        cudaDataType Btype,
                        int ldb,
                        long long int strideB,
                        void const *beta,
                        DT *C,
                        cudaDataType Ctype,
                        int ldc,
                        long long int strideC,
                        int batchCount,
                        cudaDataType computeType,
                        cublasGemmAlgo_t algo,
                        cudaStream_t stream,
                        int batch_ratio_a,
                        int batch_ratio_b,
                        int batch_ratio_c,
                        bool bwd) {
  if (batch_ratio_a == 1 && batch_ratio_b == 1 && batch_ratio_c == 1) {
    checkCUDA(cublasGemmStridedBatchedEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         alpha,
                                         A,
                                         Atype,
                                         lda,
                                         strideA,
                                         B,
                                         Btype,
                                         ldb,
                                         strideB,
                                         beta,
                                         C,
                                         Ctype,
                                         ldc,
                                         strideC,
                                         batchCount,
                                         computeType,
                                         algo));
  } else {
    const DT **h_A_array = new const DT *[batchCount];
    const DT **h_B_array = new const DT *[batchCount];
    DT **h_C_array = new DT *[batchCount];
    for (int batch = 0; batch < batchCount; batch++) {
      h_A_array[batch] = A + (batch / batch_ratio_a) * strideA;
      h_B_array[batch] = B + (batch / batch_ratio_b) * strideB;
      h_C_array[batch] = C + (batch / batch_ratio_c) * strideC;
    }
    assert(sizeof(DT *) == sizeof(void *));
    if (!bwd) {
      // Copy pointer arrays to device
      checkCUDA(cudaMemcpyAsync(meta->d_A_array,
                                h_A_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_B_array,
                                h_B_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_C_array,
                                h_C_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));

      checkCUDA(cublasGemmBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    meta->d_A_array,
                                    Atype,
                                    lda,
                                    meta->d_B_array,
                                    Btype,
                                    ldb,
                                    beta,
                                    meta->d_C_array,
                                    Ctype,
                                    ldc,
                                    batchCount,
                                    computeType,
                                    algo));
    } else {
      checkCUDA(cudaMemcpyAsync(meta->d_A_array2,
                                h_A_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_B_array2,
                                h_B_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));
      checkCUDA(cudaMemcpyAsync(meta->d_C_array2,
                                h_C_array,
                                batchCount * sizeof(DT *),
                                cudaMemcpyHostToDevice,
                                stream));

      checkCUDA(cublasGemmBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    meta->d_A_array2,
                                    Atype,
                                    lda,
                                    meta->d_B_array2,
                                    Btype,
                                    ldb,
                                    beta,
                                    meta->d_C_array2,
                                    Ctype,
                                    ldc,
                                    batchCount,
                                    computeType,
                                    algo));
    }
  }
}

// todo(gabriele): review this function
// the params should persist after the function returns
template <typename DT>
void set_wrapper_mha_fwd_1_params_peft(IncMultiHeadSelfAttentionMeta const *m,
                                       BatchConfig const *bc,
                                       DT *attn_heads,
                                       int shard_id,
                                       at::Tensor &q,
                                       at::Tensor &k,
                                       at::Tensor &v,
                                       std::optional<at::Tensor> &out_,
                                       std::optional<at::Tensor> &alibi_slopes_,
                                       float &p_dropout,
                                       float &softmax_scale,
                                       bool &is_causal,
                                       int &window_size_left,
                                       int &window_size_right,
                                       float &softcap,
                                       bool &return_softmax,
                                       std::optional<at::Generator> &gen_) {
  // todo(gabriele): check if alibi_slopes_ generation is correct
  // The slopes should be consistent with `apply_position_bias_qkprd kernel`
  if (*m->position_bias) {
    at::Tensor alibi_slopes = at::empty({m->num_q_heads}, at::kFloat);
    float *slopes_ptr = alibi_slopes.data_ptr<float>();
    for (int head_idx = 0; head_idx < m->num_q_heads; head_idx++) {
      int global_head_idx = head_idx + (m->num_q_heads * shard_id);
      float base = (float)(global_head_idx + 1) * 8.0f / m->global_num_q_heads;
      slopes_ptr[head_idx] = 1.0f / std::pow(2.0f, base);
    }
    alibi_slopes_ = alibi_slopes;
  }

  int req_idx = bc->finetuning_request_index();
  // signed long batch_size = 1;
  signed long seqlen_q = bc->requestsInfo[req_idx].num_tokens_in_batch;
  signed long seqlen_k =
      bc->requestsInfo[req_idx].first_token_depth_in_request +
      bc->requestsInfo[req_idx].num_tokens_in_batch;
  signed long num_heads = m->num_q_heads;
  signed long num_heads_k = m->num_kv_heads;
  signed long head_size = m->qProjSize;
  int tokens_previous_requests =
      bc->requestsInfo[req_idx].first_token_offset_in_batch;
  int num_new_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
  int total_tokens = bc->requestsInfo[req_idx].first_token_depth_in_request +
                     bc->requestsInfo[req_idx].num_tokens_in_batch;
  int tokens_previous_steps = total_tokens - num_new_tokens;

  p_dropout = m->flash_attn_p_dropout;
  softmax_scale = (*m->qk_prod_scaling) ? (1.0f / sqrt(m->kProjSize)) : 1.0f;
  is_causal = m->flash_attn_is_causal;
  window_size_left = m->flash_attn_window_size_left;
  window_size_right = m->flash_attn_window_size_right;
  softcap = m->flash_attn_softcap;
  return_softmax = m->flash_attn_return_softmax;

  // only support head_size aligned with 8 for now
  // todo(yingyi): do we need to support non-8 aligned head_size?
  // refer to flash-attention/flash_attn/flash_attn_interface.py:L836
  assert(head_size % 8 == 0 && "head_size must be aligned with 8");

  // todo(yingyi): might need to pass empty_like tensor and copy data back??
  // Get raw pointer of the output tensor [vProjSize, num_q_heads,
  // num_new_tokens] which is (head_size, num_q_heads, num_new_tokens)
  DT *out_ptr = static_cast<DT *>(attn_heads) +
                tokens_previous_requests * m->num_q_heads * m->vProjSize;
  // Store the output tensor to the result attn heads
  // out size: (batch_size, seqlen_q, num_heads, head_size)
  at::Tensor out =
      createTorchTensorFromCuda<DT>(out_ptr, {head_size, num_heads, seqlen_q});
  out = out.permute({2, 1, 0}).unsqueeze(0);
  out_ = out;

  // cuda layout: [qProjSize, num_q_heads, num_new_tokens]
  auto q_ptr = static_cast<DT *>(m->query_activation_buffer) +
               tokens_previous_steps * m->qProjSize * m->num_q_heads;
  // cuda layout: [kProjSize, num_kv_heads, total_tokens]
  auto k_ptr = static_cast<DT *>(m->keyCachePeft);
  // cuda layout: [vProjSize, num_kv_heads, total_tokens]
  auto v_ptr = static_cast<DT *>(m->valueCachePeft);

  // re-organize q, k, v tensor to match the layout of flash-attn
  // q size: (batch_size, seqlen_q, num_heads, head_size)
  q = createTorchTensorFromCuda<DT>(q_ptr, {head_size, num_heads, seqlen_q});
  q = q.permute({2, 1, 0}).unsqueeze(0);
  // k size: (batch_size, seqlen_k, num_heads_k, head_size)
  k = createTorchTensorFromCuda<DT>(k_ptr, {head_size, num_heads_k, seqlen_k});
  k = k.permute({2, 1, 0}).unsqueeze(0);
  // v size: (batch_size, seqlen_k, num_heads_k, head_size)
  v = createTorchTensorFromCuda<DT>(v_ptr, {head_size, num_heads_k, seqlen_k});
  v = v.permute({2, 1, 0}).unsqueeze(0);

  auto const sizes = q.sizes();
  if (m->inference_debugging) {
    std::string q_fpath = get_peft_dbg_folder(m, shard_id) + ".fwd_q.pt";
    std::string k_fpath = get_peft_dbg_folder(m, shard_id) + ".fwd_k.pt";
    std::string v_fpath = get_peft_dbg_folder(m, shard_id) + ".fwd_v.pt";
    torch::save(q.clone().detach(), q_fpath);
    torch::save(k.clone().detach(), k_fpath);
    torch::save(v.clone().detach(), v_fpath);

    if (*m->position_bias && m->inference_debugging) {
      std::string alibi_slopes_fpath =
          get_peft_dbg_folder(m, shard_id) + ".fwd_alibi_slopes.pt";
      torch::save(alibi_slopes_.value().clone().detach(), alibi_slopes_fpath);
    }
  }
}

// todo(yingyi):init dq_, dk_, dv_ by peft pre-allocated buffer
template <typename DT>
void set_wrapper_mha_bwd_1_params_peft(IncMultiHeadSelfAttentionMeta const *m,
                                       BatchConfig const *bc,
                                       int shard_id,
                                       DT *input_grad_ptr,
                                       DT const *output_grad_ptr,
                                       at::Tensor &dout,
                                       at::Tensor &q,
                                       at::Tensor &k,
                                       at::Tensor &v,
                                       at::Tensor &out,
                                       at::Tensor &softmax_lse,
                                       std::optional<at::Tensor> &dq_,
                                       std::optional<at::Tensor> &dk_,
                                       std::optional<at::Tensor> &dv_,
                                       std::optional<at::Tensor> &alibi_slopes_,
                                       float &p_dropout,
                                       float &softmax_scale,
                                       bool &is_causal,
                                       int &window_size_left,
                                       int &window_size_right,
                                       float &softcap,
                                       bool &deterministic,
                                       std::optional<at::Generator> &gen_,
                                       std::optional<at::Tensor> &rng_state) {
  // todo(gabriele): check if the data is correct
  int req_idx = bc->finetuning_request_index();
  // signed long batch_size = 1;
  signed long seqlen_q = bc->requestsInfo[req_idx].num_tokens_in_batch;
  signed long seqlen_k =
      bc->requestsInfo[req_idx].first_token_depth_in_request +
      bc->requestsInfo[req_idx].num_tokens_in_batch;
  // int num_new_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
  // int total_tokens = bc->requestsInfo[req_idx].first_token_depth_in_request +
  //                    bc->requestsInfo[req_idx].num_tokens_in_batch;
  int tokens_previous_requests =
      bc->requestsInfo[req_idx].first_token_offset_in_batch;

  signed long num_heads = m->num_q_heads;
  signed long num_heads_k = m->num_kv_heads;
  signed long head_size = m->qProjSize;
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  p_dropout = m->flash_attn_p_dropout;
  softmax_scale = (*m->qk_prod_scaling) ? (1.0f / sqrt(m->kProjSize)) : 1.0f;
  is_causal = m->flash_attn_is_causal;
  window_size_left = m->flash_attn_window_size_left;
  window_size_right = m->flash_attn_window_size_right;
  softcap = m->flash_attn_softcap;
  deterministic = false; // m->inference_debugging; // only for debugging

  // recompute alibi slopes (should be the same as in flash_peft_bwd_kernel)
  if (*m->position_bias) {
    at::Tensor alibi_slopes = at::empty({m->num_q_heads}, at::kFloat);
    float *slopes_ptr = alibi_slopes.data_ptr<float>();
    for (int head_idx = 0; head_idx < m->num_q_heads; head_idx++) {
      int global_head_idx = head_idx + (m->num_q_heads * shard_id);
      float base = (float)(global_head_idx + 1) * 8.0f / m->global_num_q_heads;
      slopes_ptr[head_idx] = 1.0f / std::pow(2.0f, base);
    }
    alibi_slopes_ = alibi_slopes;
  }

  // todo(gabriele): review the conversion of output_grad_ptr to at::Tensor dout
  // dout should have the same shape as out
  // (batch_size, seqlen_q, num_heads, head_size)
  // convert output_grad_ptr to at::Tensor dout
  dout = createTorchTensorFromCuda<DT>(
      static_cast<void *>(const_cast<DT *>(output_grad_ptr)),
      {head_size, num_heads, seqlen_q});
  dout = dout.permute({2, 1, 0}).unsqueeze(0);

  // Construct q, k, v, out tensor
  // cuda layout: [m->qProjSize * num_q_heads, num_new_tokens] (num_new_tokens =
  // num_tokens)
  auto q_ptr = static_cast<DT *>(m->query_activation_buffer);
  // cuda layout: [vProjSize * num_kv_heads, max_num_tokens, num_req]
  auto k_ptr = static_cast<DT *>(m->keyCachePeft);
  // cuda layout: [vProjSize * num_kv_heads, max_num_tokens, 1]
  auto v_ptr = static_cast<DT *>(m->valueCachePeft);
  auto out_ptr = static_cast<DT *>(m->flash_attn_out) +
                 tokens_previous_requests * m->num_q_heads * m->vProjSize;

  int num_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
  assert(m->handle.workSpaceSize >= m->qProjSize *
                                        (m->num_q_heads + 2 * m->num_kv_heads) *
                                        num_tokens * sizeof(DT));
  auto dv_ptr = static_cast<DT *>(m->handle.workSpace) +
                num_tokens * m->qProjSize * (m->num_q_heads + m->num_kv_heads);
  auto dk_ptr = static_cast<DT *>(m->handle.workSpace) +
                num_tokens * (m->qProjSize * m->num_q_heads);
  auto dq_ptr = static_cast<DT *>(m->handle.workSpace);

  // re-organize q, k, v tensor to match the layout of flash-attn
  // q size: (batch_size, seqlen_q, num_heads, head_size)
  q = createTorchTensorFromCuda<DT>(q_ptr, {head_size, num_heads, seqlen_q});
  q = q.permute({2, 1, 0}).unsqueeze(0);
  // k size: (batch_size, seqlen_k, num_heads_k, head_size)
  k = createTorchTensorFromCuda<DT>(k_ptr, {head_size, num_heads_k, seqlen_k});
  k = k.permute({2, 1, 0}).unsqueeze(0);
  // v size: (batch_size, seqlen_k, num_heads_k, head_size)
  v = createTorchTensorFromCuda<DT>(v_ptr, {head_size, num_heads_k, seqlen_k});
  v = v.permute({2, 1, 0}).unsqueeze(0);
  out =
      createTorchTensorFromCuda<DT>(out_ptr, {head_size, num_heads, seqlen_q});
  out = out.permute({2, 1, 0}).unsqueeze(0);

  // tensor dv shape (batch_size, seqlen_k, num_heads_k, head_size)
  // tensor dq shape (batch_size, seqlen_q, num_heads, head_size)
  // tensor dk shape (batch_size, seqlen_k, num_heads_k, head_size)
  auto dq =
      createTorchTensorFromCuda<DT>(dq_ptr, {head_size, num_heads, seqlen_q});
  auto dk =
      createTorchTensorFromCuda<DT>(dk_ptr, {head_size, num_heads_k, seqlen_k});
  auto dv =
      createTorchTensorFromCuda<DT>(dv_ptr, {head_size, num_heads_k, seqlen_k});
  dq = dq.permute({2, 1, 0}).unsqueeze(0);
  dk = dk.permute({2, 1, 0}).unsqueeze(0);
  dv = dv.permute({2, 1, 0}).unsqueeze(0);

  // todo(gabriele): i fix the data layout for dq, dk, dv, but it's not
  // contiguous at last dimension to flash-attn refuse it we could try to fix it
  // by the memory layout to match the q, k, v layout (and use the commented-out
  // code above) but we need to rewrite the rope bwd kernel and input_gradient
  // calculation kernel to match the new layout auto dq =
  //     createTorchTensorFromCuda<DT>(dq_ptr, {seqlen_q, head_size,
  //     num_heads});
  // auto dk =
  //     createTorchTensorFromCuda<DT>(dk_ptr, {seqlen_k, head_size,
  //     num_heads_k});
  // auto dv =
  //     createTorchTensorFromCuda<DT>(dv_ptr, {seqlen_k, head_size,
  //     num_heads_k});
  // dq = dq.permute({0, 2, 1}).unsqueeze(0);
  // dk = dk.permute({0, 2, 1}).unsqueeze(0);
  // dv = dv.permute({0, 2, 1}).unsqueeze(0);

  dq_ = dq;
  dk_ = dk;
  dv_ = dv;

  // checkpoint: dq, dk, dv data_ptr
  // std::cout << "1. the address of dq is: " << dq.data_ptr() << std::endl;
  // std::cout << "1. the address of dk is: " << dk.data_ptr() << std::endl;
  // std::cout << "1. the address of dv is: " << dv.data_ptr() << std::endl;

  auto opts = q.options();
  softmax_lse =
      torch::from_blob(static_cast<float *>(m->flash_attn_softmax_lse),
                       {1, num_heads, seqlen_q},
                       opts.dtype(at::kFloat));
  // todo(yingyi): review ths later
  // no rng_state for zero p_dropout
  rng_state = std::nullopt;
  // rng_state = torch::zeros({2}, opts.dtype(torch::kInt64));
  // rng_state.value().data_ptr<int64_t>()[0] = m->flash_attn_rng_state_0;
  // rng_state.value().data_ptr<int64_t>()[1] = m->flash_attn_rng_state_1;

  // only for results alignment
  if (m->inference_debugging) {
    std::string q_fpath = get_peft_dbg_folder(m, shard_id) + ".bwd_q.pt";
    torch::save(q.clone().detach(), q_fpath);

    std::string k_fpath = get_peft_dbg_folder(m, shard_id) + ".bwd_k.pt";
    torch::save(k.clone().detach(), k_fpath);

    std::string v_fpath = get_peft_dbg_folder(m, shard_id) + ".bwd_v.pt";
    torch::save(v.clone().detach(), v_fpath);

    std::string dout_fpath = get_peft_dbg_folder(m, shard_id) + ".dout.pt";
    torch::save(dout.clone().detach(), dout_fpath);

    std::string softmax_lse_fpath =
        get_peft_dbg_folder(m, shard_id) + ".bwd_softmax_lse.pt";
    torch::save(softmax_lse.clone().detach(), softmax_lse_fpath);

    if (alibi_slopes_.has_value()) {
      std::string alibi_slopes_fpath =
          get_peft_dbg_folder(m, shard_id) + ".bwd_alibi_slopes.pt";
      torch::save(alibi_slopes_.value().clone().detach(), alibi_slopes_fpath);
    }

    // std::string rng_state_fpath =
    //     get_peft_dbg_folder(m, shard_id) + ".bwd_rng_state.pt";
    // torch::save(rng_state.value().clone().detach(), rng_state_fpath);
  }
}

std::vector<at::Tensor> _wrapper_mha_fwd_1(
    at::Tensor
        &q, // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    at::Tensor const &
        k, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    at::Tensor const &
        v, // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    std::optional<at::Tensor> &out_, // batch_size x seqlen_q x num_heads x
                                     // round_multiple(head_size, 8)
    std::optional<at::Tensor>
        &alibi_slopes_, // num_heads or batch_size x num_heads
    float const p_dropout,
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const return_softmax,
    std::optional<at::Generator> gen_,
    cudaStream_t stream) {
  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.device()};

  auto [cc_major, cc_minor] =
      flash::get_compute_capability(flash::get_current_device());
  bool is_sm8x_min = cc_major >= 8;
  TORCH_CHECK(is_sm8x_min,
              "FlashAttention only supports Ampere GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  auto const sizes = q.sizes();

  int const batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  int const head_size = sizes[3];
  int const seqlen_k = k.size(1);
  int const num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  if (softcap > 0.f) {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
    is_causal = false;
  }
  if (is_causal) {
    window_size_right = 0;
  }

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  int const seqlenq_ngroups_swapped =
      seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
      window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 &&
      !alibi_slopes_.has_value();
  int const ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
            .transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
    if (seqlenq_ngroups_swapped) {
      out = out.reshape({batch_size, num_heads_k, ngroups, head_size})
                .transpose(1, 2);
    }
  } else {
    out = torch::empty_like(q);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  auto opts = q.options();

  auto softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty(
        {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
  } else {
    p = torch::empty({0}, opts);
  }

  flash::Flash_fwd_params params;
  flash::set_params_fprop(params,
                          batch_size,
                          seqlen_q,
                          seqlen_k,
                          seqlen_q_rounded,
                          seqlen_k_rounded,
                          num_heads,
                          num_heads_k,
                          head_size,
                          head_size_rounded,
                          q,
                          k,
                          v,
                          out,
                          /*cu_seqlens_q_d=*/nullptr,
                          /*cu_seqlens_k_d=*/nullptr,
                          /*seqused_k=*/nullptr,
                          return_softmax ? p.data_ptr() : nullptr,
                          softmax_lse.data_ptr(),
                          p_dropout,
                          softmax_scale,
                          window_size_left,
                          window_size_right,
                          softcap);

  // Keep references to these tensors to extend their lifetime
  at::Tensor softmax_lse_accum, out_accum;
  std::tie(softmax_lse_accum, out_accum) =
      flash::set_params_splitkv(params,
                                batch_size,
                                num_heads,
                                head_size,
                                seqlen_k,
                                seqlen_q,
                                head_size_rounded,
                                p_dropout,
                                /*num_splits*/ 0,
                                flash::get_num_sm(flash::get_current_device()),
                                opts);

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());

  if (p_dropout > 0.0) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  if (seqlen_k > 0) {
    // auto stream = at::cuda::getCurrentCUDAStream().stream();
    flash::run_mha_fwd(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape(
        {batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }

  return {out, softmax_lse, p, rng_state};
  // todo(yingyi): return? the results are updated at data_ptr() of each tensor
}

std::vector<at::Tensor> _wrapper_mha_bwd_1(
    at::Tensor const &dout, // batch_size x seqlen_q x num_heads, x
                            // multiple_of(head_size_og, 8)
    at::Tensor const &q,    // batch_size x seqlen_q x num_heads x head_size
    at::Tensor const &k,    // batch_size x seqlen_k x num_heads_k x head_size
    at::Tensor const &v,    // batch_size x seqlen_k x num_heads_k x head_size
    at::Tensor const &out,  // batch_size x seqlen_q x num_heads x head_size
    at::Tensor const &softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>
        &dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>
        &dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>
        &dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>
        &alibi_slopes_,    // num_heads or batch_size x num_heads
    float const p_dropout, // probability to drop
    float const softmax_scale,
    bool const is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor> &rng_state,
    cudaStream_t stream) {
  if (is_causal) {
    window_size_right = 0;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.device()};

  auto [cc_major, cc_minor] =
      flash::get_compute_capability(flash::get_current_device());
  bool is_sm8x_min = cc_major >= 8;
  TORCH_CHECK(is_sm8x_min,
              "FlashAttention only supports Ampere GPUs or newer.");

  bool is_dropout = p_dropout > 0.0;
  // auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype,
              "query and dout must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1,
              "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1,
              "dout tensor must have contiguous last dimension");

  auto const sizes = q.sizes();

  int const batch_size = sizes[0];
  int const seqlen_q = sizes[1];
  int const num_heads = sizes[2];
  int const head_size = sizes[3];
  int const seqlen_k = k.size(1);
  int const num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  if (softcap > 0.f) {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  } else {
    dq = torch::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dk = torch::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dv = torch::empty_like(v);
  }

  // checkpoint: dq, dk, dv data_ptr
  // std::cout << "2. the address of dq is: " << dq.data_ptr() << std::endl;
  // std::cout << "2. the address of dk is: " << dk.data_ptr() << std::endl;
  // std::cout << "2. the address of dv is: " << dv.data_ptr() << std::endl;

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  auto opts = q.options();
  auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded},
                                opts.dtype(at::kFloat));
  at::Tensor dq_accum;
  at::Tensor dk_accum, dv_accum;
  if (loop) {
    if (!deterministic) {
      dq_accum = torch::empty(
          {batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
          opts.dtype(at::kFloat));
    } else {
      int const nsplits = (flash::get_num_sm(flash::get_current_device()) +
                           batch_size * num_heads - 1) /
                          (batch_size * num_heads);
      dq_accum = torch::zeros(
          {nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
          opts.dtype(at::kFloat));
    }
    // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
    // head_size_rounded}, opts.dtype(at::kFloat)); dv_accum =
    // torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
    // head_size_rounded}, opts.dtype(at::kFloat));
  }

  at::Tensor dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) { // MQA / GQA
    dk_expanded =
        torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    dv_expanded =
        torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  flash::Flash_bwd_params params;

  flash::set_params_dgrad(params,
                          batch_size,
                          seqlen_q,
                          seqlen_k,
                          seqlen_q_rounded,
                          seqlen_k_rounded,
                          num_heads,
                          num_heads_k,
                          head_size,
                          head_size_rounded,
                          q,
                          k,
                          v,
                          out,
                          dout,
                          dq,
                          dk_expanded,
                          dv_expanded,
                          nullptr,
                          nullptr,
                          loop ? dq_accum.data_ptr() : nullptr,
                          // loop ? dk_accum.data_ptr() : nullptr,
                          // loop ? dv_accum.data_ptr() : nullptr,
                          nullptr,
                          nullptr,
                          softmax_lse.data_ptr(),
                          softmax_d.data_ptr(),
                          p_dropout,
                          softmax_scale,
                          window_size_left,
                          window_size_right,
                          softcap,
                          deterministic,
                          /*unpadded_lse*/ false);
  params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

  // checkpoint: dq, dk, dv data_ptr
  // std::cout << "3. the address of dq is: " << dq.data_ptr() << std::endl;
  // std::cout << "3. the address of dk is: " << dk.data_ptr() << std::endl;
  // std::cout << "3. the address of dv is: " << dv.data_ptr() << std::endl;

  auto launch = &flash::run_mha_bwd;

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;

  if (rng_state.has_value()) {
    params.rng_state =
        reinterpret_cast<uint64_t *>(rng_state.value().data_ptr());
  } else if (is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    params.rng_state[0] = std::get<0>(seeds);
    params.rng_state[1] = std::get<1>(seeds);
  }

  flash::set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  if (seqlen_q > 0) {
    launch(params, stream);
  } else {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    dk_expanded.zero_();
    dv_expanded.zero_();
    softmax_d.zero_();
  }

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    at::sum_out(dk,
                at::reshape(dk_expanded,
                            {batch_size,
                             seqlen_k,
                             num_heads_k,
                             num_heads / num_heads_k,
                             head_size}),
                {3});
    at::sum_out(dv,
                at::reshape(dv_expanded,
                            {batch_size,
                             seqlen_k,
                             num_heads_k,
                             num_heads / num_heads_k,
                             head_size}),
                {3});
  }

  // checkpoint: dq, dk, dv data_ptr
  // std::cout << "4. the address of dq is: " << dq.data_ptr() << std::endl;
  // std::cout << "4. the address of dk is: " << dk.data_ptr() << std::endl;
  // std::cout << "4. the address of dv is: " << dv.data_ptr() << std::endl;

  return {dq, dk, dv, softmax_d};
  // todo(yingyi): set dq, dk, dv params by target data_ptr
}

// TODO(yingyi): fwd implementation of flash-attn
template <typename DT>
void flash_compute_attention_kernel_peft(IncMultiHeadSelfAttentionMeta *m,
                                         BatchConfig const *bc,
                                         DT *attn_heads,
                                         int shard_id,
                                         cudaStream_t peft_stream) {
  // Step 0: param check as in compute_attention_kernel_peft
  // todo(gabriele): step0 is the same as everything before step1 in
  // compute_attention_kernel_peft (Any updates to should be reflected.)
  if (bc->num_finetuning_fwd_tokens() <= 0) {
    return;
  }

  checkCUDA(cublasSetStream(m->handle.peft_blas, peft_stream));
  checkCUDNN(cudnnSetStream(m->handle.peft_dnn, peft_stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));

  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);

  assert(bc->num_finetuning_fwd_tokens() > 0);
  int req_idx = bc->finetuning_request_index();
  assert(!bc->request_completed[req_idx]);
  assert(bc->requestsInfo[req_idx].finetuning_request &&
         !bc->requestsInfo[req_idx].finetuning_backward_phase);

  int num_new_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
  int total_tokens = bc->requestsInfo[req_idx].first_token_depth_in_request +
                     bc->requestsInfo[req_idx].num_tokens_in_batch;
  assert(num_new_tokens > 0 && total_tokens > 0);

  // Copy query to m->query_activation_buffer for BWD
  // int max_peft_tokens = bc->requestsInfo[i].max_length;
  int max_peft_tokens = BatchConfig::max_sequence_length();
  size_t activation_size_needed =
      sizeof(DT) * max_peft_tokens * m->num_q_heads * m->qProjSize;
  if (activation_size_needed != m->allocated_peft_buffer_size1) {
    std::cout << "activation_size_needed: " << activation_size_needed
              << std::endl;
    std::cout << "m->allocated_peft_buffer_size1: "
              << m->allocated_peft_buffer_size1 << std::endl;
    std::cout << "max_peft_tokens: " << max_peft_tokens << std::endl;
    std::cout << "m->num_q_heads: " << m->num_q_heads << std::endl;
    std::cout << "m->qProjSize: " << m->qProjSize << std::endl;
    std::cout << "BatchConfig::max_sequence_length()"
              << BatchConfig::max_sequence_length() << std::endl;
    std::cout << "sizeof(DT)" << sizeof(DT) << std::endl;
  }
  assert(activation_size_needed == m->allocated_peft_buffer_size1);

  // end Step 0
  // ========================================================================

  // Step 1: configure params for fwd
  // ================================================
  at::Tensor q, k, v;
  std::optional<at::Tensor> out_ = std::nullopt;
  std::optional<at::Tensor> alibi_slopes_ = std::nullopt;
  std::optional<at::Generator> gen_ = std::nullopt;
  float p_dropout, softmax_scale, softcap;
  bool is_causal, return_softmax;
  int window_size_left, window_size_right;

  set_wrapper_mha_fwd_1_params_peft<DT>(m,
                                        bc,
                                        attn_heads,
                                        shard_id,
                                        q,
                                        k,
                                        v,
                                        out_,
                                        alibi_slopes_,
                                        p_dropout,
                                        softmax_scale,
                                        is_causal,
                                        window_size_left,
                                        window_size_right,
                                        softcap,
                                        return_softmax,
                                        gen_);

  auto result = _wrapper_mha_fwd_1(q,
                                   k,
                                   v,
                                   out_,
                                   alibi_slopes_,
                                   p_dropout,
                                   softmax_scale,
                                   is_causal,
                                   window_size_left,
                                   window_size_right,
                                   softcap,
                                   return_softmax,
                                   gen_,
                                   peft_stream);
  auto out = result[0];
  auto softmax_lse = result[1];
  auto p = result[2];
  auto rng_state = result[3];

  // Step 2: Handle the output tensor and cache softmax_lse for BWD
  // ========================================================================
  // print out the shapes and values of the tensors
  // save rng_state for backward context
  // m->flash_attn_rng_state_0 = rng_state[0].item<int64_t>();
  // m->flash_attn_rng_state_1 = rng_state[1].item<int64_t>();
  if (m->inference_debugging) {
    std::string out_fpath = get_peft_dbg_folder(m, shard_id) + ".fwd_out.pt";
    torch::save(out.clone().detach(), out_fpath);

    std::string softmax_lse_fpath =
        get_peft_dbg_folder(m, shard_id) + ".fwd_softmax_lse.pt";
    torch::save(softmax_lse.clone().detach(), softmax_lse_fpath);

    // std::string rng_state_fpath =
    //     get_peft_dbg_folder(m, shard_id) + ".fwd_rng_state.pt";
    // torch::save(rng_state.clone().detach(), rng_state_fpath);
  }

  // todo(yingyi): fix copy temp out to meta out
  // todo(gabriele): review the layout of flash_attn_out
  // copy out to flash_attn_out for bwd
  // same layout as out tensor (head_size, num_q_heads, num_new_tokens)
  // use decive2device memcpy async
  checkCUDA(cudaMemcpyAsync(m->flash_attn_out,
                            out.data_ptr(),
                            out.numel() * sizeof(DT),
                            cudaMemcpyDeviceToDevice,
                            peft_stream));

  // copy softmax_lse to flash_attn_softmax_lse for bwd
  // layout: (batch_size, num_heads, seqlen_q)
  checkCUDA(cudaMemcpyAsync(m->flash_attn_softmax_lse,
                            softmax_lse.data_ptr(),
                            softmax_lse.numel() * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            peft_stream));
  // end step 2
  // ========================================================================

  // step 3: (optional, only for testing)
  // invoke attention fwd from torch
  // ========================================================================
  // skip for now, leave alignments to python script
  // end step 3
  // ========================================================================
}

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qProjSize,
                                     int num_tokens,
                                     int num_q_heads,
                                     int num_kv_heads,
                                     float scaling_factor) {
  CUDA_KERNEL_LOOP(i, (qProjSize * num_q_heads) * num_tokens) {
    int token_idx = i / (qProjSize * num_q_heads);
    int offset = i % (qProjSize * num_q_heads);
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int idx = token_idx * qProjSize * tot_num_heads + offset;
    input_ptr[idx] *= scaling_factor;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_fwd(DT *input_ptr,
                               cuFloatComplex *complex_input,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               float rope_theta,
                               bool llama3_rope,
                               float factor,
                               float low_freq_factor,
                               float high_freq_factor,
                               int original_max_position_embeddings,
                               int proj_size,
                               int num_tokens,
                               int num_q_heads,
                               int num_kv_heads) {
  int half_proj = proj_size / 2;
  int q_proj_work = num_tokens * num_q_heads * half_proj;
  int kv_proj_work = num_tokens * num_kv_heads * half_proj;
  int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  CUDA_KERNEL_LOOP(i, q_proj_work + kv_proj_work) {
    bool q_tensor = i < q_proj_work;
    int num_heads = q_tensor ? num_q_heads : num_kv_heads;

    int real_i = q_tensor ? i : i - q_proj_work;
    int token_idx = real_i / (half_proj * num_heads);
    int pair_idx = real_i % half_proj;
    int head_idx = (real_i / half_proj) % num_heads;

    // input_ptr: [proj_size, tot_num_heads, num_tokens]
    int real_part_index = token_idx * proj_size * tot_num_heads +
                          (q_tensor ? 0 : proj_size * num_q_heads) +
                          head_idx * proj_size + pair_idx;
    int complex_part_index = real_part_index + half_proj;
    complex_input[i] = {(float)input_ptr[real_part_index],
                        (float)input_ptr[complex_part_index]};

    float inv_freq =
        1.0 / pow(rope_theta, (float)2.0 * pair_idx / proj_size); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / inv_freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen > low_freq_wavelen) {
        inv_freq = inv_freq / factor;
      }
      float smooth_factor =
          (original_max_position_embeddings / wavelen - low_freq_factor) /
          (high_freq_factor - low_freq_factor);
      if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
        inv_freq = ((1 - smooth_factor) * inv_freq / factor +
                    smooth_factor * inv_freq);
      }
    }

    int pos = tokenInfos[token_idx].abs_depth_in_request;
    inv_freq = inv_freq * pos;

    cuFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_bwd(DT *input_ptr,
                               cuFloatComplex *complex_input,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               float rope_theta,
                               bool llama3_rope,
                               float factor,
                               float low_freq_factor,
                               float high_freq_factor,
                               int original_max_position_embeddings,
                               int proj_size,
                               int num_tokens,
                               int num_q_heads,
                               int num_kv_heads) {
  int half_proj = proj_size / 2;
  int q_proj_work = num_tokens * num_q_heads * half_proj;
  int kv_proj_work = num_tokens * num_kv_heads * half_proj;
  int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  CUDA_KERNEL_LOOP(i, q_proj_work + kv_proj_work) {
    // compute indexes to visit first half proj_size of each of q/k tensor.
    // devQKVProj has shape [num_tokens, proj_size, tot_num_heads] in peft_bwd
    bool q_tensor = i < q_proj_work;
    int num_heads = q_tensor ? num_q_heads : num_kv_heads;

    int real_i = q_tensor ? i : i - q_proj_work;
    int token_idx = real_i / (half_proj * num_heads);
    int pair_idx = real_i % half_proj;
    int head_idx = (real_i / half_proj) % num_heads;
    assert(head_idx < num_heads);

    // int complex_part_index =
    //     (q_tensor ? 0 : num_tokens * proj_size * num_q_heads) +
    //     head_idx * proj_size * num_tokens + pair_idx * num_tokens +
    //     token_idx;
    // int real_part_index = complex_part_index + num_tokens * half_proj;
    int complex_part_index = token_idx * proj_size * tot_num_heads +
                             (q_tensor ? 0 : proj_size * num_q_heads) +
                             head_idx * proj_size + pair_idx;
    int real_part_index = complex_part_index + half_proj;

    complex_input[i] = {(float)input_ptr[real_part_index],
                        (float)input_ptr[complex_part_index]};

    float inv_freq =
        1.0 / pow(rope_theta, (float)2.0 * pair_idx / proj_size); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / inv_freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen > low_freq_wavelen) {
        inv_freq = inv_freq / factor;
      }
      float smooth_factor =
          (original_max_position_embeddings / wavelen - low_freq_factor) /
          (high_freq_factor - low_freq_factor);
      if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
        inv_freq = ((1 - smooth_factor) * inv_freq / factor +
                    smooth_factor * inv_freq);
      }
    }

    int pos = tokenInfos[token_idx].abs_depth_in_request;
    inv_freq = inv_freq * pos;

    cuFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
void apply_scaling_and_rotary(IncMultiHeadSelfAttentionMeta const *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              cudaStream_t inf_stream) {

  checkCUDA(cublasSetStream(m->handle.blas, inf_stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, inf_stream));
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);

  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;

  if (*(m->scaling_query)) {
    scaling_query_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           inf_stream>>>(output_ptr,
                                         m->qProjSize,
                                         num_tokens,
                                         m->num_q_heads,
                                         m->num_kv_heads,
                                         m->scaling_factor);
  }

  // Step 3: apply rotary embedding if needed
  if (m->rotary_embedding_meta->apply_rotary_embedding) {
    /*q&k*/
    int half_proj = m->qProjSize / 2;
    int q_proj_work = num_tokens * m->num_q_heads * half_proj;
    int kv_proj_work = num_tokens * m->num_kv_heads * half_proj;
    parallelism = q_proj_work + kv_proj_work;
    apply_rotary_embedding_fwd<<<GET_BLOCKS(parallelism),
                                 min(CUDA_NUM_THREADS, parallelism),
                                 0,
                                 inf_stream>>>(
        output_ptr,
        m->complex_input,
        m->token_infos,
        m->rotary_embedding_meta->rope_theta,
        (m->rotary_embedding_meta->rope_type == "llama3"),
        m->rotary_embedding_meta->factor,
        m->rotary_embedding_meta->low_freq_factor,
        m->rotary_embedding_meta->high_freq_factor,
        m->rotary_embedding_meta->original_max_position_embeddings,
        m->qProjSize,
        num_tokens,
        m->num_q_heads,
        m->num_kv_heads);
  }
}

struct RopeParams {
  bool apply_rotary_embedding;
  float rope_theta;
  bool llama3_rope;
  float factor;
  float low_freq_factor;
  float high_freq_factor;
  int original_max_position_embeddings;
  cuFloatComplex *complex_input;
  bool scaling_query;
  float scaling_factor;
};

template <typename DT>
struct DataPointers {
  DT const *qkv_ptr;
  // inference query buffer and unified kv cache
  half *qTmp_ptr;
  half *kvCache_ptr;
  // finetuning query buffer and kv cache
  DT *query_activation_buffer;
  DT *keyCachePeft;
  DT *valueCachePeft;
};

struct AttnGeneralParams {
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int num_tokens;
  int peft_req_idx;
  BatchConfig::PerTokenInfo const *tokenInfos;
  int32_t *kv_indptr;
  int32_t *kv_page_indices;
  bool const *request_completed;
};

template <typename DT>
__global__ void attention_prep_kernel(AttnGeneralParams general_params,
                                      DataPointers<DT> data_pointers,
                                      RopeParams rope_params) {
  int half_proj = general_params.head_dim / 2;
  int tot_num_heads =
      general_params.num_q_heads + 2 * general_params.num_kv_heads;
  int numel = half_proj * general_params.num_tokens * tot_num_heads;
  CUDA_KERNEL_LOOP(i, numel) {
    // Iterate over qkv_ptr: [head_dim, tot_num_heads, num_tokens]
    // Read data (in pairs wrt head_dim) and populate indices
    // ================================================================
    int pair_idx = i % half_proj;
    int global_head_idx = (i / half_proj) % tot_num_heads;
    int token_idx = i / (half_proj * tot_num_heads);
    bool is_q = global_head_idx < general_params.num_q_heads;
    bool is_k = global_head_idx >= general_params.num_q_heads &&
                global_head_idx <
                    general_params.num_q_heads + general_params.num_kv_heads;
    bool is_v = global_head_idx >=
                general_params.num_q_heads + general_params.num_kv_heads;
    // one and only one of is_q, is_k, is_v is true
    assert((is_q && !is_k && !is_v) || (!is_q && is_k && !is_v) ||
           (!is_q && !is_k && is_v));
    int head_idx = is_q ? global_head_idx
                        : (is_k ? global_head_idx - general_params.num_q_heads
                                : global_head_idx - general_params.num_q_heads -
                                      general_params.num_kv_heads);
    bool is_peft_token = general_params.tokenInfos[token_idx].request_index ==
                         general_params.peft_req_idx;
    int token_abs_idx =
        general_params.tokenInfos[token_idx].abs_depth_in_request;

    int src_a_offset = (token_idx * general_params.head_dim * tot_num_heads) +
                       (global_head_idx * general_params.head_dim) + pair_idx;
    int src_b_offset = src_a_offset + half_proj;
    DT a = data_pointers.qkv_ptr[src_a_offset];
    DT b = data_pointers.qkv_ptr[src_b_offset];

    // Apply transformations to the data
    // ================================================================
    // query scaling
    if (rope_params.scaling_query && is_q) {
      a *= rope_params.scaling_factor;
      b *= rope_params.scaling_factor;
    }
    // rope embedding
    if (rope_params.apply_rotary_embedding && (is_q || is_k)) {
      float inv_freq =
          1.0 / pow(rope_params.rope_theta,
                    (float)2.0 * pair_idx / general_params.head_dim);
      if (rope_params.llama3_rope) {
        float pi = CUDART_PI_F;
        float wavelen = 2 * pi / inv_freq;
        float low_freq_wavelen = rope_params.original_max_position_embeddings /
                                 rope_params.low_freq_factor;
        float high_freq_wavelen = rope_params.original_max_position_embeddings /
                                  rope_params.high_freq_factor;
        if (wavelen > low_freq_wavelen) {
          inv_freq = inv_freq / rope_params.factor;
        }
        float smooth_factor =
            (rope_params.original_max_position_embeddings / wavelen -
             rope_params.low_freq_factor) /
            (rope_params.high_freq_factor - rope_params.low_freq_factor);
        if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
          inv_freq = ((1 - smooth_factor) * inv_freq / rope_params.factor +
                      smooth_factor * inv_freq);
        }
      }
      inv_freq = inv_freq * token_abs_idx;
      cuFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};
      cuFloatComplex complex_input = {static_cast<float>(a),
                                      static_cast<float>(b)};
      complex_input = cuCmulf(complex_input, complex_pos);
      a = static_cast<DT>(complex_input.x);
      b = static_cast<DT>(complex_input.y);
    }

    // Store the data to the appropriate buffer
    // ================================================================
    if (!is_peft_token) {
      if (is_q) {
        // store to qTmp_ptr: [head_dim, num_q_heads, num_new_tokens]
        int dst_a_offset =
            (token_idx * general_params.head_dim * general_params.num_q_heads) +
            (head_idx * general_params.head_dim) + pair_idx;
        int dst_b_offset = dst_a_offset + half_proj;
        data_pointers.qTmp_ptr[dst_a_offset] = static_cast<half>(a);
        data_pointers.qTmp_ptr[dst_b_offset] = static_cast<half>(b);
      } else {
        // store to kvCache_ptr: [head_dim, num_kv_heads, page_size, 2,
        // max_num_pages]
        int const req_idx = general_params.tokenInfos[token_idx].request_index;
        assert(req_idx != general_params.peft_req_idx &&
               "Attempting to use inference KV cache for PEFT tokens");
        int req_idx_compact = 0;
        for (int j = 0; j < req_idx; j++) {
          if (!general_params.request_completed[j]) {
            req_idx_compact++;
          }
        }
        assert(req_idx_compact >= 0 && req_idx_compact <= req_idx &&
               "Invalid request index");
        int logical_page_idx = token_abs_idx / kPagesize;
        int page_idx =
            general_params
                .kv_page_indices[general_params.kv_indptr[req_idx_compact] +
                                 logical_page_idx];
        int to_k_idx = get_k_entry_offset_verify(token_abs_idx,
                                                 page_idx,
                                                 general_params.num_kv_heads,
                                                 general_params.head_dim);
        int to_v_idx = get_v_entry_offset_verify(token_abs_idx,
                                                 page_idx,
                                                 general_params.num_kv_heads,
                                                 general_params.head_dim);
        int dst_a_offset = is_k ? to_k_idx : to_v_idx;
        dst_a_offset += (head_idx * general_params.head_dim) + pair_idx;
        int dst_b_offset = dst_a_offset + half_proj;
        data_pointers.kvCache_ptr[dst_a_offset] = static_cast<half>(a);
        data_pointers.kvCache_ptr[dst_b_offset] = static_cast<half>(b);
      }
    } else {
      if (is_q) {
        // store to query_activation_buffer: [qProjSize, num_q_heads,
        // max_seq_len]
        int dst_a_offset = (token_abs_idx * general_params.head_dim *
                            general_params.num_q_heads) +
                           (head_idx * general_params.head_dim) + pair_idx;
        int dst_b_offset = dst_a_offset + half_proj;
        data_pointers.query_activation_buffer[dst_a_offset] = a;
        data_pointers.query_activation_buffer[dst_b_offset] = b;
      } else {
        // store to keyCachePeft/valueCachePeft: [head_dim, num_kv_heads,
        // max_seq_len]
        int dst_a_offset = (token_abs_idx * general_params.head_dim *
                            general_params.num_kv_heads) +
                           (head_idx * general_params.head_dim) + pair_idx;
        int dst_b_offset = dst_a_offset + half_proj;
        if (is_k) {
          data_pointers.keyCachePeft[dst_a_offset] = a;
          data_pointers.keyCachePeft[dst_b_offset] = b;
        } else {
          data_pointers.valueCachePeft[dst_a_offset] = a;
          data_pointers.valueCachePeft[dst_b_offset] = b;
        }
      }
    }
  }
}

template <typename DT>
void flashinfer_incr_attention(IncMultiHeadSelfAttentionMeta *m,
                               BatchConfig const *bc,
                               int shard_id,
                               DT *output_ptr,
                               cudaStream_t stream) {

  // global constant parameters
  uint32_t const num_q_heads = m->num_q_heads;
  uint32_t const num_kv_heads = m->num_kv_heads;
  uint32_t const head_dim = m->qProjSize;
  uint32_t const batch_size = bc->num_inference_requests();
  float const sm_scale =
      (*m->qk_prod_scaling) ? 1.0f / sqrt(m->qProjSize) : 1.0f;
  assert(batch_size > 0);
  assert(num_q_heads > 0);
  assert(num_kv_heads > 0);
  assert(head_dim > 0);
  assert(bc->num_inference_tokens() > 0);

  half *q = static_cast<half *>(m->queryTmp),
       *kv = static_cast<half *>(m->kvCache), *o = (half *)output_ptr;
  assert(q != nullptr && "q is null!");
  assert(kv != nullptr && "kv is null!");
  assert(o != nullptr && "o is null!");
  assert(m->handle.incr_attention_metadata->q_indptr != nullptr &&
         "q_indptr is null!");
  assert(m->handle.incr_attention_metadata->kv_indices != nullptr &&
         "kv_indices is null!");
  assert(m->handle.incr_attention_metadata->kv_indptr != nullptr &&
         "kv_indptr is null!");
  assert(m->handle.incr_attention_metadata->kv_last_page_len != nullptr &&
         "kv_last_page_len is null!");

  if (m->inference_debugging) {
    // qTmp: [qProjSize, num_q_heads, num_new_tokens]
    std::string fpath_q = get_fwd_dbg_folder(m, shard_id) + ".queryTmp.pt";
    at::Tensor tensor_q = createTorchTensorFromCuda<DT>(
        m->queryTmp, {m->qProjSize, (int)num_q_heads, bc->num_active_tokens()});
    torch::save(tensor_q, fpath_q.c_str());
    // kvcache: [head_dim, num_kv_heads, page_size, 2, max_num_pages]
    std::string fpath_kv = get_fwd_dbg_folder(m, shard_id) + ".kvCache.pt";
    at::Tensor tensor_kv = createTorchTensorFromCuda<DT>(
        m->kvCache,
        {m->qProjSize, (int)num_kv_heads, kPagesize, 2, m->num_kv_cache_pages});
    torch::save(tensor_kv, fpath_kv.c_str());
  }

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

  if (m->inference_debugging && false) {
    bc->save_to_file(get_fwd_dbg_folder(m, shard_id) + ".batch_config");
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".q_indptr";
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->q_indptr),
        batch_size + 1,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_indptr";
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->kv_indptr),
        batch_size + 1,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_indices";

    int num_pages;
    checkCUDA(
        cudaMemcpy(&num_pages,
                   m->handle.incr_attention_metadata->kv_indptr + batch_size,
                   sizeof(int),
                   cudaMemcpyDeviceToHost));
    save_tensor(
        static_cast<int32_t *>(m->handle.incr_attention_metadata->kv_indices),
        num_pages,
        fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".kv_last_page_len";
    save_tensor(static_cast<int32_t *>(
                    m->handle.incr_attention_metadata->kv_last_page_len),
                batch_size,
                fpath.c_str());
  }

  assert(m->handle.incr_attention_metadata->prompt_handler_collections.count(
             batch_size) != 0 &&
         "Handler is not initialized");
  void *handler =
      m->handle.incr_attention_metadata->prompt_handler_collections[batch_size];
  // printf("obtained handler\n");
  assert(sizeof(DT) == 2 && "FlashInfer only supports half precision");
  DISPATCH_HEADDIM(head_dim, HEAD_DIM, {
    // printf("Launching BatchPrefillWithPagedKVCacheWrapperDispatched\n");
    cudaError_t result =
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
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to run "
                               "IncrementalDecodingAttentionForwardKernel: " +
                               std::string(cudaGetErrorString(result)));
    }
  });
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *qkv_ptr,
                      DT *output_ptr,
                      cudaStream_t inf_stream,
                      cudaStream_t peft_stream) {

  // qkv_ptr: [qProjSize, tot_num_heads, num_new_tokens]
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);
  size_t tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;

  if (m->inference_debugging) {
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".devQKVProjArray.pt";
    at::Tensor tensor = createTorchTensorFromCuda<DT>(
        qkv_ptr, {m->qProjSize, (int)tot_num_heads, bc->num_active_tokens()});
    torch::save(tensor, fpath.c_str());
  }

  int peft_req_idx = -1;
  if (bc->num_finetuning_fwd_requests() > 0) {
    peft_req_idx = bc->finetuning_request_index();
  }
  AttnGeneralParams general_params = {
      .num_q_heads = m->num_q_heads,
      .num_kv_heads = m->num_kv_heads,
      .head_dim = m->qProjSize,
      .num_tokens = bc->num_active_tokens(),
      .peft_req_idx = peft_req_idx,
      .tokenInfos = m->token_infos,
      .kv_indptr = m->handle.incr_attention_metadata->kv_indptr,
      .kv_page_indices = m->handle.incr_attention_metadata->kv_indices,
      .request_completed = m->request_completed};
  RopeParams rope_params = {
      .apply_rotary_embedding =
          m->rotary_embedding_meta->apply_rotary_embedding,
      .rope_theta = m->rotary_embedding_meta->rope_theta,
      .llama3_rope = m->rotary_embedding_meta->rope_type == "llama3",
      .factor = m->rotary_embedding_meta->factor,
      .low_freq_factor = m->rotary_embedding_meta->low_freq_factor,
      .high_freq_factor = m->rotary_embedding_meta->high_freq_factor,
      .original_max_position_embeddings =
          m->rotary_embedding_meta->original_max_position_embeddings,
      .complex_input = m->complex_input,
      .scaling_query = *(m->scaling_query),
      .scaling_factor = m->scaling_factor};
  DataPointers<DT> data_pointers = {
      .qkv_ptr = qkv_ptr,
      .qTmp_ptr = static_cast<half *>(m->queryTmp),
      .kvCache_ptr = static_cast<half *>(m->kvCache),
      .query_activation_buffer = static_cast<DT *>(m->query_activation_buffer),
      .keyCachePeft = static_cast<DT *>(m->keyCachePeft),
      .valueCachePeft = static_cast<DT *>(m->valueCachePeft)};
  int parallelism = m->qProjSize * tot_num_heads * bc->num_active_tokens() / 2;
  attention_prep_kernel<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          inf_stream>>>(
      general_params, data_pointers, rope_params);

  if (bc->num_finetuning_fwd_tokens() > 0) {
    // wait until preparation has finished
    cudaEvent_t prep_done;
    cudaEventCreate(&prep_done);
    cudaEventRecord(prep_done, inf_stream);
    cudaStreamWaitEvent(peft_stream, prep_done, 0);

    flash_compute_attention_kernel_peft<DT>(
        m, bc, output_ptr, shard_id, peft_stream);

    assert(m->peft_token_infos != nullptr);
    assert(m->peft_token_infos_size == sizeof(BatchConfig::PerTokenInfo) *
                                           BatchConfig::max_sequence_length());
    int num_ft_tokens = bc->num_finetuning_fwd_tokens();
    int i = bc->finetuning_request_index();
    int tokens_previous_requests =
        bc->requestsInfo[i].first_token_offset_in_batch;
    int prev_steps_tokens = bc->requestsInfo[i].first_token_depth_in_request;
    for (int j = 0; j < num_ft_tokens; j++) {
      m->peft_token_infos[prev_steps_tokens + j] =
          bc->tokensInfo[tokens_previous_requests + j];
    }
  }

  // flashinfer sdpa
  assert(bc->num_finetuning_fwd_tokens() >= 0 &&
         bc->num_finetuning_bwd_tokens() >= 0);
  if (bc->num_inference_tokens() > 0) {
    flashinfer_incr_attention<DT>(m, bc, shard_id, output_ptr, inf_stream);
  }
}

// todo(yingyi): replace with flash-attn
template <typename DT>
void flash_peft_bwd_kernel(IncMultiHeadSelfAttentionMeta *m,
                           BatchConfig const *bc,
                           int shard_id,
                           DT *input_grad_ptr,
                           DT const *output_grad_ptr,
                           cudaStream_t peft_stream) {
  // Step 0: param check as in peft_bwd_kernel
  // ================================================================
  assert(!m->offload);
  checkCUDA(cublasSetStream(m->handle.peft_blas, peft_stream));
  checkCUDNN(cudnnSetStream(m->handle.peft_dnn, peft_stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();
  int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
  int num_total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                         bc->requestsInfo[i].num_tokens_in_batch;
  // Currently assume we are calculating gradients for all tokens
  // of a request
  assert(num_tokens == num_total_tokens);
  assert(num_total_tokens == bc->requestsInfo[i].max_length);
  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  // assert(bc->requestsInfo[i].first_token_offset_in_batch == 0);

  // end step 0
  // ================================================================

  // step 1: compute gradients w.r.t. QKV
  // ================================================================
  at::Tensor dout, q, k, v, out, softmax_lse;
  std::optional<at::Tensor> dq_ = std::nullopt;
  std::optional<at::Tensor> dk_ = std::nullopt;
  std::optional<at::Tensor> dv_ = std::nullopt;
  std::optional<at::Tensor> alibi_slopes_ = std::nullopt;
  float p_dropout, softmax_scale;
  bool is_causal, deterministic;
  int window_size_left, window_size_right;
  float softcap;
  std::optional<at::Generator> gen_ = std::nullopt;
  std::optional<at::Tensor> rng_state = std::nullopt;

  set_wrapper_mha_bwd_1_params_peft<DT>(m,
                                        bc,
                                        shard_id,
                                        input_grad_ptr,
                                        output_grad_ptr,
                                        dout,
                                        q,
                                        k,
                                        v,
                                        out,
                                        softmax_lse,
                                        dq_,
                                        dk_,
                                        dv_,
                                        alibi_slopes_,
                                        p_dropout,
                                        softmax_scale,
                                        is_causal,
                                        window_size_left,
                                        window_size_right,
                                        softcap,
                                        deterministic,
                                        gen_,
                                        rng_state);

  auto result = _wrapper_mha_bwd_1(dout,
                                   q,
                                   k,
                                   v,
                                   out,
                                   softmax_lse,
                                   dq_,
                                   dk_,
                                   dv_,
                                   alibi_slopes_,
                                   p_dropout,
                                   softmax_scale,
                                   is_causal,
                                   window_size_left,
                                   window_size_right,
                                   softcap,
                                   deterministic,
                                   gen_,
                                   rng_state,
                                   peft_stream);
  auto dq = result[0];
  auto dk = result[1];
  auto dv = result[2];
  auto softmax_d = result[3];

  // checkpoint: dq, dk, dv data_ptr
  // std::cout << "5. the address of dq is: " << dq.data_ptr() << std::endl;
  // std::cout << "5. the address of dk is: " << dk.data_ptr() << std::endl;
  // std::cout << "5. the address of dv is: " << dv.data_ptr() << std::endl;

  // matrix B's layout: [num_tokens, qProjsize * tot_num_heads]
  // end step 1
  // ================================================================

  // step 2: return results
  // ================================================================
  // Print the values of dq, dk, dv to files
  if (m->inference_debugging) {
    int i = bc->finetuning_request_index();
    int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    // save dv

    std::string dq_fpath = get_peft_dbg_folder(m, shard_id) + ".dq.pt";
    std::string dk_fpath = get_peft_dbg_folder(m, shard_id) + ".dk.pt";
    std::string dv_fpath = get_peft_dbg_folder(m, shard_id) + ".dv.pt";
    torch::save(dq.clone().detach(), dq_fpath);
    torch::save(dk.clone().detach(), dk_fpath);
    torch::save(
        dv.clone().detach(),
        dv_fpath); // shape: batch_size x seqlen_k x num_heads_k x head_size

    std::cout << "the address of workSpace: " << m->handle.workSpace
              << std::endl;
    std::cout << "the address of dq data: " << dq.data_ptr() << std::endl;
    std::cout << "the address of dk data: " << dk.data_ptr() << std::endl;
    std::cout << "the address of dv data: " << dv.data_ptr() << std::endl;

    auto dv_ptr = static_cast<DT *>(m->handle.workSpace) +
                  2 * num_tokens * (m->qProjSize * m->num_q_heads);
    auto dk_ptr = static_cast<DT *>(m->handle.workSpace) +
                  num_tokens * (m->qProjSize * m->num_q_heads);
    auto dq_ptr = static_cast<DT *>(m->handle.workSpace);
    std::cout << "the address of dq_ptr should be: " << dq_ptr << std::endl;
    std::cout << "the address of dk_ptr should be: " << dk_ptr << std::endl;
    std::cout << "the address of dv_ptr should be: " << dv_ptr << std::endl;

    // std::string filename =
    //     get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray_pre";
    // save_tensor(
    //     C, num_tokens * m->qProjSize * m->num_q_heads * 3, filename.c_str());
  }

  // print shape of dq
  // std::cout << "dq shape: " << dq.sizes() << std::endl;
  // std::cout << "dk shape: " << dk.sizes() << std::endl;
  // std::cout << "dv shape: " << dv.sizes() << std::endl;

  int head_size = m->qProjSize;
  int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  auto input_grad_tensor = createTorchTensorFromCuda<DT>(
      input_grad_ptr, {head_size, tot_num_heads, num_total_tokens});

  auto dq_tensor = input_grad_tensor.narrow(1, 0, m->num_q_heads);
  auto dk_tensor = input_grad_tensor.narrow(1, m->num_q_heads, m->num_kv_heads);
  auto dv_tensor = input_grad_tensor.narrow(
      1, m->num_q_heads + m->num_kv_heads, m->num_kv_heads);
  // std::cout << "dq_tensor shape: " << dq_tensor.sizes() << std::endl;
  // std::cout << "dk_tensor shape: " << dk_tensor.sizes() << std::endl;
  // std::cout << "dv_tensor shape: " << dv_tensor.sizes() << std::endl;

  dq_tensor.copy_(dq.squeeze().permute({2, 1, 0}));
  dk_tensor.copy_(dk.squeeze().permute({2, 1, 0}));
  dv_tensor.copy_(dv.squeeze().permute({2, 1, 0}));

  if (m->rotary_embedding_meta->apply_rotary_embedding) {
    checkCUDA(cudaMemcpyAsync(m->peft_token_infos_device,
                              m->peft_token_infos,
                              m->peft_token_infos_size,
                              cudaMemcpyHostToDevice,
                              peft_stream));
    assert(m->qProjSize == m->kProjSize);
    /*q&k*/
    int half_proj = m->qProjSize / 2;
    int q_proj_work = num_tokens * m->num_q_heads * half_proj;
    int kv_proj_work = num_tokens * m->num_kv_heads * half_proj;
    int parallelism = q_proj_work + kv_proj_work;
    apply_rotary_embedding_bwd<<<GET_BLOCKS(parallelism),
                                 min(CUDA_NUM_THREADS, parallelism),
                                 0,
                                 peft_stream>>>(
        input_grad_ptr,
        m->complex_input,
        m->peft_token_infos_device,
        m->rotary_embedding_meta->rope_theta,
        (m->rotary_embedding_meta->rope_type == "llama3"),
        m->rotary_embedding_meta->factor,
        m->rotary_embedding_meta->low_freq_factor,
        m->rotary_embedding_meta->high_freq_factor,
        m->rotary_embedding_meta->original_max_position_embeddings,
        m->qProjSize,
        num_tokens,
        m->num_q_heads,
        m->num_kv_heads);
  }

  // end step 2
  // ================================================================

  // step3: (optional, only for testing)
  // invoke attention bwd from torch
  // ========================================================================
  // todo(yingyi): align with bwd results
  // skip for now, leave alignments to python script
  // end step 3
  // ========================================================================
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
    GenericTensorAccessorW const &output) {
  cudaStream_t inf_stream;
  checkCUDA(get_legion_stream(&inf_stream));
  cudaStream_t peft_stream;
  checkCUDA(get_legion_stream(&peft_stream));

  // cudaEvent_t t_start, t_end;
  // if (m->profiling) {
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);
  // }

  assert(input.data_type == output.data_type);

  if (input.data_type == DT_HALF) {
    Kernels::IncMultiHeadAttention::inference_kernel(m,
                                                     bc,
                                                     shard_id,
                                                     input.get_half_ptr(),
                                                     output.get_half_ptr(),
                                                     inf_stream,
                                                     peft_stream);
  } else if (input.data_type == DT_BFLOAT16) {
    Kernels::IncMultiHeadAttention::inference_kernel(m,
                                                     bc,
                                                     shard_id,
                                                     input.get_bfloat16_ptr(),
                                                     output.get_bfloat16_ptr(),
                                                     inf_stream,
                                                     peft_stream);
  } else {
    assert(false && "Unspported data type");
  }

  // if (m->profiling) {
  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);
  // }
}

/*static*/
void IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input_grad.data_type == output_grad.data_type);

  if (input_grad.data_type == DT_HALF) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::flash_peft_bwd_kernel(
        m,
        bc,
        shard_id,
        input_grad.get_half_ptr(),
        output_grad.get_half_ptr(),
        stream);
  } else if (input_grad.data_type == DT_BFLOAT16) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::flash_peft_bwd_kernel(
        m,
        bc,
        shard_id,
        input_grad.get_bfloat16_ptr(),
        output_grad.get_bfloat16_ptr(),
        stream);
  } else {
    assert(false && "Unspported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention PEFT backward time = %.9fms\n", elapsed);
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    MemoryAllocator &inf_mem_allocator,
    MemoryAllocator &kv_cache_mem_allocator,
    MemoryAllocator &peft_mem_allocator,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    INC_DECODING_MODE,
                                    attn,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->rotary_embedding_meta,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->scaling_factor,
                                    inf_mem_allocator,
                                    kv_cache_mem_allocator,
                                    peft_mem_allocator,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
                                    attn->num_kv_cache_pages,
                                    attn->quantization_type,
                                    attn->offload) {}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    InferenceMode infer_mode,
    Op const *attn,
    int _qProjSize,
    int _kProjSize,
    int _vProjSize,
    int _oProjSize,
    RotaryEmbeddingMeta _rotary_embedding_meta,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    float _scaling_factor,
    MemoryAllocator &inf_mem_allocator,
    MemoryAllocator &kv_cache_mem_allocator,
    MemoryAllocator &peft_mem_allocator,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    int _num_kv_cache_pages,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
  // assume dimensions match for now
  qProjSize = _qProjSize;
  kProjSize = _kProjSize;
  vProjSize = _vProjSize;
  oProjSize = _oProjSize;
  assert(qProjSize == kProjSize &&
         kProjSize == vProjSize); // required for attention QK.T matmul
  size_t size_of_dt = data_type_size(attn->data_type);
  quantization_type = _quantization_type;
  offload = _offload;

  global_num_q_heads = _global_num_q_heads;
  global_num_kv_heads = _global_num_kv_heads;
  num_q_heads = _num_q_heads;
  num_kv_heads = _num_kv_heads;

  rotary_embedding_meta = new RotaryEmbeddingMeta(_rotary_embedding_meta);
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;

  num_kv_cache_pages = _num_kv_cache_pages;
  assert(num_kv_cache_pages > 0 || enable_peft_finetuning);

  // spec decoding and peft finetuning are mutually exclusive
  if (enable_peft_finetuning) {
    assert(infer_mode == INC_DECODING_MODE);
  }

  size_t inf_instance_size = 0;
  size_t kv_cache_instance_size = 0;
  size_t peft_instance_size = 0;

  // Compute total GPU memory size needed
  {
    // 1. GQA pointers for batch matmul. Used by PEFT and spec_inc if
    // num_q_heads > num_kv_heads
    if (num_q_heads > num_kv_heads && (infer_mode == BEAM_SEARCH_MODE)) {
      assert(num_q_heads % num_kv_heads == 0 &&
             "num_q_heads must be divisible by num_kv_heads");
      assert((attn->data_type == DT_FLOAT || attn->data_type == DT_HALF ||
              attn->data_type == DT_BFLOAT16) &&
             "Unsupported data type");
      gqa_ptr_array_size = num_q_heads * sizeof(void *);
      inf_instance_size += 3 * gqa_ptr_array_size; // fwd
    }

    // 2. KV cache
    key_cache_size = value_cache_size =
        num_kv_heads * kProjSize * kPagesize * num_kv_cache_pages;
    if (infer_mode == BEAM_SEARCH_MODE || infer_mode == TREE_VERIFY_MODE) {
      // a K-ary tree max node is (k^n - 1) / 2
      assert(key_cache_size == value_cache_size);
      assert(key_cache_size >=
             num_kv_heads * kProjSize *
                 BeamSearchBatchConfig::max_requests_per_batch() *
                 (BatchConfig::max_sequence_length() +
                  BatchConfig::max_spec_tree_token_num()));
    }
    kv_cache_instance_size += (key_cache_size + value_cache_size) * size_of_dt;
    if (enable_peft_finetuning) {
      // add kv cache for single sequence
      peft_key_cache_size = peft_value_cache_size =
          num_kv_heads * kProjSize * BatchConfig::max_sequence_length();
      peft_instance_size +=
          (peft_key_cache_size + peft_value_cache_size) * size_of_dt;
    }

    // 3. buffers for intermediate results
    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int max_tokens_per_batch = (infer_mode == TREE_VERIFY_MODE)
                                   ? BatchConfig::max_verify_tokens_per_batch()
                                   : BatchConfig::max_tokens_per_batch();
    // devQKVProjArray
    qkv_max_proj_size = qProjSize * tot_num_heads * max_tokens_per_batch;
    if (infer_mode == BEAM_SEARCH_MODE || infer_mode == TREE_VERIFY_MODE) {
      inf_instance_size += qkv_max_proj_size * size_of_dt;
    }
    // queryTmp and outputTmp: only for paged attention
    if (infer_mode == INC_DECODING_MODE) {
      query_tmp_size = num_q_heads * qProjSize * max_tokens_per_batch;
      inf_instance_size += (query_tmp_size)*size_of_dt;
    }
    // complex_input & complex_input_bwd
    complex_size = max_tokens_per_batch * qProjSize *
                   (num_q_heads + num_kv_heads) /
                   2; // only used for Q and K, not V
    inf_instance_size += complex_size * sizeof(cuFloatComplex);
    if (enable_peft_finetuning) {
      complex_size_bwd = BatchConfig::max_sequence_length() * qProjSize *
                         (num_q_heads + num_kv_heads) /
                         2; // only used for Q and K, not V
      peft_instance_size += complex_size_bwd * sizeof(cuFloatComplex);
    }
    // QK prods and QK prods (softmax)
    if (infer_mode == BEAM_SEARCH_MODE) {
      qk_prod_size = max_tokens_per_batch * BatchConfig::max_sequence_length() *
                     num_q_heads;
      inf_instance_size += 2 * qk_prod_size * size_of_dt;
    }
    // PEFT partial results buffers
    if (enable_peft_finetuning) {
      allocated_peft_buffer_size1 = BatchConfig::max_sequence_length() *
                                    num_q_heads * qProjSize * size_of_dt;
      flash_attn_softmax_lse_size =
          BatchConfig::max_sequence_length() * num_q_heads * sizeof(float);
      flash_attn_out_size = qProjSize * num_q_heads *
                            BatchConfig::max_sequence_length() * size_of_dt;
      peft_token_infos = (BatchConfig::PerTokenInfo *)calloc(
          1,
          sizeof(BatchConfig::PerTokenInfo) *
              BatchConfig::max_sequence_length());
      peft_token_infos_size = sizeof(BatchConfig::PerTokenInfo) *
                              BatchConfig::max_sequence_length();
      peft_instance_size += allocated_peft_buffer_size1;
      peft_instance_size += flash_attn_softmax_lse_size + flash_attn_out_size;
      peft_instance_size += peft_token_infos_size;
    }

    // 4. offload: TBD
    if (offload) {
      assert(false && "TODO");
    }
  }

  // Allocate chunk of memory
  inf_mem_allocator.create_legion_instance(
      inf_instance, inf_instance_size, "IncMultiHeadSelfAttentionMeta (inf)");
  kv_cache_mem_allocator.create_legion_instance(
      kv_cache_instance, kv_cache_instance_size, "KV Cache");
  peft_mem_allocator.create_legion_instance(
      peft_instance,
      peft_instance_size,
      "IncMultiHeadSelfAttentionMeta (peft)");

  // Assign pointers from chunk of memory
  {
    // gqa pointers
    if (num_q_heads > num_kv_heads && (infer_mode == BEAM_SEARCH_MODE)) {
      assert(num_q_heads % num_kv_heads == 0 &&
             "Num Q heads must be a multiple of num KV heads");
      d_A_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_B_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_C_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
    }

    // KV cache
    if (infer_mode == INC_DECODING_MODE) {
      kvCache = kv_cache_mem_allocator.allocate_instance_untyped(
          (key_cache_size + value_cache_size) * size_of_dt);
      keyCache = valueCache = nullptr;
    } else {
      kvCache = nullptr;
      keyCache = kv_cache_mem_allocator.allocate_instance_untyped(
          key_cache_size * size_of_dt);
      valueCache = kv_cache_mem_allocator.allocate_instance_untyped(
          value_cache_size * size_of_dt);
    }
    if (enable_peft_finetuning) {
      assert(infer_mode == INC_DECODING_MODE);
      keyCachePeft = peft_mem_allocator.allocate_instance_untyped(
          peft_key_cache_size * size_of_dt);
      valueCachePeft = peft_mem_allocator.allocate_instance_untyped(
          peft_value_cache_size * size_of_dt);

      // todo(gabriele): review the allocation of flash-attn bwd context
      // flash-attn out: (head_size, num_q_heads, num_new_tokens)
      flash_attn_out =
          peft_mem_allocator.allocate_instance_untyped(flash_attn_out_size);
      // flash-attn softmax_lse
      flash_attn_softmax_lse = peft_mem_allocator.allocate_instance_untyped(
          flash_attn_softmax_lse_size);

      // todo(gabriele): review flash-attn metadara
      flash_attn_p_dropout = 0.0f;
      flash_attn_is_causal = true;
      flash_attn_return_softmax = false;
      flash_attn_softcap = 0.0f;
      // flash_attn_rng_state_0 = 0;
      // flash_attn_rng_state_1 = 0;
      flash_attn_window_size_left = -1;
      flash_attn_window_size_right = -1;
    } else {
      keyCachePeft = valueCachePeft = nullptr;
    }
    // intermediate buffers
    // devQKVProjArray: used to store QKV proj so that we can modify them (apply
    // rope, etc)
    if (infer_mode == BEAM_SEARCH_MODE || infer_mode == TREE_VERIFY_MODE) {
      devQKVProjArray = inf_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size * size_of_dt);
    } else {
      devQKVProjArray = nullptr;
    }
    // queryTmp and outputTmp: only for paged attention
    if (infer_mode == INC_DECODING_MODE) {
      queryTmp = inf_mem_allocator.allocate_instance_untyped(query_tmp_size *
                                                             size_of_dt);
    }
    // complex input
    complex_input =
        inf_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
    complex_input_bwd =
        peft_mem_allocator.allocate_instance<cuFloatComplex>(complex_size_bwd);
    // qk_prods, qk_prods_softmax
    if (infer_mode == BEAM_SEARCH_MODE) {
      qk_prods = inf_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = inf_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
    }
    // peft partial result buffers
    if (enable_peft_finetuning) {
      query_activation_buffer = peft_mem_allocator.allocate_instance_untyped(
          allocated_peft_buffer_size1);
      peft_token_infos_device =
          (BatchConfig::PerTokenInfo *)peft_mem_allocator
              .allocate_instance_untyped(peft_token_infos_size);
    }

    token_infos = static_cast<BatchConfig::PerTokenInfo *>(
        handler.batch_config_metadata->tokens_info);
    request_infos = static_cast<BatchConfig::PerRequestInfo *>(
        handler.batch_config_metadata->requestsInfo);
    request_completed =
        static_cast<bool *>(handler.batch_config_metadata->request_completed);

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
    }
  }

  // set attention constants
  // std::cerr << "Enabling incr attention metadata for handler incr meta: "
  //           << handler.incr_attention_metadata << std::endl;
  handler.incr_attention_metadata->set_enabled(true);
  handler.incr_attention_metadata->set_num_q_heads(num_q_heads);
  handler.incr_attention_metadata->set_num_kv_heads(num_kv_heads);
  handler.incr_attention_metadata->set_head_dim(qProjSize);

  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (inf_instance != Realm::RegionInstance::NO_INST) {
    inf_instance.destroy();
  }
  if (kv_cache_instance != Realm::RegionInstance::NO_INST) {
    kv_cache_instance.destroy();
  }
  if (peft_instance != Realm::RegionInstance::NO_INST) {
    peft_instance.destroy();
  }
}

template void Kernels::IncMultiHeadAttention::run_batched_matmul<half>(
    IncMultiHeadSelfAttentionMeta const *meta,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    half const *A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    half const *B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    half *C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo,
    cudaStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::run_batched_matmul<float>(
    IncMultiHeadSelfAttentionMeta const *meta,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    float const *A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    float const *B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    float *C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo,
    cudaStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::run_batched_matmul<__ff_bfloat16>(
    IncMultiHeadSelfAttentionMeta const *meta,
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    __ff_bfloat16 const *A,
    cudaDataType Atype,
    int lda,
    long long int strideA,
    __ff_bfloat16 const *B,
    cudaDataType Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    __ff_bfloat16 *C,
    cudaDataType Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cudaDataType computeType,
    cublasGemmAlgo_t algo,
    cudaStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float *output_ptr,
    cudaStream_t inf_stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half *output_ptr,
    cudaStream_t inf_stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<__ff_bfloat16>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    __ff_bfloat16 *output_ptr,
    cudaStream_t inf_stream);

template __global__ void
    Kernels::IncMultiHeadAttention::apply_position_bias_qkprd<float>(
        float *input_ptr,
        int num_tokens,
        int num_total_tokens,
        int num_heads,
        int global_num_q_heads,
        int shard_id);

template __global__ void
    Kernels::IncMultiHeadAttention::apply_position_bias_qkprd<half>(
        half *input_ptr,
        int num_tokens,
        int num_total_tokens,
        int num_heads,
        int global_num_q_heads,
        int shard_id);

template __global__ void
    Kernels::IncMultiHeadAttention::apply_position_bias_qkprd<__ff_bfloat16>(
        __ff_bfloat16 *input_ptr,
        int num_tokens,
        int num_total_tokens,
        int num_heads,
        int global_num_q_heads,
        int shard_id);
}; // namespace FlexFlow
