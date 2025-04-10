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
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/hip_helper.h"
#include "hip/hip_complex.h"
#include <hip/hip_math_constants.h>
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

template <typename T>
__device__ __forceinline__ T
    WARP_SHFL(unsigned mask, T var, int srcLane, int width = warpSize) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_sync(mask, var, srcLane, width);
#else
  return __shfl(var, srcLane, width);
#endif
}

template <typename T>
__device__ __forceinline__ T
    WARP_SHFL_XOR(unsigned mask, T var, int laneMask, int width = warpSize) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_xor_sync(mask, var, laneMask, width);
#else
  return __shfl_xor(var, laneMask, width);
#endif
}

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
    // kCache_ptr: [head_dim, num_kv_heads, max_seq_len, max_batch_size]
    // vCache_ptr: [head_dim, num_kv_heads, max_seq_len, max_batch_size]

    // i is iterating over one set of key/val projections from the input
    int token_idx = i / (head_dim * num_kv_heads);
    int head_idx = (i / head_dim) % num_kv_heads;
    int offset = i % head_dim;

    int tot_num_heads = num_q_heads + 2 * num_kv_heads;
    int key_src_idx = token_idx * head_dim * tot_num_heads +
                      head_dim * num_q_heads + head_dim * head_idx + offset;
    int val_src_idx = key_src_idx + head_dim * num_kv_heads;

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    int dst_idx = req_id * (head_dim * num_kv_heads * max_seq_len) +
                  tok_id * head_dim * num_kv_heads + head_idx * head_dim +
                  offset;

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
                        hipblasHandle_t handle,
                        hipblasOperation_t transa,
                        hipblasOperation_t transb,
                        int m,
                        int n,
                        int k,
                        void const *alpha,
                        const DT *A,
                        hipblasDatatype_t Atype,
                        int lda,
                        long long int strideA,
                        const DT *B,
                        hipblasDatatype_t Btype,
                        int ldb,
                        long long int strideB,
                        void const *beta,
                        DT *C,
                        hipblasDatatype_t Ctype,
                        int ldc,
                        long long int strideC,
                        int batchCount,
                        hipblasDatatype_t computeType,
                        hipblasGemmAlgo_t algo,
                        hipStream_t stream,
                        int batch_ratio_a,
                        int batch_ratio_b,
                        int batch_ratio_c,
                        bool bwd) {
  if (batch_ratio_a == 1 && batch_ratio_b == 1 && batch_ratio_c == 1) {
    checkCUDA(hipblasGemmStridedBatchedEx(handle,
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
      checkCUDA(hipMemcpyAsync(meta->d_A_array,
                               h_A_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));
      checkCUDA(hipMemcpyAsync(meta->d_B_array,
                               h_B_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));
      checkCUDA(hipMemcpyAsync(meta->d_C_array,
                               h_C_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));

      checkCUDA(hipblasGemmBatchedEx(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     alpha,
                                     (void const **)meta->d_A_array,
                                     Atype,
                                     lda,
                                     (void const **)meta->d_B_array,
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
      checkCUDA(hipMemcpyAsync(meta->d_A_array2,
                               h_A_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));
      checkCUDA(hipMemcpyAsync(meta->d_B_array2,
                               h_B_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));
      checkCUDA(hipMemcpyAsync(meta->d_C_array2,
                               h_C_array,
                               batchCount * sizeof(DT *),
                               hipMemcpyHostToDevice,
                               stream));

      checkCUDA(hipblasGemmBatchedEx(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     alpha,
                                     (void const **)meta->d_A_array2,
                                     Atype,
                                     lda,
                                     (void const **)meta->d_B_array2,
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

template <typename DT>
__global__ void store_softmax_activation(DT const *qk_prods_softmax,
                                         DT *softmax_activation_buffer,
                                         int num_new_tokens,
                                         int total_tokens,
                                         int max_finetuning_seq_len,
                                         int num_q_heads) {
  CUDA_KERNEL_LOOP(i, num_new_tokens * total_tokens * num_q_heads) {
    // qk_prods_softmax: [num_new_tokens, total_tokens, num_q_heads]
    // softmax activation buffer: [MAX_FINETUNING_LENGTH(num_new_tokens),
    // MAX_FINETUNING_LENGTH(total_tokens), num_q_heads]
    int tokens_previous_steps = total_tokens - num_new_tokens;
    int new_tokens_idx = i % num_new_tokens;
    int total_tokens_idx = (i / num_new_tokens) % total_tokens;
    int head_idx = i / (num_new_tokens * total_tokens);
    int src_idx = head_idx * num_new_tokens * total_tokens +
                  total_tokens_idx * num_new_tokens + new_tokens_idx;
    int dst_idx = head_idx * max_finetuning_seq_len * max_finetuning_seq_len +
                  total_tokens_idx * max_finetuning_seq_len +
                  (tokens_previous_steps + new_tokens_idx);

    softmax_activation_buffer[dst_idx] = qk_prods_softmax[src_idx];
  }
}

template <typename DT>
void compute_attention_kernel_prompt(IncMultiHeadSelfAttentionMeta *m,
                                     BatchConfig const *bc,
                                     DT *attn_heads,
                                     int shard_id,
                                     hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  hipblasDatatype_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  miopenDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  hipblasDatatype_t compute_type = cublas_data_type;

  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  int num_processed_prompt_tokens = 0;
  for (int req_idx = 0; req_idx < bc->max_requests_per_batch(); req_idx++) {
    if (bc->request_completed[req_idx] || is_decoding_request(bc, req_idx) ||
        is_finetuning_bwd_request(bc, req_idx)) {
      continue;
    }
    int num_new_tokens = bc->requestsInfo[req_idx].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[req_idx].first_token_depth_in_request +
                       bc->requestsInfo[req_idx].num_tokens_in_batch;
    if (num_new_tokens <= 0) {
      continue;
    }
    // Copy query to m->query_activation_buffer if we need to compute
    // PEFT backward
    if (bc->requestsInfo[req_idx].finetuning_request &&
        !bc->requestsInfo[req_idx].finetuning_backward_phase) {
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
      int parallelism = m->qProjSize * m->num_q_heads * num_new_tokens;
      int tokens_previous_steps = total_tokens - num_new_tokens;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(store_query_cache),
                         GET_BLOCKS(parallelism),
                         min(CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         static_cast<DT *>(m->devQKVProjArray),
                         static_cast<DT *>(m->query_activation_buffer),
                         num_new_tokens,
                         num_processed_prompt_tokens,
                         tokens_previous_steps,
                         m->qProjSize,
                         m->num_q_heads,
                         m->num_kv_heads);
    }
    // Step 1: compute query-key product QK.T/sqrt(d_k)
    {
      DT alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        // Scale by sqrt(d_k) as per the original attention paper
        alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
      }
      // after transpositions
      int m_ = num_new_tokens;
      int n = total_tokens;
      int k = m->qProjSize;
      // before transpositions
      int lda = m->qProjSize * tot_num_heads;
      int ldb = m->kProjSize * m->num_kv_heads;
      int ldc = num_new_tokens;
      // N.B. strides are applied before transpose operations
      int strideA = m->qProjSize;
      int strideB = m->kProjSize;
      int strideC = num_new_tokens * total_tokens;

      // matrix A: devQKVProjArray
      // matrix A's layout: [qProjSize, tot_num_heads, num_new_tokens]
      // To get query projection, skip over Q entries from previous requests
      DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                    bc->requestsInfo[req_idx].first_token_offset_in_batch *
                        m->qProjSize * (m->num_q_heads + 2 * m->num_kv_heads);
      // matrix B: key cache
      // matrix B's layout: [kProjSize, num_kv_heads, total_tokens]
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      DT const *B = static_cast<DT *>(m->keyCache) +
                    req_idx * (m->kProjSize * m->num_kv_heads *
                               BatchConfig::max_sequence_length());
      // matrix C: qk_prods (current req only)
      // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
      DT *C = static_cast<DT *>(m->qk_prods);
      run_batched_matmul<DT>(m,
                             m->handle.blas,
                             HIPBLAS_OP_T,
                             HIPBLAS_OP_N,
                             m_,
                             n,
                             k,
                             &alpha,
                             A,
                             cublas_data_type,
                             lda,
                             strideA,
                             B,
                             cublas_data_type,
                             ldb,
                             strideB,
                             &beta,
                             C,
                             cublas_data_type,
                             ldc,
                             strideC,
                             m->num_q_heads,
                             compute_type,
                             HIPBLAS_GEMM_DEFAULT,
                             stream,
                             1,
                             m->num_q_heads / m->num_kv_heads,
                             1);
      if (m->inference_debugging) {
        std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".qk_prods";
        save_tensor(static_cast<DT const *>(m->qk_prods),
                    num_new_tokens * total_tokens * m->num_q_heads,
                    fpath.c_str());
      }
    }
    // Step 2: Add alibi position bias to qk production
    // matrix C: qk_prods
    // matrix C's layout: [num_new_tokens, total_tokens, num_heads]
    // To get C, skip over QK.T products from previous requests
    {
      // matrix C: qk_prods (current req only)
      // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
      DT *C = static_cast<DT *>(m->qk_prods);
      if (*m->position_bias) {
        size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_position_bias_qkprd),
                           GET_BLOCKS(parallelism),
                           min((size_t)CUDA_NUM_THREADS, parallelism),
                           0,
                           stream,
                           C,
                           num_new_tokens,
                           total_tokens,
                           m->num_q_heads,
                           m->global_num_q_heads,
                           shard_id);
      }
    }

    // Step 3: Apply causal mask. Fill all elements above diagonal in qk prods
    // with -inf to force causal attention.
    {
      assert(num_new_tokens <= total_tokens);
      size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
      if (entries_above_diagonal > 0) {
        // matrix C: qk_prods (current req only)
        // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
        DT *C = static_cast<DT *>(m->qk_prods);
        size_t parallelism = m->num_q_heads * entries_above_diagonal;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(fill_entries_above_diagonal),
                           GET_BLOCKS(parallelism),
                           min((size_t)CUDA_NUM_THREADS, parallelism),
                           0,
                           stream,
                           C,
                           num_new_tokens,
                           total_tokens,
                           m->num_q_heads,
                           entries_above_diagonal,
                           static_cast<DT>(-INFINITY));
      }
      if (m->inference_debugging) {
        std::string fpath =
            get_fwd_dbg_folder(m, shard_id) + ".qk_prods.masked";
        save_tensor(static_cast<DT const *>(m->qk_prods),
                    num_new_tokens * total_tokens * m->num_q_heads,
                    fpath.c_str());
      }
    }

    // Step 4: Compute Softmax(QK.T/sqrt(d_k))
    {
      // Before modifying the parameters below, make sure to read the following
      // description of the HIPDNN_TENSOR_NCHW tensor layout, from
      // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#hipdnnTensorFormat_t:
      // This tensor format specifies that the data is laid out in the following
      // order: batch size, feature maps, rows, columns. The strides are
      // implicitly defined in such a way that the data are contiguous in memory
      // with no padding between images, feature maps, rows, and columns; the
      // columns are the inner dimension and the images are the outermost
      // dimension.
      int n_param = m->num_q_heads;
      int c_param = total_tokens;
      int h_param = 1;
      int w_param = num_new_tokens;
      checkCUDNN(miopenSet4dTensorDescriptor(
          m->qk_tensor, cudnn_data_type, n_param, c_param, h_param, w_param));
      float softmax_alpha = 1.0f, softmax_beta = 0.0f;
      // matrix C: qk_prods (current req only)
      // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
      DT *C = static_cast<DT *>(m->qk_prods);
      // matrix C_softmax: qk_prods_softmax (current req only)
      // matrix C_softmax's layout: [num_new_tokens, total_tokens, num_q_heads]
      DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
      // The softmax operation below is executed according to the
      // MIOPEN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
      // softmax operation is computed per spatial location (H,W) per image (N)
      // across dimension C.
      checkCUDNN(miopenSoftmaxForward_V2(m->handle.dnn,
                                         &softmax_alpha,
                                         m->qk_tensor,
                                         C,
                                         &softmax_beta,
                                         m->qk_tensor,
                                         C_softmax,
                                         MIOPEN_SOFTMAX_ACCURATE,
                                         MIOPEN_SOFTMAX_MODE_CHANNEL));
      // Copy C_softmax to m->softmax_activation_buffer if we need to compute
      // PEFT backward
      if (bc->requestsInfo[req_idx].finetuning_request) {
        int max_peft_tokens = BatchConfig::max_sequence_length();
        int max_dataset_entry_size = bc->requestsInfo[req_idx].max_length;
        size_t activation_size_needed =
            sizeof(DT) * max_peft_tokens * max_peft_tokens * m->num_q_heads;
        assert(activation_size_needed == m->allocated_peft_buffer_size2);
        int parallelism = m->num_q_heads * total_tokens * num_new_tokens;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(store_softmax_activation),
                           GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream,
                           static_cast<DT *>(m->qk_prods_softmax),
                           static_cast<DT *>(m->softmax_activation_buffer),
                           num_new_tokens,
                           total_tokens,
                           max_dataset_entry_size,
                           m->num_q_heads);
      }
      if (m->inference_debugging) {
        std::string fpath =
            get_fwd_dbg_folder(m, shard_id) + ".qk_prods_softmax";
        save_tensor(static_cast<DT const *>(m->qk_prods_softmax),
                    num_new_tokens * total_tokens * m->num_q_heads,
                    fpath.c_str());
      }
    }

    // Step 5: Matmul softmax(QK.T/sqrt(d_k)) by V. Implemented as V @
    // softmax(QK.T/sqrt(d_k)).T
    {
      DT alpha = 1.0f, beta = 0.0f;
      // after transpositions
      int m_ = m->vProjSize;
      int n = num_new_tokens;
      int k = total_tokens;
      // before transpositions
      int lda = m_ * m->num_kv_heads;
      int ldb = n;
      int ldc = m_ * m->num_q_heads;
      // N.B. strides are applied before transpose operations
      int strideA = m->vProjSize;
      int strideB = num_new_tokens * total_tokens;
      int strideC = m->vProjSize;
      // matrix A: value cache
      // matrix A's layout: [vProjSize, num_kv_heads, total_tokens]
      // To get A, skip over V.T entries from previous requests (all heads +
      // padding)
      DT *A = static_cast<DT *>(m->valueCache) +
              req_idx * (m->vProjSize * m->num_kv_heads *
                         BatchConfig::max_sequence_length());
      // matrix B: qk_prods_softmax (current req only)
      // matrix B's layout: [num_new_tokens, total_tokens, num_q_heads]
      // To get B, skip over softmax(QK.T/sqrt(d_k)) entries from previous
      // requests (all heads)
      DT *B = static_cast<DT *>(m->qk_prods_softmax);
      // matrix C: attn heads
      // matrix C's layout: [vProjSize, num_q_heads, num_new_tokens]
      // To get C, skip over softmax(QK.T/sqrt(d_k))V products from previous
      // requests
      // store the result attn heads, also skip the genration tokens
      DT *C = static_cast<DT *>(attn_heads) +
              (bc->requestsInfo[req_idx].first_token_offset_in_batch) *
                  m->num_q_heads * m->vProjSize;
      run_batched_matmul<DT>(m,
                             m->handle.blas,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_T,
                             m_,
                             n,
                             k,
                             &alpha,
                             A,
                             cublas_data_type,
                             lda,
                             strideA,
                             B,
                             cublas_data_type,
                             ldb,
                             strideB,
                             &beta,
                             C,
                             cublas_data_type,
                             ldc,
                             strideC,
                             m->num_q_heads,
                             compute_type,
                             HIPBLAS_GEMM_DEFAULT,
                             stream,
                             m->num_q_heads / m->num_kv_heads,
                             1,
                             1);
      if (m->inference_debugging) {
        std::string fpath =
            get_fwd_dbg_folder(m, shard_id) + ".qk_prods_softmax";
        save_tensor(static_cast<DT const *>(attn_heads),
                    num_new_tokens * m->num_q_heads * m->vProjSize,
                    fpath.c_str());
      }
    }
    num_processed_prompt_tokens += num_new_tokens;
  }
  if (num_processed_prompt_tokens !=
      (bc->num_active_tokens() - bc->num_generation_tokens)) {
    bc->print();
    printf("num_processed_prompt_tokens: %i\n", num_processed_prompt_tokens);
    printf("num_tokens: %i\n", bc->num_active_tokens());
    printf("bc->num_generation_tokens: %i\n", bc->num_generation_tokens);
  }
  assert(num_processed_prompt_tokens ==
         (bc->num_active_tokens() - bc->num_generation_tokens));
}

// gridDim = num_heads
// blockDim = num_tokens/num_request * head_size
// QKV tensor layout: |QKV| * num_new_tokens. |Q=K=V=head_size * num_heads|
// one thread process one head_size
template <typename DT,
          int THREADS_PER_BLOCK,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE>
__global__ void compute_attention_kernel_generation_kernel(
    DT const *query,
    DT const *key_cache,
    DT const *value_cache,
    DT *output_ptr,
    float const scale,
    int max_seq_length,
    int per_head_size,
    int num_q_heads,
    int num_kv_heads,
    BatchConfig::PerRequestInfo *request_infos) {

  int total_num_heads = num_q_heads + 2 * num_kv_heads;

  // q, k
  using Q_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using K_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using V_vec = typename VEC_V<DT>::Type;
  using Out_sum = typename Vec_fp32_<V_vec>::Type;

  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  // eg.  if head_size = 128, thread_per_key = 4, with float32 precision
  // then K_VEC_SIZE = 1,  QK_VEC_SIZE = 4
  //  K_ELTS_PER_THREAD = 128 / 4 = 32
  //  K_VECS_PER_THREAD = 32 / 1 = 32
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(DT);
  // constexpr int QK_VEC_SIZE = 16 / sizeof(DT);
  // // constexpr int QK_VEC_SIZE = sizeof(Qk_vec_k) / sizeof(DT);
  constexpr int K_ELTS_PER_THREAD = Dh / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;
  // constexpr int QK_ELTS_IN_16B = 16 / sizeof(DT);

  // thread id
  int const tidx = threadIdx.x;
  // head id
  int const head_idx = blockIdx.x;
  int const kv_head_idx = head_idx / (num_q_heads / num_kv_heads);
  // request idx
  int const request_idx = blockIdx.y;

  int const batch_config_request_id =
      request_infos[request_idx].batch_config_request_id;

  int const first_step = 0;

  int const tlength =
      request_infos[batch_config_request_id].first_token_depth_in_request +
      request_infos[batch_config_request_id].num_tokens_in_batch;

  // shared memory objects
  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *out_smem = reinterpret_cast<float *>(smem_);

  float qk_max = -FLT_MAX;

  // first WARPS_PER_BLOCK for store qk_max, second WARPS_PER_BLOCK for sum
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  const DT *q_ptr = query + request_idx * per_head_size * total_num_heads +
                    head_idx * per_head_size;
  __shared__ Q_vec q_vecs[THREADS_PER_KEY][K_VECS_PER_THREAD];
  // DT const *q_ptr =
  //     query + request_idx * Dh * QKV_WEIGHT_NUM + head_idx * per_head_size;

  // q tensor in this thread
  // if THREADS_PER_KEY is 4, first thread load 0, 4, 8, 12..., total
  // K_VECS_PER_THREAD elements
  // QK_vec_k: 32->1, 64->2, 128->4... head_size
  // K_vec_k: 4->1, 2->2, 1->4 threads_per_key

  // the start offset of the element eg. (0, 1, 2, 3) * K_VEC_SIZE
  int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;
  int ki_o = tidx % THREADS_PER_KEY;
  // the first key's offset for this thread
  // ko = 0, 0, 0, 0, 1, 1, 1, 1, ....
  int ko = tidx / THREADS_PER_KEY;
  // load q tensor
  Q_vec q_vec[K_VECS_PER_THREAD];
#pragma unroll
  for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
    q_vecs[ki_o][ii] = *reinterpret_cast<Q_vec const *>(
        q_ptr + ki + ii * THREADS_PER_KEY * K_VEC_SIZE);
  }
  __syncthreads();
  // first iter = 128 / 4 = 32
  // K_VECS_PER_THREAD = 32
  //  K_PER_ITER how many keys in this loop
  //  The number of timesteps loaded per iteration.
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  //   // The number of keys per warp.
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  DT const *k_cache_batch = key_cache +
                            batch_config_request_id *
                                (per_head_size * num_kv_heads) *
                                max_seq_length +
                            ki;

  int ti_end =
      div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;
  // get k, perform qk proj

  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    K_vec k[K_VECS_PER_THREAD];
    int const ti_circ = ti % max_seq_length;
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * THREADS_PER_KEY * K_VEC_SIZE;
      if (ti < tlength) {
        k[ii] = *reinterpret_cast<K_vec const *>(
            k_cache_batch + ti_circ * (per_head_size * num_kv_heads) +
            kv_head_idx * per_head_size + jj);
      }
      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
    }
    float qk = scale * Qk_dot<DT, THREADS_PER_KEY>::dot(q_vecs[ki_o], k);
    // // todo add positional embedding to the qk production
    // // Store the product to shared memory. There's one qk value per
    // timestep.
    // // Update the max.
    if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
      // todo add alobi here
      bool const mask = ti_circ >= tlength;
      if (mask) {
        assert(false);
      }
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      qk_smem[ti - first_step] = mask ? 0.f : qk;
    }
  }

  __syncthreads();

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, WARP_SHFL_XOR(uint32_t(-1), qk_max, mask));
  }

  // Decompose the thread index into warp and lane.
  int const warp = tidx / WARP_SIZE;
  int const lane = tidx % WARP_SIZE;

  // The warp leader writes the max to shared memory.
  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  // Make sure the products are in shared memory.
  __syncthreads();

  // The warps finalize the reduction.
  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, WARP_SHFL_XOR(uint32_t(-1), qk_max, mask));
  }

  // Broadcast to all the threads in the warp.
  qk_max = WARP_SHFL(uint32_t(-1), qk_max, 0);

  float exp_sum = 0.f;
  for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti - first_step] - qk_max);
    exp_sum += logit;
    qk_smem[ti - first_step] = logit;
  }

  // Compute the sum.
  exp_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], exp_sum);

  // softmax
  float inv_sum = __fdividef(1.f, exp_sum + 1.e-6);
  for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
    qk_smem[ti - first_step] *= inv_sum;
  }

  __syncthreads();
  // if (blockIdx.y == 0 && blockIdx.x == 0 && tidx == 0) {
  //   printf("softmax %.10f\n", qk_smem[0]);
  // }

  // value projection
  constexpr int V_VEC_SIZE = 16 / sizeof(DT);
  // A vector of V elements for the current timestep.
  // using V_vec_k = typename V_vec_k_<DT, V_VEC_SIZE>::Type;
  // using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;

  // The value computed by this thread.
  int vo = tidx / THREADS_PER_VALUE;
  // The hidden dimensions computed by this particular thread.
  int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

  Out_sum out;
  zero(out);

  // The base pointer for the value in the cache buffer.
  DT const *v_cache_batch = value_cache +
                            batch_config_request_id * max_seq_length *
                                (per_head_size * num_kv_heads) +
                            vi;

  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = first_step + vo; ti < tlength; ti += V_PER_ITER) {
      // Load the values from the cache.
      int const ti_circ = ti % max_seq_length;

      V_vec v = *reinterpret_cast<V_vec const *>(
          v_cache_batch + ti_circ * (per_head_size * num_kv_heads) +
          kv_head_idx * per_head_size);
      float logit = qk_smem[ti - first_step];
      out = FlexFlow::fma(logit, cast_to_float(v), out);
    }
  }

  //   // Make sure we can start writing to shared memory.
  __syncthreads();

  // Run the final reduction amongst the different groups computing different
  // partial outputs.
  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {

      // The midpoint in the number of active groups.
      int midpoint = active_groups / 2;

      // The upper part of active threads store to shared memory.
      if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
        *reinterpret_cast<Out_sum *>(out_smem + (vo - midpoint) * Dh + vi) =
            out;
      }
      __syncthreads();

      // The bottom warps update their values.
      if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out = add(*reinterpret_cast<Out_sum const *>(out_smem + vo * Dh + vi),
                  out);
      }
      __syncthreads();
    }
  }

  // Output the final values.
  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
    convert_from_float(
        *reinterpret_cast<V_vec *>(output_ptr +
                                   request_idx * (per_head_size * num_q_heads) +
                                   head_idx * per_head_size + vi),
        out);
  }
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
                               hipFloatComplex *complex_input,
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
      float pi = HIP_PI_F;
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

    hipFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = hipCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_bwd(DT *input_ptr,
                               hipFloatComplex *complex_input,
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
  // int tot_num_heads = num_q_heads + 2 * num_kv_heads;
  CUDA_KERNEL_LOOP(i, q_proj_work + kv_proj_work) {
    // compute indexes to visit first half proj_size of each of q/k tensor.
    // devQKVProj has shape [num_tokens, proj_size, tot_num_heads] in peft_bwd
    bool q_tensor = i < q_proj_work;
    int num_heads = q_tensor ? num_q_heads : num_kv_heads;
    int real_i = q_tensor ? i : i - q_proj_work;

    int token_idx = real_i % num_tokens;
    int pair_idx = (real_i / num_tokens) % half_proj;
    int head_idx = real_i / (num_tokens * half_proj);
    assert(head_idx < num_heads);

    int complex_part_index =
        (q_tensor ? 0 : num_tokens * proj_size * num_q_heads) +
        head_idx * proj_size * num_tokens + pair_idx * num_tokens + token_idx;
    int real_part_index = complex_part_index + num_tokens * half_proj;

    complex_input[i] = {(float)input_ptr[real_part_index],
                        (float)input_ptr[complex_part_index]};

    float inv_freq =
        1.0 / pow(rope_theta, (float)2.0 * pair_idx / proj_size); // θ_i

    if (llama3_rope) {
      float pi = HIP_PI_F;
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

    hipFloatComplex complex_pos = {cos(inv_freq), sin(inv_freq)};

    complex_input[i] = hipCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = (DT)complex_input[i].x;
    input_ptr[complex_part_index] = (DT)complex_input[i].y;
  }
}

template <typename DT>
void apply_scaling_and_rotary(IncMultiHeadSelfAttentionMeta const *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              hipStream_t stream) {

  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);

  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;

  if (m->scaling_query) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scaling_query_kernel),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
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
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(apply_rotary_embedding_fwd),
        GET_BLOCKS(parallelism),
        min(CUDA_NUM_THREADS, parallelism),
        0,
        stream,
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

template <typename DT>
void update_kv_cache_kernel(IncMultiHeadSelfAttentionMeta const *m,
                            BatchConfig const *bc,
                            hipStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  int tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  int head_dim = m->qProjSize;
  if (num_tokens > 0) {
    int parallelism = head_dim * tot_num_heads * num_tokens;
    // devQKVProj has shape [qProjSize, tot_num_heads, num_new_tokens]
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_kv_cache),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       static_cast<DT *>(m->devQKVProjArray),
                       static_cast<DT *>(m->keyCache),
                       static_cast<DT *>(m->valueCache),
                       m->token_infos,
                       num_tokens,
                       BatchConfig::max_sequence_length(),
                       head_dim,
                       m->num_q_heads,
                       m->num_kv_heads);
  }
}

#define LAUNCH_ATTENTION_SCORE_KERNEL(                                         \
    DT, Dh, Dh_MAX, THDS_PER_KEY, THREADS_PER_VALUE, THDS_PER_BLOCK, stream)   \
  smem_sz = smem_size_in_bytes<DT>(m->qProjSize,                               \
                                   BatchConfig::max_sequence_length(),         \
                                   THREADS_PER_VALUE,                          \
                                   THDS_PER_BLOCK);                            \
  compute_attention_kernel_generation_kernel<DT,                               \
                                             THDS_PER_BLOCK,                   \
                                             Dh,                               \
                                             Dh_MAX,                           \
                                             THDS_PER_KEY,                     \
                                             THREADS_PER_VALUE>                \
      <<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                             \
          static_cast<DT *>(m->devQKVProjArray),                               \
          static_cast<DT *>(m->keyCache),                                      \
          static_cast<DT *>(m->valueCache),                                    \
          output_ptr,                                                          \
          scale,                                                               \
          BatchConfig::max_sequence_length(),                                  \
          m->qProjSize,                                                        \
          m->num_q_heads,                                                      \
          m->num_kv_heads,                                                     \
          m->request_infos)

template <typename DT>
void compute_attention_kernel_generation(IncMultiHeadSelfAttentionMeta const *m,
                                         BatchConfig const *bc,
                                         DT *output_ptr,
                                         hipStream_t stream) {
  dim3 grid(m->num_q_heads, bc->num_generation_tokens);
  int const per_head_size = m->qProjSize;
  float scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  size_t smem_sz;
  if (per_head_size == 64) {
    constexpr int THREADS_PER_VALUE_64 = threads_per_value_t<DT, 64>::value;
    LAUNCH_ATTENTION_SCORE_KERNEL(
        DT, 64, 64, 4, THREADS_PER_VALUE_64, 128, stream);
  } else if (per_head_size == 128) {
    constexpr int THREADS_PER_VALUE_128 = threads_per_value_t<DT, 128>::value;
    LAUNCH_ATTENTION_SCORE_KERNEL(
        DT, 128, 128, 4, THREADS_PER_VALUE_128, 128, stream);
  } else {
    assert(false && "a unsupported head size");
  }
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *qkv_ptr,
                      DT *output_ptr,
                      hipStream_t stream) {

  // phase 0: copy calculated qkv into devQKVProjArray
  // [qProjSize, tot_num_heads, num_new_tokens]
  assert(m->qProjSize == m->kProjSize && m->qProjSize == m->vProjSize);
  size_t tot_num_heads = m->num_q_heads + 2 * m->num_kv_heads;
  size_t qkv_proj_size = m->qProjSize * tot_num_heads * bc->num_active_tokens();

  checkCUDA(hipMemcpyAsync(m->devQKVProjArray,
                           qkv_ptr,
                           qkv_proj_size * sizeof(DT),
                           hipMemcpyDeviceToDevice,
                           stream));

  if (m->inference_debugging) {
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".devQKVProjArray";
    save_tensor(static_cast<DT const *>(m->devQKVProjArray),
                qkv_proj_size,
                fpath.c_str());
  }

  // phase 1: Implement kernel to apply rotary embedding and scaling
  apply_scaling_and_rotary(
      m, bc, shard_id, static_cast<DT *>(m->devQKVProjArray), stream);

  if (m->inference_debugging) {
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".post_rope";
    save_tensor(static_cast<DT const *>(m->devQKVProjArray),
                qkv_proj_size,
                fpath.c_str());
  }

  update_kv_cache_kernel<DT>(m, bc, stream);

  if (m->inference_debugging) {
    size_t key_cache_size = m->kProjSize * m->num_kv_heads *
                            BatchConfig::max_sequence_length() *
                            BatchConfig::max_requests_per_batch();
    std::string fpath = get_fwd_dbg_folder(m, shard_id) + ".key_cache";
    save_tensor(
        static_cast<DT const *>(m->keyCache), key_cache_size, fpath.c_str());
    fpath = get_fwd_dbg_folder(m, shard_id) + ".value_cache";
    save_tensor(
        static_cast<DT const *>(m->valueCache), key_cache_size, fpath.c_str());
  }

  if (bc->num_generation_tokens > 0) {
    // phase 3: Compute attention score for generation tokens
    compute_attention_kernel_generation<DT>(m, bc, output_ptr, stream);
  }

  if (bc->num_tokens > bc->num_generation_tokens) {
    // phase 4: Compute attention score for prompt tokens;
    compute_attention_kernel_prompt<DT>(m, bc, output_ptr, shard_id, stream);
  }

  if (bc->num_finetuning_fwd_tokens() > 0) {
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

__global__ void transposeAdd_half_kernel(
    half *out, half const *in, int width, int height, half alpha, half beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

__global__ void transposeAdd_float_kernel(float *out,
                                          float const *in,
                                          int width,
                                          int height,
                                          float alpha,
                                          float beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

template <typename DT>
void transposeAdd(DT *out,
                  const DT *in,
                  int width,
                  int height,
                  float alpha,
                  float beta,
                  hipStream_t stream) {
  assert(false && "Unsupported data type");
}

template <>
void transposeAdd<float>(float *out,
                         float const *in,
                         int width,
                         int height,
                         float alpha,
                         float beta,
                         hipStream_t stream) {
  transposeAdd_float_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, alpha, beta);
}

template <>
void transposeAdd<__ff_bfloat16>(__ff_bfloat16 *out,
                                __ff_bfloat16 const *in,
                                int width,
                                int height,
                                float alpha,
                                float beta,
                                hipStream_t stream) {
  transposeAdd_float_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, __float2bfloat16(alpha), __float2bfloat16(beta));
}

template <>
void transposeAdd<half>(half *out,
                        half const *in,
                        int width,
                        int height,
                        float alpha,
                        float beta,
                        hipStream_t stream) {
  transposeAdd_half_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, __float2half(alpha), __float2half(beta));
}

template <typename DT>
void peft_bwd_kernel(IncMultiHeadSelfAttentionMeta const *m,
                     BatchConfig const *bc,
                     int shard_id,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     hipStream_t stream) {
  assert(!m->offload);
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  hipblasDatatype_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  miopenDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  hipblasDatatype_t compute_type = cublas_data_type;

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

  // Step 1: copy gradient before final projection into workspace
  {
    int m_ = m->vProjSize * m->num_q_heads;
    int n_ = num_tokens;
    DT *C = static_cast<DT *>(m->handle.workSpace);
    checkCUDA(hipMemcpyAsync(
        C,
        output_grad_ptr +
            bc->requestsInfo[i].first_token_offset_in_batch * m->oProjSize,
        m_ * n_ * sizeof(DT),
        hipMemcpyDeviceToDevice,
        stream));
    if (m->inference_debugging) {
      // save result to file for checking
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".o_proj.input_gradient_0";
      save_tensor(C, m_ * n_, filename.c_str());
    }
  }
  // Step 2: compute gradients w.r.t. value
  {
    float alpha = 1.0f, beta = 0.0f;
    // matrix A: qk_prods_softmax
    // matrix A's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->qk_prods_softmax);
    // matrix B: attn_heads gradients
    // matrix B's layout: [vProjSize * num_q_heads, num_new_tokens]
    DT const *B = static_cast<DT *>(m->handle.workSpace);
    // matrix C: gradients for value (saved as part of m->devQKVProjArray)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    // note that we first need to compute the gradients wrt each q_heads, then
    // we can sum the gradients corresponding to each group of q_heads to obtain
    // the gradients wrt each value head
    DT *C = static_cast<DT *>(m->devQKVProjArray) +
            2 * num_tokens *
                (m->qProjSize * m->num_q_heads); // skip over regions reserved
                                                 // for Q and K gradients
    // after transpositions
    int m_ = num_tokens;   // total_tokens
    int n_ = m->vProjSize; // num_new_tokens
    int k_ = num_tokens;   // num_new_tokens
    // before transpositions
    int lda = num_tokens; // num_new_tokens
    int ldb = m->vProjSize * m->num_q_heads;
    int ldc = num_tokens; // total_tokens
    // N.B. strides are applied before transpose operations
    int strideA = num_tokens * num_tokens; // num_new_tokens * total_tokens
    int strideB = m->vProjSize;
    int strideC = num_tokens * m->vProjSize;
    checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                          HIPBLAS_OP_T,
                                          HIPBLAS_OP_T,
                                          m_,
                                          n_,
                                          k_,
                                          &alpha,
                                          A,
                                          cublas_data_type,
                                          lda,
                                          strideA,
                                          B,
                                          cublas_data_type,
                                          ldb,
                                          strideB,
                                          &beta,
                                          C,
                                          cublas_data_type,
                                          ldc,
                                          strideC,
                                          m->num_q_heads,
                                          compute_type,
                                          HIPBLAS_GEMM_DEFAULT));
    // save result to file for checking
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".v_proj.input_gradient_0";
      save_tensor(C, m_ * n_ * m->num_q_heads, filename.c_str());
      std::string filename2 =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax";
      save_tensor(A, m_ * k_ * m->num_q_heads, filename2.c_str());
    }
  }
  // Step 3: compute gradients w.r.t. the qk_prods_softmax tensor
  {
    float alpha = 1.0f, beta = 0.0f;
    // matrix A: attn_heads gradients
    // matrix A's layout: [vProjSize * num_q_heads, num_new_tokens]
    DT const *A = static_cast<DT *>(m->handle.workSpace);
    // matrix B: value cache
    // matrix B's layout: [vProjSize * num_kv_heads, max_num_tokens, num_req]
    DT const *B =
        static_cast<DT *>(m->valueCache) +
        i * m->vProjSize * m->num_kv_heads * BatchConfig::max_sequence_length();
    // matrix C: qk_prods_softmax gradients
    // matrix C's layout: [num_new_tokens, total_tokens, num_q_heads]
    DT *C = static_cast<DT *>(m->qk_prods_softmax);
    // after transposition & striding
    int m_ = num_tokens; // num_new_tokens
    int n_ = num_tokens;
    int k_ = m->vProjSize;
    // before transposition and striding
    int lda = m->vProjSize * m->num_q_heads;
    int ldb = m->vProjSize * m->num_kv_heads;
    int ldc = num_tokens; // num_new_tokens
    int strideA = m->vProjSize;
    int strideB = m->vProjSize;
    int strideC = num_tokens * num_tokens; // num_new_tokens * total_tokens

    run_batched_matmul<DT>(m,
                           m->handle.blas,
                           HIPBLAS_OP_T,
                           HIPBLAS_OP_N,
                           m_,
                           n_,
                           k_,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           HIPBLAS_GEMM_DEFAULT,
                           stream,
                           1,
                           m->num_q_heads / m->num_kv_heads,
                           1,
                           true);
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      std::string filename2 = get_peft_dbg_folder(m, shard_id) + ".vcache";
      save_tensor(B,
                  m->vProjSize * m->num_kv_heads *
                      BatchConfig::max_sequence_length(),
                  filename2.c_str());
    }
  }
  // Step 4: softmax backpropagation
  {
    float alpha = 1.0f, beta = 0.0f;
    int n_param = m->num_q_heads;
    int c_param = num_tokens;
    int h_param = 1;
    int w_param = num_tokens;
    checkCUDNN(miopenSet4dTensorDescriptor(
        m->qk_tensor, cudnn_data_type, n_param, c_param, h_param, w_param));
    checkCUDNN(miopenSoftmaxBackward_V2(m->handle.dnn,
                                        &alpha,
                                        m->qk_tensor,
                                        m->softmax_activation_buffer,
                                        m->qk_tensor,
                                        m->qk_prods_softmax,
                                        &beta,
                                        m->qk_tensor,
                                        m->qk_prods,
                                        MIOPEN_SOFTMAX_ACCURATE,
                                        MIOPEN_SOFTMAX_MODE_CHANNEL));

    if (m->inference_debugging) {
      DT *C = static_cast<DT *>(m->qk_prods);
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad_in";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      filename =
          get_peft_dbg_folder(m, shard_id) + ".softmax_activation_buffer";
      save_tensor(static_cast<DT *>(m->softmax_activation_buffer),
                  num_tokens * num_tokens * m->num_q_heads,
                  filename.c_str());
    }

    //  TODO: fill all elements above diagonal to force causal attention
    size_t entries_above_diagonal = num_tokens * (num_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(fill_entries_above_diagonal),
                         GET_BLOCKS(parallelism),
                         min((size_t)CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         static_cast<DT *>(m->qk_prods),
                         num_tokens,
                         num_tokens,
                         m->num_q_heads,
                         entries_above_diagonal,
                         DT(0.0f));
    }
    if (m->inference_debugging) {
      DT *C = static_cast<DT *>(m->qk_prods);
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad_in.masked";
      save_tensor(
          C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
    }
  }
  // Step 5: compute gradients w.r.t. key
  {
    float alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = 1.0f / sqrt(m->kProjSize);
    }
    // matrix A: gradients w.r.t. qk_prods
    // matrix A's layout: [num_new_tokens, num_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->qk_prods);
    // matrix B: query activation (in query_activation_buffer)
    // matrix B's layout: [m->qProjSize * num_q_heads, num_new_tokens]
    DT const *B = static_cast<DT *>(m->query_activation_buffer);
    // matrix C: gradients for key (saved as part of m->devQKVProjArray)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    // note that we first need to compute the gradients wrt each q_heads, then
    // we can sum the gradients corresponding to each group of q_heads to obtain
    // the gradients wrt each key head
    DT *C = static_cast<DT *>(m->devQKVProjArray) +
            num_tokens *
                (m->qProjSize *
                 m->num_q_heads); // skip over regions reserved for Q gradients
    // after transposition & striding
    int m_ = num_tokens;
    int n_ = m->kProjSize;
    int k_ = num_tokens; // num_new_tokens
    // before transposition and striding
    int lda = num_tokens; // num_new_tokens
    int ldb = m->kProjSize * m->num_q_heads;
    int ldc = num_tokens;
    int strideA = num_tokens * num_tokens;
    int strideB = m->kProjSize;
    int strideC = num_tokens * m->kProjSize;
    checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                          HIPBLAS_OP_T,
                                          HIPBLAS_OP_T,
                                          m_,
                                          n_,
                                          k_,
                                          &alpha,
                                          A,
                                          cublas_data_type,
                                          lda,
                                          strideA,
                                          B,
                                          cublas_data_type,
                                          ldb,
                                          strideB,
                                          &beta,
                                          C,
                                          cublas_data_type,
                                          ldc,
                                          strideC,
                                          m->num_q_heads,
                                          compute_type,
                                          HIPBLAS_GEMM_DEFAULT));
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".query_activation";
      save_tensor(
          B, m->qProjSize * m->num_q_heads * num_tokens, filename.c_str());
      std::string filename2 =
          get_peft_dbg_folder(m, shard_id) + ".devkproj_pre";
      save_tensor(
          C, num_tokens * (m->qProjSize * m->num_q_heads), filename2.c_str());
    }
  }
  // Step 6: compute gradients w.r.t query
  {
    float alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = 1.0f / sqrt(m->kProjSize);
    }
    // matrix A: gradients w.r.t. qk_prods
    // matrix A's layout: [num_new_tokens, num_tokens, num_q_heads]
    DT const *A = static_cast<DT *>(m->qk_prods);
    // matrix B: key cache
    // matrix B's layout: [vProjSize * num_kv_heads, max_num_tokens, num_req]
    DT const *B =
        static_cast<DT *>(m->keyCache) +
        i * m->kProjSize * m->num_kv_heads * BatchConfig::max_sequence_length();
    // matrix C: gradients for query (saved as part of m->devQKVProjArray)
    // matrix C's layout: [num_tokens, qProjsize * num_q_heads, 3]
    DT *C = static_cast<DT *>(m->devQKVProjArray);
    // after transposition & striding
    int m_ = num_tokens; // num_new_tokens
    int n_ = m->qProjSize;
    int k_ = num_tokens;
    // before transposition and striding
    int lda = num_tokens; // num_new_tokens
    int ldb = m->qProjSize * m->num_kv_heads;
    int ldc = num_tokens;
    int strideA = num_tokens * num_tokens;
    int strideB = m->qProjSize;
    int strideC = num_tokens * m->qProjSize;
    run_batched_matmul<DT>(m,
                           m->handle.blas,
                           HIPBLAS_OP_N,
                           HIPBLAS_OP_T,
                           m_,
                           n_,
                           k_,
                           &alpha,
                           A,
                           cublas_data_type,
                           lda,
                           strideA,
                           B,
                           cublas_data_type,
                           ldb,
                           strideB,
                           &beta,
                           C,
                           cublas_data_type,
                           ldc,
                           strideC,
                           m->num_q_heads,
                           compute_type,
                           HIPBLAS_GEMM_DEFAULT,
                           stream,
                           1,
                           m->num_q_heads / m->num_kv_heads,
                           1,
                           true);
    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray_pre";
      save_tensor(
          C, num_tokens * m->qProjSize * m->num_q_heads * 3, filename.c_str());
    }
  }

  // Step 7: perform rotary position embeddings (RoPE) bwd
  {
    if (m->rotary_embedding_meta->apply_rotary_embedding) {
      checkCUDA(hipMemcpyAsync(m->peft_token_infos_device,
                               m->peft_token_infos,
                               m->peft_token_infos_size,
                               hipMemcpyHostToDevice,
                               stream));

      assert(m->qProjSize == m->kProjSize);
      /*q&k*/
      int half_proj = m->qProjSize / 2;
      int q_proj_work = num_tokens * m->num_q_heads * half_proj;
      int kv_proj_work = num_tokens * m->num_kv_heads * half_proj;
      int parallelism = q_proj_work + kv_proj_work;
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(apply_rotary_embedding_bwd),
          GET_BLOCKS(parallelism),
          min(CUDA_NUM_THREADS, parallelism),
          0,
          stream,
          static_cast<DT *>(m->devQKVProjArray),
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
      DT *C = static_cast<DT *>(m->devQKVProjArray);
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray";
        save_tensor(C,
                    num_tokens * m->qProjSize * m->num_q_heads * 3,
                    filename.c_str());
      }
    }

    // matrix C: gradients for key (saved as part of m->devQKVProjArray)
    // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
    DT *C = static_cast<DT *>(m->devQKVProjArray) +
            num_tokens *
                (m->qProjSize *
                 m->num_q_heads); // skip over regions reserved for Q gradients
    if (m->inference_debugging) {
      std::string filename = get_peft_dbg_folder(m, shard_id) + ".devkproj";
      save_tensor(
          C, num_tokens * (m->qProjSize * m->num_q_heads), filename.c_str());
    }
  }

  // Step 8: compute gradients w.r.t. input
  {
    float alpha = 1.0f, beta = 0.0f;
    if (!m->reset_input_grads[0]) {
      beta = 1.0f;
    }
    // matrix B: gradients w.r.t. QKV (concatenated in devQKVArray)
    // matrix B's layout: [num_tokens, qProjsize * num_heads, 3]
    DT const *B = static_cast<DT *>(m->devQKVProjArray);
    // matrix C: gradients w.r.t. input
    // matrix C's layout: [m->qProjSize * m->num_q_heads, num_tokens]
    DT *C = input_grad_ptr + bc->requestsInfo[i].first_token_offset_in_batch *
                                 m->qProjSize * m->num_q_heads;
    // int m_ = m->qProjSize * m->num_q_heads;
    int n_ = num_tokens;
    int k_ = m->qProjSize * (m->num_q_heads + 2 * m->num_kv_heads);

    // The original version uses existing result and attention's projection to
    // do further calculation in a way different than the usual dense layer,
    // they are off by a transpose. So an explicit transpose is needed here.
    // The add here is just for gradient accumulation.
    transposeAdd(C, B, n_, k_, alpha, beta, stream);

    if (m->inference_debugging) {
      std::string filename =
          get_peft_dbg_folder(m, shard_id) + ".self_attn.input_gradient_0";
      save_tensor(
          C, num_tokens * m->qProjSize * m->num_q_heads, filename.c_str());
    }
  }
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  assert(input.data_type == output.data_type);

  if (input.data_type == DT_HALF) {
    Kernels::IncMultiHeadAttention::inference_kernel(
        m, bc, shard_id, input.get_half_ptr(), output.get_half_ptr(), stream);
  } else if (input.data_type == DT_FLOAT) {
    Kernels::IncMultiHeadAttention::inference_kernel(
        m, bc, shard_id, input.get_float_ptr(), output.get_float_ptr(), stream);
  } else {
    assert(false && "Unspported data type");
  }

  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);
  }
}

/*static*/
void IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &output_grad) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  assert(input_grad.data_type == output_grad.data_type);

  if (input_grad.data_type == DT_HALF) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_half_ptr(),
                                                    output_grad.get_half_ptr(),
                                                    stream);
  } else if (input_grad.data_type == DT_FLOAT) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_float_ptr(),
                                                    output_grad.get_float_ptr(),
                                                    stream);
  } else {
    assert(false && "Unspported data type");
  }
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(miopenSetStream(handler.dnn, stream));
  checkCUDNN(miopenCreateTensorDescriptor(&qk_tensor));
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

  rotary_embedding_meta =
      (RotaryEmbeddingMeta *)calloc(1, sizeof(RotaryEmbeddingMeta));
  *rotary_embedding_meta = _rotary_embedding_meta;
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

  assert(num_q_heads % num_kv_heads == 0 &&
         "num_q_heads must be divisible by num_kv_heads");
  if (num_q_heads > num_kv_heads) {
    // grouped query attention
    assert(attn->data_type == DT_FLOAT ||
           attn->data_type == DT_HALF && "Unsupported data type");
    gqa_ptr_array_size = num_q_heads * sizeof(void *);
  }

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = infer_mode == TREE_VERIFY_MODE
                                   ? BatchConfig::max_verify_tokens_per_batch()
                                   : BatchConfig::max_tokens_per_batch();
    size_t qkv_max_proj_size =
        max_tokens_per_batch *
        (qProjSize * num_q_heads + kProjSize * num_kv_heads +
         vProjSize * num_kv_heads);
    size_t query_tmp_size = 0, key_cache_size = 0, value_cache_size = 0;
    switch (infer_mode) {
      case INC_DECODING_MODE: {
        key_cache_size = num_kv_heads * kProjSize *
                         BatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length();
        value_cache_size = num_kv_heads * vProjSize *
                           BatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length();
        break;
      }
      case BEAM_SEARCH_MODE:
      case TREE_VERIFY_MODE: {
        // a K-ary tree max node is (k^n - 1) / 2
        key_cache_size = num_kv_heads * kProjSize *
                         BeamSearchBatchConfig::max_requests_per_batch() *
                         (BatchConfig::max_sequence_length() +
                          BatchConfig::max_spec_tree_token_num());
        value_cache_size = num_kv_heads * vProjSize *
                           BeamSearchBatchConfig::max_requests_per_batch() *
                           (BatchConfig::max_sequence_length() +
                            BatchConfig::max_spec_tree_token_num());
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    size_t requestinfo_size = BatchConfig::max_requests_per_batch();
    // size_t tokeninfo_size = max_tokens_per_batch;
    size_t qk_prod_size =
        max_tokens_per_batch * BatchConfig::max_sequence_length() * num_q_heads;
    size_t attn_heads_size = max_tokens_per_batch * num_q_heads * vProjSize;
    size_t complex_size = (max_tokens_per_batch * (qProjSize * num_q_heads +
                                                   kProjSize * num_kv_heads)) /
                          2;
    if (enable_peft_finetuning) {
      allocated_peft_buffer_size1 = BatchConfig::max_sequence_length() *
                                    num_q_heads * qProjSize * size_of_dt;
      allocated_peft_buffer_size2 = BatchConfig::max_sequence_length() *
                                    BatchConfig::max_sequence_length() *
                                    num_q_heads * size_of_dt;
      peft_token_infos = (BatchConfig::PerTokenInfo *)calloc(
          1,
          sizeof(BatchConfig::PerTokenInfo) *
              BatchConfig::max_sequence_length());
      peft_token_infos_size = sizeof(BatchConfig::PerTokenInfo) *
                              BatchConfig::max_sequence_length();
    } else {
      allocated_peft_buffer_size1 = 0;
      allocated_peft_buffer_size2 = 0;
      peft_token_infos = nullptr;
      peft_token_infos_size = 0;
    }
    size_t totalSize = (qkv_max_proj_size + query_tmp_size + key_cache_size +
                        value_cache_size + 2 * qk_prod_size + attn_heads_size) *
                           size_of_dt +
                       complex_size * sizeof(hipFloatComplex) +
                       3 * gqa_ptr_array_size;
    if (enable_peft_finetuning) {
      totalSize += allocated_peft_buffer_size1 + allocated_peft_buffer_size2;
      totalSize += peft_token_infos_size;
      totalSize += 3 * gqa_ptr_array_size;
    }
    if (offload) {
      // assert that we have enough reserved work space left
      size_t totalSharedSize =
          infer_mode == TREE_VERIFY_MODE
              ? totalSize - (query_tmp_size + key_cache_size +
                             value_cache_size + qkv_max_proj_size) *
                                size_of_dt
              : totalSize -
                    (query_tmp_size + key_cache_size + value_cache_size) *
                        size_of_dt;

      size_t inf_instance_size =
          size_of_dt *
          (infer_mode == TREE_VERIFY_MODE
               ? query_tmp_size + key_cache_size + value_cache_size +
                     qkv_max_proj_size
               : query_tmp_size + key_cache_size + value_cache_size);

      assert(inf_mem_allocator.reserved_total_size -
                 inf_mem_allocator.reserved_allocated_size >=
             totalSharedSize);
      inf_mem_allocator.create_legion_instance(
          inf_instance, inf_instance_size, "IncMultiHeadSelfAttentionMeta");
    } else {
      inf_mem_allocator.create_legion_instance(
          inf_instance, totalSize, "IncMultiHeadSelfAttentionMeta");
    }

    // in tree_verify, enable devQKVProjArray;
    if (!offload || infer_mode == TREE_VERIFY_MODE) {
      devQKVProjArray = inf_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size * size_of_dt);
    } else {
      devQKVProjArray = inf_mem_allocator.allocate_reserved_untyped(
          qkv_max_proj_size * size_of_dt);
      // offset += qkv_max_proj_size * size_of_dt;
    }

    // use key value cache in all mode.
    keyCache = inf_mem_allocator.allocate_instance_untyped(key_cache_size *
                                                           size_of_dt);
    valueCache = inf_mem_allocator.allocate_instance_untyped(value_cache_size *
                                                             size_of_dt);

    // gqa pointers
    if (num_q_heads > num_kv_heads) {
      assert(num_q_heads % num_kv_heads == 0 &&
             "Num Q heads must be a multiple of num KV heads");
      d_A_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_B_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      d_C_array = (void **)inf_mem_allocator.allocate_instance_untyped(
          gqa_ptr_array_size);
      if (enable_peft_finetuning) {
        d_A_array2 = (void **)inf_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
        d_B_array2 = (void **)inf_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
        d_C_array2 = (void **)inf_mem_allocator.allocate_instance_untyped(
            gqa_ptr_array_size);
      }
    }

    token_infos = static_cast<BatchConfig::PerTokenInfo *>(
        handler.batch_config_metadata->tokens_info);
    request_infos = static_cast<BatchConfig::PerRequestInfo *>(
        handler.batch_config_metadata->requestsInfo);

    if (offload) {
      qk_prods = inf_mem_allocator.allocate_reserved_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = inf_mem_allocator.allocate_reserved_untyped(
          qk_prod_size * size_of_dt);
      // attn_heads =
      // inf_mem_allocator.allocate_reserved_untyped(attn_heads_size *
      //                                                          size_of_dt);
      complex_input =
          inf_mem_allocator.allocate_reserved<hipFloatComplex>(complex_size);
    } else {
      qk_prods = inf_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = inf_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
      // attn_heads =
      // inf_mem_allocator.allocate_instance_untyped(attn_heads_size *
      //                                                          size_of_dt);
      complex_input =
          inf_mem_allocator.allocate_instance<hipFloatComplex>(complex_size);
    }

    if (enable_peft_finetuning) {
      query_activation_buffer = inf_mem_allocator.allocate_instance_untyped(
          allocated_peft_buffer_size1);
      softmax_activation_buffer = inf_mem_allocator.allocate_instance_untyped(
          allocated_peft_buffer_size2);
      peft_token_infos_device = (BatchConfig::PerTokenInfo *)
                                    inf_mem_allocator.allocate_instance_untyped(
                                        peft_token_infos_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
    }
    if (!offload) {
      assert(inf_mem_allocator.reserved_total_size ==
             inf_mem_allocator.reserved_allocated_size);
    }
  }

  checkCUDA(hipStreamSynchronize(stream));
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (inf_instance != Realm::RegionInstance::NO_INST) {
    inf_instance.destroy();
  }
}

template void Kernels::IncMultiHeadAttention::run_batched_matmul<half>(
    IncMultiHeadSelfAttentionMeta const *meta,
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    half const *A,
    hipblasDatatype_t Atype,
    int lda,
    long long int strideA,
    half const *B,
    hipblasDatatype_t Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    half *C,
    hipblasDatatype_t Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    hipblasDatatype_t computeType,
    hipblasGemmAlgo_t algo,
    hipStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::run_batched_matmul<float>(
    IncMultiHeadSelfAttentionMeta const *meta,
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    float const *A,
    hipblasDatatype_t Atype,
    int lda,
    long long int strideA,
    float const *B,
    hipblasDatatype_t Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    float *C,
    hipblasDatatype_t Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    hipblasDatatype_t computeType,
    hipblasGemmAlgo_t algo,
    hipStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void Kernels::IncMultiHeadAttention::run_batched_matmul<__ff_bfloat16>(
    IncMultiHeadSelfAttentionMeta const *meta,
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    void const *alpha,
    __ff_bfloat16 const *A,
    hipblasDatatype_t Atype,
    int lda,
    long long int strideA,
    __ff_bfloat16 const *B,
    hipblasDatatype_t Btype,
    int ldb,
    long long int strideB,
    void const *beta,
    __ff_bfloat16 *C,
    hipblasDatatype_t Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    hipblasDatatype_t computeType,
    hipblasGemmAlgo_t algo,
    hipStream_t stream,
    int batch_ratio_a,
    int batch_ratio_b,
    int batch_ratio_c,
    bool bwd);

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<float>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        float *output_ptr,
        hipStream_t stream);

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<__ff_bfloat16>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        __ff_bfloat16 *output_ptr,
        hipStream_t stream);

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        half *output_ptr,
        hipStream_t stream);

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<__ff_bfloat16>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        __ff_bfloat16 *output_ptr,
        hipStream_t stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float *output_ptr,
    hipStream_t stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half *output_ptr,
    hipStream_t stream);

template void Kernels::IncMultiHeadAttention::apply_scaling_and_rotary<__ff_bfloat16>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    __ff_bfloat16 *output_ptr,
    hipStream_t stream);

}; // namespace FlexFlow
