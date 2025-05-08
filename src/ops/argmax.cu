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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/argmax.h"
#include "flexflow/utils/cuda_helper.h"
#include <cub/block/block_reduce.cuh>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <cfloat>
#include <type_traits>

namespace FlexFlow {

// Data structure to hold a value-index pair.
struct ArgmaxPair {
  float value;
  int index;
};

// Functor that returns the pair with the larger value.
struct ArgmaxOp {
  __device__ __forceinline__ ArgmaxPair operator()(ArgmaxPair const &a,
                                                   ArgmaxPair const &b) const {
    return (a.value >= b.value) ? a : b;
  }
};

// Single inline helper function to convert a value of type T to float.
template <typename T>
__device__ __forceinline__ float toFloat(T x) {
  if constexpr (std::is_same<T, float>::value) {
    return x;
  } else if constexpr (std::is_same<T, __half>::value) {
    return __half2float(x);
  }
}

// Templated kernel that computes the argmax over the first dimension
// (vocab_size) for each column (i.e. for each batch element). The tensor is
// assumed to be stored in column-major order. The index of the maximum value is
// written to `output`, and if `prob_ptr` is not null, the maximum value (in
// float) is written there.
template <typename T, int BLOCK_SIZE>
__global__ void argmaxKernel(T const *__restrict__ input,
                             int vocab_size,
                             int batch_size,
                             int *__restrict__ output,
                             float *__restrict__ prob_ptr) {
  int col = blockIdx.x;
  if (col >= batch_size) {
    return; // safeguard
  }

  // Pointer to the start of the column.
  T const *col_ptr = input + col * vocab_size;

  // Each thread processes a subset of the column.
  float thread_max = -FLT_MAX;
  int thread_idx = 0;
  for (int i = threadIdx.x; i < vocab_size; i += BLOCK_SIZE) {
    float val = toFloat(col_ptr[i]);
    if (val > thread_max || isnan(val)) {
      thread_max = isnan(val) ? -FLT_MAX : val; // Handle NaN values
      thread_idx = i;
    }
  }

  // Prepare candidate for block reduction.
  ArgmaxPair thread_data;
  thread_data.value = thread_max;
  thread_data.index = thread_idx;

  // Use CUB's block reduction to compute the maximum and its index.
  typedef cub::BlockReduce<ArgmaxPair, BLOCK_SIZE> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  ArgmaxPair block_result =
      BlockReduceT(temp_storage).Reduce(thread_data, ArgmaxOp());

  // Thread 0 writes the results.
  if (threadIdx.x == 0) {
    output[col] = block_result.index;
    if (prob_ptr != nullptr) {
      prob_ptr[col] = block_result.value;
    }
  }
}

// Templated host wrapper for launching the kernel asynchronously on a given
// stream. Note: d_probs is always a float pointer.
template <typename T>
void launchArgmaxKernel(
    T const *d_input,
    int vocab_size,
    int batch_size,
    int *d_output,
    float *d_probs, // optional pointer for max values (always float)
    cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(CUDA_NUM_THREADS);
  argmaxKernel<T, CUDA_NUM_THREADS><<<grid, block, 0, stream>>>(
      d_input, vocab_size, batch_size, d_output, d_probs);
}

template <typename DT>
__global__ void compute_sparse_categorical_crossentropy_loss(
    DT const *logits,
    BatchConfig::TokenId const *labels,
    float *loss,
    int num_tokens,
    int num_classes) {
  float const LOG_MIN_VALUE = 0.00000001f;
  CUDA_KERNEL_LOOP(b, num_tokens) {
    float my_logit =
        max((float)logits[b * num_classes + labels[b]], LOG_MIN_VALUE);
    atomicAdd(loss, -log(my_logit));
  }
}

template <typename DT>
void inference_kernel_spatial_sharing(ArgMaxMeta const *m,
                                      BatchConfig const *bc,
                                      DT const *input_ptr,
                                      int *indices_ptr,
                                      int const num_classes,
                                      float *loss,
                                      cudaStream_t main_stream) {
  assert(!m->beam_search);
  assert(input_ptr != nullptr);
  assert(indices_ptr != nullptr);
  assert(loss != nullptr);
  assert(bc != nullptr);
  
  // launch finetuning fwd tokens kernel if there are any finetuning fwd tokens
  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaEventRecord(m->handle.peft_fwd_can_start, main_stream)); 
    checkCUDA(cudaStreamWaitEvent(m->handle.peft_fwd_stream, m->handle.peft_fwd_can_start, 0));

    // launchArgmaxKernel(input_ptr + num_classes * bc->num_inference_tokens(), 
    //                   num_classes, 
    //                   bc->num_finetuning_fwd_tokens(), 
    //                   indices_ptr + bc->num_inference_tokens(), 
    //                   nullptr, 
    //                   m->handle.peft_fwd_stream);

    // print_tensor(indices_ptr, batch_size, "indices_ptr: ");

    // compute cross-entropy loss if there is a finetuning request
    assert(loss != nullptr);
    BatchConfig::TokenId token_ids[BatchConfig::MAX_NUM_TOKENS];
    int i = bc->finetuning_request_index();
    assert(bc->requestsInfo[i].peft_model_id != PEFTModelID::NO_ID);
    assert(!bc->requestsInfo[i].finetuning_backward_phase);
    int num_finetuning_tokens = bc->requestsInfo[i].num_tokens_in_batch - 1;
    assert(num_finetuning_tokens + 1 == bc->num_finetuning_fwd_tokens());
    int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    for (int j = 0; j < num_finetuning_tokens; j++) {
      token_ids[j] = bc->tokensInfo[j + first_token_offset + 1].token_id;
    }
    checkCUDA(
        cudaMemcpyAsync(m->handle.workSpace,
                        token_ids,
                        sizeof(BatchConfig::TokenId) * num_finetuning_tokens,
                        cudaMemcpyHostToDevice,
                        m->handle.peft_fwd_stream));
    // copy loss to d_loss
    checkCUDA(cudaMemsetAsync(m->d_loss, 0, sizeof(float), m->handle.peft_fwd_stream));
    compute_sparse_categorical_crossentropy_loss<<<
        GET_BLOCKS(num_finetuning_tokens),
        min(CUDA_NUM_THREADS, num_finetuning_tokens),
        0,
        m->handle.peft_fwd_stream>>>(input_ptr + first_token_offset * num_classes,
                  static_cast<BatchConfig::TokenId *>(m->handle.workSpace),
                  m->d_loss,
                  num_finetuning_tokens,
                  num_classes);
    // copy value from d_loss to loss
    checkCUDA(cudaMemcpyAsync(
        loss, m->d_loss, sizeof(float), cudaMemcpyDeviceToHost, m->handle.peft_fwd_stream));
    *loss = *loss / (float)num_finetuning_tokens;

    checkCUDA(cudaEventRecord(m->handle.peft_fwd_done, m->handle.peft_fwd_stream));
  }

  // launch inference kernel if there are inference tokens
  if (bc->num_inference_tokens() > 0) {
    launchArgmaxKernel(
      input_ptr, num_classes, bc->num_inference_tokens(), indices_ptr, nullptr, main_stream);
  }

  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaStreamWaitEvent(main_stream, m->handle.peft_fwd_done, 0));
  }
}

/*static*/
template <typename DT>
void ArgMax::inference_kernel(ArgMaxMeta const *m,
                            BatchConfig const *bc,
                            DT const *input_ptr,
                            int *indices_ptr,
                            float *prob_ptr,
                            int *parent,
                            int const num_classes,
                            float *loss,
                            cudaStream_t stream) {
  if (m->peft_support_mode == SPATIAL_SHARING || m->peft_support_mode == SPATIAL_SHARING_LIMITED) {
    inference_kernel_spatial_sharing(m,
                                     bc,
                                     input_ptr,
                                     indices_ptr,
                                     num_classes,
                                     loss,
                                     stream);
    return;
  }

  if (m->beam_search) {
    // set all parents id zero in arg top1 case.
    checkCUDA(cudaMemsetAsync(parent, 0, bc->num_active_tokens() * sizeof(int), stream));
  }

  if (bc->num_inference_tokens() > 0) {
    launchArgmaxKernel(
        input_ptr, num_classes, bc->num_inference_tokens(), indices_ptr, prob_ptr, stream);
  }

  // print_tensor(indices_ptr, batch_size, "indices_ptr: ");

  // compute cross-entropy loss if there is a finetuning request
  assert(loss != nullptr);
  BatchConfig::TokenId token_ids[BatchConfig::MAX_NUM_TOKENS];
  if (bc->num_finetuning_fwd_requests() > 0) {
    assert(bc->num_finetuning_fwd_tokens() >= 1);
    int i = bc->finetuning_request_index();
    assert(bc->requestsInfo[i].peft_model_id != PEFTModelID::NO_ID);
    assert(!bc->requestsInfo[i].finetuning_backward_phase);
    int num_finetuning_tokens = bc->requestsInfo[i].num_tokens_in_batch - 1;
    assert(num_finetuning_tokens + 1 == bc->num_finetuning_fwd_tokens());
    int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    for (int j = 0; j < num_finetuning_tokens; j++) {
      token_ids[j] = bc->tokensInfo[j + first_token_offset + 1].token_id;
    }
    checkCUDA(
        cudaMemcpyAsync(m->handle.workSpace,
                        token_ids,
                        sizeof(BatchConfig::TokenId) * num_finetuning_tokens,
                        cudaMemcpyHostToDevice,
                        stream));
    // copy loss to d_loss
    checkCUDA(cudaMemsetAsync(m->d_loss, 0, sizeof(float), stream));
    compute_sparse_categorical_crossentropy_loss<<<
        GET_BLOCKS(num_finetuning_tokens),
        min(CUDA_NUM_THREADS, num_finetuning_tokens),
        0,
        stream>>>(input_ptr + first_token_offset * num_classes,
                  static_cast<BatchConfig::TokenId *>(m->handle.workSpace),
                  m->d_loss,
                  num_finetuning_tokens,
                  num_classes);
    // copy value from d_loss to loss
    checkCUDA(cudaMemcpyAsync(
        loss, m->d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
    *loss = *loss / (float)num_finetuning_tokens;
  }
}

/*static*/
void ArgMax::inference_kernel_wrapper(ArgMaxMeta const *m,
                                    BatchConfig const *bc,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorW const &indices,
                                    GenericTensorAccessorW const &parent,
                                    float *loss) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  int num_classes = input.domain.hi()[0] - input.domain.lo()[0] + 1;

  if (input.data_type == DT_HALF) {
    ArgMax::inference_kernel<half>(m,
                                 bc,
                                 input.get_half_ptr(),
                                 indices.get_int32_ptr(),
                                 m->beam_search ? m->probs : nullptr,
                                 m->beam_search ? parent.get_int32_ptr()
                                                : nullptr,
                                 num_classes,
                                 loss,
                                 stream);

  } else if (input.data_type == DT_FLOAT) {
    ArgMax::inference_kernel<float>(m,
                                  bc,
                                  input.get_float_ptr(),
                                  indices.get_int32_ptr(),
                                  m->beam_search ? m->probs : nullptr,
                                  m->beam_search ? parent.get_int32_ptr()
                                                 : nullptr,
                                  num_classes,
                                  loss,
                                  stream);
  } else {
    assert(false && "Unsupported data type");
  }

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[ArgMax] forward time = %.2lfms\n", elapsed);
  }
}

ArgMaxMeta::ArgMaxMeta(FFHandler handler,
                       Op const *op,
                       Legion::Domain const &input_domain,
                       Legion::Domain const &output_domain,
                       GenericTensorAccessorW input,
                       int batch_size,
                       int total_ele,
                       MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, op) {
  DataType data_type = op->data_type;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // size_t d_offsets_size = batch_size;
  size_t prob_size = batch_size;
  assert(data_type == DT_FLOAT || data_type == DT_HALF);
  size_t total_size = prob_size * sizeof(float);
  gpu_mem_allocator.create_legion_instance(
      reserveInst, total_size, "ArgMaxMeta");
  probs = gpu_mem_allocator.allocate_instance<float>(prob_size);

  // allocate space for loss on device
  gpu_mem_allocator.create_legion_instance(
      reserveInst, sizeof(float), "ArgMaxMeta");
  d_loss = gpu_mem_allocator.allocate_instance<float>(1);
}

ArgMaxMeta::~ArgMaxMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}
}; // namespace FlexFlow
