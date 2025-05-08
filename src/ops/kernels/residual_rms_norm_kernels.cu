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
#include "flexflow/ops/kernels/residual_rms_norm_kernels.h"
#include "flexflow/ops/residual_rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

#define C10_WARP_SIZE 32

ResidualRMSNormMeta::ResidualRMSNormMeta(FFHandler handler,
                                         ResidualRMSNorm const *rms,
                                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, rms) {
  eps = rms->eps;

  inplace_residual = rms->inplace_residual;
  in_dim = rms->inputs[0]->dims[0].size / rms->inputs[0]->dims[0].degree;
  size_t data_size = data_type_size(rms->weights[0]->data_type);

  size_t rms_ptr_size = 0;
  allocated_peft_buffer_size = 0;
  if (peft_finetuning_enabled(peft_support_mode)) {
    rms_ptr_size = rms->effective_batch_size * sizeof(float);
    allocated_peft_buffer_size =
        BatchConfig::max_sequence_length() * in_dim * data_size;
  }
  size_t totalSize = rms_ptr_size + allocated_peft_buffer_size;
  gpu_mem_allocator.create_legion_instance(
      reserveInst, totalSize, "ResidualRMSNormMeta");
  if (peft_finetuning_enabled(peft_support_mode)) {
    rms_ptr = gpu_mem_allocator.allocate_instance_untyped(rms_ptr_size);
    input_activation =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  } else {
    rms_ptr = input_activation = nullptr;
  }
}
ResidualRMSNormMeta::~ResidualRMSNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

namespace Kernels {
namespace ResidualRMSNorm {

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value,
                                            unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T *shared) {
  int const lid = threadIdx.x % C10_WARP_SIZE;
  int const wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / C10_WARP_SIZE)) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T>
__global__ void ResidualRMSNormFusedForwardKernel(int64_t data_dim,
                                                  float eps,
                                                  T const *X1,
                                                  T const *X2,
                                                  T *X_out,
                                                  float *rms,
                                                  int first_ft_token_idx,
                                                  T const *weights,
                                                  T *output) {
  __shared__ float v_shared[C10_WARP_SIZE];
  const int64_t i = blockIdx.x; // token idx
  const int64_t base_idx = i * data_dim;

  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < data_dim; j += blockDim.x) {
    int64_t const index = base_idx + j;
    float x1 = static_cast<float>(X1[index]);
    float x2 = static_cast<float>(X2[index]);
    float x_out = x1 + x2;
    X_out[index] = static_cast<T>(x_out);
    sum += x_out * x_out;
  }
  sum = BlockReduceSum<float>(sum, v_shared);

  float rms_val = 0.0f;
  if (threadIdx.x == 0) {
    rms_val = rsqrt((sum / static_cast<float>(data_dim)) + eps);
    v_shared[0] = rms_val;
  }
  __syncthreads();
  rms_val = v_shared[0];

  // store rms value for peft finetuning tokens. These tokens are assumed to be
  // contiguous and at the end of the batch rms already points to the first
  // available slot (excluding the rms already computed in previous partial fwd
  // passes)
  if (i >= first_ft_token_idx) {
    rms[i] = rms_val; // Store the RMS value for this token
  }

  for (int64_t j = threadIdx.x; j < data_dim; j += blockDim.x) {
    const int64_t index = base_idx + j;
    float const input_val = static_cast<float>(X_out[index]);
    float const weight_val = static_cast<float>(weights[j]);
    output[index] = static_cast<T>(input_val * rms_val * weight_val);
  }
}

template <typename T>
void inference_kernel_spatial_sharing(ResidualRMSNormMeta const *m,
                                      BatchConfig const *bc,
                                      T const *input1_ptr,
                                      T const *input2_ptr,
                                      T const *weight_ptr,
                                      T *residual_output_ptr,
                                      T *output_ptr,
                                      cudaStream_t main_stream) {
  // launch finetuning fwd tokens kernel if there are any finetuning fwd tokens
  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaEventRecord(m->handle.peft_fwd_can_start, main_stream)); 
    checkCUDA(cudaStreamWaitEvent(m->handle.peft_fwd_stream, m->handle.peft_fwd_can_start, 0));

    ResidualRMSNormFusedForwardKernel<T>
      <<<bc->num_finetuning_fwd_tokens(), std::min(CUDA_NUM_THREADS, m->in_dim), 0, m->handle.peft_fwd_stream>>>(
          m->in_dim,
          m->eps,
          input1_ptr + m->in_dim * bc->num_inference_tokens(),
          input2_ptr + m->in_dim * bc->num_inference_tokens(),
          residual_output_ptr + m->in_dim * bc->num_inference_tokens(),
          static_cast<float *>(m->rms_ptr) + bc->requestsInfo[bc->finetuning_request_index()].first_token_depth_in_request,
          0 /*first_ft_token_idx*/, 
          weight_ptr,
          output_ptr + m->in_dim * bc->num_inference_tokens());

    checkCUDA(cudaEventRecord(m->handle.peft_fwd_done, m->handle.peft_fwd_stream));
  }

  // launch inference kernel if there are inference tokens
  if (bc->num_inference_tokens() > 0) {
    ResidualRMSNormFusedForwardKernel<T>
      <<<bc->num_inference_tokens(), std::min(CUDA_NUM_THREADS, m->in_dim), 0, main_stream>>>(
          m->in_dim,
          m->eps,
          input1_ptr,
          input2_ptr,
          residual_output_ptr,
          nullptr /*rms_ptr*/,
          bc->num_inference_tokens() /*first_ft_token_idx*/, 
          weight_ptr,
          output_ptr);
  }

  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaStreamWaitEvent(main_stream, m->handle.peft_fwd_done, 0));
  }
}

template <typename T>
void inference_kernel(ResidualRMSNormMeta const *m,
                      BatchConfig const *bc,
                      T const *input1_ptr,
                      T const *input2_ptr,
                      T const *weight_ptr,
                      T *residual_output_ptr,
                      T *output_ptr,
                      cudaStream_t stream) {
  if (m->peft_support_mode == SPATIAL_SHARING || m->peft_support_mode == SPATIAL_SHARING_LIMITED) {
    inference_kernel_spatial_sharing(m, bc, input1_ptr, input2_ptr, weight_ptr, residual_output_ptr, output_ptr, stream);
    return;
  }
  int num_tokens = bc->num_active_tokens();
  int data_dim = m->in_dim;
  if (num_tokens <= 0) {
    // No tokens to process
    return;
  }
  int first_ft_token_idx =
      bc->num_active_tokens() - bc->num_finetuning_fwd_tokens();
  float *rms_ptr = nullptr;
  if (bc->num_finetuning_fwd_tokens() > 0) {
    int i = bc->finetuning_request_index();
    int tokens_previous_steps =
        bc->requestsInfo[i].first_token_depth_in_request;
    rms_ptr = static_cast<float *>(m->rms_ptr) + tokens_previous_steps;
  }
  ResidualRMSNormFusedForwardKernel<T>
      <<<num_tokens, std::min(CUDA_NUM_THREADS, data_dim), 0, stream>>>(
          data_dim,
          m->eps,
          input1_ptr,
          input2_ptr,
          residual_output_ptr,
          rms_ptr,
          first_ft_token_idx,
          weight_ptr,
          output_ptr);
}

void forward_kernel_wrapper(ResidualRMSNormMeta const *m,
                            GenericTensorAccessorR const &input1,
                            GenericTensorAccessorR const &input2,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &residual_output,
                            GenericTensorAccessorW const &output) {
  assert(false && "Not implemented yet");
}

template <typename DT>
void store_peft_activations(ResidualRMSNormMeta const *m,
                            BatchConfig const *bc,
                            size_t in_dim,
                            DT const *residual_output_ptr,
                            cudaStream_t stream) {
  assert(peft_finetuning_enabled(m->peft_support_mode));
  assert(bc->num_finetuning_fwd_tokens() >= 1);

  int num_ft_tokens = bc->num_finetuning_fwd_tokens();
  int i = bc->finetuning_request_index();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  int tokens_previous_steps = bc->requestsInfo[i].first_token_depth_in_request;
  assert(bc->requestsInfo[i].num_tokens_in_batch == num_ft_tokens);

  size_t batch_offset = in_dim * tokens_previous_requests;
  size_t request_offset = in_dim * tokens_previous_steps;
  size_t data_size = in_dim * num_ft_tokens * sizeof(DT);
  assert(m->allocated_peft_buffer_size >=
         BatchConfig::max_sequence_length() * in_dim * sizeof(DT));

  checkCUDA(
      cudaMemcpyAsync(static_cast<DT *>(m->input_activation) + request_offset,
                      residual_output_ptr + batch_offset,
                      data_size,
                      cudaMemcpyDeviceToDevice,
                      stream));
}

void inference_kernel_wrapper(ResidualRMSNormMeta *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input1,
                              GenericTensorAccessorR const &input2,
                              GenericTensorAccessorR const &weight,
                              GenericTensorAccessorW const &residual_output,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input1.data_type == input2.data_type);
  assert(output.data_type == input1.data_type);
  assert(weight.data_type == output.data_type);
  assert(residual_output.data_type == output.data_type);
  int in_dim = input1.domain.hi()[0] - input1.domain.lo()[0] + 1;

  if (output.data_type == DT_HALF) {
    inference_kernel(m,
                     bc,
                     input1.get_half_ptr(),
                     input2.get_half_ptr(),
                     weight.get_half_ptr(),
                     residual_output.get_half_ptr(),
                     output.get_half_ptr(),
                     stream);
    if (bc->num_finetuning_fwd_requests() > 0) {
      store_peft_activations(
          m, bc, in_dim, residual_output.get_half_ptr(), stream);
    }
  } else if (output.data_type == DT_FLOAT) {
    inference_kernel(m,
                     bc,
                     input1.get_float_ptr(),
                     input2.get_float_ptr(),
                     weight.get_float_ptr(),
                     residual_output.get_float_ptr(),
                     output.get_float_ptr(),
                     stream);
    if (bc->num_finetuning_fwd_requests() > 0) {
      store_peft_activations(
          m, bc, in_dim, residual_output.get_float_ptr(), stream);
    }
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
    printf("[ResidualRMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

template <typename T>
__global__ void
    ResidualRMSNormFusedBackwardKernel(int64_t data_dim,
                                       T const *output_grad_0_ptr,
                                       T const *output_grad_1_ptr,
                                       T const *residual_output_rms_input_ptr,
                                       T const *weight_ptr,
                                       float const *rms_ptr,
                                       T *input_grad_0_ptr,
                                       T *input_grad_1_ptr,
                                       bool reset_input_grad1,
                                       bool reset_input_grad2) {
  __shared__ float ds_storage[C10_WARP_SIZE];
  __shared__ float norm_val_shared;

  const int64_t i = blockIdx.x; // token idx

  float const rms_val = rms_ptr[i];
  float const rms_cubed = rms_val * rms_val * rms_val;
  float const inv_dim = 1.0f / static_cast<float>((int)data_dim);

  float ds = 0.0f;
  for (int64_t j = threadIdx.x; j < data_dim; j += blockDim.x) {
    const int64_t index = i * data_dim + j;
    float const out_grad = static_cast<float>(output_grad_1_ptr[index]);
    float const in_val =
        static_cast<float>(residual_output_rms_input_ptr[index]);
    float const w_val = static_cast<float>(weight_ptr[j]);
    ds += out_grad * in_val * w_val;
  }
  ds = BlockReduceSum<float>(ds, ds_storage);
  if (threadIdx.x == 0) {
    norm_val_shared = -ds * rms_cubed * inv_dim;
  }

  __syncthreads();

  for (int64_t j = threadIdx.x; j < data_dim; j += blockDim.x) {
    const int64_t index = i * data_dim + j;
    float const out_grad = static_cast<float>(output_grad_1_ptr[index]);
    float const in_val =
        static_cast<float>(residual_output_rms_input_ptr[index]);
    float const w_val = static_cast<float>(weight_ptr[j]);
    float const dX_val = rms_val * out_grad * w_val + norm_val_shared * in_val;

    T in_grad_0_val =
        static_cast<T>(dX_val); // This is the gradient for input 0
    if (reset_input_grad1) {
      input_grad_0_ptr[index] = in_grad_0_val;
    } else {
      in_grad_0_val += output_grad_0_ptr[index];
      input_grad_0_ptr[index] = in_grad_0_val;
    }
    if (reset_input_grad2) {
      input_grad_1_ptr[index] = in_grad_0_val;
    } else {
      input_grad_1_ptr[index] += in_grad_0_val;
    }
  }
}

template <typename T>
void peft_bwd_kernel(ResidualRMSNormMeta const *m,
                     BatchConfig const *bc,
                     T const *output_grad_0_ptr,
                     T const *output_grad_1_ptr,
                     T *input_grad_0_ptr,
                     T *input_grad_1_ptr,
                     T const *weight_ptr,
                     cudaStream_t stream) {

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();

  int num_tokens = bc->num_finetuning_bwd_tokens();
  if (num_tokens <= 0) {
    return;
  }
  int data_dim = m->in_dim;
  ResidualRMSNormFusedBackwardKernel<T>
      <<<num_tokens, std::min(data_dim, CUDA_NUM_THREADS), 0, stream>>>(
          data_dim,
          output_grad_0_ptr,
          output_grad_1_ptr,
          static_cast<T *>(m->input_activation),
          weight_ptr,
          static_cast<float *>(m->rms_ptr),
          input_grad_0_ptr,
          input_grad_1_ptr,
          m->reset_input_grads[0],
          m->reset_input_grads[1]);
}

void backward_kernel_wrapper(
    ResidualRMSNormMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &residual_output_rms_input,
    GenericTensorAccessorW const &residual_input0_grad,
    GenericTensorAccessorW const &residual_input1_grad,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &weight_grad) {
  assert(false && "Not implemented yet");
}

void peft_bwd_kernel_wrapper(ResidualRMSNormMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorR const &output_grad_0,
                             GenericTensorAccessorR const &output_grad_1,
                             GenericTensorAccessorW const &input_grad_0,
                             GenericTensorAccessorW const &input_grad_1,
                             GenericTensorAccessorR const &weight) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  assert(output_grad_1.data_type == input_grad_0.data_type);
  assert(input_grad_0.data_type == input_grad_1.data_type);
  assert(input_grad_1.data_type == weight.data_type);

  if (output_grad_1.data_type == DT_HALF) {
    peft_bwd_kernel(m,
                    bc,
                    m->reset_input_grads[0] ? nullptr
                                            : output_grad_0.get_half_ptr(),
                    output_grad_1.get_half_ptr(),
                    input_grad_0.get_half_ptr(),
                    input_grad_1.get_half_ptr(),
                    weight.get_half_ptr(),
                    stream);
  } else if (output_grad_1.data_type == DT_FLOAT) {
    peft_bwd_kernel(m,
                    bc,
                    m->reset_input_grads[0] ? nullptr
                                            : output_grad_0.get_float_ptr(),
                    output_grad_1.get_float_ptr(),
                    input_grad_0.get_float_ptr(),
                    input_grad_1.get_float_ptr(),
                    weight.get_float_ptr(),
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
    printf("[ResidualRMSNorm] backward time (CF) = %.2fms\n", elapsed);
  }
}

} // namespace ResidualRMSNorm
} // namespace Kernels
} // namespace FlexFlow
