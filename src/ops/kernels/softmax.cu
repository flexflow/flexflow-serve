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

#include "flexflow/ops/kernels/softmax_kernels.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Domain;

SoftmaxMeta::SoftmaxMeta(FFHandler handler,
                         Softmax const *softmax,
                         Domain const &input_domain,
                         bool is_last_op,
                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, softmax) {
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain4SoftMax(
      inputTensor, input_domain, softmax->data_type));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain4SoftMax(
      outputTensor, input_domain, softmax->data_type));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorPeftFwd));
  checkCUDNN(cudnnSetTensorDescriptorFromDomain4SoftMax(
      outputTensorPeftFwd, input_domain, softmax->data_type));
  dim = softmax->dim;
  profiling = softmax->profiling;
  inference_debugging = softmax->inference_debugging;
  if (peft_finetuning_enabled(peft_support_mode) && is_last_op) {
    allocated_peft_buffer_size =
        input_domain.get_volume() * data_type_size(softmax->data_type);
    gpu_mem_allocator.create_legion_instance(
        reserveInst, allocated_peft_buffer_size, "SoftmaxMeta");
    output_grad_ptr =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  } else {
    allocated_peft_buffer_size = 0;
    output_grad_ptr = nullptr;
  }
  std::strcpy(op_name, softmax->name);
}

namespace Kernels {
namespace Softmax {

void forward_kernel_wrapper(SoftmaxMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->output_type[0] == DT_FLOAT) {
    Internal::forward_kernel(
        m, input.get_float_ptr(), output.get_float_ptr(), stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::forward_kernel(
        m, input.get_half_ptr(), output.get_half_ptr(), stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] forward time = %.2fms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(SoftmaxMeta const *m,
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
  assert(input_grad.domain == output_grad.domain);
  if (m->output_type[0] == DT_FLOAT) {
    Internal::backward_kernel(m,
                              input_grad.get_float_ptr(),
                              output_grad.get_float_ptr(),
                              output_grad.domain.get_volume(),
                              stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::backward_kernel(m,
                              input_grad.get_half_ptr(),
                              output_grad.get_half_ptr(),
                              output_grad.domain.get_volume(),
                              stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_output_grad.ptr, acc_output_grad.rect.volume(),
    // "[Softmax:backward:output_grad]");
    // print_tensor<float>(acc_input_grad.ptr, acc_input_grad.rect.volume(),
    // "[Softmax:backward:input_grad]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug("Softmax backward time = %.2fms\n", elapsed);
  }
}

void inference_kernel_wrapper(SoftmaxMeta *m,
                              BatchConfig const *bc,
                              bool is_last_op,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (bc->num_active_tokens() <= 0) {
    return;
  }
  int num_classes = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  if (m->output_type[0] == DT_FLOAT) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_float_ptr(),
                               output.get_float_ptr(),
                               num_classes,
                               stream);
    if (is_last_op && bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations(
          m, bc, num_classes, output.get_float_ptr(), stream);
    }
  } else if (m->output_type[0] == DT_HALF) {
    Internal::inference_kernel(m,
                               bc,
                               input.get_half_ptr(),
                               output.get_half_ptr(),
                               num_classes,
                               stream);
    if (is_last_op && bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations(
          m, bc, num_classes, output.get_half_ptr(), stream);
    }
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

void peft_bwd_kernel_wrapper(SoftmaxMeta const *m,
                             BatchConfig const *bc,
                             GenericTensorAccessorW const &input_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  int num_classes = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  if (m->output_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel(
        m, bc, input_grad.get_float_ptr(), num_classes, stream);
  } else if (m->output_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel(
        m, bc, input_grad.get_half_ptr(), num_classes, stream);
  } else {
    assert(false && "Unsupported data type");
  }
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    // print_tensor<float>(acc_input.ptr, acc_input.rect.volume(),
    // "[Softmax:forward:input]"); print_tensor<float>(acc_output.ptr,
    // acc_output.rect.volume(), "[Softmax:forward:output]");
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    log_measure.debug(
        "%s [Softmax] inference time = %.2fms\n", m->op_name, elapsed);
  }
}

namespace Internal {
template <typename DT>
void forward_kernel(SoftmaxMeta const *m,
                    DT const *input_ptr,
                    DT *output_ptr,
                    cudaStream_t stream) {
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m->inputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));
}

template <typename DT>
void backward_kernel(SoftmaxMeta const *m,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     size_t num_elements,
                     cudaStream_t stream) {
  checkCUDA(cudaMemcpyAsync(input_grad_ptr,
                            output_grad_ptr,
                            num_elements * sizeof(DT),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename DT>
void inference_kernel_spatial_sharing(SoftmaxMeta const *m,
                                    BatchConfig const *bc,
                                    DT const *input_ptr,
                                    DT *output_ptr,
                                    int num_classes,
                                    cudaStream_t main_stream) {
  // launch finetuning fwd tokens kernel if there are any finetuning fwd tokens
  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaEventRecord(m->handle.peft_fwd_can_start, main_stream)); 
    checkCUDA(cudaStreamWaitEvent(m->handle.peft_fwd_stream, m->handle.peft_fwd_can_start, 0));

    checkCUDNN(cudnnSetStream(m->handle.peft_fwd_dnn, m->handle.peft_fwd_stream));
    float alpha = 1.0f, beta = 0.0f;
    cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensorPeftFwd,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          bc->num_finetuning_fwd_tokens(),
                                          num_classes,
                                          1,
                                          1));
    checkCUDNN(cudnnSoftmaxForward(m->handle.peft_fwd_dnn,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  m->outputTensorPeftFwd,
                                  input_ptr + num_classes * bc->num_inference_tokens(),
                                  &beta,
                                  m->outputTensorPeftFwd,
                                  output_ptr + num_classes * bc->num_inference_tokens()));

    checkCUDA(cudaEventRecord(m->handle.peft_fwd_done, m->handle.peft_fwd_stream));
  }

  // launch inference kernel if there are inference tokens
  if (bc->num_inference_tokens() > 0) {

    checkCUDNN(cudnnSetStream(m->handle.dnn, main_stream));
    float alpha = 1.0f, beta = 0.0f;
    cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          bc->num_inference_tokens(),
                                          num_classes,
                                          1,
                                          1));
    checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                  CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  m->outputTensor,
                                  input_ptr,
                                  &beta,
                                  m->outputTensor,
                                  output_ptr));
  }

  if (bc->num_finetuning_fwd_tokens() > 0) {
    checkCUDA(cudaStreamWaitEvent(main_stream, m->handle.peft_fwd_done, 0));
  }
}

template <typename DT>
void inference_kernel(SoftmaxMeta const *m,
                      BatchConfig const *bc,
                      DT const *input_ptr,
                      DT *output_ptr,
                      int num_classes,
                      cudaStream_t stream) {
  if (m->peft_support_mode == SPATIAL_SHARING || m->peft_support_mode == SPATIAL_SHARING_LIMITED) {
    inference_kernel_spatial_sharing(m, bc, input_ptr, output_ptr, num_classes, stream);
    return;
  }
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  // fprintf(stderr, "Error in cudnnSetTensor4dDescriptor: %s\n", e.what());
  // printf("num_active_tokens: %d, num_classes: %d\n",
  //         bc->num_active_tokens(), num_classes);
  // printf("input_ptr: %p, output_ptr: %p, m->outputTensor: %p\n", input_ptr, output_ptr, m->outputTensor);
  // std::cerr << "bc: " << *bc << std::endl;
  // try {
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          cudnn_data_type,
                                          bc->num_active_tokens(),
                                          num_classes,
                                          1,
                                          1));
  // } catch (const std::exception &e) {
  //   fprintf(stderr, "Error in cudnnSetTensor4dDescriptor: %s\n", e.what());
  //   fprintf(stderr, "num_active_tokens: %d, num_classes: %d\n",
  //           bc->num_active_tokens(), num_classes);
  //   fprintf(stderr, "input_ptr: %p, output_ptr: %p\n", input_ptr, output_ptr);
  //   std::cerr << "bc: " << *bc << std::endl;
  //   assert(false);
  // }
  checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 m->outputTensor,
                                 input_ptr,
                                 &beta,
                                 m->outputTensor,
                                 output_ptr));
}

template <typename DT>
void store_peft_activations(SoftmaxMeta *m,
                            BatchConfig const *bc,
                            int num_classes,
                            DT *output_ptr,
                            cudaStream_t stream) {
  assert(peft_finetuning_enabled(m->peft_support_mode));
  assert(m->output_grad_ptr != nullptr);

  int num_ft_tokens = bc->num_finetuning_fwd_tokens();
  int i = bc->finetuning_request_index();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  int prev_steps_tokens = bc->requestsInfo[i].first_token_depth_in_request;
  assert(bc->requestsInfo[i].num_tokens_in_batch == num_ft_tokens);

  // shift labels by 1 position to the left (ignore first token label)
  for (int j = 0; j < num_ft_tokens - 1; j++) {
    m->peft_token_ids[prev_steps_tokens + j] =
        bc->tokensInfo[tokens_previous_requests + j + 1].token_id;
  }

  size_t batch_offset = num_classes * tokens_previous_requests;
  size_t req_offset = num_classes * prev_steps_tokens;
  size_t data_size = num_classes * num_ft_tokens * sizeof(DT);
  assert(m->allocated_peft_buffer_size >= data_size);
  checkCUDA(cudaMemcpyAsync(static_cast<DT *>(m->output_grad_ptr) + req_offset,
                            output_ptr + batch_offset,
                            data_size,
                            cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename DT>
__global__ void sparse_categorical_crossentropy_loss_peft_backward(
    DT *input_grad,
    DT const *output_grad,
    BatchConfig::TokenId const *token_ids,
    int num_tokens,
    int num_classes) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_classes) {
    int class_idx = i % num_classes;
    int token_idx = i / num_classes;
    input_grad[i] = output_grad[i];
    if (class_idx == token_ids[token_idx]) {
      input_grad[i] = input_grad[i] - (DT)1.0f;
    }
  }
}

template <typename DT>
void peft_bwd_kernel(SoftmaxMeta const *m,
                     BatchConfig const *bc,
                     DT *input_grad_ptr,
                     int num_classes,
                     cudaStream_t stream) {
  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();

  int num_bwd_tokens = bc->requestsInfo[i].num_tokens_in_batch - 1;

  DT scale_factor = 1.0 / (bc->requestsInfo[i].num_tokens_in_batch);
  // ignore last token
  checkCUDA(cudaMemsetAsync(input_grad_ptr + num_bwd_tokens * num_classes,
                            0,
                            num_classes * sizeof(DT),
                            stream));
  checkCUDA(cudaMemcpyAsync(m->handle.workSpace,
                            m->peft_token_ids,
                            sizeof(BatchConfig::TokenId) * num_bwd_tokens,
                            cudaMemcpyHostToDevice,
                            stream));
  sparse_categorical_crossentropy_loss_peft_backward<<<
      GET_BLOCKS(num_bwd_tokens * num_classes),
      CUDA_NUM_THREADS,
      0,
      stream>>>(input_grad_ptr,
                static_cast<DT *>(m->output_grad_ptr),
                static_cast<BatchConfig::TokenId const *>(m->handle.workSpace),
                num_bwd_tokens,
                num_classes);
  // scale
  scale_kernel<<<GET_BLOCKS(num_bwd_tokens * num_classes),
                 CUDA_NUM_THREADS,
                 0,
                 stream>>>(
      input_grad_ptr, num_bwd_tokens * num_classes, DT(0.0), scale_factor);
}

} // namespace Internal
} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow
