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
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/ops/lora_linear_params.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

LinearMeta::LinearMeta(FFHandler handler,
                       int batch_size,
                       Linear const *li,
                       MemoryAllocator gpu_mem_allocator,
                       int weightSize)
    : OpMeta(handler, li), weight_ptr(nullptr), activation(li->activation),
      kernel_reg_type(li->kernel_reg_type),
      kernel_reg_lambda(li->kernel_reg_lambda), use_bias(li->use_bias),
      add_bias_only_once(li->add_bias_only_once),
      quantization_type(li->quantization_type), offload(li->offload) {
  weight_ptr_type = this->input_type[0];
  trainable_inputs[0] = li->trainable_inputs[0];

  DataType data_type = li->data_type;
  size_t data_size = data_type_size(data_type);
  // allocate weight and bias in the reserve space for cpu offloading
  if (li->offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(
        weightSize * data_type_size(data_type));
    if (li->quantization_type != DT_NONE) {
      quantized_weightSize = get_quantization_to_byte_size(
          data_type, li->quantization_type, weightSize);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }
  }
  // peft activation
  size_t out_dim =
      li->outputs[0]->dims[0].size / li->outputs[0]->dims[0].degree;
  if (enable_peft_finetuning &&
      (activation == AC_MODE_RELU || activation == AC_MODE_SIGMOID)) {
    // Allocate space for storing the output activations for PEFT finetuning
    // during inference
    allocated_peft_buffer_size =
        data_size * BatchConfig::max_sequence_length() * out_dim;
  } else {
    allocated_peft_buffer_size = 0;
    output_activation_buffer = nullptr;
  }
  size_t bias_size = use_bias ? batch_size * data_size : 0;

  size_t totalSize = bias_size + allocated_peft_buffer_size;
  gpu_mem_allocator.create_legion_instance(
      reserveInst, totalSize, "LinearMeta");
  // Allocate an all-one's vector
  if (use_bias) {
    one_ptr = gpu_mem_allocator.allocate_instance_untyped(bias_size);
    int parallelism = batch_size;
    cudaStream_t stream;
    checkCUDA(get_legion_stream(&stream));
    if (data_type == DT_FLOAT) {
      Kernels::Linear::Internal::
          build_one_ptr<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>((float *)one_ptr, batch_size);
    } else if (data_type == DT_HALF) {
      Kernels::Linear::Internal::
          build_one_ptr<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>((half *)one_ptr, batch_size);
    } else if (data_type == DT_BFLOAT16) {
      Kernels::Linear::Internal::
          build_one_ptr<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>((__ff_bfloat16 *)one_ptr, batch_size);
    } else {
      assert(false && "Unsupported data type");
    }
  } else {
    one_ptr = nullptr;
  }
  if (enable_peft_finetuning) {
    output_activation_buffer =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  } else {
    output_activation_buffer = nullptr;
  }
  // Allocate descriptors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

LinearMeta::~LinearMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

bool lora_applies_to_this_layer(LinearMeta const *m,
                                LoraLinearConfig const &config) {
  for (std::string s : config.target_modules) {
    std::string n(m->op_name);
    if (n.find(s) != std::string::npos) {
      return true;
    }
  }
  return false;
}

namespace Kernels {
namespace Linear {

bool use_activation(ActiMode mode) {
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
    case AC_MODE_NONE:
      return false;
    default:
      assert(0);
      break;
  }
  return false;
}

void init_kernel(LinearMeta *m, int batch_size, int channel) {
  if (use_activation(m->activation)) {
    cudnnActivationMode_t mode;
    switch (m->activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(
        m->actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(
        cudnnSetTensor4dDescriptor(m->outputTensor,
                                   CUDNN_TENSOR_NCHW,
                                   ff_to_cudnn_datatype(m->output_type[0]),
                                   batch_size,
                                   channel,
                                   1,
                                   1));
  }
}

void forward_kernel_wrapper(LinearMeta const *m,
                            void const *input_ptr,
                            void *output_ptr,
                            void const *weight_ptr,
                            void const *bias_ptr,
                            int in_dim,
                            int out_dim,
                            int batch_size) {
  assert(false && "Not implemented yet");
}

void inference_kernel_wrapper(LinearMeta *m,
                              BatchConfig const *bc,
                              void const *input_ptr,
                              void *output_ptr,
                              void const *weight_ptr,
                              void const *bias_ptr,
                              int in_dim,
                              int out_dim,
                              int batch_size) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  if (m->input_type[0] == DT_FLOAT) {
    Internal::inference_kernel<float>(m,
                                      input_ptr,
                                      output_ptr,
                                      weight_ptr,
                                      bias_ptr,
                                      in_dim,
                                      out_dim,
                                      batch_size,
                                      stream);
    if ((m->activation == AC_MODE_RELU || m->activation == AC_MODE_SIGMOID) &&
        bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations<float>(
          m, bc, out_dim, static_cast<float *>(output_ptr), stream);
    }
  } else if (m->input_type[0] == DT_HALF) {
    Internal::inference_kernel<half>(m,
                                     input_ptr,
                                     output_ptr,
                                     weight_ptr,
                                     bias_ptr,
                                     in_dim,
                                     out_dim,
                                     batch_size,
                                     stream);
    if ((m->activation == AC_MODE_RELU || m->activation == AC_MODE_SIGMOID) &&
        bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations<half>(
          m, bc, out_dim, static_cast<half *>(output_ptr), stream);
    }
  } else if (m->input_type[0] == DT_BFLOAT16) {
    Internal::inference_kernel<float>(m,
                                      input_ptr,
                                      output_ptr,
                                      weight_ptr,
                                      bias_ptr,
                                      in_dim,
                                      out_dim,
                                      batch_size,
                                      stream);
    if ((m->activation == AC_MODE_RELU || m->activation == AC_MODE_SIGMOID) &&
        bc->num_finetuning_fwd_requests() > 0) {
      Internal::store_peft_activations<__ff_bfloat16>(
          m, bc, out_dim, static_cast<__ff_bfloat16 *>(output_ptr), stream);
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
    printf("%s [Linear] inference time = %.2lfms\n", m->op_name, elapsed);
  }
}

void peft_bwd_kernel_wrapper(LinearMeta const *m,
                             BatchConfig const *bc,
                             void *input_grad_ptr,
                             void *output_grad_ptr,
                             void const *weight_ptr,
                             int in_dim,
                             int out_dim) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  if (m->input_type[0] == DT_FLOAT) {
    Internal::peft_bwd_kernel<float>(m,
                                     bc,
                                     input_grad_ptr,
                                     output_grad_ptr,
                                     weight_ptr,
                                     in_dim,
                                     out_dim,
                                     stream);
  } else if (m->input_type[0] == DT_HALF) {
    Internal::peft_bwd_kernel<half>(m,
                                    bc,
                                    input_grad_ptr,
                                    output_grad_ptr,
                                    weight_ptr,
                                    in_dim,
                                    out_dim,
                                    stream);
  } else if (m->input_type[0] == DT_BFLOAT16) {
    // cublas scale type: https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasGemmEx#cublasgemmex
    Internal::peft_bwd_kernel<float>(m,
                                             bc,
                                             input_grad_ptr,
                                             output_grad_ptr,
                                             weight_ptr,
                                             in_dim,
                                             out_dim,
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
    printf("%s [Linear] PEFT Bwd time = %.2lfms\n", m->op_name, elapsed);
  }
}

void backward_kernel_wrapper(LinearMeta const *m,
                             void const *input_ptr,
                             void *input_grad_ptr,
                             void const *output_ptr,
                             void *output_grad_ptr,
                             void const *kernel_ptr,
                             void *kernel_grad_ptr,
                             void *bias_grad_ptr,
                             int in_dim,
                             int out_dim,
                             int batch_size) {
  assert(false && "Not implemented yet");
}

namespace Internal {

template <typename DT>
__global__ void AddBiasWithReLU(DT *output_ptr,
                                DT const *bias_ptr,
                                int out_dim,
                                int batch_size) {
  CUDA_KERNEL_LOOP(i, out_dim * batch_size) {
    int bias_idx = i % out_dim;
    DT value = output_ptr[i] + bias_ptr[bias_idx];
    output_ptr[i] = ((float)value > 0.0f) ? value : (DT)0.0f;
  }
}

template <typename DT>
void inference_kernel(LinearMeta const *m,
                      void const *input_ptr,
                      void *output_ptr,
                      void const *weight_ptr,
                      void const *bias_ptr,
                      int in_dim,
                      int out_dim,
                      int batch_size,
                      ffStream_t stream) {
  // additional processing for uploading weights
  if (m->offload) {
    // Note that we update weight_ptr when uploading weight
    if (m->quantization_type != DT_NONE) {
      cudaMemcpyAsync(m->quantized_weight_ptr,
                      weight_ptr,
                      m->quantized_weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
      if (m->quantization_type == DT_INT4) {
        int parallelism = in_dim * out_dim / 2;
        decompress_int4_general_weights<DT>
            <<<GET_BLOCKS(parallelism),
               min(CUDA_NUM_THREADS, parallelism),
               0,
               stream>>>(m->quantized_weight_ptr,
                         static_cast<DT *>(m->weight_ptr),
                         in_dim,
                         in_dim * out_dim);
      } else {
        assert(m->quantization_type == DT_INT8);
        int parallelism = in_dim * out_dim;
        decompress_int8_general_weights<DT>
            <<<GET_BLOCKS(parallelism),
               min(CUDA_NUM_THREADS, parallelism),
               0,
               stream>>>(m->quantized_weight_ptr,
                         static_cast<DT *>(m->weight_ptr),
                         in_dim,
                         in_dim * out_dim);
      }
    } else {
      cudaMemcpyAsync(m->weight_ptr,
                      weight_ptr,
                      in_dim * out_dim * sizeof(DT),
                      cudaMemcpyHostToDevice,
                      stream);
    }
  }
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  // CUDA_R_32F for bf16:
  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasGemmEx#cublasgemmex
  DT alpha = 1.0f, beta = 0.0f;
  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = m->offload
                                   ? ff_to_cuda_datatype(m->weight_ptr_type)
                                   : ff_to_cuda_datatype(m->weight_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);
  assert(input_type == weight_type && weight_type == output_type);
  cudaDataType_t compute_type =
      output_type == CUDA_R_16BF ? CUDA_R_32F : output_type;
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         out_dim,
                         batch_size,
                         in_dim,
                         &alpha,
                         m->offload ? m->weight_ptr : weight_ptr,
                         weight_type,
                         in_dim,
                         input_ptr,
                         input_type,
                         in_dim,
                         &beta,
                         output_ptr,
                         output_type,
                         out_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // use_bias = True
  if (bias_ptr != NULL) {
    // fuse bias and relu
    if (m->activation == AC_MODE_RELU) {
      int parallelism = out_dim * batch_size;
      AddBiasWithReLU<<<GET_BLOCKS(parallelism), CUDA_NUM_THREADS, 0, stream>>>(
          static_cast<DT *>(output_ptr),
          static_cast<DT const *>(bias_ptr),
          out_dim,
          batch_size);
      return;
    }
    checkCUDA(cublasGemmEx(m->handle.blas,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           out_dim,
                           batch_size,
                           1,
                           &alpha,
                           bias_ptr,
                           weight_type,
                           1,
                           static_cast<DT *>(m->one_ptr),
                           weight_type,
                           1,
                           &alpha,
                           output_ptr,
                           output_type,
                           out_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (use_activation(m->activation)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn,
                                      m->actiDesc,
                                      &alpha,
                                      m->outputTensor,
                                      output_ptr,
                                      &beta,
                                      m->outputTensor,
                                      output_ptr));
  } else if (m->activation == AC_MODE_GELU) {
    size_t elements = (size_t)out_dim * (size_t)batch_size;
    constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
    constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS, 0, stream>>>(
        elements, B, C, (float *)output_ptr);
  } else if (m->activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
  }
}

template <typename DT>
void store_peft_activations(LinearMeta const *m,
                            BatchConfig const *bc,
                            size_t out_dim,
                            DT *output_ptr,
                            cudaStream_t stream) {
  int i = bc->finetuning_request_index();
  int num_ft_tokens = bc->num_finetuning_fwd_tokens();
  int tokens_previous_requests =
      bc->requestsInfo[i].first_token_offset_in_batch;
  int tokens_previous_steps = bc->requestsInfo[i].first_token_depth_in_request;
  size_t data_size = out_dim * num_ft_tokens * sizeof(DT);
  size_t batch_offset = out_dim * tokens_previous_requests;
  size_t request_offset = out_dim * tokens_previous_steps;
  assert(bc->num_finetuning_fwd_tokens() >= 1);
  assert(bc->requestsInfo[i].peft_model_id != PEFTModelID::NO_ID);
  assert(!bc->requestsInfo[i].finetuning_backward_phase);
  assert(bc->requestsInfo[i].num_tokens_in_batch == num_ft_tokens);
  assert(m->allocated_peft_buffer_size >= data_size);

  checkCUDA(cudaMemcpyAsync(static_cast<DT *>(m->output_activation_buffer) +
                                request_offset,
                            output_ptr + batch_offset,
                            data_size,
                            cudaMemcpyDeviceToDevice,
                            stream));
}

template <typename DT>
void peft_bwd_kernel(LinearMeta const *m,
                     BatchConfig const *bc,
                     void *input_grad_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     int in_dim,
                     int out_dim,
                     ffStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.peft_blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.peft_dnn, stream));

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();
  int num_peft_tokens = bc->num_finetuning_bwd_tokens();
  assert(bc->num_finetuning_bwd_requests() == 1);

  cudaDataType_t input_type = ff_to_cuda_datatype(m->input_type[0]);
  cudaDataType_t weight_type = ff_to_cuda_datatype(m->weight_type[0]);
  cudaDataType_t output_type = ff_to_cuda_datatype(m->output_type[0]);

  input_grad_ptr = static_cast<DT *>(input_grad_ptr);
  output_grad_ptr = static_cast<DT *>(output_grad_ptr);
  cudaDataType_t compute_type = output_type == CUDA_R_16BF ? CUDA_R_32F : output_type;
  int output_size = out_dim * num_peft_tokens;
  if (m->activation == AC_MODE_RELU) {
    relu_backward_kernel(m->output_type[0],
                         output_grad_ptr,
                         m->output_activation_buffer,
                         output_size,
                         stream);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward_kernel(m->output_type[0],
                            output_grad_ptr,
                            m->output_activation_buffer,
                            output_size,
                            stream);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }

  // Compute data gradient
  // NOTE: we use beta=1 for input_grad to accumulate gradients when needed
  DT alpha = 1.0f;
  DT beta = m->reset_input_grads[0] ? 0.0f : 1.0f;
  std::string peft_model_config_str =
      std::string(bc->requestsInfo[i].peft_model_config_str);
  LoraLinearConfig lora_config =
      LoraLinearConfig::deserialize_from_json_string(peft_model_config_str);
  bool lora_applies = lora_applies_to_this_layer(m, lora_config);
  // if the request does not have any active lora in the current layer, reset
  // beta to 0
  if (lora_applies) {
    beta = 1.0f;
  }

  if (input_grad_ptr != NULL) {
    checkCUDA(cublasGemmEx(m->handle.peft_blas,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           in_dim,
                           num_peft_tokens,
                           out_dim,
                           &alpha,
                           kernel_ptr,
                           weight_type,
                           in_dim,
                           output_grad_ptr,
                           output_type,
                           out_dim,
                           &beta,
                           input_grad_ptr,
                           input_type,
                           in_dim,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
}

template <typename DT>
__global__ void build_one_ptr(DT *one_ptr, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    one_ptr[i] = static_cast<DT>(1.0f);
  }
}

} // namespace Internal
} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow
