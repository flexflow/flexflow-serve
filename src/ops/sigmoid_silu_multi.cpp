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

#include "flexflow/ops/sigmoid_silu_multi.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

SigmoidSiluMultiMeta::SigmoidSiluMultiMeta(FFHandler handle,
                                           SigmoidSiluMulti const *ssm,
                                           MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handle, ssm) {
  profiling = ssm->profiling;
  inference_debugging = ssm->inference_debugging;
  enable_peft_finetuning = ssm->enable_peft_finetuning;
  if (enable_peft_finetuning) {
    size_t in_dim = ln->inputs[0]->dims[0].size / ln->inputs[0]->dims[0].degree;
    allocated_peft_buffer_size = 2 * data_type_size(data_type) *
                                 BatchConfig::max_sequence_length() * in_dim;
    gpu_mem_allocator.create_legion_instance(reserveInst,
                                             allocated_peft_buffer_size);
    input_activation =
        gpu_mem_allocator.allocate_instance_untyped(allocated_peft_buffer_size);
  }
}

SigmoidSiluMultiMeta::~SigmoidSiluMultiMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

template <typename T>
__global__ void SigmoidSiluMultiKernel(int num_elements,
                                       T const *input1_ptr,
                                       T const *input2_ptr,
                                       T *output_ptr) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float sigmoid_val = static_cast<float>(input1_ptr[i]);
    sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_val));
    output_ptr[i] = input1_ptr[i] * T(sigmoid_val) * input2_ptr[i];
  }
}

template <typename T>
__global__ void SigmoidSiluMultiBackwardKernel(int num_elements,
                                               T const *output_grad_ptr,
                                               T const *input1_ptr,
                                               T const *input2_ptr,
                                               T *input1_grad_ptr,
                                               T *input2_grad_ptr,
                                               bool reset_input_grad1,
                                               bool reset_input_grad2) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    float sigmoid_val = static_cast<float>(input1_ptr[i]);
    sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_val));

    if (reset_input_grad2) {
      input2_grad_ptr[i] =
          output_grad_ptr[i] * (input1_ptr[i] * T(sigmoid_val));
    } else {
      input2_grad_ptr[i] +=
          output_grad_ptr[i] * (input1_ptr[i] * T(sigmoid_val));
    }
    T ss_grad_val = output_grad_ptr[i] * input2_ptr[i];
    if (reset_input_grad1) {
      input1_grad_ptr[i] = ss_grad_val * T(sigmoid_val);
    } else {
      input1_grad_ptr[i] += ss_grad_val * T(sigmoid_val);
    }
    T sig_grad = ss_grad_val * input1_ptr[i];

    float x1_grad_val = static_cast<float>(sig_grad);
    x1_grad_val = x1_grad_val * sigmoid_val * (1.0f - sigmoid_val);
    input1_grad_ptr[i] += T(x1_grad_val);
  }
}

/*static*/
void SigmoidSiluMulti::inference_kernel_wrapper(
    SigmoidSiluMultiMeta *m,
    BatchConfig const *bc,
    GenericTensorAccessorR const &input1,
    GenericTensorAccessorR const &input2,
    GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_elements = input1.domain.get_volume();
  assert(input2.domain.get_volume() == num_elements);
  assert(output.domain.get_volume() == num_elements);

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  // save input activation if needed for PEFT
  if (bc->num_finetuning_bwd_requests() > 0) {
    // Check that we have at most one request that requires peft_bwd
    assert(bc->num_finetuning_bwd_tokens() >= 1);
    int i = bc->finetuning_request_index();
    assert(bc->requestsInfo[i].peft_model_id != PEFTModelID::NO_ID);
    assert(!bc->requestsInfo[i].finetuning_backward_phase);
    int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
    int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    assert(num_peft_tokens == bc->num_finetuning_fwd_tokens());
    int first_token_offset = bc->requestsInfo[i].first_token_offset_in_batch;
    size_t input_tensor_size =
        data_type_size(m->input_type[0]) * num_peft_tokens * in_dim;
    assert(m->allocated_peft_buffer_size == 2 * input_tensor_size);
    // copy input activation
    if (m->input_type[0] == DT_FLOAT) {
      checkCUDA(
          hipMemcpyAsync(m->input_activation,
                         input1.get_float_ptr() + first_token_offset * in_dim,
                         input_tensor_size,
                         hipMemcpyDeviceToDevice,
                         stream));
      checkCUDA(hipMemcpyAsync(
          (void *)((char *)m->input_activation + input_tensor_size),
          input2.get_float_ptr() + first_token_offset * in_dim,
          input_tensor_size,
          hipMemcpyDeviceToDevice,
          stream));
    } else if (m->input_type[0] == DT_HALF) {
      checkCUDA(
          hipMemcpyAsync(m->input_activation,
                         input1.get_half_ptr() + first_token_offset * in_dim,
                         input_tensor_size,
                         hipMemcpyDeviceToDevice,
                         stream));
      checkCUDA(hipMemcpyAsync(
          (void *)((char *)m->input_activation + input_tensor_size),
          input2.get_half_ptr() + first_token_offset * in_dim,
          input_tensor_size,
          hipMemcpyDeviceToDevice,
          stream));
    } else {
      assert(false && "unsupport datatype in layernorm");
    }
  }

  if (m->input_type[0] == DT_FLOAT) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       input1.domain.get_volume(),
                       input1.get_float_ptr(),
                       input2.get_float_ptr(),
                       output.get_float_ptr());
  } else if (m->input_type[0] == DT_HALF) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       input1.domain.get_volume(),
                       input1.get_half_ptr(),
                       input2.get_half_ptr(),
                       output.get_half_ptr());
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }

  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("[SigmoidSiluMulti] forward time (CF) = %.9fms\n", elapsed);
  }
}

/*static*/
void SigmoidSiluMulti::backward_kernel_wrapper(
    SigmoidSiluMultiMeta const *m,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input1,
    GenericTensorAccessorR const &input2,
    GenericTensorAccessorW const &input1_grad,
    GenericTensorAccessorW const &input2_grad) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  int num_elements = output_grad.domain.get_volume();
  assert(input1.domain.get_volume() == num_elements);
  assert(input2.domain.get_volume() == num_elements);
  assert(input1_grad.domain.get_volume() == num_elements);
  assert(input2_grad.domain.get_volume() == num_elements);

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  if (m->input_type[0] == DT_FLOAT) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiBackwardKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       output_grad.domain.get_volume(),
                       output_grad.get_float_ptr(),
                       input1.get_float_ptr(),
                       input2.get_float_ptr(),
                       input1_grad.get_float_ptr(),
                       input2_grad.get_float_ptr(),
                       m->reset_input_grads[0],
                       m->reset_input_grads[1]);
  } else if (m->input_type[0] == DT_HALF) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiBackwardKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       output_grad.domain.get_volume(),
                       output_grad.get_half_ptr(),
                       input1.get_half_ptr(),
                       input2.get_half_ptr(),
                       input1_grad.get_half_ptr(),
                       input2_grad.get_half_ptr(),
                       m->reset_input_grads[0],
                       m->reset_input_grads[1]);
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("[SigmoidSiluMulti] backward time (CF) = %.9fms\n", elapsed);
  }
}

/*static*/
void SigmoidSiluMulti::peft_bwd_kernel_wrapper(
    SigmoidSiluMultiMeta const *m,
    BatchConfig const *bc,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorW const &input1_grad,
    GenericTensorAccessorW const &input2_grad) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  assert(input1_grad.domain.get_volume() == output_grad.domain.get_volume());
  assert(input2_grad.domain.get_volume() == input1_grad.domain.get_volume());

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
  }

  assert(
      bc->peft_bwd_applies_to_this_layer(m->layer_guid.transformer_layer_id));
  int i = bc->finetuning_request_index();

  int num_peft_tokens = bc->requestsInfo[i].num_tokens_in_batch;
  int in_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  int num_elements = in_dim * num_peft_tokens;

  if (m->input_type[0] == DT_FLOAT) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiBackwardKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       num_elements,
                       output_grad.get_float_ptr(),
                       static_cast<float const *>(m->input_activation),
                       static_cast<float const *>(m->input_activation) +
                           num_peft_tokens * in_dim,
                       input1_grad.get_float_ptr(),
                       input2_grad.get_float_ptr(),
                       m->reset_input_grads[0],
                       m->reset_input_grads[1]);
  } else if (m->input_type[0] == DT_HALF) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(SigmoidSiluMultiBackwardKernel),
                       GET_BLOCKS(num_elements),
                       min(CUDA_NUM_THREADS, num_elements),
                       0,
                       stream,
                       num_elements,
                       output_grad.get_half_ptr(),
                       static_cast<half const *>(m->input_activation),
                       static_cast<half const *>(m->input_activation) +
                           num_peft_tokens * in_dim,
                       input1_grad.get_half_ptr(),
                       input2_grad.get_half_ptr(),
                       m->reset_input_grads[0],
                       m->reset_input_grads[1]);
  } else {
    assert(false && "unsupport datatype in SigmoidSiluMulti");
  }
  if (m->profiling) {
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
    printf("[SigmoidSiluMulti] peft_bwd time (CF) = %.9fms\n", elapsed);
  }
}

}; // namespace FlexFlow
