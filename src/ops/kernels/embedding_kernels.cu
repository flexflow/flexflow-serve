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

#include "flexflow/ops/kernels/embedding_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::PhysicalRegion;
using Legion::Rect;
using Legion::Runtime;
using Legion::Task;

namespace Kernels {
namespace Embedding {

/*static*/
void forward_kernel_wrapper(EmbeddingMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight,
                            int in_dim,
                            int out_dim,
                            int batch_size) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  if (input.data_type == DT_INT32) {
    if (weight.data_type == DT_HALF) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_half_ptr(),
                               weight.get_half_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_BFLOAT16) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_bfloat16_ptr(),
                               weight.get_bfloat16_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_FLOAT) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_float_ptr(),
                               weight.get_float_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_DOUBLE) {
      Internal::forward_kernel(input.get_int32_ptr(),
                               output.get_double_ptr(),
                               weight.get_double_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  } else if (input.data_type == DT_INT64) {
    if (weight.data_type == DT_HALF) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_half_ptr(),
                               weight.get_half_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_BFLOAT16) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_bfloat16_ptr(),
                               weight.get_bfloat16_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else if (weight.data_type == DT_DOUBLE) {
      Internal::forward_kernel(input.get_int64_ptr(),
                               output.get_double_ptr(),
                               weight.get_double_ptr(),
                               in_dim,
                               out_dim,
                               batch_size,
                               m->aggr,
                               output.domain.get_volume(),
                               stream);
    } else {
      assert(false && "Unsupported DataType in Embedding");
    }
  } else {
    assert(false && "Unsupported DataType in Embedding");
  }
  if (m->profiling) {
    checkCUDA(cudaDeviceSynchronize());
    // print_tensor<TI>(input_ptr, input_domain.get_volume(),
    // "[Embedding:forward:input]"); print_tensor<float>(kernel_ptr,
    // kernel_domain.get_volume(), "[Embedding:forward:weight]");
    // print_tensor<float>(output_ptr, output_domain.get_volume(),
    // "[Embedding:forward:output]");
  }
}

/*static*/
void backward_kernel_wrapper(EmbeddingMeta const *m,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorW const &weight_grad,
                             int in_dim,
                             int out_dim,
                             int batch_size) {
  assert(false && "No longer supported");
}

namespace Internal {

template <typename TI, typename TD>
__global__ void embed_forward_no_aggr(
    TI const *input, TD *output, TD const *embed, int out_dim, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    TI wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}

template <typename TI, typename TD>
__global__ void embed_forward_with_aggr(TI const *input,
                                        TD *output,
                                        TD const *embed,
                                        int out_dim,
                                        int in_dim,
                                        int batch_size,
                                        AggrMode aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      TI wordIdx = input[idx * in_dim + j];
      output[i] = output[i] + embed[wordIdx * out_dim + off];
      if (aggr == AGGR_MODE_SUM) {
      } else {
        assert(aggr == AGGR_MODE_AVG);
        output[i] = output[i] * scale;
      }
    }
  }
}

/*static*/
template <typename TI, typename TD>
void forward_kernel(TI const *input_ptr,
                    TD *output_ptr,
                    TD const *weight_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size,
                    AggrMode aggr,
                    int outputSize,
                    cudaStream_t stream) {
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  assert(weight_ptr != nullptr);

  if (aggr == AGGR_MODE_NONE) {
    embed_forward_no_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, out_dim, batch_size);
  } else {
    assert(aggr == AGGR_MODE_AVG || aggr == AGGR_MODE_SUM);
    embed_forward_with_aggr<TI, TD>
        <<<GET_BLOCKS(outputSize), CUDA_NUM_THREADS, 0, stream>>>(input_ptr,
                                                                  output_ptr,
                                                                  weight_ptr,
                                                                  out_dim,
                                                                  in_dim,
                                                                  batch_size,
                                                                  aggr);
  }
}

} // namespace Internal
} // namespace Embedding
} // namespace Kernels
}; // namespace FlexFlow
