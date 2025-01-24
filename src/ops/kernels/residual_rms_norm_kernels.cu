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

#include "flashinfer/utils.cuh"
#include <numeric>

#include "flashinfer/math.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/residual_rms_norm_kernels.h"
#include "flexflow/ops/residual_rms_norm.h"
#include "flexflow/utils/cuda_helper.h"
#include <cublas_v2.h>

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;

#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDANumThreads = 256;

ResidualRMSNormMeta::ResidualRMSNormMeta(FFHandler handler,
                                         ResidualRMSNorm const *rms,
                                         MemoryAllocator &gpu_mem_allocator)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  alpha = 1.0f;
  beta = 0.0f;

  in_dim = rms->data_dim;
  batch_size = rms->effective_batch_size;
  num_elements = in_dim * batch_size;

  DataType data_type = rms->weights[0]->data_type;
  size_t rms_ptr_size = batch_size;
  size_t norm_ptr_size = num_elements;
  size_t totalSize = (rms_ptr_size + norm_ptr_size) * data_type_size(data_type);
  gpu_mem_allocator.create_legion_instance(
      reserveInst, totalSize, "ResidualRMSNormMeta");
  rms_ptr = gpu_mem_allocator.allocate_instance_untyped(
      rms_ptr_size * data_type_size(data_type));
  norm_ptr = gpu_mem_allocator.allocate_instance_untyped(
      norm_ptr_size * data_type_size(data_type));
}
ResidualRMSNormMeta::~ResidualRMSNormMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

// Adopted from flashinfer
// (https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/norm.cuh)
// Main modification is for non-inplace computation
template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(T const *__restrict__ input,
                                      T const *__restrict__ residual,
                                      T const *__restrict__ weight,
                                      T *__restrict__ output,
                                      T *__restrict__ residual_output,
                                      const uint32_t d,
                                      float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> input_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_output_vec;
    input_vec.fill(0);
    residual_vec.fill(0);
    residual_output_vec.fill(0);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * d + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * d + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_output_vec[j] = (T)x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_output_vec.store(residual_output + bx * d +
                                i * num_threads * VEC_SIZE +
                                thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += flashinfer::math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = flashinfer::math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    flashinfer::vec_t<T, VEC_SIZE> weight_vec;
    flashinfer::vec_t<T, VEC_SIZE> residual_output_vec;
    flashinfer::vec_t<T, VEC_SIZE> output_vec;
    weight_vec.fill(0);
    residual_output_vec.fill(0);
    output_vec.fill(0);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
      residual_output_vec.load(residual_output + bx * d +
                               i * num_threads * VEC_SIZE +
                               thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] =
          float(residual_output_vec[j]) * rms_rcp * float(weight_vec[j]);
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * d + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

template <typename T>
cudaError_t FusedAddRMSNorm(T const *input,
                            T const *residual,
                            T const *weight,
                            T *output,
                            T *residual_output,
                            uint32_t batch_size,
                            uint32_t d,
                            float eps = 1e-5,
                            cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = flashinfer::ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  void *args[] = {
      &input, &residual, &weight, &output, &residual_output, &d, &eps};

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
    FLASHINFER_CUDA_CALL(cudaLaunchKernel(
        (void *)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return cudaSuccess;
}

namespace Kernels {
namespace ResidualRMSNorm {
template <typename T>
void forward_kernel(ResidualRMSNormMeta const *m,
                    T const *input1_ptr,
                    T const *input2_ptr,
                    T const *weight_ptr,
                    T *residual_output_ptr,
                    T *output_ptr,
                    int batch_size,
                    cudaStream_t stream) {
  assert(batch_size <= m->batch_size);
  // use active batch size
  std::pair<int, int> kernel1_parallelism =
      std::make_pair(batch_size, kCUDABlockReduceNumThreads);
  std::pair<int, int> kernel2_parallelism =
      std::make_pair(batch_size, kCUDANumThreads);

  int num_blocks =
      std::max(kernel1_parallelism.first, kernel2_parallelism.first);
  int num_threads =
      std::max(kernel1_parallelism.second, kernel2_parallelism.second);

  checkCUDA(FusedAddRMSNorm<T>(input1_ptr,
                               input2_ptr,
                               weight_ptr,
                               output_ptr,
                               residual_output_ptr,
                               batch_size,
                               m->in_dim,
                               m->eps,
                               stream));
}

void forward_kernel_wrapper(ResidualRMSNormMeta const *m,
                            GenericTensorAccessorR const &input1,
                            GenericTensorAccessorR const &input2,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &residual_output,
                            GenericTensorAccessorW const &output,
                            int batch_size) {
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
  if (output.data_type == DT_HALF) {
    forward_kernel(m,
                   input1.get_half_ptr(),
                   input2.get_half_ptr(),
                   weight.get_half_ptr(),
                   residual_output.get_half_ptr(),
                   output.get_half_ptr(),
                   batch_size,
                   stream);
  } else if (output.data_type == DT_FLOAT) {
    forward_kernel(m,
                   input1.get_float_ptr(),
                   input2.get_float_ptr(),
                   weight.get_float_ptr(),
                   residual_output.get_float_ptr(),
                   output.get_float_ptr(),
                   batch_size,
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
    printf("[ResidualRMSNorm] forward time (CF) = %.2fms\n", elapsed);
  }
}

} // namespace ResidualRMSNorm
} // namespace Kernels
} // namespace FlexFlow
