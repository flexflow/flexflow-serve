#ifndef GEMM_IMPL_H
#define GEMM_IMPL_H

#include <cublasLt.h>
#include <cublas_v2.h>

namespace Internal {

/* TODO: Consider appropriate case to use Lt */
// #if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11040))
//     // Strangely, if mat2 has only 1 row or column, we get
//     // CUBLAS_STATUS_INVALID_VALUE error from cublasLtMatmulAlgoGetHeuristic.
//     // self.dim() == 1 && result.dim() == 2 && self.sizes()[0] ==
//     mat2_sizes[1]
//     // is to use lt interface only when self is bias.
//     // for cuda 11.4, cublasLtMatmul is activated
//     // the last two conditions is to skip 16b transA and non-trans-B having
//     // leading dim >> rows when they are sliced from a large tensor
//     // see
//     fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul if
//     (!disable_addmm_cuda_lt) {
//       useLtInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
//           result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
//           self.is_contiguous() && result.is_contiguous() &&
//           (scalar_type == at::ScalarType::Double ||
//            scalar_type == at::ScalarType::Float ||
//            scalar_type == at::ScalarType::Half ||
//            scalar_type == at::ScalarType::BFloat16) &&
// #if (defined(CUDA_VERSION) && CUDA_VERSION >= 12010)
//           mat2_sizes[0] > 1 && mat2_sizes[1] > 1;
// #else
//           mat2_sizes[0] > 1 && mat2_sizes[1] > 1 &&
//           mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32 &&
//           mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32 &&
//           // avoid leading dim >> rows bugs
//           ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
//            (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
//            (scalar_type != at::ScalarType::Half &&
//             scalar_type != at::ScalarType::BFloat16)) &&
//           ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
//            (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
//            (scalar_type != at::ScalarType::Half &&
//             scalar_type != at::ScalarType::BFloat16));
// #endif
//     }
// #endif

#define USE_CUBLASLT

#ifdef USE_CUBLASLT
template <typename Dtype>
inline void gemm_internal_cublaslt(cublasLtHandle_t handle,
                                   cudaDeviceProp *prop,
                                   void *workspace,
                                   size_t workspace_size,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   int64_t m,
                                   int64_t n,
                                   int64_t k,
                                   Dtype alpha,
                                   Dtype const *a,
                                   int64_t lda,
                                   Dtype const *b,
                                   int64_t ldb,
                                   Dtype beta,
                                   Dtype *c,
                                   int64_t ldc,
                                   cudaStream_t stream);
#else
template <typename Dtype>
inline void gemm_internal_cublas(cublasHandle_t handle,
                                 cudaDeviceProp *prop,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 Dtype alpha,
                                 Dtype const *a,
                                 int64_t lda,
                                 Dtype const *b,
                                 int64_t ldb,
                                 Dtype beta,
                                 Dtype *c,
                                 int64_t ldc,
                                 cudaStream_t stream);
#endif

// Wrapper for gemm
// Adopted from pytorch:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp
class GemmEngine {
public:
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // defaultlt setting workspace size to 1M.
  GemmEngine(cublasHandle_t blas_,
             cublasLtHandle_t blasLt_,
             cudaDeviceProp *device_prop_ = nullptr,
             size_t workspace_size_ = 1024 * 1024);
  void assign_workspace(void *workspace_, size_t workspace_size_);

  template <typename Dtype>
  void gemm_internal(cublasOperation_t transa,
                     cublasOperation_t transb,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     Dtype alpha,
                     Dtype const *a,
                     int64_t lda,
                     Dtype const *b,
                     int64_t ldb,
                     Dtype beta,
                     Dtype *c,
                     int64_t ldc,
                     cudaStream_t stream);

public:
  cublasHandle_t blas;
  cublasLtHandle_t blasLt;
  cudaDeviceProp *device_prop;
  size_t workspace_size; // in bytes
  void *workspace;
};

} // namespace Internal

#endif // GEMM_IMPL_H
