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

#include "flexflow/ops/kernels/gemm_impl.h"
#include "flexflow/utils/cuda_helper.h"
#include <memory>

namespace Internal {

GemmEngine::GemmEngine(cublasHandle_t blas_,
                       cublasLtHandle_t blasLt_,
                       cudaDeviceProp *device_prop_,
                       size_t workspace_size_) {
  blas = blas_;
  blasLt = blasLt_;
  if (device_prop_ == nullptr) {
    device_prop = new cudaDeviceProp;
  } else {
    device_prop = device_prop_;
  }
  workspace_size = workspace_size_;
  workspace = nullptr;
}

void GemmEngine::assign_workspace(void *workspace_, size_t workspace_size_) {
  assert(workspace_size_ >= workspace_size);
  workspace = workspace_;
}

template <typename Dtype>
void GemmEngine::gemm_internal(cublasOperation_t transa,
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
                               cudaStream_t stream) {
  static_assert(false && sizeof(Dtype), "gemm_internal: not implemented");
}

#ifdef USE_CUBLASLT
/* Implementations for gemm_internal_cublaslt */
template <typename T, cublasStatus_t (*destructor)(T *)>
struct CuBlasLtDeleter {
  void operator()(T *x) {
    if (x != nullptr) {
      checkCUDA(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T *)>
class CuBlasLtDescriptor {
public:
  T *descriptor() const {
    return descriptor_.get();
  }
  T *descriptor() {
    return descriptor_.get();
  }

protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor
    : public CuBlasLtDescriptor<cublasLtMatmulDescOpaque_t,
                                &cublasLtMatmulDescDestroy> {
public:
  CuBlasLtMatmulDescriptor(cublasComputeType_t compute_type,
                           cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    checkCUDA(
        cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    checkCUDA(::cublasLtMatmulDescSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatrixLayout
    : public CuBlasLtDescriptor<cublasLtMatrixLayoutOpaque_t,
                                &cublasLtMatrixLayoutDestroy> {
public:
  CuBlasLtMatrixLayout(cudaDataType_t type,
                       uint64_t rows,
                       uint64_t cols,
                       int64_t ld,
                       bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    checkCUDA(cublasLtMatrixLayoutCreate(
        &raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatrixLayoutAttribute_t attr,
                           const T value) {
    checkCUDA(::cublasLtMatrixLayoutSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatmulPreference
    : public CuBlasLtDescriptor<cublasLtMatmulPreferenceOpaque_t,
                                &cublasLtMatmulPreferenceDestroy> {
public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    checkCUDA(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulPreferenceAttributes_t attr,
                           const T value) {
    checkCUDA(::cublasLtMatmulPreferenceSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

inline uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}

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
                                   cudaStream_t stream) {
  assert(workspace != nullptr && "workspace must be provided.");
  cudaDataType_t abcType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  if constexpr (std::is_same_v<Dtype, double>) {
    abcType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_R_64F;
  } else if constexpr (std::is_same_v<Dtype, float>) {
    computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
  } else if constexpr (std::is_same_v<Dtype, half>) {
    abcType = CUDA_R_16F;
    computeType = CUBLAS_COMPUTE_16F;
  } else {
    static_assert(false && sizeof(Dtype),
                  "bgemm_internal_cublaslt: not implemented");
  }

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  CuBlasLtMatrixLayout Adesc(abcType, m, k, lda, transa == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Bdesc(abcType, k, n, ldb, transb == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Cdesc(abcType, m, n, ldc);

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                          workspace_size);

  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(c));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                          a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                          b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                          c_alignment);

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  checkCUDA(cublasLtMatmulAlgoGetHeuristic(handle,
                                           computeDesc.descriptor(),
                                           Adesc.descriptor(),
                                           Bdesc.descriptor(),
                                           Cdesc.descriptor(),
                                           Cdesc.descriptor(),
                                           preference.descriptor(),
                                           1,
                                           &heuristicResult,
                                           &returnedResult));
  if (returnedResult == 0) {
    assert(false && "cuBLASLt failed to find a valid algorithm.");
  }

  checkCUDA(cublasLtMatmul(handle,
                           computeDesc.descriptor(),
                           &alpha,
                           a,
                           Adesc.descriptor(),
                           b,
                           Bdesc.descriptor(),
                           &beta,
                           c,
                           Cdesc.descriptor(),
                           c,
                           Cdesc.descriptor(),
                           &heuristicResult.algo,
                           workspace,
                           workspace_size,
                           stream));
}
#else
/* Implementations for gemm_internal_cublas */
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
                                 cudaStream_t stream) {
  static_assert(false && sizeof(Dtype),
                "gemm_internal_cublas: not implemented");
}

template <>
void gemm_internal_cublas<double>(cublasHandle_t handle,
                                  cudaDeviceProp *prop,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int64_t m,
                                  int64_t n,
                                  int64_t k,
                                  double alpha,
                                  double const *a,
                                  int64_t lda,
                                  double const *b,
                                  int64_t ldb,
                                  double beta,
                                  double *c,
                                  int64_t ldc,
                                  cudaStream_t stream) {
  checkCUDA(cublasDgemm(
      handle, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<float>(cublasHandle_t handle,
                                 cudaDeviceProp *prop,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 float alpha,
                                 float const *a,
                                 int64_t lda,
                                 float const *b,
                                 int64_t ldb,
                                 float beta,
                                 float *c,
                                 int64_t ldc,
                                 cudaStream_t stream) {
  checkCUDA(cublasSgemm(
      handle, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
  // checkCUDA(cublasGemmEx(
  //     handle,
  //     transa,
  //     transb,
  //     m,
  //     n,
  //     k,
  //     &alpha,
  //     a,
  //     CUDA_R_32F,
  //     lda,
  //     b,
  //     CUDA_R_32F,
  //     ldb,
  //     &beta,
  //     c,
  //     CUDA_R_32F,
  //     ldc,
  //     CUBLAS_COMPUTE_32F_FAST_16F,
  //     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void gemm_internal_cublas<half>(cublasHandle_t handle,
                                cudaDeviceProp *prop,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                half alpha,
                                half const *a,
                                int64_t lda,
                                half const *b,
                                int64_t ldb,
                                half beta,
                                half *c,
                                int64_t ldc,
                                cudaStream_t stream) {
  if (prop->major >= 5) {
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    // cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    // if (!at::globalContext().allowFP16ReductionCuBLAS()) {
    //   cublas_flags = static_cast<cublasMath_t>(cublas_flags |
    //   CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    // }
    // checkCUDA(cublasSetMathMode(handle, cublas_flags));
    checkCUDA(cublasGemmEx(handle,
                           transa,
                           transb,
                           m,
                           n,
                           k,
                           &alpha,
                           a,
                           CUDA_R_16F,
                           lda,
                           b,
                           CUDA_R_16F,
                           ldb,
                           &beta,
                           c,
                           CUDA_R_16F,
                           ldc,
                           CUBLAS_COMPUTE_16F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // checkCUDA(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    float falpha = alpha;
    float fbeta = beta;
    checkCUDA(cublasSgemmEx(handle,
                            transa,
                            transb,
                            m,
                            n,
                            k,
                            &falpha,
                            a,
                            CUDA_R_16F,
                            lda,
                            b,
                            CUDA_R_16F,
                            ldb,
                            &fbeta,
                            c,
                            CUDA_R_16F,
                            ldc));
  }
}
#endif

template <>
void GemmEngine::gemm_internal(cublasOperation_t transa,
                               cublasOperation_t transb,
                               int64_t m,
                               int64_t n,
                               int64_t k,
                               double alpha,
                               double const *a,
                               int64_t lda,
                               double const *b,
                               int64_t ldb,
                               double beta,
                               double *c,
                               int64_t ldc,
                               cudaStream_t stream) {
#ifdef USE_CUBLASLT
  gemm_internal_cublaslt(blasLt,
                         device_prop,
                         workspace,
                         workspace_size,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         a,
                         lda,
                         b,
                         ldb,
                         beta,
                         c,
                         ldc,
                         stream);
#else
  gemm_internal_cublas(blas,
                       device_prop,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       alpha,
                       a,
                       lda,
                       b,
                       ldb,
                       beta,
                       c,
                       ldc,
                       stream);
#endif
}

template <>
void GemmEngine::gemm_internal(cublasOperation_t transa,
                               cublasOperation_t transb,
                               int64_t m,
                               int64_t n,
                               int64_t k,
                               float alpha,
                               float const *a,
                               int64_t lda,
                               float const *b,
                               int64_t ldb,
                               float beta,
                               float *c,
                               int64_t ldc,
                               cudaStream_t stream) {
#ifdef USE_CUBLASLT
  gemm_internal_cublaslt(blasLt,
                         device_prop,
                         workspace,
                         workspace_size,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         a,
                         lda,
                         b,
                         ldb,
                         beta,
                         c,
                         ldc,
                         stream);
#else
  gemm_internal_cublas(blas,
                       device_prop,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       alpha,
                       a,
                       lda,
                       b,
                       ldb,
                       beta,
                       c,
                       ldc,
                       stream);
#endif
}

template <>
void GemmEngine::gemm_internal(cublasOperation_t transa,
                               cublasOperation_t transb,
                               int64_t m,
                               int64_t n,
                               int64_t k,
                               half alpha,
                               half const *a,
                               int64_t lda,
                               half const *b,
                               int64_t ldb,
                               half beta,
                               half *c,
                               int64_t ldc,
                               cudaStream_t stream) {
#ifdef USE_CUBLASLT
  gemm_internal_cublaslt(blasLt,
                         device_prop,
                         workspace,
                         workspace_size,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         a,
                         lda,
                         b,
                         ldb,
                         beta,
                         c,
                         ldc,
                         stream);
#else
  gemm_internal_cublas(blas,
                       device_prop,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       alpha,
                       a,
                       lda,
                       b,
                       ldb,
                       beta,
                       c,
                       ldc,
                       stream);
#endif
}
} // namespace Internal
