#ifndef _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class EmbeddingMeta : public OpMeta {
public:
  EmbeddingMeta(FFHandler handle, Op const *op);
  DataType input_data_type;
  AggrMode aggr;
};

namespace Kernels {
namespace Embedding {
void inference_kernel_wrapper(EmbeddingMeta const *m,
                              BatchConfig const *bc,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight,
                            int in_dim,
                            int out_dim);
void backward_kernel_wrapper(EmbeddingMeta const *m,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorW const &weight_grad,
                             int in_dim,
                             int out_dim,
                             int batch_size);

namespace Internal {
template <typename TI, typename TD>
void forward_kernel(EmbeddingMeta const *m,
                    BatchConfig const *bc,
                    TI const *input_ptr,
                    TD *output_ptr,
                    TD const *weight_ptr,
                    int in_dim,
                    int out_dim,
                    // int batch_size,
                    // AggrMode aggr,
                    // int outputSize,
                    cudaStream_t stream);
template <typename TI, typename TD>
void forward_kernel_spatial_sharing(EmbeddingMeta const *m,
                                    BatchConfig const *bc,
                                    TI const *input_ptr,
                                    TD *output_ptr,
                                    TD const *weight_ptr,
                                    int in_dim,
                                    int out_dim,
                                    cudaStream_t main_stream);
} // namespace Internal
} // namespace Embedding
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H