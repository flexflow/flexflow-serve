#ifndef _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
#define _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H

#include "flexflow/ffconst.h"
#include "flexflow/fftype.h"
#include "flexflow/parallel_tensor.h"

namespace FlexFlow {

struct SpecIncMultiHeadSelfAttentionParams {
  LayerID layer_guid;
  int embed_dim, num_q_heads, num_kv_heads, kdim, vdim,
      tensor_parallelism_degree;
  float dropout, scaling_factor;
  bool qkv_bias, final_bias, add_zero_attn, scaling_query, qk_prod_scaling,
      position_bias;
  RotaryEmbeddingMeta rotary_embedding_meta;
  bool streaming_cache;
  char name[MAX_OPNAME];
  bool is_valid(ParallelTensorShape const &) const;
};

bool operator==(SpecIncMultiHeadSelfAttentionParams const &,
                SpecIncMultiHeadSelfAttentionParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SpecIncMultiHeadSelfAttentionParams> {
  size_t
      operator()(FlexFlow::SpecIncMultiHeadSelfAttentionParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SPEC_INC_MULTIHEAD_SELF_ATTENTION_PARAMS_H
