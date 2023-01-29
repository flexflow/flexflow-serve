#ifndef _FLEXFLOW_SOFTMAX_PARAMS_H
#define _FLEXFLOW_SOFTMAX_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct SoftmaxParams {
public:
  bool is_valid(ParallelTensorShape const &) const;

  using AsConstTuple = std::tuple<int>;
  AsConstTuple as_tuple() const;
public:
  int dim;
};
bool operator==(SoftmaxParams const &, SoftmaxParams const &);
bool operator<(SoftmaxParams const &, SoftmaxParams const &);

} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::SoftmaxParams> {
  size_t operator()(FlexFlow::SoftmaxParams const &) const;
};
} // namespace std

#endif // _FLEXFLOW_SOFTMAX_PARAMS_H
