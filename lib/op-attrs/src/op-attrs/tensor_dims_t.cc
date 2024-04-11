// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/tensor_dims_t.struct.toml

#include "op-attrs/tensor_dims_t.h"

namespace FlexFlow {
TensorDims::TensorDims(::FlexFlow::FFOrdered<size_t> const &ff_ordered)
    : ff_ordered(ff_ordered) {}
bool TensorDims::operator==(TensorDims const &other) const {
  return std::tie(this->ff_ordered) == std::tie(other.ff_ordered);
}
bool TensorDims::operator!=(TensorDims const &other) const {
  return std::tie(this->ff_ordered) != std::tie(other.ff_ordered);
}
bool TensorDims::operator<(TensorDims const &other) const {
  return std::tie(this->ff_ordered) < std::tie(other.ff_ordered);
}
bool TensorDims::operator>(TensorDims const &other) const {
  return std::tie(this->ff_ordered) > std::tie(other.ff_ordered);
}
bool TensorDims::operator<=(TensorDims const &other) const {
  return std::tie(this->ff_ordered) <= std::tie(other.ff_ordered);
}
bool TensorDims::operator>=(TensorDims const &other) const {
  return std::tie(this->ff_ordered) >= std::tie(other.ff_ordered);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::TensorDims>::operator()(
    FlexFlow::TensorDims const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::FFOrdered<size_t>>{}(x.ff_ordered) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::TensorDims
    adl_serializer<FlexFlow::TensorDims>::from_json(json const &j) {
  return {j.at("ff_ordered").template get<::FlexFlow::FFOrdered<size_t>>()};
}
void adl_serializer<FlexFlow::TensorDims>::to_json(
    json &j, FlexFlow::TensorDims const &v) {
  j["__type"] = "TensorDims";
  j["ff_ordered"] = v.ff_ordered;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(TensorDims const &x) {
  std::ostringstream oss;
  oss << "<TensorDims";
  oss << " ff_ordered=" << x.ff_ordered;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, TensorDims const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
