// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/parallel_dim_t.struct.toml

#include "op-attrs/parallel_dim_t.h"

namespace FlexFlow {
ParallelDim::ParallelDim(size_t const &size,
                         int const &degree,
                         bool const &is_replica_dim)
    : size(size), degree(degree), is_replica_dim(is_replica_dim) {}
bool ParallelDim::operator==(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) ==
         std::tie(other.size, other.degree, other.is_replica_dim);
}
bool ParallelDim::operator!=(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) !=
         std::tie(other.size, other.degree, other.is_replica_dim);
}
bool ParallelDim::operator<(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) <
         std::tie(other.size, other.degree, other.is_replica_dim);
}
bool ParallelDim::operator>(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) >
         std::tie(other.size, other.degree, other.is_replica_dim);
}
bool ParallelDim::operator<=(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) <=
         std::tie(other.size, other.degree, other.is_replica_dim);
}
bool ParallelDim::operator>=(ParallelDim const &other) const {
  return std::tie(this->size, this->degree, this->is_replica_dim) >=
         std::tie(other.size, other.degree, other.is_replica_dim);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ParallelDim>::operator()(
    FlexFlow::ParallelDim const &x) const {
  size_t result = 0;
  result ^=
      std::hash<size_t>{}(x.size) + 0x9e3779b9 + (result << 6) + (result >> 2);
  result ^=
      std::hash<int>{}(x.degree) + 0x9e3779b9 + (result << 6) + (result >> 2);
  result ^= std::hash<bool>{}(x.is_replica_dim) + 0x9e3779b9 + (result << 6) +
            (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::ParallelDim
    adl_serializer<FlexFlow::ParallelDim>::from_json(json const &j) {
  return {j.at("size").template get<size_t>(),
          j.at("degree").template get<int>(),
          j.at("is_replica_dim").template get<bool>()};
}
void adl_serializer<FlexFlow::ParallelDim>::to_json(
    json &j, FlexFlow::ParallelDim const &v) {
  j["__type"] = "ParallelDim";
  j["size"] = v.size;
  j["degree"] = v.degree;
  j["is_replica_dim"] = v.is_replica_dim;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::ParallelDim> Arbitrary<FlexFlow::ParallelDim>::arbitrary() {
  return gen::construct<FlexFlow::ParallelDim>(
      gen::arbitrary<size_t>(), gen::arbitrary<int>(), gen::arbitrary<bool>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(ParallelDim const &x) {
  std::ostringstream oss;
  oss << "<ParallelDim";
  oss << " size=" << x.size;
  oss << " degree=" << x.degree;
  oss << " is_replica_dim=" << x.is_replica_dim;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ParallelDim const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
