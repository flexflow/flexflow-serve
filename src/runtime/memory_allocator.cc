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

#include "flexflow/utils/memory_allocator.h"
#include "flexflow/mapper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;
using Realm::RegionInstance;
using namespace Legion;
using namespace Mapping;

Legion::Logger log_ff_mem_allocator("MemoryAllocator");

MemoryAllocator::MemoryAllocator(Memory _memory)
    : memory(_memory), reserved_ptr(nullptr), instance_ptr(nullptr),
      reserved_total_size(0), reserved_allocated_size(0),
      instance_total_size(0), instance_allocated_size(0),
      log_instance_creation(false) {
  InputArgs const &command_args = HighLevelRuntime::get_input_args();
  char **argv = command_args.argv;
  int argc = command_args.argc;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--log-instance-creation")) {
      log_instance_creation = true;
      break;
    }
  }
}

void MemoryAllocator::create_legion_instance(RegionInstance &inst,
                                             size_t size,
                                             char const *task_name) {
  // Assert that we have used up previously created region instance
  assert(instance_total_size == instance_allocated_size);
  Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                 Realm::Point<1, coord_t>(size - 1));
  std::vector<size_t> field_sizes;
  field_sizes.push_back(sizeof(char));
  Realm::RegionInstance::create_instance(
      inst, memory, bounds, field_sizes, 0, Realm::ProfilingRequestSet())
      .wait();
  if (log_instance_creation) {
    log_ff_mem_allocator.print(
        "Created instance in memory_kind: %s memory_id: %llx size: %zu "
        "(capacity %lu) task_name: %s",
        Legion::Mapping::Utilities::to_string(memory.kind()),
        memory.id,
        size,
        memory.capacity(),
        ((task_name != NULL) ? task_name : "unknown"));
  }
  instance_ptr = inst.pointer_untyped(0, 0);
  instance_total_size = size;
  instance_allocated_size = 0;
}

void MemoryAllocator::register_reserved_work_space(void *base, size_t size) {
  // Assert that we haven't allocated anything before
  assert(reserved_total_size == 0);
  reserved_ptr = base;
  reserved_total_size = size;
  reserved_allocated_size = 0;
}

}; // namespace FlexFlow
