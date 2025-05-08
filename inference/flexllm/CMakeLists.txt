cmake_minimum_required(VERSION 3.10)

project(FlexFlow_PeftTrain)

# Normal PEFT
set(project_target1 peft_train_unwrapped)
set(CPU_SRC1
  ${FLEXFLOW_CPP_DRV_SRC}
  peft_train.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC1} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target1} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target1} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target1} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target1} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target1} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})


set(TARGET_PATH "${project_target1}")

# Configure the normal execution wrapper.
# Here, LAUNCHER is simply "exec" so that it runs the executable normally.
set(LAUNCHER "exec")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../inference_wrapper.in"
  "${CMAKE_CURRENT_BINARY_DIR}/peft_train"
  @ONLY
)

file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/peft_train"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
     GROUP_READ GROUP_EXECUTE
     WORLD_READ WORLD_EXECUTE
)

# Configure the debugging launcher wrapper.
# Here, LAUNCHER is set to "gdb --args" so that it runs under gdb.
set(LAUNCHER "gdb -ex run --args")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../inference_wrapper.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gdb_peft_train"
  @ONLY
)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/gdb_peft_train"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
     GROUP_READ GROUP_EXECUTE
     WORLD_READ WORLD_EXECUTE
)


set(BIN_DEST "bin")
install(TARGETS ${project_target1} DESTINATION ${BIN_DEST})
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/peft_train" DESTINATION ${BIN_DEST})
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/gdb_peft_train" DESTINATION ${BIN_DEST})




# Overhead test
set(project_target2 overhead_test_unwrapped)
set(CPU_SRC2
  ${FLEXFLOW_CPP_DRV_SRC}
  overhead_test.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target2} ${CPU_SRC2})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target2} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC2} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target2} ${CPU_SRC2})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target2} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target2} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target2} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target2} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target2} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target2} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})


set(TARGET_PATH "${project_target2}")

# Configure the normal execution wrapper.
# Here, LAUNCHER is simply "exec" so that it runs the executable normally.
set(LAUNCHER "exec")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../inference_wrapper.in"
  "${CMAKE_CURRENT_BINARY_DIR}/overhead_test"
  @ONLY
)

file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/overhead_test"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
     GROUP_READ GROUP_EXECUTE
     WORLD_READ WORLD_EXECUTE
)

# Configure the debugging launcher wrapper.
# Here, LAUNCHER is set to "gdb --args" so that it runs under gdb.
set(LAUNCHER "gdb -ex run --args")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../inference_wrapper.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gdb_overhead_test"
  @ONLY
)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/gdb_overhead_test"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
     GROUP_READ GROUP_EXECUTE
     WORLD_READ WORLD_EXECUTE
)


set(BIN_DEST "bin")
install(TARGETS ${project_target2} DESTINATION ${BIN_DEST})
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/overhead_test" DESTINATION ${BIN_DEST})
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/gdb_overhead_test" DESTINATION ${BIN_DEST})
