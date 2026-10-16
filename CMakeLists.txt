cmake_minimum_required(VERSION 3.18...3.27)
# CUDA architectures: override with the CUDAARCHS environment variable or -DCMAKE_CUDA_ARCHITECTURES (e.g. "native" or
# "120" for a quick local build). The default embeds SASS and PTX for A100=80, RTX 40xx=89, H100=90, RTX 50xx=120.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 80;89;90;120)
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)
message(STATUS "Project name: ${SKBUILD_PROJECT_NAME}")

# Find Python
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Find FFI include directory
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Development builds compile only a few (k, n_dim) template instantiations, for much faster compilation
option(GRAPHGP_CUDA_DEV_BUILD "Compile only k <= 16 and n_dim in {2, 3}" OFF)

# Add the shared library target
add_library(${SKBUILD_PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME}/xla_bindings.cu)
target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME} ${XLA_DIR})
set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(GRAPHGP_CUDA_DEV_BUILD)
  target_compile_definitions(${SKBUILD_PROJECT_NAME} PRIVATE GRAPHGP_CUDA_DEV_BUILD)
endif()
install(TARGETS ${SKBUILD_PROJECT_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
