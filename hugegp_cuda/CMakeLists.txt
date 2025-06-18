cmake_minimum_required(VERSION 3.18...3.27)
project(hugegp_cuda LANGUAGES CXX CUDA)
message(STATUS "Project name: hugegp_cuda")

# Find Python
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Find FFI include directory
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# Add the shared library target
add_library(hugegp_cuda SHARED bindings.cu)
target_include_directories(hugegp_cuda PRIVATE ${XLA_DIR})
set_target_properties(hugegp_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)