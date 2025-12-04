cmake_minimum_required(VERSION 3.18...3.27)
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

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80 89 90) # A100=80 RTX4050=89 H100=90

# Add the shared library target
add_library(${SKBUILD_PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME}/xla_bindings.cu)
target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME} ${XLA_DIR})
set_target_properties(${SKBUILD_PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS ${SKBUILD_PROJECT_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
