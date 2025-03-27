cmake_minimum_required(VERSION 3.10)
project(ParallelProgrammingA1)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Add executable and link OpenCL
add_executable(parallel_a1
  src/main.cpp
  src/Clmg.h
  src/Utils.h
)

target_include_directories(parallel_a1 PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(parallel_a1 OpenCL::OpenCL)