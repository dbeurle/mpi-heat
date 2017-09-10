
cmake_minimum_required(VERSION 3.8)

project(mpi-heat CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
endif()

add_executable(heat-equation hybrid.cpp)

target_link_libraries(heat-equation ${MPI_LIBRARIES})
