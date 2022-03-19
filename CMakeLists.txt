cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(WLOP VERSION 1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -pthread -fopenmp")

add_compile_options(-std=c++14)
add_compile_options(-fpermissive)
add_compile_options(-fPIC)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Setup project directories
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE( OpenMP REQUIRED)
if(OpenMP_CXX_FOUND) 
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
	message("Missed OpenMP!")
endif()

find_package(Eigen3 QUIET)
include_directories(${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES("./include") 

add_executable(main.exe ./test/main.cpp ./test/KDTreeFlann.cpp)
# add_executable(main.exe ./test/main.cpp)
target_link_libraries(main.exe OpenMP::OpenMP_CXX)   