cmake_minimum_required(VERSION 3.10)
project(assignment1)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()


# Project sources
add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../ext/eigen")


# Use C++11 version of the standard
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

# Place the output binary at the root of the build folder
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Folder where data files are stored (meshes & stuff)
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
target_compile_definitions(${PROJECT_NAME} PUBLIC -DDATA_DIR=\"${DATA_DIR}\")