# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE_INIT Release)

project(dvr-course-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(common/cmake/common.cmake)

add_subdirectory(common)
add_subdirectory(ex00_hello_dvr_course)
add_subdirectory(ex01_let_there_be_voxels)
add_subdirectory(ex02_woodcock)
add_subdirectory(ex03_multi_volume)
add_subdirectory(ex04_lights_on)
add_subdirectory(ex05_tets_n_friends)
add_subdirectory(ex06_need_for_speed)
add_subdirectory(ex07_render_graph)
add_subdirectory(ex09_anari)

# cmdline tools used for data wrangling, etc.
# we don't usually need these to run the examples
option(BUILD_CMDLINE_TOOLS "Build with cmdline tools" OFF)
if(BUILD_CMDLINE_TOOLS)
  add_subdirectory(tools/vtu2bin)
  add_subdirectory(tools/vtu2vdb)
endif()
