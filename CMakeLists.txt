# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(dvr-course-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(common/cmake/common.cmake)

add_subdirectory(common)
add_subdirectory(ex00_hello_dvr_course)
add_subdirectory(ex01_let_there_be_voxels)
add_subdirectory(ex02_woodcock)
add_subdirectory(ex03_multi_volume)
# add_subdirectory(ex04_lights_on)
add_subdirectory(ex05_hey_icon)
