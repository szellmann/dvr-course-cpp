# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(dvr_course-common LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE fb.cu pipeline.cu)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE external)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
configure_cuda_src(fb.cu)
configure_cuda_src(pipeline.cu)
