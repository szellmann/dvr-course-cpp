# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(raw2vdb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVDB REQUIRED COMPONENTS openvdb)

add_executable(${PROJECT_NAME} raw2vdb.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  dvr_course-common
  OpenVDB::openvdb
)
