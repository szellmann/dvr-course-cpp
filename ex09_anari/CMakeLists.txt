# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(ex04_anari LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(anari REQUIRED)
#add_library(${PROJECT_NAME}_device)

add_executable(${PROJECT_NAME}_app)
target_sources(${PROJECT_NAME}_app PRIVATE app/hostCode.cpp)
target_link_libraries(${PROJECT_NAME}_app PRIVATE dvr_course-common anari::anari)
