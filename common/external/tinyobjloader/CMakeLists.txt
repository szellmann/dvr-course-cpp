# Copyright 2025-2026 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

project(tinyobjloader)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_LIST_DIR})


