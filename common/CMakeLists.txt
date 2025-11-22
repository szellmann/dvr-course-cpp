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

option(DVR_COURSE_PIPELINE_INTERACTIVE "Use interactive viewer instead of png output" OFF)
if (DVR_COURSE_PIPELINE_INTERACTIVE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE INTERACTIVE=1)
  include(FetchContent)

  # SDL3
  find_package(SDL3 QUIET)
  if (NOT TARGET SDL3::SDL3)
    mark_as_advanced(SDL3_DIR)
    set(dvr_course_sdl3_SOURCE ${CMAKE_BINARY_DIR}/deps/source/dvr_course_sdl3)
    FetchContent_Populate(
      NAME dvr_course_sdl3
      URL https://github.com/libsdl-org/SDL/archive/refs/tags/release-3.2.12.zip
      SOURCE_DIR ${dvr_course_sdl3_SOURCE}
    )
    set(dvr_course_sdl3_LOCATION ${dvr_course_sdl3_SOURCE})
    add_subdirectory(
      ${dvr_course_sdl3_LOCATION}
      ${CMAKE_CURRENT_BINARY_DIR}/deps/build/dvr_course_sdl3
      EXCLUDE_FROM_ALL
    )
  else()
    mark_as_advanced(CLEAR SDL3_DIR)
  endif()
  target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3)
  
  # ImGui
  set(dvr_course_imgui_SOURCE ${CMAKE_BINARY_DIR}/deps/source/dvr_course_imgui)
  FetchContent_Populate(
    NAME dvr_course_imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.7-docking.zip
    SOURCE_DIR ${dvr_course_imgui_SOURCE}
  )
  set(dvr_course_imgui_LOCATION ${dvr_course_imgui_SOURCE})
  target_sources(${PROJECT_NAME} PRIVATE
    ${dvr_course_imgui_LOCATION}/imgui.cpp
    ${dvr_course_imgui_LOCATION}/imgui_draw.cpp
    ${dvr_course_imgui_LOCATION}/imgui_demo.cpp
    ${dvr_course_imgui_LOCATION}/imgui_tables.cpp
    ${dvr_course_imgui_LOCATION}/imgui_widgets.cpp
    ${dvr_course_imgui_LOCATION}/backends/imgui_impl_sdl3.cpp
    ${dvr_course_imgui_LOCATION}/backends/imgui_impl_sdlrenderer3.cpp
    ${dvr_course_imgui_LOCATION}/misc/cpp/imgui_stdlib.cpp
  )
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${dvr_course_imgui_LOCATION}
    ${dvr_course_imgui_LOCATION}/backends
    ${dvr_course_imgui_LOCATION}/misc/cpp)
endif()



