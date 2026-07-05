# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.12)
project(dvr_course-common LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/common.cmake)

add_library(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(DVR_COURSE_WITH_OWL "Use OWL as ray tracing API (for hardware ray tracing)" ON)
if (DVR_COURSE_WITH_OWL)
  find_package(owl REQUIRED)
  if (NOT OWL_IS_FAKE)
    set(DVR_COURSE_WITH_CUDA ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES DVR_COURSE_WITH_CUDA ON)
    enable_language(CUDA)
  else()
    set(DVR_COURSE_WITH_FAKE_OWL ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES DVR_COURSE_WITH_FAKE_OWL ON)
  endif()
endif()

target_sources(${PROJECT_NAME} PRIVATE fb.cu pipeline.cu transfunc.cu)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC external)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
configure_cuda_src(fb.cu)
configure_cuda_src(pipeline.cu)
configure_cuda_src(transfunc.cu)

if (DVR_COURSE_WITH_CUDA)
  target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_CUDA=1)
  set_source_files_properties(pipeline.cu
    PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr")
endif()

if (DVR_COURSE_WITH_OWL)
  # public define "RTCORE", also used by example programs:
  target_compile_definitions(${PROJECT_NAME} PUBLIC RTCORE=1)
  target_link_libraries(${PROJECT_NAME} PUBLIC owl::owl)
  if (OWL_IS_FAKE)
    target_compile_definitions(${PROJECT_NAME} PUBLIC OWL_IS_FAKE=1)
  endif()
endif()

option(DVR_COURSE_PIPELINE_INTERACTIVE "Use interactive viewer instead of png output" OFF)
if (DVR_COURSE_PIPELINE_INTERACTIVE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE INTERACTIVE=1)
  include(FetchContent)

  # OpenGL
  find_package(OpenGL REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

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
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.7-docking.zip
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

  # UI sources
  target_sources(${PROJECT_NAME} PRIVATE alpha_editor.cpp tfe.cpp)
endif()

add_subdirectory(external)



