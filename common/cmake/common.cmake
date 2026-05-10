# Copyright 2025-2025 Stefan Zellmann
# SPDX-License-Identifier: Apache-2.0

set(DVR_COURSE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if (TARGET dvr_course-common)
  get_target_property(DVR_COURSE_WITH_CUDA dvr_course-common DVR_COURSE_WITH_CUDA)
  get_target_property(DVR_COURSE_WITH_OWL dvr_course-common DVR_COURSE_WITH_OWL)
  get_target_property(DVR_COURSE_WITH_FAKE_OWL dvr_course-common DVR_COURSE_WITH_FAKE_OWL)
endif()

macro(configure_cuda_src src)
  if (NOT DVR_COURSE_WITH_CUDA)
    set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
    if (NOT  WIN32)
      set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
    endif()
  endif()
endmacro()
