#
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(validation)

# Build options
option(ENCODE_PERCENTAGE "Encode percentage rate mode" OFF)
option(ENABLE_WRITE "Output writing mode" OFF)
option(PROFILING "Profiling mode" OFF)
option(DEBUG "Debug mode" OFF)

if(ENCODE_PERCENTAGE)
    add_definitions(-D__BITC__EN_ENCODE_PERCENTAGE_RATE__=1)
endif()

if(ENABLE_WRITE)
    add_definitions(-D__BITC__EN_OUT_WRITE__=1)
endif()

if(PROFILING)
    add_definitions(-D__BITC__EN_PROFILING__=1)
endif()

if(DEBUG)
    add_definitions(-D__BITC__EN_DBG__=1)
endif()

# Adding sources
file(GLOB SOURCES "../src/*.cpp" "src/*.cpp")

add_executable(
    bitc
    ${SOURCES}
)

target_include_directories(bitc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/vpux/compiler/bitc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/
)
