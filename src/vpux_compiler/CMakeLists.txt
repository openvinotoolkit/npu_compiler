#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#
# Build configuration
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(TARGET_NAME "npu_mlir_compiler")
set(SCHEMA_TARGET_NAME "${TARGET_NAME}_schema")
set(OBJ_TARGET_NAME "${TARGET_NAME}_obj")
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")
set(gen_base_dst_include_dir "src/vpux_compiler/include/vpux/compiler")

#
# SCHEMA target
#
if(ENABLE_NPU_MONO)
    if(NOT DEFINED GRAPHFILE_SCHEMA_SUBMODULE_PATH)
        message(FATAL_ERROR "Graphfile schema submodule path is not set while `npu_mono` was activated")
    endif()
    set(SCHEMA_SOURCE_DIR ${GRAPHFILE_SCHEMA_SUBMODULE_PATH}/src/schema)
else()
    set(SCHEMA_SOURCE_DIR ${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/elf/src/schema) # Legacy path
endif(ENABLE_NPU_MONO)


vpux_add_flatc_target(${SCHEMA_TARGET_NAME}
    SRC_DIR "${SCHEMA_SOURCE_DIR}"
    DST_DIR "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated"
    ARGS "--gen-name-strings")
set_target_properties(${SCHEMA_TARGET_NAME} PROPERTIES FOLDER "src")

# These headers are needed by Unit tests even if ENABLE_MLIR_COMPILER=OFF
add_subdirectory(include/vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers)
add_subdirectory(include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers)

if(NOT ENABLE_MLIR_COMPILER)
    return()
endif()

#
# Embed VPUNN models
#

find_package(Git REQUIRED)

execute_process(
    COMMAND ${GIT_EXECUTABLE} lfs pull
    WORKING_DIRECTORY "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel")

vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7_159.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7")
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_2_7_159.fast.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_2_7_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_2_7_FAST")

vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_40_159_strict.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_4_0.hpp.inc"
    VARIABLE_NAME "COST_MODEL_4_0")
vpux_embed_bin_file(
    SOURCE_FILE "${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/thirdparty/vpucostmodel/models/vpu_40_159_strict.fast.vpunn"
    HEADER_FILE "${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPU/generated/cost_model_data_4_0_fast.hpp.inc"
    VARIABLE_NAME "COST_MODEL_4_0_FAST")

#
# OBJECT target
#

file(GLOB SOURCES CONFIGURE_DEPENDS
        src/*.cpp)

find_package(MLIR REQUIRED CONFIG)
if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
    set_llvm_flags()
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
endif()

include_directories(SYSTEM
    ${PROJECT_SOURCE_DIR}/src/vpux_compiler/tblgen  # Include .td files
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

include(cmake/vpux_tblgen.cmake)
# Keep MLIR auto-generated docs in binary folder to be ignored by git
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(tblgen)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)
# GraphFile schema is used in tblgen and therefore in entire compiler
include_directories(
    ${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated
    $<TARGET_PROPERTY:flatbuffers,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:npu_elf,INTERFACE_INCLUDE_DIRECTORIES>
)

include(cmake/add_npu_library.cmake)

set(VPUX_COMPILER_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_version.cpp)

# Construct source files after BITCOMPACTOR configuration
add_subdirectory(src)
get_property(VPU_SRC_LIBS GLOBAL PROPERTY NPU_SRC_LIB_LIST)

add_library(mlir-dependencies INTERFACE)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(mlir-dependencies
    INTERFACE
        ${conversion_libs}
        MLIRExecutionEngine
        MLIROptLib)
target_include_directories(mlir-dependencies
    SYSTEM INTERFACE
        ${MLIR_INCLUDE_DIRS}
        ${VPUNN_INCLUDE_DIRS})

if(BUILD_SHARED_LIBS)
    set(COMPILER_TARGET ${OBJ_TARGET_NAME})
    add_library(${OBJ_TARGET_NAME} OBJECT ${SOURCES})
else()
    set(COMPILER_TARGET ${TARGET_NAME})
    add_library(${TARGET_NAME} STATIC ${SOURCES})

    target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_STATIC_LIBRARY)
endif()

add_dependencies(${COMPILER_TARGET} mlir-doc)

ov_build_target_faster(${COMPILER_TARGET} PCH PUBLIC "src/precomp.hpp")
enable_warnings_as_errors(${COMPILER_TARGET} WIN_STRICT)
replace_noerror(${COMPILER_TARGET})

target_compile_definitions(${COMPILER_TARGET}
    PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN
        "DEVICE_NAME=\"${NPU_DEVICE_NAME}\""
        "LIBRARY_OUTPUT_DIRECTORY=\"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}\"")
target_include_directories(${COMPILER_TARGET}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated>
        $<TARGET_PROPERTY:flatbuffers,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:npu_elf,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(${COMPILER_TARGET}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen>
)
target_link_libraries(${COMPILER_TARGET}
    PUBLIC
        openvino::npu_al
    PRIVATE
        npu_profiling_utils
)
ov_link_system_libraries(${COMPILER_TARGET}
    PRIVATE
        openvino::runtime
)
target_link_libraries(${COMPILER_TARGET} PUBLIC ${VPU_SRC_LIBS})

#
# NPUReg40XX dialect library
#

if (ENABLE_DIALECT_SHARED_LIBRARIES)
    file(GLOB_RECURSE NPUReg40XX_DIALECT_SOURCES CONFIGURE_DEPENDS
            "src/NPU40XX/dialect/NPUReg40XX/*.cpp"
            "include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/*.hpp")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NPUReg40XX_DIALECT_SOURCES} ${NPUReg40XX_DIALECT_HEADERS})

    add_library(NPUReg40XX_dialect SHARED ${NPUReg40XX_DIALECT_SOURCES})

    target_include_directories(NPUReg40XX_dialect
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/vpux/compiler/dialect>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/tblgen>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${IE_MAIN_VPUX_PLUGIN_SOURCE_DIR}/sw_runtime_kernels>)
    target_link_libraries(NPUReg40XX_dialect
        PUBLIC
            mlir-dependencies
            npu_elf
            npu_llvm_utils)
endif()

#
# Update version file
#
ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${COMPILER_TARGET})

#
# MODULE library
#

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_TARGET_NAME})
    if(WIN32)
        ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
        ov_add_vs_version_file(NAME ${TARGET_NAME} FILEDESCRIPTION "NPU Compiler")
    endif()
endif()

#
# STATIC target
#

if(BUILD_SHARED_LIBS)
    add_library(${STATIC_TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJ_TARGET_NAME}>)
    # Link the same to propagate dependencies
    target_link_libraries(${STATIC_TARGET_NAME} PUBLIC ${OBJ_TARGET_NAME})
else()
    bundle_static_library(${COMPILER_TARGET} inferenceStatic)
    add_library(${STATIC_TARGET_NAME} ALIAS ${COMPILER_TARGET})
endif()

#
# targets install
#

# Note not included in OpenVINOTargets export!

if(BUILD_SHARED_LIBS)
install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT})
endif()
