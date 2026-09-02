#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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
set(STATIC_TARGET_NAME "${TARGET_NAME}_static")
set(gen_base_dst_include_dir "src/vpux_compiler/include/vpux/compiler")

# These headers are needed by Unit tests
add_subdirectory(include/vpux/compiler/NPU37XX/dialect/NPUReg37XX/firmware_headers)
add_subdirectory(include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers)

#
# STATIC target
#

file(GLOB SOURCES CONFIGURE_DEPENDS
        src/*.cpp
        src/init/*.cpp)

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

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/interface/include
    ${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/include
)
include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)
# GraphFile schema is used in tblgen and therefore in entire compiler
include_directories(
    ${PROJECT_BINARY_DIR}/${gen_base_dst_include_dir}/dialect/VPUIP/generated
    $<TARGET_PROPERTY:npu_elf,INTERFACE_INCLUDE_DIRECTORIES>
)

include(cmake/add_npu_library.cmake)

set(VPUX_COMPILER_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_version.cpp)

# compiler repo git commit hash
set(COMMIT_HASH_PATTERN ${CMAKE_CURRENT_SOURCE_DIR}/include/vpux/compiler/compiler_hash.hpp.in)
set(COMMIT_HASH_FILE ${CMAKE_CURRENT_BINARY_DIR}/include/vpux/compiler/compiler_hash.hpp)
set(COMMIT_HASH_CACHE ${CMAKE_CURRENT_BINARY_DIR}/include/vpux/compiler/compiler_hash_cache)

add_custom_target(update_compiler_hash ALL
    COMMAND ${CMAKE_COMMAND}
        -D REPO_DIR=${PROJECT_SOURCE_DIR}
        -D COMMIT_HASH_PATTERN=${COMMIT_HASH_PATTERN}
        -D COMMIT_HASH_FILE=${COMMIT_HASH_FILE}
        -D COMMIT_HASH_CACHE=${COMMIT_HASH_CACHE}
        -P ${PROJECT_SOURCE_DIR}/cmake/compiler_commit_hash.cmake
    COMMENT "Updating compiler commit hash..."
    BYPRODUCTS ${COMMIT_HASH_FILE} ${COMMIT_HASH_CACHE}
)

add_library(compiler_commit_hash INTERFACE)
target_include_directories(compiler_commit_hash INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/include/vpux/compiler)
add_dependencies(compiler_commit_hash update_compiler_hash)

#
# Compiler interface library (OBJECT)
# This lightweight library provides the public API for network metadata and compiler interface
# It can be used by runtime components without depending on the full compiler implementation
#

add_library(npu_compiler_interface OBJECT
    interface/src/network_metadata.cpp
)

target_include_directories(npu_compiler_interface
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/vpux_driver_compiler/include>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/vpux_driver_compiler>
)

# Link only essential dependencies that are part of the public API
target_link_libraries(npu_compiler_interface
    PUBLIC
        npu_core_utils
        openvino::runtime
)

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
        ${MLIR_INCLUDE_DIRS})

set(COMPILER_TARGET ${STATIC_TARGET_NAME})
add_library(${STATIC_TARGET_NAME} STATIC ${SOURCES})

#
# Separate compiler module library
#
set(COMPILER_LIB openvino_intel_npu_compiler)
add_library(${COMPILER_LIB} MODULE $<TARGET_OBJECTS:${STATIC_TARGET_NAME}>)
target_link_libraries(${COMPILER_LIB} PRIVATE ${STATIC_TARGET_NAME})
if(WIN32)
    ov_add_api_validator_post_build_step(TARGET ${COMPILER_LIB})
    if(ENABLE_PRODUCTION_BUILD)
        add_link_options("/INTEGRITYCHECK")
    endif()
endif()

if(UNIX)
    target_link_options(${COMPILER_LIB} PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/compiler.exports)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/npu_compiler_version.cmake)
npu_compiler_configure_version(${COMPILER_LIB})

install_target_with_debug_files(${COMPILER_LIB} ${INSTALL_DESTINATION}/lib ${INSTALL_COMPONENT})

install(TARGETS ${COMPILER_LIB}
        ARCHIVE DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        )

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
        $<TARGET_PROPERTY:npu_elf,INTERFACE_INCLUDE_DIRECTORIES>
)
target_include_directories(${COMPILER_TARGET}
    SYSTEM PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/vpux_compiler/tblgen>
)

target_link_libraries(${COMPILER_TARGET}
    PUBLIC
        npu_compiler_interface
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
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/sw_runtime_kernels>)
    target_link_libraries(NPUReg40XX_dialect
        PUBLIC
            mlir-dependencies
            npu_elf)
endif()

#
# Update version file
#
ov_add_version_defines(${VPUX_COMPILER_VERSION_FILE} ${COMPILER_TARGET})

#
# Dependency check
#
if(ENABLE_DEP_CHECK)
    # Append compiler dependencies
    depcheck_collect_deps(src include "vpux/compiler" ${VPU_SRC_LIBS})
    # Add dep_check target to ALL
    add_depcheck_target(dep_check)
endif()
