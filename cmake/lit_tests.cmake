#
# Copyright (C) 2022-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

find_package(Python3 QUIET)

function(vpux_setup_lit_tool)
    set(extra_tools_copy_cmd)
    if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
        set(extra_tools ${ARGN})
        set(LLVM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/llvm-project/llvm")
        if(MSVC)
            set(FILECHECK_EXECUTION "FileCheck.exe")
            set(NOT_EXECUTION "not.exe")
        else()
            set(FILECHECK_EXECUTION "FileCheck")
            set(NOT_EXECUTION "not")
        endif()
        list(APPEND extra_tools_copy_cmd
                COMMAND
                ${CMAKE_COMMAND} -E copy
                "${MLIR_BINARY_PKG_DIR}/bin/${FILECHECK_EXECUTION}"
                "$<TARGET_FILE_DIR:npuUnitTests>/"
        )
        install(PROGRAMS "${MLIR_BINARY_PKG_DIR}/bin/${FILECHECK_EXECUTION}"
                DESTINATION tests
                COMPONENT tests
                EXCLUDE_FROM_ALL)
        list(APPEND extra_tools_copy_cmd
                COMMAND
                ${CMAKE_COMMAND} -E copy
                "${MLIR_BINARY_PKG_DIR}/bin/${NOT_EXECUTION}"
                "$<TARGET_FILE_DIR:npuUnitTests>/"
        )
        install(PROGRAMS "${MLIR_BINARY_PKG_DIR}/bin/${NOT_EXECUTION}"
                DESTINATION tests
                COMPONENT tests
                EXCLUDE_FROM_ALL)
    else()
        set(extra_tools FileCheck ${ARGN})
    endif()
    foreach(tool IN LISTS extra_tools)
        list(APPEND extra_tools_copy_cmd
            COMMAND
                ${CMAKE_COMMAND} -E copy
                    "$<TARGET_FILE:${tool}>"
                    "$<TARGET_FILE_DIR:npuUnitTests>/"
        )
        install(PROGRAMS "$<TARGET_FILE:${tool}>"
                DESTINATION tests
                COMPONENT tests
                EXCLUDE_FROM_ALL)
    endforeach()

    add_custom_target(copy_lit_tool ALL
        COMMAND
            ${CMAKE_COMMAND} -E remove_directory
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/lit-tool"
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/lit-tool"
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory
                "${LLVM_SOURCE_DIR}/utils/lit/lit"
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/lit-tool/lit"
        COMMAND
            ${CMAKE_COMMAND} -E copy
                "${LLVM_SOURCE_DIR}/utils/lit/LICENSE.TXT"
                "${LLVM_SOURCE_DIR}/utils/lit/lit.py"
                "${LLVM_SOURCE_DIR}/utils/lit/README.rst"
                "${LLVM_SOURCE_DIR}/utils/lit/setup.py"
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/lit-tool/"
        ${extra_tools_copy_cmd}
        DEPENDS ${extra_tools}
        COMMENT "[LIT] Copy LIT tool"
    )
    set_target_properties(copy_lit_tool PROPERTIES FOLDER "tests")
endfunction()

function(vpux_setup_lit_tests TEST_NAME)
    set(options)
    set(oneValueArgs ROOT)
    set(multiValueArgs PATTERNS VARS PARAMS PARAMS_DEFAULT_VALUES SUBSTITUTIONS EXTRA_SOURCES)
    cmake_parse_arguments(LIT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT LIT_ROOT)
        set(LIT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    file(GLOB_RECURSE SOURCES RELATIVE ${LIT_ROOT} CONFIGURE_DEPENDS ${LIT_PATTERNS} ${LIT_EXTRA_SOURCES})
    source_group(TREE ${LIT_ROOT} FILES ${SOURCES})

    set(SUFFIXES)
    foreach(pat IN LISTS LIT_PATTERNS)
        string(REPLACE "*" "" suf ${pat})
        set(SUFFIXES "'${suf}', ${SUFFIXES}")
    endforeach()

    set(EXTRA_DECLARATIONS)
    foreach(var IN LISTS LIT_VARS)
        set(EXTRA_DECLARATIONS "${EXTRA_DECLARATIONS}\nconfig.${var} = ${${var}}")
    endforeach()

    list(LENGTH LIT_PARAMS PARAMS_LENGTH)
    math(EXPR PARAMS_LENGTH "${PARAMS_LENGTH} - 1")

    if (${PARAMS_LENGTH} GREATER_EQUAL 0)
        foreach(var RANGE ${PARAMS_LENGTH})
            list(GET LIT_PARAMS ${var} param)
            list(GET LIT_PARAMS_DEFAULT_VALUES ${var} param_default_value)
            set(EXTRA_DECLARATIONS "${EXTRA_DECLARATIONS}\nconfig.${param} = lit_config.params['${param}'] if '${param}' in lit_config.params else '${param_default_value}'")
        endforeach()
    endif()

    set(EXTRA_SUBSTITUTIONS)
    foreach(var IN LISTS LIT_SUBSTITUTIONS LIT_PARAMS)
        set(EXTRA_SUBSTITUTIONS "${EXTRA_SUBSTITUTIONS}\nconfig.substitutions.append(('%${var}%', config.${var}))")
    endforeach()

    set(EXTRA_AVAILABLE_FEATURES)
    foreach(var IN LISTS LIT_PARAMS)
        set(EXTRA_AVAILABLE_FEATURES "${EXTRA_AVAILABLE_FEATURES}\nconfig.available_features.add(('${var}-' + config.${var}))")
    endforeach()

    # Add `dev-build` as an available feature to lit tests, which can be used in the REQUIRES field of the tests
    # This is useful for disabling incompatible tests in non-developer environments (e.g. lit-tests that use backend descriptors,
    # whose printing / parsing logic behaves differently in developer vs non-developer builds)
    if(ENABLE_DEVELOPER_BUILD)
        set(EXTRA_AVAILABLE_FEATURES "${EXTRA_AVAILABLE_FEATURES}\nconfig.available_features.add(('dev-build'))")
    endif()

    if(ENABLE_PREBUILT_LLVM_MLIR_LIBS)
        set(LLVM_LIBRARY_DIR ${MLIR_BINARY_PKG_DIR}/lib)
        set(LLVM_TOOLS_BINARY_DIR ${MLIR_BINARY_PKG_DIR}/bin)
        set(LLVM_RUNTIME_OUTPUT_INTDIR ${MLIR_BINARY_PKG_DIR}/bin)
        set(LLVM_SHLIB_OUTPUT_INTDIR ${MLIR_BINARY_PKG_DIR}/lib)
        set(LLVM_NATIVE_ARCH "X86")
    else()
        get_directory_property(LLVM_LIBRARY_DIR DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_LIBRARY_DIR)
        get_directory_property(LLVM_TOOLS_BINARY_DIR DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_TOOLS_BINARY_DIR)
        get_directory_property(LLVM_RUNTIME_OUTPUT_INTDIR DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_RUNTIME_OUTPUT_INTDIR)
        get_directory_property(LLVM_SHLIB_OUTPUT_INTDIR DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_SHLIB_OUTPUT_INTDIR)
        get_directory_property(LLVM_BINDINGS DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_BINDINGS)
        get_directory_property(LLVM_NATIVE_ARCH DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_NATIVE_ARCH)
        get_directory_property(TARGET_TRIPLE DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION TARGET_TRIPLE)
        get_directory_property(LD64_EXECUTABLE DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LD64_EXECUTABLE)
        get_directory_property(LLVM_BUILD_MODE DIRECTORY ${LLVM_SOURCE_DIR} DEFINITION LLVM_BUILD_MODE)
    endif()
    set(LTDL_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(EXEEXT ${CMAKE_EXECUTABLE_SUFFIX})
    string(CONCAT LLVM_LIT_PATH_FUNCTION
            "# Allow generated file to be relocatable.\n"
            "from pathlib import Path\n"
            "def path(p):\n"
            "    if not p: return ''\n"
            "    return str((Path(__file__).parent / p).resolve())\n")

    configure_lit_site_cfg(
        "${PROJECT_SOURCE_DIR}/cmake/lit.site.cfg.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
        @ONLY
    )

    set(tests_copy_cmd)
    if(LIT_TESTS_USE_LINKS)
        list(APPEND tests_copy_cmd
            COMMAND
                ${CMAKE_COMMAND} -E create_symlink
                    "${LIT_ROOT}/${TEST_NAME}"
                    "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}/${TEST_NAME}"
        )
    else()
        foreach(file IN LISTS SOURCES)
            list(APPEND tests_copy_cmd
                COMMAND
                    ${CMAKE_COMMAND} -E copy
                        "${LIT_ROOT}/${file}"
                        "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}/${file}"
            )
        endforeach()
    endif()

    add_custom_target(copy_${TEST_NAME}_tests ALL
        COMMAND
            ${CMAKE_COMMAND} -E remove_directory
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}"
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}"
        COMMAND
            ${CMAKE_COMMAND} -E copy
                "${PROJECT_SOURCE_DIR}/cmake/lit.cfg.py"
                "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}/"
        ${tests_copy_cmd}
        SOURCES ${SOURCES}
        COMMENT "[LIT] Copy ${TEST_NAME} LIT tests"
    )
    set_target_properties(copy_${TEST_NAME}_tests PROPERTIES FOLDER "tests")

    if(NOT Python3_FOUND)
        message(WARNING "Python3 is not found, LIT tests ${TEST_NAME} disabled")
    else()
        add_test(NAME LIT-${TEST_NAME}
            COMMAND
                ${Python3_EXECUTABLE}
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/lit-tool/lit.py"
                -v
                "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/${TEST_NAME}"
        )
        set_tests_properties(LIT-${TEST_NAME} PROPERTIES
            LABELS "NPU;LIT"
        )
        if(UNIX)
            add_test(NAME LIT-${TEST_NAME}-ALL
                COMMAND
                    "$<TARGET_FILE_DIR:npuUnitTests>/lit-tests/run_all_lit_tests.sh"
            )
            set_tests_properties(LIT-${TEST_NAME}-ALL PROPERTIES
                LABELS "NPU;LIT;Linux"
            )
        endif()
    endif()
endfunction()
