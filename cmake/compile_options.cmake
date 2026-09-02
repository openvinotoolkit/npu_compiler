#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include("${CMAKE_CURRENT_LIST_DIR}/compile_options_llvm.cmake")

# put flags allowing dynamic symbols into target
macro(replace_compile_visibility_options)
    # Replace compiler flags
    foreach(flag IN ITEMS "-fvisibility=default" "-fvisibility=hidden" "-rdynamic" "-export-dynamic")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
    endforeach()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -rdynamic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -export-dynamic")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -export-dynamic")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -export-dynamic")
endmacro()

macro(replace_noerror TARGET_NAME)
    # TODO(E#78994): better way to wrap up code which uses deprecated declarations
    if(NOT MSVC)
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wno-error=deprecated-declarations
        )
    endif()
    # TODO(E#83264): consider making it enabled
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wno-error=covered-switch-default
        )
    endif()
endmacro()

if(MSVC)
    # Wile cmake default is /Zi OV, overrides /Zi with /Z7
    # We need /Z7 to avoid pdb creation issues with Ninja build and/or ccache
    # Add /debug:fastlink to link step to avoid pdb files exceeding 4GB limit
    # Note with fastlink object files are required for full debug information!
    foreach(link_flag_var
        CMAKE_EXE_LINKER_FLAGS_DEBUG
        CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
        CMAKE_MODULE_LINKER_FLAGS_DEBUG
        CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO
        CMAKE_SHARED_LINKER_FLAGS_DEBUG
        CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO
        )
        string(REGEX REPLACE "/debug" "/debug:fastlink" ${link_flag_var} "${${link_flag_var}}")
    endforeach()

    # Optimize global data
    add_compile_options(/Zc:inline /Gw)
    # Use compiler intrinsincs
    add_compile_options(/Oi)

    # Note: CCache requires /Z7 which is set above
    if(ENABLE_CCACHE_FOR_VISUAL_STUDIO)
        message(STATUS "CCache for Visual Studio Generator is going to be used")
        cmake_minimum_required(VERSION 3.21 FATAL_ERROR) # for file(COPY_FILE)

        # see https://github.com/ccache/ccache/wiki/MS-Visual-Studio#usage-with-cmake
        find_program(CCACHE_PATH ccache REQUIRED)
        if(CCACHE_PATH)
            file(COPY_FILE ${CCACHE_PATH} ${CMAKE_BINARY_DIR}/cl.exe
                 ONLY_IF_DIFFERENT)
            set(CMAKE_VS_GLOBALS
                "CLToolExe=cl.exe"
                "CLToolPath=${CMAKE_BINARY_DIR}"
                "UseMultiToolTask=true"
            )
        endif()
    endif()
endif()

function(enable_warnings_as_errors TARGET_NAME)

    cmake_parse_arguments(WARNIGS "WIN_STRICT;SKIP_SYSTEM" "" "" ${ARGN})

    if(MSVC)
        # Enforce standards conformance on MSVC
        target_compile_options(${TARGET_NAME}
            PRIVATE
                /permissive-
        )

        if(WARNIGS_WIN_STRICT)
            # TODO(E#220632): check and fix warnings to avoid error c2220
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /WX /W3 /wd4244 /wd4267
            )

            # Disable 3rd-party components warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /experimental:external /external:anglebrackets /external:W0
            )

            if(ENABLE_UNSUPPRESSED_WARNINGS_FOR_MSVC)
                # Retrieve all compile options currently applied to the target
                get_target_property(current_options ${TARGET_NAME} COMPILE_OPTIONS)
                if(current_options)
                    # Remove Warning-as-Error
                    list(REMOVE_ITEM current_options "/WX")
                    # Remove all warning suppression flags (/wdXXXX or -wdXXXX)
                    list(FILTER current_options EXCLUDE REGEX "^[-/]wd")
                    # Update target properties with the filtered list
                    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_OPTIONS "${current_options}")
                endif()
            endif()
        endif()
    else()
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wall -Wextra -Werror -Werror=suggest-override
        )
        if(NOT WARNIGS_SKIP_SYSTEM)
            # Set SYSTEM property on OpenVINO dependencies to suppress extra warnings from their headers
            # when building via OPENVINO_EXTRA_MODULES
            get_target_property(deps ${TARGET_NAME} LINK_LIBRARIES)
            foreach(lib IN LISTS deps)
                if(${lib} MATCHES "^openvino::")
                    get_target_property(orig ${lib} ALIASED_TARGET)
                    if (TARGET ${orig})
                        set_target_properties(${orig} PROPERTIES SYSTEM TRUE)
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endfunction()

macro(enable_split_dwarf)
    if ((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_compile_options(-gsplit-dwarf)
            if (COMMAND check_linker_flag)
                check_linker_flag(CXX "-Wl,--gdb-index" LINKER_SUPPORTS_GDB_INDEX)
                if (LINKER_SUPPORTS_GDB_INDEX)
                    foreach(_flag_var CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
                        set(${_flag_var} "${${_flag_var}} -Wl,--gdb-index")
                    endforeach()
                endif()
            endif()
            set(LLVM_USE_SPLIT_DWARF ON)
        endif()
    endif()
endmacro()

function(append_avx2_flags TARGET_NAME)
    if(ENABLE_AVX2)
        ov_avx2_optimization_flags(avx2_flags)
        target_compile_options(${TARGET_NAME} PUBLIC "${avx2_flags}")
    endif()
endfunction()

# the implementation is taken from llvm/cmake/modules/HandleLLVMOptions.cmake
macro(enable_asserts)
    if(NOT MSVC)
        add_compile_definitions(_DEBUG)
    endif()
    if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
        add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
        if (MSVC)
            foreach (flags_var_to_scrub
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS_MINSIZEREL
                CMAKE_C_FLAGS_RELEASE
                CMAKE_C_FLAGS_RELWITHDEBINFO
                CMAKE_C_FLAGS_MINSIZEREL)
              string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
                  "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
            endforeach()
        endif()
    endif()
    add_compile_definitions(_GLIBCXX_ASSERTIONS)
    add_compile_definitions(_LIBCPP_ENABLE_ASSERTIONS)
endmacro()

macro(enable_color_diagnostics)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_COLOR_DIAGNOSTICS ON)
  else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_compile_options(-fdiagnostics-color=always)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      add_compile_options(-fcolor-diagnostics)
    endif()
  endif()
endmacro()
