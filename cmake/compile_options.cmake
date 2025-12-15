#
# Copyright (C) 2022-2025 Intel Corporation.
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

endif()

function(enable_warnings_as_errors TARGET_NAME)

    cmake_parse_arguments(WARNIGS "WIN_STRICT" "" "" ${ARGN})

    if(MSVC)
        # Enforce standards conformance on MSVC
        target_compile_options(${TARGET_NAME}
            PRIVATE
                /permissive-
        )

        if(WARNIGS_WIN_STRICT)
            if(BUILD_COMPILER_FOR_DRIVER)
                target_compile_options(${TARGET_NAME}
                    PRIVATE
                        /W3
                )
            else()
                # Use W3 instead of Wall, since W4 introduces some hard-to-fix warnings
                target_compile_options(${TARGET_NAME}
                    PRIVATE
                        /WX /W3 /wd4244 /wd4267
                        # TODO(E#86977): check and fix warnings to avoid error c2220
                )
            endif()
            # Disable 3rd-party components warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /experimental:external /external:anglebrackets /external:W0
            )
        endif()
    else()
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wall -Wextra -Werror -Werror=suggest-override
        )
    endif()
endfunction()

macro(enable_split_dwarf)
    if ((CMAKE_BUILD_TYPE STREQUAL "Debug") OR (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_compile_options(-gsplit-dwarf)
            if (COMMAND check_linker_flag)
                check_linker_flag(CXX "-Wl,--gdb-index" LINKER_SUPPORTS_GDB_INDEX)
                append_if(LINKER_SUPPORTS_GDB_INDEX "-Wl,--gdb-index"
                CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
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
