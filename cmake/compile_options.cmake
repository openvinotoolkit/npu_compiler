#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

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

macro(replace_noerror)
    # foreach(flag IN ITEMS "-Wno-error")
    foreach(flag IN ITEMS "-Wno-error=unused-variable" "-Wno-error=unused-but-set-variable" "-Wno-error=deprecated-declarations")
        string(REPLACE ${flag} "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        string(REPLACE ${flag} "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
    endforeach()
endmacro()

function(enable_warnings_as_errors TARGET_NAME)
    if(NOT TREAT_WARNING_AS_ERROR)
        return()
    endif()

    cmake_parse_arguments(WARNIGS "WIN_STRICT" "" "" ${ARGN})

    if(MSVC)
        # Enforce standards conformance on MSVC
        target_compile_options(${TARGET_NAME}
            PRIVATE
                /permissive-
        )

        if(WARNIGS_WIN_STRICT)
            # Use W3 instead of Wall, since W4 introduces some hard-to-fix warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /WX /W3
            )

            # Disable 3rd-party components warnings
            target_compile_options(${TARGET_NAME}
                PRIVATE
                    /experimental:external /external:anglebrackets /external:W0
            )
        endif()
    else()
        target_compile_options(${TARGET_NAME}
            PRIVATE
                -Wall -Wextra -Werror
        )
    endif()
endfunction()

# Links provided libraries and include their INTERFACE_INCLUDE_DIRECTORIES as SYSTEM
function(link_system_libraries TARGET_NAME)
    set(MODE PRIVATE)

    foreach(arg IN LISTS ARGN)
        if(arg MATCHES "(PRIVATE|PUBLIC|INTERFACE)")
            set(MODE ${arg})
        else()
            if(TARGET "${arg}")
                target_include_directories(${TARGET_NAME}
                    SYSTEM ${MODE}
                        $<TARGET_PROPERTY:${arg},INTERFACE_INCLUDE_DIRECTORIES>
                        $<TARGET_PROPERTY:${arg},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
                )
            endif()

            target_link_libraries(${TARGET_NAME}
                ${MODE}
                    ${arg}
            )
        endif()
    endforeach()
endfunction()

function(vpux_enable_clang_format TARGET_NAME)
    add_clang_format_target("${TARGET_NAME}_clang_format"
        FOR_TARGETS ${TARGET_NAME}
        ${ARGN}
    )

    if(ENABLE_DEVELOPER_BUILD)
        if(TARGET "${TARGET_NAME}_clang_format_fix")
            add_dependencies(${TARGET_NAME} "${TARGET_NAME}_clang_format_fix")
        endif()
    else()
        if(TARGET "${TARGET_NAME}_clang_format")
            add_dependencies(${TARGET_NAME} "${TARGET_NAME}_clang_format")
        endif()
    endif()
endfunction()
