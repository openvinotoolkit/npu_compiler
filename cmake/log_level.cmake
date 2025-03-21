#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

function(set_log_level)
    if(NOT DEFINED BUILD_LOG_LEVEL)
        if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT ENABLE_DEVELOPER_BUILD)
            set(DEFAULT_LOG_LEVEL "LOG_INFO")
        else()
            set(DEFAULT_LOG_LEVEL "LOG_TRACE")
        endif()
        set(BUILD_LOG_LEVEL ${DEFAULT_LOG_LEVEL} CACHE STRING "Build log level")
    endif()
    if(BUILD_LOG_LEVEL STREQUAL "LOG_NONE")
        set(LOG_LEVEL_VALUE 0)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_FATAL")
        set(LOG_LEVEL_VALUE 1)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_ERROR")
        set(LOG_LEVEL_VALUE 2)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_WARNING")
        set(LOG_LEVEL_VALUE 3)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_INFO")
        set(LOG_LEVEL_VALUE 4)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_DEBUG")
        set(LOG_LEVEL_VALUE 5)
    elseif(BUILD_LOG_LEVEL STREQUAL "LOG_TRACE")
        set(LOG_LEVEL_VALUE 6)
    else()
        message(FATAL_ERROR "Unknown log level: `${BUILD_LOG_LEVEL}`")
    endif()
    # Add a global definition since logger might be used without explicit cmake dependency
    add_compile_definitions(BUILD_LOG_LEVEL=${LOG_LEVEL_VALUE})
endfunction()
