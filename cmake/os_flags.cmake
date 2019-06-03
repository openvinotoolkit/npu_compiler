# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

macro(disable_deprecated_warnings)
    if(WIN32)
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES Intel)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-warning:1478")
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996") # disable warning on deprecated API
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    endif()
endmacro()

if (WIN32)
    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-warning:2586,177,3180,1740,1786,47,161")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") # Too many warnings
        endif()
    endif()


    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Z7")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")

    if(ENABLE_DEBUG_SYMBOLS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Z7")

        set(DEBUG_SYMBOLS_LINKER_FLAGS "/DEBUG")
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
            # Keep default /OPT values. See /DEBUG reference for details.
            set(DEBUG_SYMBOLS_LINKER_FLAGS "${DEBUG_SYMBOLS_LINKER_FLAGS} /OPT:REF /OPT:ICF")
        endif()

        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()
endif()
