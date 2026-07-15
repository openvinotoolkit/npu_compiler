# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Collects all .cpp files from root_dir, excluding files in nested directories
# that contain their own CMakeLists.txt files.
#
# Args:
#   out_var: Variable name to store the list of collected .cpp files
#   root_dir: Root directory to search for .cpp files
#
function(collect_cpp_excluding_nested_cmakelists out_var root_dir)
    file(TO_CMAKE_PATH "${root_dir}" _root_dir)

    # Collect all candidate .cpp files
    file(GLOB_RECURSE _all_cpp CONFIGURE_DEPENDS
        "${_root_dir}/*.cpp"
    )

    # Find nested CMakeLists.txt files
    file(GLOB_RECURSE _nested_cmakelists CONFIGURE_DEPENDS
        "${_root_dir}/*/CMakeLists.txt"
    )

    set(_filtered_cpp "${_all_cpp}")
    set(_exclude_regexes)

    foreach(_cmakelists IN LISTS _nested_cmakelists)
        get_filename_component(_cmake_dir "${_cmakelists}" DIRECTORY)
        file(TO_CMAKE_PATH "${_cmake_dir}" _cmake_dir)

        # Keep the root directory itself; exclude only nested subprojects
        if(_cmake_dir STREQUAL _root_dir)
            continue()
        endif()

        # Escape regex special chars in the path
        string(REGEX REPLACE "([][+.*^$(){}|\\\\?])" "\\\\\\1" _cmake_dir_escaped "${_cmake_dir}")

        # Match any file under that directory
        list(APPEND _exclude_regexes "^${_cmake_dir_escaped}/")
    endforeach()

    foreach(_regex IN LISTS _exclude_regexes)
        list(FILTER _filtered_cpp EXCLUDE REGEX "${_regex}")
    endforeach()

    set(${out_var} "${_filtered_cpp}" PARENT_SCOPE)
endfunction()
