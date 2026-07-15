#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Install library related files to components
# Note: The status of BUILD_COMPILER_FOR_DRIVER will
# generate different compilers and install their targets in different locations.
function(install_target_with_debug_files target destination component)
    if(WIN32)
        # Install library to component
        install(TARGETS ${target}
            LIBRARY DESTINATION ${destination} COMPONENT ${component}
            RUNTIME DESTINATION ${destination} COMPONENT ${component})

        # Install PDB files for Windows
        if(BUILD_COMPILER_FOR_DRIVER)
            # For CID build, PDB files exist and are required.
            install(FILES $<TARGET_PDB_FILE:${target}>
                    DESTINATION ${INSTALL_DESTINATION}/pdb
                    COMPONENT ${component})
        else()
            # Sometimes, the PDB file for npu_compiler may not exist.
            # So, use `OPTIONAL` option to prevent the install command from reporting an error
            # when the openvino_intel_npu_compiler.pdb file is missing.
            install(FILES $<TARGET_PDB_FILE:${target}>
                    OPTIONAL
                    DESTINATION ${INSTALL_DESTINATION}/pdb
                    COMPONENT ${component})
        endif()
    else()
        install(TARGETS ${target}
            LIBRARY DESTINATION ${destination} COMPONENT ${component}
            RUNTIME DESTINATION ${destination} COMPONENT ${component})
    endif()
endfunction()
