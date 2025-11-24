#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_NPU_MONO)
    message(FATAL_ERROR "Remove VPUXLoader target after monorepo integration")
endif()

set(TARGET_NAME VPUXLoader)

set(ELF_DIR "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/")
set(LOADER_DIR_INC "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/loader/include/")
set(LOADER_DIR_HPI_INC "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/hpi_component/include/")
set(LOADER_DIR_HPI_SRC "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/hpi_component/src")
set(LOADER_DIR_SRC "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/loader/src")
set(LOADER_DIR "${PROJECT_SOURCE_DIR}/thirdparty/elf/vpux_elf/loader/")

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

set(Loader_Library_SRCS
    ${LOADER_DIR_INC}/vpux_headers/metadata.hpp
    ${LOADER_DIR_INC}/vpux_loader/vpux_loader.hpp
    ${LOADER_DIR_SRC}/vpux_loader.cpp
    ${ELF_DIR}/core/include/vpux_elf/reader.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer.hpp
    ${ELF_DIR}/core/include/vpux_elf/accessor.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/relocation_entry.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/symbol_entry.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/vpu_extensions.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/data_types.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/elf_header.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/elf_structs.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/relocation_entry.hpp
    ${ELF_DIR}/core/include/vpux_elf/types/section_header.hpp
    ${ELF_DIR}/core/include/vpux_elf/utils/utils.hpp
    ${ELF_DIR}/core/include/vpux_elf/utils/error.hpp
    ${ELF_DIR}/core/include/vpux_elf/utils/log.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/binary_data_section.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/empty_section.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/relocation.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/relocation_section.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/section.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/string_section.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/symbol.hpp
    ${ELF_DIR}/core/include/vpux_elf/writer/symbol_section.hpp
)

add_library(${TARGET_NAME} STATIC ${Loader_Library_SRCS})

target_link_libraries(${TARGET_NAME}
    PUBLIC
        npu_elf
)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${LOADER_DIR_INC}/
        ${LOADER_DIR_SRC}/
        ${LOADER_DIR}/
        ${ELF_DIR}/core/include/vpux_elf/
        ${ELF_DIR}/core/src/
)

target_compile_definitions(npu_elf PUBLIC HOST_BUILD)

ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#--------------------------------------------------------------------------------------------
# npu_elf lib + loader lib
#--------------------------------------------------------------------------------------------
#
# loader folder | -> include/ | -> vpux_headers/ -> 1 hpp
#               |             | -> vpux_loader/  -> 1 hpp

install(DIRECTORY "${LOADER_DIR_INC}/vpux_headers"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_INC}/vpux_loader"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

#
# hpi_component | -> common/ -> 2 hpp
#               | -> 3720/   -> 1 hpp
#               | -> 4000/   -> 1 hpp

install(DIRECTORY "${LOADER_DIR_HPI_INC}/common"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_INC}/3720"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_INC}/4000"
        DESTINATION cid/vpux_elf/loader/include/
        COMPONENT ${CID_COMPONENT})
#
# loader folder | -> src/ | 1 cpp

install(DIRECTORY "${LOADER_DIR_SRC}/"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

#
# hpi_component | src | common -> 1 cpp
#               |     |  3720  -> 1 cpp
#               |     |  4000  -> 1 cpp


install(DIRECTORY "${LOADER_DIR_HPI_SRC}/common"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_SRC}/3720"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})

install(DIRECTORY "${LOADER_DIR_HPI_SRC}/4000"
        DESTINATION cid/vpux_elf/loader/src/
        COMPONENT ${CID_COMPONENT})
# core folder | -> include/ -> vpux_elf/ | -> types/
#             |                          | -> utils/
#             |                          | -> writer/
#             |                          | -> + 3 hpp

# 3 hpp
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf"
        DESTINATION cid/vpux_elf/core/include/
        COMPONENT ${CID_COMPONENT})

# types
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/types"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# utils
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/utils"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# writer
install(DIRECTORY "${ELF_DIR}/core/include/vpux_elf/writer"
        DESTINATION cid/vpux_elf/core/include/vpux_elf/
        COMPONENT ${CID_COMPONENT})

# core folder | -> src/ -> | -> types/
#             |            | -> utils/
#             |            | -> writer/
#             |            | -> + 2 cpp

# 2 cpp
install(DIRECTORY "${ELF_DIR}/core/src"
        DESTINATION cid/vpux_elf/core/
        COMPONENT ${CID_COMPONENT})

# types
install(DIRECTORY "${ELF_DIR}/core/src/types"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

# utils
install(DIRECTORY "${ELF_DIR}/core/src/utils"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

# writer
install(DIRECTORY "${ELF_DIR}/core/src/writer"
        DESTINATION cid/vpux_elf/core/src/
        COMPONENT ${CID_COMPONENT})

# loader lib
install(TARGETS VPUXLoader
        CONFIGURATIONS Release
        LIBRARY DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT})
install(TARGETS VPUXLoader
        CONFIGURATIONS Debug
        LIBRARY DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT})
install(TARGETS VPUXLoader
        CONFIGURATIONS RelWithDebInfo
        LIBRARY DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT})

# elf lib
install(TARGETS npu_elf
        CONFIGURATIONS Release
        LIBRARY DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Release COMPONENT ${CID_COMPONENT})
install(TARGETS npu_elf
        CONFIGURATIONS Debug
        LIBRARY DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/Debug COMPONENT ${CID_COMPONENT})
install(TARGETS npu_elf
        CONFIGURATIONS RelWithDebInfo
        LIBRARY DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        ARCHIVE DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT}
        RUNTIME DESTINATION cid/vpux_elf/lib/RelWithDebInfo COMPONENT ${CID_COMPONENT})
