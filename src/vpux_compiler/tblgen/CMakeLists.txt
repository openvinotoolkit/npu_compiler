#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(vpux/compiler/dialect/core)
add_subdirectory(vpux/compiler/dialect/config)
add_subdirectory(vpux/compiler/dialect/net)
add_subdirectory(vpux/compiler/conversion)
add_subdirectory(vpux/compiler/ShaveCodeGen)
add_subdirectory(vpux/compiler/dialect/const)
add_subdirectory(vpux/compiler/dialect/IE)
add_subdirectory(vpux/compiler/dialect/VPU)
add_subdirectory(vpux/compiler/NPU37XX/dialect/VPUIP)
add_subdirectory(vpux/compiler/NPU37XX/dialect/VPUIPDPU)
add_subdirectory(vpux/compiler/NPU40XX/dialect/VPUIP)
add_subdirectory(vpux/compiler/dialect/VPUIP)
add_subdirectory(vpux/compiler/dialect/VPUIPDPU)
add_subdirectory(vpux/compiler/NPU40XX/dialect/VPUIPDPU)
add_subdirectory(vpux/compiler/dialect/VPURT)
add_subdirectory(vpux/compiler/dialect/ELFNPU37XX)
add_subdirectory(vpux/compiler/dialect/VPUMI37XX)
add_subdirectory(vpux/compiler/dialect/VPUMI40XX)
add_subdirectory(vpux/compiler/dialect/VPURegMapped)
add_subdirectory(vpux/compiler/dialect/VPUASM)
add_subdirectory(vpux/compiler/NPU40XX/dialect/ELF)
add_subdirectory(vpux/compiler/NPU37XX/dialect/NPUReg37XX)
add_subdirectory(vpux/compiler/NPU40XX/dialect/NPUReg40XX)
add_subdirectory(vpux/compiler/dialect/HostExec)
