#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(bytecode)
add_vpux_ops_granular(bytecode GENERIC ops/ arithmetic)
add_vpux_ops_granular(bytecode GENERIC ops/ comparison)
add_vpux_ops_granular(bytecode GENERIC ops/ conversion)
add_vpux_ops_granular(bytecode GENERIC ops/ bitwise)
add_vpux_ops_granular(bytecode GENERIC ops/ conditional)
add_vpux_ops_granular(bytecode GENERIC ops/ buffer)
add_vpux_ops_granular(bytecode GENERIC ops/ control_flow)
add_vpux_ops_granular(bytecode GENERIC ops/ external)
add_vpux_ops_granular(bytecode GENERIC ops/ kernel_submission)
add_vpux_ops_granular(bytecode GENERIC ops/ metadata)
add_vpux_ops_granular(bytecode GENERIC ops/ section)
add_vpux_ops_granular(bytecode GENERIC ops/ register)
add_vpux_ops_interface(bytecode dialect/bytecode/)
add_vpux_pass(bytecode bytecode dialect/bytecode/)
add_vpux_attribute(bytecode ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(bytecode)
