#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(IE)
add_vpux_ops_granular(IE GENERIC ops/ activation)
add_vpux_ops_granular(IE GENERIC ops/ arithmetic)
add_vpux_ops_granular(IE GENERIC ops/ bitwise)
add_vpux_ops_granular(IE GENERIC ops/ comparison)
add_vpux_ops_granular(IE GENERIC ops/ control_flow)
add_vpux_ops_granular(IE GENERIC ops/ convolution)
add_vpux_ops_granular(IE GENERIC ops/ data_movement)
add_vpux_ops_granular(IE GENERIC ops/ data_type)
add_vpux_ops_granular(IE GENERIC ops/ eltwise)
add_vpux_ops_granular(IE GENERIC ops/ image)
add_vpux_ops_granular(IE GENERIC ops/ logical)
add_vpux_ops_granular(IE GENERIC ops/ normalization)
add_vpux_ops_granular(IE GENERIC ops/ pooling)
add_vpux_ops_granular(IE GENERIC ops/ recurrent)
add_vpux_ops_granular(IE GENERIC ops/ reduce)
add_vpux_ops_granular(IE GENERIC ops/ shape_manipulation)
add_vpux_ops_granular(IE GENERIC ops/ specialized)
add_vpux_ops_interface(IE dialect/IE/)
add_vpux_attr_interface(IE dialect/IE/)
add_vpux_attribute(IE ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_pass(IE IE dialect/IE/)
add_vpux_rewrite(convert IECvt)
add_vpux_rewrite(reorder IERdr)
