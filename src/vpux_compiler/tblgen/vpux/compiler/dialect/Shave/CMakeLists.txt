#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(Shave)
add_vpux_ops_granular(Shave GENERIC ops/ meta-ops)
add_vpux_ops_interface(Shave dialect/Shave/)
add_vpux_attribute(Shave ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_pass(Shave Shave dialect/Shave/)
