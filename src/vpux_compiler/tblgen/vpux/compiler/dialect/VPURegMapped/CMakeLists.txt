#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPURegMapped)
add_vpux_ops_interface(VPURegMapped dialect/VPURegMapped/)
add_vpux_attribute(VPURegMapped ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(VPURegMapped)
add_vpux_pass(VPURegMapped VPURegMapped dialect/VPURegMapped/)
