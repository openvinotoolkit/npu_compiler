#
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(NPUReg60XX)
add_vpux_ops(NPUReg60XX GENERIC)
add_vpux_ops_interface(NPUReg60XX dialect/NPUReg60XX/)
add_vpux_attribute(NPUReg60XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(NPUReg60XX)
add_npu_reg_type(NPU60XX)
