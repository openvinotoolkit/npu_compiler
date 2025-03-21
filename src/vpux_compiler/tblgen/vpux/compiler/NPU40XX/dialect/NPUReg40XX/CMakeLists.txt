#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(NPUReg40XX)
add_vpux_ops_interface(NPUReg40XX dialect/NPUReg40XX/)
add_vpux_attribute(NPUReg40XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
# E#135032: add_vpux_type to be replaced with add_npu_reg_type
add_vpux_type(NPUReg40XX)
add_npu_reg_type(NPU40XX)
