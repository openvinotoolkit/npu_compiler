#
# Copyright (C) 2023-2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(NPUReg50XX)
add_vpux_ops(NPUReg50XX GENERIC)
add_vpux_ops_interface(NPUReg50XX dialect/NPUReg50XX/)
add_vpux_attribute(NPUReg50XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(NPUReg50XX)
add_npu_reg_type(NPU50XX)
add_vpux_pass(NPUReg50XX NPUReg50XX NPUReg50XX/dialect/NPUReg50XX/)
