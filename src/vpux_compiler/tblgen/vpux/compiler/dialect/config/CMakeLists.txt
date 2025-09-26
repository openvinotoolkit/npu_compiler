#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#

add_vpux_dialect(config)
add_vpux_ops(config GENERIC)
add_vpux_attribute(config ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_ops_interface(config dialect/config/)
