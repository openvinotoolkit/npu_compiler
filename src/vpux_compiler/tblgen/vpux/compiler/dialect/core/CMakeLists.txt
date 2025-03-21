#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(Core)
add_vpux_attr_interface(Core core/)
add_vpux_ops_interface(Core core/)
add_vpux_type_interface(Core core/)
add_vpux_pass(Core Core core/)
