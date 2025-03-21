#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_pass(Conversion Conversion conversion/)
add_vpux_rewrite(convert_layers_to_VPU Conversion)
add_vpux_rewrite(convert_eltwise_layers_to_math EltwiseLayersToMathConversion)
