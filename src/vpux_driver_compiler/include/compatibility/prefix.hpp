//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// VPUX generates these prefixes; OpenVINO consumes them when restoring
// runtime graph names. Keep values and semantics synced with OV.
//

#pragma once

#include <string>
#include <string_view>

namespace intel_npu {

//
// Prefixes used to identify special inputs/outputs
//

constexpr std::string_view READVALUE_PREFIX = "vpux_ie_read_value_";
constexpr std::string_view ASSIGN_PREFIX = "vpux_ie_assign_";
constexpr std::string_view SHAPE_TENSOR_PREFIX = "vpux_ie_shape_";
constexpr std::string_view INIT_INPUT_WEIGHTS_PREFIX = "vpux_ow_";
constexpr std::string_view INIT_OUTPUT_WEIGHTS_PREFIX = "vpux_tw_";
constexpr std::string_view MAIN_INPUT_WEIGHTS_PREFIX = "vpux_tw_";

}  // namespace intel_npu
