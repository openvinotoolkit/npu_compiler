//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

namespace vpux::IE {

// Marker attribute set on IE.DynamicDequantize ops that were created by passes
// such as ConsolidateWeightsDequantization to replace Dequantize ops.
// The bridge pass (ConvertConstantDynamicDequantizeToDequantize) uses this to
// distinguish synthetic ops that need conversion back to Dequantize from
// pre-existing DynamicDequantize ops.
inline constexpr StringLiteral SYNTHETIC_DYN_DEQUANT_ATTR = "vpux.synthetic_dyn_dequant";

// Marker attribute set on IE.DynamicDequantize ops created by the weights import
// (WeightsDequantizeToDynamicDequantize) in place of what the FakeQuantize import would have produced.
// The bridge pass (ConvertDynamicDequantizeToFakeQuantize) matches ONLY ops carrying this marker, so it
// reconstructs FakeQuantize for exactly these ops and leaves every other DynamicDequantize (model-original,
// Gather/KV-cache, or synthetic-from-Dequantize) untouched.
inline constexpr StringLiteral WEIGHTS_IMPORT_DYN_DEQUANT_ATTR = "vpux.weights_import_dyn_dequant";

}  // namespace vpux::IE
