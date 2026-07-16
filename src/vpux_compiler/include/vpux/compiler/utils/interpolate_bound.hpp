//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <cstdint>

// INTERPOLATE_SCALES_BOUND value is used to set the upper bound for scales as parameter scenarios.
constexpr double INTERPOLATE_SCALES_BOUND = 8.0;
constexpr double INTERPOLATE_LARGE_INPUT_SCALES_BOUND = 4.0;
// Byte size of tensor<1x3x810x1440xf16>.
// For input shape bigger than this, the upper bound for scales will pick 4.0
// to avoid large memory consumption and runtime error.
// So far the output shape of interpolate scale param case will be within 1 x 3 x 2160 x 3840
constexpr int64_t INTERPOLATE_LARGE_INPUT_BYTES_BOUND = 3 * 810 * 1440 * 2;

inline double getInterpolateScalesBound(vpux::NDTypeInterface inputType) {
    const auto hasUnboundedDynamicShape =
            inputType.getShape().isDynamic() && !mlir::isa<vpux::Core::BoundedTensorType>(inputType);
    if (hasUnboundedDynamicShape) {
        return INTERPOLATE_LARGE_INPUT_SCALES_BOUND;
    }

    if (inputType.getTotalAllocSize().count() >= INTERPOLATE_LARGE_INPUT_BYTES_BOUND) {
        return INTERPOLATE_LARGE_INPUT_SCALES_BOUND;
    }
    return INTERPOLATE_SCALES_BOUND;
}
