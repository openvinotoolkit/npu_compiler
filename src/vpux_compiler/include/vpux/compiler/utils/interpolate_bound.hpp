//
// Copyright (C) 2022-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

// INTERPOLATE_SCALES_BOUND value is used to set the upper bound for scales-as-parameter scenarios.
constexpr double INTERPOLATE_SCALES_BOUND = 8.0;
constexpr double INTERPOLATE_LARGE_INPUT_SCALES_BOUND = 4.0;
// Logical element-count threshold for the same reference tensor shape used by the old byte-based heuristic.
// The decision now depends only on shape volume, so it stays stable across precision conversion.
constexpr int64_t INTERPOLATE_LARGE_INPUT_ELEMENTS_BOUND = 3 * 810 * 1440;
constexpr int64_t INTERPOLATE_4K = 3 * 2160 * 3840;

inline double getInterpolateScalesBound(vpux::NDTypeInterface inputType) {
    const auto hasUnboundedDynamicShape =
            inputType.getShape().isDynamic() && !mlir::isa<vpux::Core::BoundedTensorType>(inputType);
    if (hasUnboundedDynamicShape) {
        return INTERPOLATE_LARGE_INPUT_SCALES_BOUND;
    }

    const auto logicalShape =
            inputType.getShape().isDynamic() ? vpux::getBoundedShape(inputType) : inputType.getShape();
    const auto inputTotalSize = logicalShape.totalSize();
    const auto bound = inputTotalSize >= INTERPOLATE_LARGE_INPUT_ELEMENTS_BOUND ? INTERPOLATE_LARGE_INPUT_SCALES_BOUND
                                                                                : INTERPOLATE_SCALES_BOUND;

    // Limit dynamic Interpolate output by 4K:
    // Both spatial axes (H, W) are scaled by the same bound, so the output volume grows as bound^2.
    // Clamp it further so the worst-case output never exceeds a 4K-equivalent element count.
    // This works well for symmetric interpolation. Asymmetric cases would need a more precise per-axis bound.
    const auto maxScaleFor4K = std::sqrt(static_cast<double>(INTERPOLATE_4K) / static_cast<double>(inputTotalSize));
    return std::min(bound, std::max(maxScaleFor4K, 1.0));
}
