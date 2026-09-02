//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

#include <cstdint>

namespace vpux {

inline constexpr int64_t SDPA_MIN_SOFTMAX_SIZE = int64_t{1} << 16;
inline constexpr double K_SDPA_OUTPUT_INPUT_SIZE_MULTIPLIER = 9.2;

inline bool isSdpaSoftmaxSizeLargeEnough(ShapeRef softmaxOutputShape, int64_t minSoftmaxSize = SDPA_MIN_SOFTMAX_SIZE) {
    return softmaxOutputShape.totalSize() >= minSoftmaxSize;
}

inline bool isSdpaSoftmaxBottleneckByShape(ShapeRef softmaxOutputShape, ShapeRef userOutputShape,
                                           double outputInputSizeMultiplier = K_SDPA_OUTPUT_INPUT_SIZE_MULTIPLIER) {
    return static_cast<double>(softmaxOutputShape.totalSize()) >
           static_cast<double>(userOutputShape.totalSize()) * outputInputSizeMultiplier;
}

inline bool isSdpaSoftmaxDecompositionBeneficialByShape(
        ShapeRef softmaxOutputShape, ShapeRef userOutputShape, int64_t minSoftmaxSize = SDPA_MIN_SOFTMAX_SIZE,
        double outputInputSizeMultiplier = K_SDPA_OUTPUT_INPUT_SIZE_MULTIPLIER) {
    return isSdpaSoftmaxSizeLargeEnough(softmaxOutputShape, minSoftmaxSize) &&
           isSdpaSoftmaxBottleneckByShape(softmaxOutputShape, userOutputShape, outputInputSizeMultiplier);
}

}  // namespace vpux
