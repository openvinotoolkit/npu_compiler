//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include <vpux/compiler/utils/error.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux::IE::arch40xx;

//
// D2SToTransposedConvVerifier
//

// Heuristic here basically decides whether we should use DPU lowering or fallback to DMA approach.
// In testing, we've found that if input channels is higher than the threshold, we start to see
// regressions. We've also found that for block sizes >= 4, we see fairly large regressions as
// compared to DMA approach.

// In future, we should use a better heuristic: E#158117

// For more information on heuristic, see:
// - E#125463
// - E#113159

constexpr int64_t BENEFICIAL_INPUT_CHANNEL_MAX = 256;

mlir::LogicalResult D2SToTransposedConvVerifier::isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                                                        IE::DepthToSpaceOp d2sOp) const {
    if (d2sOp.getBlockSize() >= 4) {
        return matchFailed(log, rewriter, d2sOp, "mapping D2S to DPU is not beneficial: blockSize({0}) >= 4",
                           d2sOp.getBlockSize());  // Better to map larger block size to DMA.
    }

    if (d2sOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return matchFailed(
                log, rewriter, d2sOp,
                "mapping D2S to DPU is not beneficial: mode == BLOCKS_FIRST");  //  Better to map BLOCKS_FIRST to DMA.
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(d2sOp.getInput().getType());

    auto inputShape = inputType.getShape();
    auto inputChannels = inputShape[Dims4D::Act::C];

    if (inputChannels > BENEFICIAL_INPUT_CHANNEL_MAX) {
        return matchFailed(log, rewriter, d2sOp, "mapping D2S to DPU is not beneficial: inputChannels({0}) > {1}",
                           inputChannels, BENEFICIAL_INPUT_CHANNEL_MAX);
    }

    return mlir::success();  // Is efficient to map to DPU.
}
