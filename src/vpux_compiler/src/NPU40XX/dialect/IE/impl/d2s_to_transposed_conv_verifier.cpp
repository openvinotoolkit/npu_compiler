//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include <vpux/compiler/utils/error.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux::IE::arch40xx;

//
// D2SToTransposedConvVerifier
//

// For more information on heuristic, see:
// - E#125463
// - E#113159

constexpr int64_t BENEFICIAL_INPUT_CHANNEL_MAX = 256;

mlir::LogicalResult D2SToTransposedConvVerifier::isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                                                        IE::DepthToSpaceOp d2sOp) const {
    if (d2sOp.getBlockSize() >= 4) {
        return matchFailed(log, rewriter, d2sOp, "mapping D2S to DPU is not benefical: blockSize({0}) >= 4",
                           d2sOp.getBlockSize());  // Better to map larger block size to DMA.
    }

    if (d2sOp.getMode() == IE::DepthToSpaceMode::BLOCKS_FIRST) {
        return matchFailed(
                log, rewriter, d2sOp,
                "mapping D2S to DPU is not benefical: mode == BLOCKS_FIRST");  //  Better to map BLOCKS_FIRST to DMA.
    }

    auto inputType = d2sOp.getInput().getType().cast<NDTypeInterface>();
    auto inputShape = inputType.getShape();
    auto inputChannels = inputShape[Dims4D::Act::C];

    if (inputChannels > BENEFICIAL_INPUT_CHANNEL_MAX) {
        return matchFailed(log, rewriter, d2sOp, "mapping D2S to DPU is not benefical: inputChannels({1}) > {2}",
                           inputChannels, BENEFICIAL_INPUT_CHANNEL_MAX);
    }

    return mlir::success();  // Is efficient to map to DPU.
}
