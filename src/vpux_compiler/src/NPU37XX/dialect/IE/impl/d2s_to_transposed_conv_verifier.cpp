//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include <vpux/compiler/utils/error.hpp>
#include <vpux/utils/core/error.hpp>

using namespace vpux::IE::arch37xx;

//
// D2SToTransposedConvVerifier
//

// For more information on heuristic, see:
// - E#125463
// - E#113159

mlir::LogicalResult D2SToTransposedConvVerifier::isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                                                        IE::DepthToSpaceOp d2sOp) const {
    if (d2sOp.getBlockSize() >= 4) {
        return matchFailed(log, rewriter, d2sOp, "mapping D2S to DPU is not benefical: blockSize({0}) >= 4",
                           d2sOp.getBlockSize());
    }

    auto outputType = d2sOp.getOutput().getType().cast<NDTypeInterface>();
    auto outputChannels = outputType.getShape()[Dims4D::Act::C];

    auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    if (outputChannels < alignment) {
        return matchFailed(log, rewriter, d2sOp,
                           "mapping D2S to DPU is not benefical: outputChannels({0}) < alignment({1})", outputChannels,
                           alignment);
    }

    return mlir::success();
}
