//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

bool IE::arch50xx::isMixPrecisionSupported(mlir::Operation* origOp, const bool, Logger log) {
    if (!mlir::isa<IE::LayerWithPostOpInterface>(origOp)) {
        return false;
    }

    // Check that the kernel size are not exceding the NCE HW limits
    if (VPU::NCEInvariant::verifyKernel(origOp, log).failed()) {
        return false;
    }

    // If the Eltwise operands have different shapes the operation will be mapped on SHAVE, which does not support mixed
    // precision operations
    if (mlir::isa<IE::AddOp, IE::MultiplyOp, IE::SubtractOp>(origOp)) {
        const auto shape1 = getShape(origOp->getOperand(0));
        const auto shape2 = getShape(origOp->getOperand(1));
        if (shape1 != shape2) {
            return false;
        }
    }

    const auto hasLeakyReLUConsumer = llvm::any_of(origOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    // Thus, mixed precision is supported only when consumers and post-ops are not leaky ReLU
    return !hasLeakyReLUConsumer && !hasLeakyReLUPostOp(origOp);
}

bool IE::arch50xx::checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput,
                               bool isFloatInput) {
    VPUX_UNUSED(isFloatInput);
    const auto postOp = layerWithPostOp.getPostOp();

    // Because in the PPE pipeline the quantization scale happens before post-op effects are applied, the following
    // limitation occurs: If we have per axis quantization at output this would produce per axis Clamp intervals and
    // LeakyReLU alphas which would not be supported.
    return !isPerAxisQuantizedOutput || mlir::isa<IE::ReluAttr>(postOp);
}
