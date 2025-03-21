//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

//
// IE::arch37xx
//

bool IE::arch37xx::isMixPrecisionSupported(mlir::Operation* origOp, const bool isPReLUSupported, Logger log) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::AddOp, IE::AvgPoolOp, IE::TransposedConvolutionOp,
                   IE::MatMulOp>(origOp)) {
        return false;
    }

    // Check that the kernel size are not exceding the NCE HW limits
    if (VPU::NCEInvariant::verifyKernel(origOp, log).failed()) {
        return false;
    }

    // If the Add operands have different shapes the operation will be mapped on SHAVE, which does not support mixed
    // precision operations
    if (mlir::isa<IE::AddOp>(origOp)) {
        auto addOp = mlir::dyn_cast<IE::AddOp>(origOp);
        const auto shape1 = getShape(addOp.getInput1());
        const auto shape2 = getShape(addOp.getInput2());
        if (shape1 != shape2)
            return false;
    }

    // Float input with quantized output supports leaky ReLU when quantize out is per-tensor.
    // Further checks are not necessary, bail out.
    if (isPReLUSupported) {
        return true;
    }

    // HW limitations below do not apply to NPU37XX
    // However, leaky ReLU does not work accurately in quant in / float out mode.
    // In quant in / float out flow, PReLU alpha coefficient can only be represented as prelu_mult.
    // prelu_shift is not available in such configuration.
    // Therefore, it becomes problematic to express rational negative slopes.
    // See E#58368 for details.
    const auto hasLeakyReLUConsumer = llvm::any_of(origOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    // Thus, mixed precision is supported only when consumers and post-ops are not leaky ReLU
    return !hasLeakyReLUConsumer && !hasLeakyReLUPostOp(origOp);
}

bool IE::arch37xx::checkPostOp(IE::LayerWithPostOpInterface layerWithPostOp, bool isPerAxisQuantizedOutput,
                               bool isFloatInput, mlir::Location loc) {
    auto postOpName = layerWithPostOp.getPostOp().value().getStringRef();
    auto postOpDictAttr = layerWithPostOp.getPostOpAttrs();
    if (postOpDictAttr != nullptr) {
        // On NPU37XX, NPU40XX, the prelu alpha multiplier used for integer input is unsigned, on
        // floating input it is signed. If input is floating, output is integer, quantize output need to be per tensor,
        // this will check in mix-precision pass
        if (!isFloatInput && postOpName == IE::LeakyReluOp::getOperationName()) {
            IE::LeakyReluOp::Adaptor leakyRelu(std::nullopt, nullptr, toProperties<IE::LeakyReluOp>(postOpDictAttr));
            if (leakyRelu.verify(loc).succeeded()) {
                const auto alpha = leakyRelu.getNegativeSlope().convertToDouble();
                if (alpha < 0.0) {
                    return false;
                }
            }
        } else if (isPerAxisQuantizedOutput && postOpName != IE::ReLUOp::getOperationName()) {
            return false;
        }
    }
    return true;
}
