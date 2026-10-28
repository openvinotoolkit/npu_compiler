//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/layer_post_ops_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace VPU {

static bool checkIfUsersFakeQuantize(mlir::Operation* op) {
    for (auto user : op->getUsers()) {
        if (!mlir::isa<IE::FakeQuantizeOp>(user)) {
            return false;
        }
    }

    return true;
}

bool checkForQuantization(mlir::Operation* op, mlir::Type type) {
    // Either a shaped type or element type
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
    if (ndType != nullptr) {
        type = ndType.getElementType();
    }

    // since FusePostOps is called also after LowPrecisionPipeline
    const auto isQuantizedElemType = mlir::isa<mlir::quant::QuantizedType>(type);

    auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
    auto isFakeQuantizeOpOutput = checkIfUsersFakeQuantize(op);

    return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
}

bool checkForQuantization(mlir::Operation* op, mlir::Operation* postOp) {
    const auto operandType = mlir::cast<vpux::NDTypeInterface>(postOp->getOperand(0).getType());
    // since FusePostOps is called also after LowPrecisionPipeline
    const auto isQuantizedElemType = mlir::isa<mlir::quant::QuantizedType>(operandType.getElementType());

    auto isFakeQuantizeOpInput = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op->getOperand(0).getDefiningOp());
    auto isFakeQuantizeOpOutput = checkIfUsersFakeQuantize(postOp);
    return (isFakeQuantizeOpOutput && isFakeQuantizeOpInput) || isQuantizedElemType;
}

namespace {

bool isPerChannelFakeQuantize(mlir::Operation* op) {
    auto fq = mlir::dyn_cast<IE::FakeQuantizeOp>(op);
    if (fq == nullptr) {
        return false;
    }

    auto inLow = fq.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHigh = fq.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLow = fq.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHigh = fq.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(inLow == nullptr || inHigh == nullptr || outLow == nullptr || outHigh == nullptr,
                    "Got FakeQuantize with non-constant parameters, loc: {0}", fq->getLoc());

    return !inLow.getContentAttr().isSplat() || !inHigh.getContentAttr().isSplat() ||
           !outLow.getContentAttr().isSplat() || !outHigh.getContentAttr().isSplat();
}

}  // namespace

bool hasPerChannelQuantizedOutput(mlir::Operation* op) {
    return llvm::any_of(op->getUsers(), isPerChannelFakeQuantize);
}

bool hasPerChannelQuantizedOutput(mlir::Value output) {
    return llvm::any_of(output.getUsers(), isPerChannelFakeQuantize);
}

}  // namespace VPU
}  // namespace vpux
