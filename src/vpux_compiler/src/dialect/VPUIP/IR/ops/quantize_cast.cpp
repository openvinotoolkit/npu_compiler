//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/Diagnostics.h>

using namespace vpux;

mlir::Value VPUIP::QuantizeCastOp::getViewSource() {
    return getInput();
}

namespace {

bool isSupportedQuantizeCastType(mlir::Location loc, mlir::Type inputType, mlir::Type outputType,
                                 LogCb logCb = emptyLogCb) {
    auto inputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(inputType);
    auto outputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(outputType);
    if (inputNDType == nullptr || outputNDType == nullptr) {
        logCb(formatv("QuantizeCastOp input and output must be ND types: in type = {0}, out type = {1}", inputType,
                      outputType));
        return false;
    }

    auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
    auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputType);
    if (inputDistType != nullptr && outputDistType != nullptr &&
        inputDistType.getDistribution() != outputDistType.getDistribution()) {
        logCb(formatv("QuantizeCastOp input and output must have the same distribution attribute"));
        return false;
    }

    if (inputNDType.getStrides() != outputNDType.getStrides()) {
        logCb(formatv("QuantizeCastOp input and output must have the same strides, but got {0} and {1}",
                      inputNDType.getStrides(), outputNDType.getStrides()));
        return false;
    }

    if (inputNDType.getShape() != outputNDType.getShape()) {
        logCb(formatv("QuantizeCastOp input and output must have the same shape, but got {0} and {1}",
                      inputNDType.getShape(), outputNDType.getShape()));
        return false;
    }

    SmallVector<std::string> diagnosticMessages;
    const auto isValid = [&]() {
        mlir::ScopedDiagnosticHandler diagnosticGuard(loc.getContext(), [&diagnosticMessages](mlir::Diagnostic& diag) {
            diagnosticMessages.push_back(diag.str());
            return mlir::success();
        });
        return mlir::succeeded(
                vpux::isQuantizeCastValid(loc, inputNDType.getElementType(), outputNDType.getElementType()));
    }();
    for (const auto& msg : diagnosticMessages) {
        logCb(formatv("{0}", msg));
    }
    return isValid;
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::QuantizeCastOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr) {
        return std::nullopt;
    }

    // QuantizeCast changes only the element type.
    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto dstElemType = origOutputNDType.getElementType();
    auto newInputStrides = newInputNDType.getStrides();
    auto typeComps = TypeComponents().setElementType(dstElemType).setStrides(newInputStrides);
    auto outType = newInputNDType.changeTypeComponents(typeComps);
    const auto outputType = mlir::cast<mlir::Type>(outType);
    if (!isSupportedQuantizeCastType(getLoc(), newInputType, outputType)) {
        return std::nullopt;
    }
    return outputType;
}

std::optional<mlir::Type> VPUIP::QuantizeCastOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    auto desiredOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(desiredOutputType);
    if (desiredOutputNDType == nullptr) {
        return std::nullopt;
    }
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    // Reverse direction restores the original storage element type while
    // following the desired output's shape/order/space.
    const auto desiredOutputStrides = desiredOutputNDType.getStrides();
    auto typeComps = TypeComponents().setElementType(origInputNDType.getElementType()).setStrides(desiredOutputStrides);
    auto inType = desiredOutputNDType.changeTypeComponents(typeComps);
    auto inputType = mlir::cast<mlir::Type>(inType);
    if (!isSupportedQuantizeCastType(getLoc(), inputType, desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

mlir::OpFoldResult VPUIP::QuantizeCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}

//
// FuseQuantizeCastOps
//

namespace {

class FuseQuantizeCastOps final : public mlir::OpRewritePattern<VPUIP::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCastOps::matchAndRewrite(VPUIP::QuantizeCastOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto producerQuantizeCastOp = origOp.getInput().getDefiningOp<VPUIP::QuantizeCastOp>();
    if (producerQuantizeCastOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::QuantizeCastOp>(origOp, origOp.getType(), producerQuantizeCastOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCastOps>(ctx);
}

mlir::LogicalResult vpux::VPUIP::QuantizeCastOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedQuantizeCastType(getLoc(), getInput().getType(), getOutput().getType(), logCb));
}
