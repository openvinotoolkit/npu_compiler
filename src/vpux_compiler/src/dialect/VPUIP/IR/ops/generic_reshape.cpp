//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

bool isSupportedGenericReshapeType(mlir::Type inputType, mlir::Type outputType, LogCb logCb = emptyLogCb) {
    auto inputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(inputType);
    auto outputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(outputType);
    if (inputNDType == nullptr || outputNDType == nullptr) {
        logCb(formatv("GenericReshape input and output must be ND types: in type = {0}, out type = {1}", inputType,
                      outputType));
        return false;
    }

    auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(inputType);
    auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(outputType);
    if (inputDistType != nullptr && outputDistType != nullptr &&
        !VPUIP::isDistributedCompatibleAfterShapeChangeForViewOps<VPUIP::DistributedBufferType>(inputDistType,
                                                                                                outputDistType)) {
        logCb(formatv("Reshape has incompatible output shape as clustering: in type = {0}, out type = {1}",
                      inputDistType, outputDistType));
        return false;
    }

    if (inputNDType.getNumElements() != outputNDType.getNumElements()) {
        logCb(formatv("Reshape input and output must have the same number of elements"));
        return false;
    }

    if (!VPUIP::isInAndOutStridesCompatible(inputNDType, outputNDType)) {
        logCb(formatv("Incompatible strides between input {0} and output {1}", inputNDType, outputNDType));
        return false;
    }

    return true;
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::GenericReshapeOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };
    return mlir::success(isSupportedGenericReshapeType(getInput().getType(), getOutput().getType(), logCb));
}

mlir::Value VPUIP::GenericReshapeOp::getViewSource() {
    return getInput();
}

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::GenericReshapeOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr) {
        return std::nullopt;
    }

    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outShape = origOutputNDType.getShape();

    auto outType = VPUIP::inferReshapeOutputType(newInputNDType, origInputNDType, origOutputNDType, outShape);
    if (!outType.has_value()) {
        return std::nullopt;
    }

    auto outNDType = mlir::cast<vpux::NDTypeInterface>(outType.value());
    if (isSupportedGenericReshapeType(mlir::cast<mlir::Type>(newInputNDType), mlir::cast<mlir::Type>(outNDType))) {
        return mlir::cast<mlir::Type>(outNDType);
    }

    // A shape-only reshape can still need stride repair after the input type is
    // rewritten. Example: if the new input has one non-compact stride dim and
    // the reshape preserves the same left/right memory products around that dim,
    // update the output strides to keep the view relation valid.
    auto strideUpdatedOutType = VPUIP::updateStridesForReshape(newInputNDType, outNDType);
    if (mlir::failed(strideUpdatedOutType)) {
        return std::nullopt;
    }
    if (!isSupportedGenericReshapeType(mlir::cast<mlir::Type>(newInputNDType),
                                       mlir::cast<mlir::Type>(strideUpdatedOutType.value()))) {
        return std::nullopt;
    }
    return mlir::cast<mlir::Type>(strideUpdatedOutType.value());
}

std::optional<mlir::Type> VPUIP::GenericReshapeOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    auto desiredOutputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(desiredOutputType);
    if (desiredOutputNDType == nullptr) {
        return std::nullopt;
    }
    auto origInputNDType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto origOutputNDType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto inputType =
            VPUIP::inferReshapeInputType(getOperation(), desiredOutputNDType, origInputNDType, origOutputNDType);
    if (!inputType.has_value() || !isSupportedGenericReshapeType(inputType.value(), desiredOutputType)) {
        return std::nullopt;
    }
    return inputType;
}

mlir::OpFoldResult VPUIP::GenericReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(cst).transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPUIP::GenericReshapeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::GenericReshapeOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPUIP::GenericReshapeOp origOp,
                                                  mlir::PatternRewriter& rewriter) const {
    auto producerReshapeOp = origOp.getInput().getDefiningOp<VPUIP::GenericReshapeOp>();
    if (producerReshapeOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, origOp.getOutput().getType(),
                                                         producerReshapeOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::GenericReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseReshapes>(ctx);
}
