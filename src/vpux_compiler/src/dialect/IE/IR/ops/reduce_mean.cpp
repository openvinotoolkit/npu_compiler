//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void IE::ReduceMeanOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Type outputType,
                             mlir::Value input, mlir::Value axes, mlir::ArrayAttr axesValue, mlir::UnitAttr keepDims) {
    return build(odsBuilder, odsState, outputType, input, axes, axesValue, keepDims, /*outputPadding=*/nullptr,
                 /*inputPadding=*/nullptr);
}

void IE::ReduceMeanOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                             mlir::Value axes, mlir::ArrayAttr axesValue, mlir::UnitAttr keepDims) {
    return build(odsBuilder, odsState, input, axes, axesValue, keepDims, /*outputPadding=*/nullptr,
                 /*inputPadding=*/nullptr);
}

mlir::LogicalResult vpux::IE::ReduceMeanOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceMeanOpAdaptor reduceMean(operands, attrs, prop);
    if (mlir::failed(reduceMean.verify(loc))) {
        return mlir::failure();
    }
    if (reduceMean.getAxes() != nullptr && reduceMean.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceMean.getAxes() == nullptr && !reduceMean.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceMean.getInput();
    const auto keepDims = reduceMean.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceMean);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes,
                                               reduceMean.getInputPaddingAttr(), reduceMean.getOutputPaddingAttr());
}

mlir::LogicalResult vpux::IE::ReduceMeanOp::verify() {
    const auto op = getOperation();

    if (mlir::failed(IE::checkPadding(getInputPaddingAttr(), getInput().getType()))) {
        return errorAt(op, "Input padding {0} incompatible with input type {1}", getInputPaddingAttr(),
                       getInput().getType());
    }
    if (mlir::failed(IE::checkPadding(getOutputPaddingAttr(), getOutput().getType()))) {
        return errorAt(op, "Output padding {0} incompatible with output type {1}", getOutputPaddingAttr(),
                       getOutput().getType());
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceMeanOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        if (getInputPaddingAttr() == nullptr && getOutputPaddingAttr() == nullptr) {
            return getInput();
        }

        // In case the operation has padding, check if the non-padded shapes are the same. If they are, the operation
        // can be folded as there is nothing to reduce on the given axes
        auto inputShape = SmallVector<int64_t>(mlir::cast<NDTypeInterface>(getInput().getType()).getShape().raw());
        if (getInputPaddingAttr() != nullptr) {
            auto inputPadding = parseIntArrayAttr<int64_t>(getInputPaddingAttr());
            for (size_t i = 0; i < inputShape.size(); ++i) {
                inputShape[i] -= inputPadding[i];
            }
        }
        auto outputShape = SmallVector<int64_t>(mlir::cast<NDTypeInterface>(getOutput().getType()).getShape().raw());
        if (getOutputPaddingAttr() != nullptr) {
            auto outputPadding = parseIntArrayAttr<int64_t>(getOutputPaddingAttr());
            for (size_t i = 0; i < outputShape.size(); ++i) {
                outputShape[i] -= outputPadding[i];
            }
        }
        if (inputShape == outputShape) {
            return getInput();
        }
    }

    return nullptr;
}

void vpux::IE::ReduceMeanOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceMeanOp>>(context);
}
