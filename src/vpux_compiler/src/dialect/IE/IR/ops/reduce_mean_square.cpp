//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceMeanSquareOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceMeanSquareOpAdaptor reduceMeanSquare(operands, attrs, prop);
    if (mlir::failed(reduceMeanSquare.verify(loc))) {
        return mlir::failure();
    }
    if (reduceMeanSquare.getAxes() != nullptr && reduceMeanSquare.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceMeanSquare.getAxes() == nullptr && !reduceMeanSquare.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceMeanSquare.getInput();
    const auto keepDims = reduceMeanSquare.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceMeanSquare);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceMeanSquareOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

void vpux::IE::ReduceMeanSquareOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                               mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceMeanSquareOp>>(context);
}
