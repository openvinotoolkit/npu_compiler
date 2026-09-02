//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceLogicalOrOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceLogicalOrOpAdaptor reduceLogicalOr(operands, attrs, prop);
    if (mlir::failed(reduceLogicalOr.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceLogicalOr.getInput();
    const auto keepDims = reduceLogicalOr.getKeepDims();
    auto axesValue = parseIntArrayAttr<int64_t>(reduceLogicalOr.getAxesValue());
    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceLogicalOrOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
