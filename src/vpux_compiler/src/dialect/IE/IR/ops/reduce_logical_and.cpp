//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceLogicalAndOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceLogicalAndOpAdaptor reduceLogicalAnd(operands, attrs, prop);
    if (mlir::failed(reduceLogicalAnd.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceLogicalAnd.getInput();
    const auto keepDims = reduceLogicalAnd.getKeepDims();
    auto axesValue = parseIntArrayAttr<int64_t>(reduceLogicalAnd.getAxesValue());
    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceLogicalAndOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
