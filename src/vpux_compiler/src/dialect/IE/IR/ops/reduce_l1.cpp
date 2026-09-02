//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceL1Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceL1OpAdaptor reduceL1(operands, attrs, prop);
    if (mlir::failed(reduceL1.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceL1.getInput();
    const auto keepDims = reduceL1.getKeepDims();
    auto axesValue = parseIntArrayAttr<int64_t>(reduceL1.getAxesValue());
    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceL1Op::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
