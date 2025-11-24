//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::SDPAExtendedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SDPAExtendedOpAdaptor sdpaExtended(operands, attrs, prop);
    if (mlir::failed(sdpaExtended.verify(loc))) {
        return mlir::failure();
    }
    const auto inQType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputQ().getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inKType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputK().getType());
    const auto inKShape = inKType.getShape().raw();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputV().getType());
    const auto inVShape = inVType.getShape().raw();

    const auto isTransposedV = inKShape[rank - 2] != inVShape[rank - 2];
    const auto Ev = isTransposedV ? inVShape[rank - 2] : inVShape[rank - 1];
    SmallVector<int64_t> outShape(inQShape.begin(), inQShape.end());
    outShape[rank - 1] = Ev;
    inferredReturnShapes.emplace_back(outShape, inQType.getElementType());

    return mlir::success();
}
