//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::OneHotOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::OneHotOpAdaptor oneHot(operands, attrs, prop);
    if (mlir::failed(oneHot.verify(loc))) {
        return mlir::failure();
    }

    auto inType = mlir::cast<vpux::NDTypeInterface>(oneHot.getInput().getType());
    mlir::Type outType = oneHot.getOutputType();
    auto anyRankedTensorType = inType.changeElemType(outType);

    SmallVector<int64_t> outShape =
            to_small_vector(mlir::cast<vpux::NDTypeInterface>(oneHot.getInput().getType()).getShape());
    const auto axis = oneHot.getAxis();
    int64_t depth = oneHot.getDepthAttr().getInt();
    if (axis < 0) {
        outShape.insert(outShape.end() + 1 + axis, depth);
    } else {
        outShape.insert(outShape.begin() + axis, depth);
    }

    inferredReturnTypes.push_back(anyRankedTensorType.changeShape(ShapeRef(outShape)));

    return mlir::success();
}
