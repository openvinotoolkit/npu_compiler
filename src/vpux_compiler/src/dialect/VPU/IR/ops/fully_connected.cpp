//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::FullyConnectedOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::FullyConnectedOpAdaptor fullyConnected(operands, attrs, prop);
    if (mlir::failed(fullyConnected.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(fullyConnected.getInput().getType());
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(fullyConnected.getWeights().getType());
    const auto inShape = inType.getShape().raw();
    const auto weightsShape = weightsType.getShape().raw();
    const auto inRank = inShape.size();
    const auto weightsRank = weightsShape.size();

    if (weightsRank != 2 || inRank != 2) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    outShape.push_back(inShape[0]);
    outShape.push_back(weightsShape[0]);

    const auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
