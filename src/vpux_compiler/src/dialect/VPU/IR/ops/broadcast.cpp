//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

namespace {

SmallVector<int64_t> getResultShapeBidirectional(SmallVector<int64_t>& inShape, SmallVector<int64_t>& targetShape) {
    const auto targetPaddedRank = std::max(inShape.size(), targetShape.size());

    SmallVector<int64_t> resultShape(targetPaddedRank);

    while (inShape.size() < targetPaddedRank) {
        inShape.insert(inShape.begin(), 1);
    }

    while (targetShape.size() < targetPaddedRank) {
        targetShape.insert(targetShape.begin(), 1);
    }

    for (size_t i = 0; i < targetPaddedRank; ++i) {
        VPUX_THROW_UNLESS(inShape[i] == 1 || targetShape[i] == 1 || inShape[i] == targetShape[i],
                          "Broadcast incorrect target shape. Expecting either 1 or {0}. Got {1}", inShape[i],
                          targetShape[i]);

        resultShape[i] = std::max(inShape[i], targetShape[i]);
    }

    return resultShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::BroadcastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::BroadcastOpAdaptor broadcast(operands, attrs, prop);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(broadcast.getInput().getType());
    auto inShape = to_small_vector(inType.getShape().raw());
    const auto broadcastMode = broadcast.getMode().value();

    auto outShape = IE::constInputToData(loc, broadcast.getTargetShape()).value();
    if (broadcastMode == IE::BroadcastType::BIDIRECTIONAL) {
        outShape = getResultShapeBidirectional(inShape, outShape);
    }

    auto outType = inType.changeShape(ShapeRef(outShape));
    outType = outType.changeDimsOrder(DimsOrder::fromNumDims(outShape.size()));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
