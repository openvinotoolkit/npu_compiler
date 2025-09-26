//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

using namespace vpux;

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

mlir::LogicalResult vpux::IE::BroadcastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BroadcastOpAdaptor broadcast(operands, attrs, prop);
    if (mlir::failed(broadcast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(broadcast.getInput().getType());

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inType), "{0} doesn't support dynamic shapes",
                      IE::BroadcastOp::getOperationName());

    auto inShape = to_small_vector(inType.getShape());
    const auto broadcastMode = broadcast.getMode().has_value() ? broadcast.getMode().value() : IE::BroadcastType::NUMPY;

    auto outShape = IE::constInputToData(loc, broadcast.getTargetShape()).value();
    if (broadcastMode == IE::BroadcastType::BIDIRECTIONAL) {
        outShape = getResultShapeBidirectional(inShape, outShape);
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::BroadcastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    // move broadcast to const attribute.
    if (auto ephemeral = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto contentAttr = static_cast<Const::ContentAttr>(ephemeral);
        const auto inputShape = to_small_vector(getShape(getInput()));
        const auto outputShape = to_small_vector(getShape(getOutput()));
        const auto broadcastType = getMode().value_or(IE::BroadcastType::NUMPY);
        SmallVector<int64_t> broadcastAxes;

        VPUX_THROW_WHEN(inputShape.size() > outputShape.size(),
                        "Invalid BroadcastOp, inputShape size {0}..outputShape size {1}", inputShape.size(),
                        outputShape.size());

        // Finds the axes over which the broadcasting rules apply. For example:
        // NUMPY and BIDIRECTIONAL: input 16x1x1, output 1x16x50x50 will return the axes [0, 2, 3]
        // EXPLICIT:                input 16, output 1x16x50x50, axesMapping 1 will return the axes [0, 2, 3]
        if (broadcastType == IE::BroadcastType::BIDIRECTIONAL || broadcastType == IE::BroadcastType::NUMPY) {
            broadcastAxes = vpux::IE::getBroadcastAxesNumpyBidirectional(inputShape, outputShape);
        } else if (broadcastType == IE::BroadcastType::EXPLICIT) {
            auto axesMapping = IE::constInputToData(getLoc(), getAxesMapping()).value();
            broadcastAxes = vpux::IE::getBroadcastAxesExplicit(axesMapping, outputShape);
        } else {
            VPUX_THROW("Invalid broadcast type..{0}", broadcastType);
        }

        auto adjustedInputShape = inputShape;
        for (const auto& axis : broadcastAxes) {
            if (adjustedInputShape.size() < outputShape.size()) {
                adjustedInputShape.insert(adjustedInputShape.begin() + axis, 1);
            }
        }

        auto contentAttrSetup = contentAttr.transform();

        if (adjustedInputShape != inputShape) {
            contentAttrSetup = contentAttrSetup.reshape(ShapeRef(adjustedInputShape));
        }

        for (const auto& dim : enumerate(outputShape)) {
            if (dim.value() > 1 && dim.value() != adjustedInputShape[dim.index()]) {
                contentAttrSetup = contentAttrSetup.broadcast(Dim(dim.index()), outputShape[dim.index()]);
            }
        }

        return contentAttrSetup.get();
    }

    return nullptr;
}
