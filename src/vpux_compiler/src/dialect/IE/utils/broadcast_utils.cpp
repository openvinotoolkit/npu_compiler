//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux {
namespace IE {

SmallVector<int64_t> getBroadcastAxesNumpyBidirectional(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape) {
    SmallVector<int64_t> broadcastAxes;
    const auto startAxis = static_cast<int64_t>(outputShape.size()) - static_cast<int64_t>(inputShape.size());
    VPUX_THROW_UNLESS(startAxis >= 0, "Broadcast axes not known deterministically");
    for (int64_t i = 0; i < static_cast<int64_t>(outputShape.size()); i++) {
        if (i < startAxis || outputShape[i] != inputShape[i - startAxis]) {
            broadcastAxes.push_back(i);
        }
    }
    return broadcastAxes;
}

SmallVector<int64_t> getBroadcastAxesExplicit(ArrayRef<int64_t> axesMapping, ArrayRef<int64_t> outputShape) {
    SmallVector<int64_t> broadcastAxes(outputShape.size());
    std::iota(broadcastAxes.begin(), broadcastAxes.end(), 0);
    for (auto i = axesMapping.rbegin(); i != axesMapping.rend(); ++i) {
        broadcastAxes.erase(broadcastAxes.begin() + *i);
    }
    return broadcastAxes;
}

mlir::Value createShapeConstForBroadcast(mlir::OpBuilder& rewriter, mlir::Location loc, ShapeRef shape) {
    const auto ctx = rewriter.getContext();
    const auto shapeStorageType = mlir::RankedTensorType::get({static_cast<int64_t>(shape.size())}, getSInt64Type(ctx));

    return Const::createConst(rewriter, loc, shapeStorageType, shape.raw(), [&](Const::ContentSetup& setup) {
        return setup.castElemType(getSInt32Type(ctx));
    });
}

mlir::Value createBroadcast(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, ShapeRef targetShape,
                            mlir::Value axisMapping, IE::BroadcastTypeAttr broadcastTypeAttr) {
    const auto ctx = builder.getContext();
    if (broadcastTypeAttr == nullptr) {
        broadcastTypeAttr = IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY);
    }

    auto targetShapeConst = createShapeConstForBroadcast(builder, appendLoc(loc, "shape"), targetShape);
    return builder.createOrFold<IE::BroadcastOp>(loc, input, targetShapeConst, axisMapping, broadcastTypeAttr);
}

mlir::Value createDynamicBroadcast(mlir::OpBuilder& rewriter, mlir::Location loc, mlir::Value input, mlir::Value other,
                                   mlir::Value axisMapping, IE::BroadcastTypeAttr broadcastTypeAttr) {
    // Broadcasts the input to the dynamic shape of another value:
    //  [input]  [other]
    //     |        |
    //     |     ShapeOf
    //     |       /
    // DynamicBroadcast

    const auto ctx = rewriter.getContext();
    if (broadcastTypeAttr == nullptr) {
        broadcastTypeAttr = IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY);
    }

    const auto targetType = mlir::cast<NDTypeInterface>(other.getType());

    const auto shapeAttr = getIntArrayAttr(ctx, targetType.getShape());
    const auto boundsAttr =
            llvm::TypeSwitch<mlir::Type, mlir::ArrayAttr>(targetType)
                    .Case<Core::BoundedTensorType>([&](const auto boundedTensor) {
                        return getIntArrayAttr(ctx, boundedTensor.getBounds().raw());
                    })
                    .Default([](const auto type) {
                        VPUX_THROW("Got static shaped type: {0}, a regular Broadcast could probably be used", type);
                        return nullptr;
                    });

    const auto shapeDstElemType = getSInt32Type(ctx);
    const auto targetShape =
            IE::ShapeOfOp::create(rewriter, appendLoc(loc, "shape_of"), other, shapeDstElemType).getOutput();

    return rewriter.createOrFold<IE::DynamicBroadcastOp>(loc, input, targetShape, axisMapping, broadcastTypeAttr,
                                                         shapeAttr, boundsAttr);
}

Const::ReshapeAttr makeReshape(mlir::MLIRContext* ctx, ShapeRef newShape) {
    return Const::ReshapeAttr::get(getIntArrayAttr(ctx, newShape));
}

Const::BroadcastAttr makeBroadcast(mlir::MLIRContext* ctx, Dim axis, int64_t value) {
    return Const::BroadcastAttr::get(getIntAttr(ctx, axis.ind()), getIntAttr(ctx, value));
}

mlir::LogicalResult broadcastAlignShapes(mlir::MLIRContext* ctx, Const::Content& x, Const::Content& y,
                                         const Logger& log) {
    if (x.isSplat()) {
        auto reshaper = makeReshape(ctx, y.getType().getShape());
        x = reshaper.transform(x);
        return mlir::success();
    }
    if (y.isSplat()) {
        auto reshaper = makeReshape(ctx, x.getType().getShape());
        y = reshaper.transform(y);
        return mlir::success();
    }

    const auto xShape = x.getType().getShape();
    const auto yShape = y.getType().getShape();
    if (xShape.size() != yShape.size()) {
        log.trace("Shape sizes of buffers differ: {0} vs {1}", xShape.size(), yShape.size());
        return mlir::failure();
    }

    for (size_t i = 0; i < xShape.size(); ++i) {
        const auto dim = Dim(i);
        const auto xSize = xShape[dim];
        const auto ySize = yShape[dim];
        if (xSize == ySize) {
            continue;
        }
        if (xSize != 1 && ySize != 1) {
            log.trace("Cannot broadcast ambiguous shapes at dim #{0}: {1}, {2}", i, xShape, yShape);
            return mlir::failure();
        }

        if (const bool needToBroadcastX = ySize > 1; needToBroadcastX) {
            auto broadcaster = makeBroadcast(ctx, dim, ySize);
            x = broadcaster.transform(x);
        } else {
            auto broadcaster = makeBroadcast(ctx, dim, xSize);
            y = broadcaster.transform(y);
        }
    }

    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
