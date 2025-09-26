//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

mlir::LogicalResult vpux::IE::SpaceToBatch::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SpaceToBatchAdaptor spb(operands, attrs, prop);
    if (mlir::failed(spb.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(spb.getInput().getType());
    const auto inputShape = inputType.getShape().raw();

    SmallVector<int64_t> blockShapeVal = {0};
    SmallVector<int64_t> padsBeginVal = {0};
    SmallVector<int64_t> padsEndVal = {0};

    if (spb.getBlockShape() != nullptr || spb.getBlockShapeValue().has_value()) {
        auto blockShape = getConstOrArrAttrValue(spb.getBlockShape(), spb.getBlockShapeValueAttr());
        if (mlir::failed(blockShape)) {
            return mlir::failure();
        }
        blockShapeVal = blockShape.value();
    }

    if (spb.getPadsBegin() != nullptr || spb.getPadsBeginValue().has_value()) {
        auto padsBegin = getConstOrArrAttrValue(spb.getPadsBegin(), spb.getPadsBeginValueAttr());
        if (mlir::failed(padsBegin)) {
            return mlir::failure();
        }
        padsBeginVal = padsBegin.value();
    }

    if (spb.getPadsEnd() != nullptr || spb.getPadsEndValue().has_value()) {
        auto padsEnd = getConstOrArrAttrValue(spb.getPadsEnd(), spb.getPadsEndValueAttr());
        if (mlir::failed(padsEnd)) {
            return mlir::failure();
        }
        padsEndVal = padsEnd.value();
    }

    if (inputShape.size() < 3 || inputShape.size() > 5) {
        return errorAt(loc, "Input tensor rank only support 3, 4 or 5. Got {0}D tensor", inputShape.size());
    }

    if (inputShape.size() != blockShapeVal.size() || inputShape.size() != padsBeginVal.size() ||
        inputShape.size() != padsEndVal.size()) {
        return errorAt(loc,
                       "blockShape, padsBegin, padsEnd shape[N] should be equal to the size of Input shape. Got "
                       "blockShape [{0}], padsBegin [{1}], padsEnd [{2}]",
                       blockShapeVal.size(), padsBeginVal.size(), padsEndVal.size());
    }

    auto outShape = SmallVector<int64_t>(inputShape.size());

    outShape[0] = inputShape[0] *
                  std::accumulate(blockShapeVal.begin(), blockShapeVal.end(), int64_t(1), std::multiplies<int64_t>());

    for (size_t i = 1; i < inputShape.size(); i++) {
        outShape[i] = (inputShape[i] + padsBeginVal[i] + padsEndVal[i]) / blockShapeVal[i];
    }

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inputType), "{0} doesn't support dynamic shapes",
                      IE::SpaceToBatch::getOperationName());
    const auto outDesc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SpaceToBatch> {
public:
    using mlir::OpRewritePattern<IE::SpaceToBatch>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SpaceToBatch spacetobatch, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SpaceToBatch spacetobatch,
                                                        mlir::PatternRewriter& rewriter) const {
    if (spacetobatch.getBlockShapeValue().has_value() || spacetobatch.getPadsBeginValue().has_value() ||
        spacetobatch.getPadsEndValue().has_value()) {
        return mlir::failure();
    }

    SmallVector<int64_t> blockShapeVal = {0};
    SmallVector<int64_t> padsBeginVal = {0};
    SmallVector<int64_t> padsEndVal = {0};

    if (spacetobatch.getBlockShape() != nullptr) {
        const auto blockShape = getConstArrValue(spacetobatch.getBlockShape());
        if (mlir::failed(blockShape)) {
            return mlir::failure();
        }
        blockShapeVal = blockShape.value();
    }

    if (spacetobatch.getPadsBegin() != nullptr) {
        const auto padsBegin = getConstArrValue(spacetobatch.getPadsBegin());
        if (mlir::failed(padsBegin)) {
            return mlir::failure();
        }
        padsBeginVal = padsBegin.value();
    }

    if (spacetobatch.getPadsEnd() != nullptr) {
        const auto padsEnd = getConstArrValue(spacetobatch.getPadsEnd());
        if (mlir::failed(padsEnd)) {
            return mlir::failure();
        }
        padsEndVal = padsEnd.value();
    }

    rewriter.replaceOpWithNewOp<IE::SpaceToBatch>(
            spacetobatch, spacetobatch.getType(), spacetobatch.getInput(), nullptr, nullptr, nullptr,
            getIntArrayAttr(rewriter.getContext(), blockShapeVal), getIntArrayAttr(rewriter.getContext(), padsBeginVal),
            getIntArrayAttr(rewriter.getContext(), padsEndVal));
    return mlir::success();
}

}  // namespace

void vpux::IE::SpaceToBatch::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
