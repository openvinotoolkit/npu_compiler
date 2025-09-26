//
// Copyright (C) 2024-2025 Intel Corporation.
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

mlir::LogicalResult vpux::IE::BatchToSpace::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BatchToSpaceAdaptor bsp(operands, attrs, prop);
    if (mlir::failed(bsp.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(bsp.getInput().getType());
    const auto inputShape = inputType.getShape().raw();

    SmallVector<int64_t> blockShapeVal = {0};
    SmallVector<int64_t> cropsBeginVal = {0};
    SmallVector<int64_t> cropsEndVal = {0};

    if (bsp.getBlockShape() != nullptr || bsp.getBlockShapeValue().has_value()) {
        auto blockShape = getConstOrArrAttrValue(bsp.getBlockShape(), bsp.getBlockShapeValueAttr());
        if (mlir::failed(blockShape)) {
            return mlir::failure();
        }
        blockShapeVal = blockShape.value();
    }

    if (bsp.getCropsBegin() != nullptr || bsp.getCropsBeginValue().has_value()) {
        auto cropsBegin = getConstOrArrAttrValue(bsp.getCropsBegin(), bsp.getCropsBeginValueAttr());
        if (mlir::failed(cropsBegin)) {
            return mlir::failure();
        }
        cropsBeginVal = cropsBegin.value();
    }

    if (bsp.getCropsEnd() != nullptr || bsp.getCropsEndValue().has_value()) {
        auto cropsEnd = getConstOrArrAttrValue(bsp.getCropsEnd(), bsp.getCropsEndValueAttr());
        if (mlir::failed(cropsEnd)) {
            return mlir::failure();
        }
        cropsEndVal = cropsEnd.value();
    }

    if (inputShape.size() < 2) {
        return errorAt(loc, "Input tensor rank should be 2 or greater.");
    }

    if (inputShape.size() != blockShapeVal.size() || inputShape.size() != cropsBeginVal.size() ||
        inputShape.size() != cropsEndVal.size()) {
        return errorAt(loc,
                       "blockShape, cropsBegin, cropsEnd shape[N] should be equal to the size of Input shape. Got "
                       "blockShape [{0}], cropsBegin [{1}], cropsEnd [{2}]",
                       blockShapeVal.size(), cropsBeginVal.size(), cropsEndVal.size());
    }

    auto outShape = SmallVector<int64_t>(inputShape.size());

    outShape[0] = inputShape[0] /
                  std::accumulate(blockShapeVal.begin(), blockShapeVal.end(), int64_t(1), std::multiplies<int64_t>());

    for (size_t i = 1; i < inputShape.size(); i++) {
        outShape[i] = inputShape[i] * blockShapeVal[i] - cropsBeginVal[i] - cropsEndVal[i];
    }

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inputType), "{0} doesn't support dynamic shapes",
                      IE::BatchToSpace::getOperationName());
    const auto outDesc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::BatchToSpace> {
public:
    using mlir::OpRewritePattern<IE::BatchToSpace>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::BatchToSpace BatchToSpace, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::BatchToSpace BatchToSpace,
                                                        mlir::PatternRewriter& rewriter) const {
    if (BatchToSpace.getBlockShapeValue().has_value() || BatchToSpace.getCropsBeginValue().has_value() ||
        BatchToSpace.getCropsEndValue().has_value()) {
        return mlir::failure();
    }

    SmallVector<int64_t> blockShapeVal = {0};
    SmallVector<int64_t> cropsBeginVal = {0};
    SmallVector<int64_t> cropsEndVal = {0};

    if (BatchToSpace.getBlockShape() != nullptr) {
        const auto blockShape = getConstArrValue(BatchToSpace.getBlockShape());
        if (mlir::failed(blockShape)) {
            return mlir::failure();
        }
        blockShapeVal = blockShape.value();
    }

    if (BatchToSpace.getCropsBegin() != nullptr) {
        const auto cropsBegin = getConstArrValue(BatchToSpace.getCropsBegin());
        if (mlir::failed(cropsBegin)) {
            return mlir::failure();
        }
        cropsBeginVal = cropsBegin.value();
    }

    if (BatchToSpace.getCropsEnd() != nullptr) {
        const auto cropsEnd = getConstArrValue(BatchToSpace.getCropsEnd());
        if (mlir::failed(cropsEnd)) {
            return mlir::failure();
        }
        cropsEndVal = cropsEnd.value();
    }

    rewriter.replaceOpWithNewOp<IE::BatchToSpace>(
            BatchToSpace, BatchToSpace.getType(), BatchToSpace.getInput(), nullptr, nullptr, nullptr,
            getIntArrayAttr(rewriter.getContext(), blockShapeVal),
            getIntArrayAttr(rewriter.getContext(), cropsBeginVal), getIntArrayAttr(rewriter.getContext(), cropsEndVal));
    return mlir::success();
}

}  // namespace

void vpux::IE::BatchToSpace::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
