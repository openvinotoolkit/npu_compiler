//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getDefaultIndex())) {
        return errorAt(*this, "default_index should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::EmbeddingBagOffsetsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::EmbeddingBagOffsetsSumOpAdaptor embeddingBag(operands, attrs, prop);
    if (mlir::failed(embeddingBag.verify(loc))) {
        return mlir::failure();
    }

    const auto inTypeEmbTable = embeddingBag.getEmbTable().getType().cast<mlir::ShapedType>();

    SmallVector<int64_t> outShape(to_small_vector(inTypeEmbTable.getShape()));

    if (embeddingBag.getOffsets() != nullptr) {
        const auto inTypeOffsets = embeddingBag.getOffsets().getType().cast<mlir::ShapedType>();
        const auto offsetsShape = inTypeOffsets.getShape();
        outShape[0] = checked_cast<int64_t>(offsetsShape[0]);
    } else if (embeddingBag.getOffsetsValue().has_value()) {
        const auto offsetsAttr = parseIntArrayAttr<int32_t>(embeddingBag.getOffsetsValue().value());
        outShape[0] = offsetsAttr.size();
    } else
        return errorAt(loc, "Offsets input was not provided properly");

    inferredReturnShapes.emplace_back(outShape, inTypeEmbTable.getElementType());
    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp EmbeddingBagOffsetsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp embeddingBagOffsetsSumOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingBagOffsetsSumOp.getDefaultIndex(), rewriter);

    if ((embeddingBagOffsetsSumOp.getDefaultIndexValueAttr() == nullptr) && (defaultIndexAttr == nullptr)) {
        int32_t defaultValueDefaultIndex = -1;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    if (defaultIndexAttr == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::EmbeddingBagOffsetsSumOp>(
            embeddingBagOffsetsSumOp, embeddingBagOffsetsSumOp.getType(), embeddingBagOffsetsSumOp.getEmbTable(),
            embeddingBagOffsetsSumOp.getIndices(), embeddingBagOffsetsSumOp.getOffsets(), nullptr /*defaultIndex*/,
            embeddingBagOffsetsSumOp.getPerSampleWeights(), nullptr /*indicesAttr*/, nullptr /*offsetsAttr*/,
            defaultIndexAttr, nullptr);

    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingBagOffsetsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                     mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
