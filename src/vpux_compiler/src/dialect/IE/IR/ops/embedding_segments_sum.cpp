//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getNumSegments())) {
        return errorAt(*this, "num_segments should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getDefaultIndex())) {
        return errorAt(*this, "default_index should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

namespace {

mlir::FailureOr<int64_t> extractNumSegments(mlir::Location loc,
                                            IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum) {
    if (embeddingSegmentsSum.getNumSegments() != nullptr) {
        auto numSegmentsConst = embeddingSegmentsSum.getNumSegments().getDefiningOp<Const::DeclareOp>();
        if (numSegmentsConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for numSegments");
        }

        if (const auto& attr = numSegmentsConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(loc, "numSegments value must be a scalar");
        }

        const auto numSegmentsContent = numSegmentsConst.getContent();
        return numSegmentsContent.getSplatValue<int64_t>();
    } else if (embeddingSegmentsSum.getNumSegmentsValue().has_value()) {
        return embeddingSegmentsSum.getNumSegmentsValue().value();
    }
    return errorAt(loc, "NumSegments was not provided");
}

}  // namespace

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs, prop);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = embeddingSegmentsSum.getEmbTable().getType().cast<mlir::ShapedType>();

    const auto numSegments = extractNumSegments(loc, embeddingSegmentsSum);
    if (mlir::failed(numSegments)) {
        return mlir::failure();
    }

    int64_t numSegmentsVal = checked_cast<int64_t>(*numSegments);

    SmallVector<int64_t> outShape(to_small_vector(inType.getShape()));
    outShape[0] = numSegmentsVal;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {
class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp EmbeddingSegmentsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::EmbeddingSegmentsSumOp embeddingSegmentsSumOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto numSegmentsAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getNumSegments(), rewriter);
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getDefaultIndex(), rewriter);

    if ((embeddingSegmentsSumOp.getNumSegments() == nullptr) && (embeddingSegmentsSumOp.getDefaultIndex() == nullptr)) {
        return mlir::failure();
    }

    if (defaultIndexAttr == nullptr) {
        int32_t defaultValueDefaultIndex = -1;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    rewriter.replaceOpWithNewOp<IE::EmbeddingSegmentsSumOp>(
            embeddingSegmentsSumOp, embeddingSegmentsSumOp.getType(), embeddingSegmentsSumOp.getEmbTable(),
            embeddingSegmentsSumOp.getIndices(), embeddingSegmentsSumOp.getSegmentIds(), nullptr, nullptr,
            embeddingSegmentsSumOp.getPerSampleWeights(), nullptr, nullptr, numSegmentsAttr, defaultIndexAttr, nullptr);
    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingSegmentsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                   mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
