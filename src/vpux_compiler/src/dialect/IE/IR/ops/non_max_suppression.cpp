//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

std::optional<int64_t> extractMaxOutputBoxesPerClass(IE::NonMaxSuppressionOpAdaptor nms) {
    if (nms.getMaxOutputBoxesPerClass() != nullptr) {
        auto maxBoxesConst = nms.getMaxOutputBoxesPerClass().getDefiningOp<Const::DeclareOp>();
        if (maxBoxesConst != nullptr && maxBoxesConst.getContentAttr().isSplat()) {
            const auto maxBoxesContent = maxBoxesConst.getContent();
            return maxBoxesContent.getSplatValue<int64_t>();
        }
        return std::nullopt;
    }
    if (nms.getMaxOutputBoxesPerClassValueAttr() != nullptr) {
        return nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    }
    return std::nullopt;
}

double extractNMSAttrValue(mlir::Value constName, mlir::FloatAttr attrName) {
    double attrValue = 0.0f;
    if (constName != nullptr) {
        vpux::Const::DeclareOp attrConst = constName.getDefiningOp<Const::DeclareOp>();
        if (attrConst != nullptr && attrConst.getContentAttr().isSplat()) {
            vpux::Const::Content attrContent = attrConst.getContent();
            attrValue = attrContent.getSplatValue<float>();
        }
    } else if (attrName != nullptr) {
        attrValue = attrName.getValueAsDouble();
    }
    return attrValue;
}

}  // namespace

mlir::LogicalResult vpux::IE::NonMaxSuppressionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonMaxSuppressionOpAdaptor nms(operands, attrs, prop);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    const auto inScoresType = mlir::cast<vpux::NDTypeInterface>(nms.getInBoxScores().getType());
    const auto inScoresShapeInfo = ShapeInfo::fromNDType(inScoresType);
    const auto hasBounds = inScoresShapeInfo.isDynamic();
    const auto& sizeSource = hasBounds ? inScoresShapeInfo.bounds : inScoresShapeInfo.shape;
    const auto numBatches = sizeSource[0];
    const auto numClasses = sizeSource[1];
    const auto maxOutputBoxesPerClass = extractMaxOutputBoxesPerClass(nms);
    const auto numBoxes = (!maxOutputBoxesPerClass.has_value() || maxOutputBoxesPerClass.value() == 0)
                                  ? sizeSource[2]
                                  : std::min(sizeSource[2], maxOutputBoxesPerClass.value());

    const SmallVector<int64_t> dynamicOutShape{mlir::ShapedType::kDynamic, 3};
    const SmallVector<int64_t> staticOutShape{numBatches * numClasses * numBoxes, 3};

    TensorAttr outTensorAttr = nullptr;
    SmallVector<int64_t> outShape = staticOutShape;

    if (hasBounds) {
        // BoundedTensorType input: dynamic output with bounds computed from input bounds.
        Bounds bounds(staticOutShape);
        outTensorAttr = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(staticOutShape.size()), nullptr, bounds);
        outShape = dynamicOutShape;
    }
    // TODO: NMS is a dynamism-producing op in E#223694 — its output is inherently dynamic even for static inputs
    // (number of selected boxes is data-dependent). Per OV NMS-9 spec, the static branch should also
    // produce a BoundedTensorType output with upper bound numBatches * numClasses * numBoxes. It is kept
    // static as a compatibility shim while ConvertNMS9ToNMSIEInternal is enabled in the frontend;
    // once that pass is disabled, downstream ops and passes need updating to handle the dynamic output.

    const SmallVector<int64_t> validOutputsShape{1};
    auto s32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    inferredReturnShapes.emplace_back(outShape, s32Type, outTensorAttr);
    inferredReturnShapes.emplace_back(outShape, inScoresType.getElementType(), outTensorAttr);
    inferredReturnShapes.emplace_back(validOutputsShape, s32Type);

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    using mlir::OpRewritePattern<IE::NonMaxSuppressionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp nmsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::NonMaxSuppressionOp nmsOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (nmsOp.getMaxOutputBoxesPerClassValue().has_value() && nmsOp.getIouThresholdValue().has_value() &&
        nmsOp.getScoreThresholdValue().has_value() && nmsOp.getSoftNmsSigmaValue().has_value()) {
        return mlir::failure();
    }

    const auto maxBoxesOpt = extractMaxOutputBoxesPerClass(nmsOp);
    if (!maxBoxesOpt.has_value()) {
        return mlir::failure();
    }
    int64_t maxBoxesPerClassValue = maxBoxesOpt.value();

    double iouThresholdValue = extractNMSAttrValue(nmsOp.getIouThreshold(), nmsOp.getIouThresholdValueAttr());

    double scoreThresholdValue = extractNMSAttrValue(nmsOp.getScoreThreshold(), nmsOp.getScoreThresholdValueAttr());

    double softNMSSigmaValue = extractNMSAttrValue(nmsOp.getSoftNmsSigma(), nmsOp.getSoftNmsSigmaValueAttr());

    rewriter.replaceOpWithNewOp<IE::NonMaxSuppressionOp>(
            nmsOp, nmsOp.getInBoxCoords(), nmsOp.getInBoxScores(), nullptr, nullptr, nullptr, nullptr,
            nmsOp.getBoxEncoding(), nmsOp.getSortResultDescending(), rewriter.getI64IntegerAttr(maxBoxesPerClassValue),
            rewriter.getF64FloatAttr(iouThresholdValue), rewriter.getF64FloatAttr(scoreThresholdValue),
            rewriter.getF64FloatAttr(softNMSSigmaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::NonMaxSuppressionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
