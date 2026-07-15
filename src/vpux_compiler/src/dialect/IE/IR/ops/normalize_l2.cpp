//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::NormalizeL2Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NormalizeL2OpAdaptor normalizeL2(operands, attrs, prop);
    if (mlir::failed(normalizeL2.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = Const::getConstOrArrAttrValue(normalizeL2.getAxes(), normalizeL2.getAxesValueAttr());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(normalizeL2.getData().getType());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::NormalizeL2Op::verify() {
    const auto inRank = mlir::cast<mlir::ShapedType>(getData().getType()).getRank();
    const auto axes_tensor = getAxes();
    const auto axes_attribute = getAxesValueAttr();
    if (!(axes_tensor || axes_attribute)) {
        return mlir::failure();
    }
    const auto axes = Const::getConstOrArrAttrValue(axes_tensor, axes_attribute);
    auto axesVec{axes.value()};
    for (auto& axis : axesVec) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(*this, "Axes values should be unique");
    }

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::NormalizeL2Op> {
public:
    using mlir::OpRewritePattern<IE::NormalizeL2Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NormalizeL2Op normalizeL2Op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::NormalizeL2Op normalizeL2Op,
                                                        mlir::PatternRewriter& rewriter) const {
    if (normalizeL2Op.getAxesValue()) {
        return mlir::failure();
    }

    const auto axesValue = Const::getConstOrArrAttrValue(normalizeL2Op.getAxes(), normalizeL2Op.getAxesValueAttr());

    if (mlir::failed(axesValue)) {
        return mlir::failure();
    }

    // Normalize negative axes to positive
    const auto inRank = mlir::cast<mlir::ShapedType>(normalizeL2Op.getData().getType()).getRank();
    auto axesVec = axesValue.value();
    llvm::for_each(axesVec, [&](auto& axis) {
        axis = axis < 0 ? axis + inRank : axis;
    });

    const auto axesAttr = getIntArrayAttr(rewriter.getContext(), axesVec);

    rewriter.replaceOpWithNewOp<IE::NormalizeL2Op>(normalizeL2Op, normalizeL2Op.getData(), nullptr, axesAttr,
                                                   normalizeL2Op.getEps(), normalizeL2Op.getEpsMode());

    return mlir::success();
}

}  // namespace

void vpux::IE::NormalizeL2Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
