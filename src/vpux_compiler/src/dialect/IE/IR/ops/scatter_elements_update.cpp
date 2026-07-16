//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::ScatterElementsUpdateOp::verify() {
    if (getAxis() != nullptr) {
        auto axisNumElements = mlir::cast<vpux::NDTypeInterface>(getAxis().getType()).getNumElements();
        if (axisNumElements != 1) {
            return errorAt(*this, "Axis should have only 1 element, while it has {0}", axisNumElements);
        }

        if (getAxisValue().has_value()) {
            return errorAt(*this, "Ambiguous axis representation");
        }
    }

    if (getAxis() == nullptr && !getAxisValue().has_value()) {
        return errorAt(*this, "Axis was not provided");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::ScatterElementsUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterElementsUpdateOpAdaptor scatterElementsUpdate(operands, attrs, prop);
    if (mlir::failed(scatterElementsUpdate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(scatterElementsUpdate.getInput().getType());

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ScatterElementsUpdateOp> {
public:
    using mlir::OpRewritePattern<IE::ScatterElementsUpdateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterElementsUpdateOp scatterElementsUpdateOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ScatterElementsUpdateOp scatterElementsUpdateOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto axis = scatterElementsUpdateOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = axis.getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr || !axisConst.getContentAttr().isSplat()) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    auto axisValue = static_cast<int64_t>(axisContent.getSplatValue<int32_t>());
    const auto inputShape = getShape(scatterElementsUpdateOp.getInput());
    const auto rank = inputShape.size();
    // convert negative axis to positive
    if (axisValue < 0) {
        axisValue += static_cast<int64_t>(rank);
    }
    rewriter.replaceOpWithNewOp<IE::ScatterElementsUpdateOp>(
            scatterElementsUpdateOp, scatterElementsUpdateOp.getType(), scatterElementsUpdateOp.getInput(),
            scatterElementsUpdateOp.getIndices(), scatterElementsUpdateOp.getUpdates(), nullptr,
            rewriter.getI64IntegerAttr(axisValue), scatterElementsUpdateOp.getReductionAttr(),
            scatterElementsUpdateOp.getUseInitValAttr());
    return mlir::success();
}

class FoldScatterSplatInputEqualsUpdates final : public mlir::OpRewritePattern<IE::ScatterElementsUpdateOp> {
public:
    using mlir::OpRewritePattern<IE::ScatterElementsUpdateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterElementsUpdateOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FoldScatterSplatInputEqualsUpdates::matchAndRewrite(IE::ScatterElementsUpdateOp op,
                                                                        mlir::PatternRewriter& rewriter) const {
    if (op.getReduction() != IE::ScatterElementsUpdateReductionType::NONE) {
        return mlir::failure();
    }

    auto inputVal = vpux::Const::getSplatValue<double>(op.getInput());
    auto updateVal = vpux::Const::getSplatValue<double>(op.getUpdates());
    if (mlir::failed(inputVal) || mlir::failed(updateVal)) {
        return mlir::failure();
    }

    if (!isDoubleEqual(inputVal.value(), updateVal.value())) {
        return mlir::failure();
    }

    rewriter.replaceOp(op, op.getInput());
    return mlir::success();
}

}  // namespace

void vpux::IE::ScatterElementsUpdateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                    mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<FoldScatterSplatInputEqualsUpdates>(context);
}
