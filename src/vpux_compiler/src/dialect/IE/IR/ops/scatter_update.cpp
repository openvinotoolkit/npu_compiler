//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ScatterUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterUpdateOpAdaptor scatterUpdate(operands, attrs, prop);
    if (mlir::failed(scatterUpdate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(scatterUpdate.getInput().getType());

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ScatterUpdateOp> {
public:
    using mlir::OpRewritePattern<IE::ScatterUpdateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterUpdateOp scatterUpdateOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ScatterUpdateOp scatterUpdateOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto axis = scatterUpdateOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = scatterUpdateOp.getAxis().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(axisConst == nullptr, "Only support constant axis");

    if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    rewriter.replaceOpWithNewOp<IE::ScatterUpdateOp>(
            scatterUpdateOp, scatterUpdateOp.getType(), scatterUpdateOp.getInput(), scatterUpdateOp.getIndices(),
            scatterUpdateOp.getUpdates(), nullptr, rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()));
    return mlir::success();
}

}  // namespace

void vpux::IE::ScatterUpdateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                            mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
