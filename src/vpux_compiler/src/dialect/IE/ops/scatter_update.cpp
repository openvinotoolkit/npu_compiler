////
//// Copyright (C) 2022 Intel Corporation.
//// SPDX-License-Identifier: Apache 2.0
////
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ScatterUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ScatterUpdateOpAdaptor scatterUpdate(operands, attrs);
    if (mlir::failed(scatterUpdate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatterUpdate.input().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// inferLayoutInfo
//
void vpux::IE::ScatterUpdateOp::inferLayoutInfo(vpux::IE::LayerLayoutInfo& info) {
    const auto inType = input().getType().cast<mlir::ShapedType>();
    info.setInput(0, DimsOrder::fromNumDims(inType.getShape().size()));
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
    auto axis = scatterUpdateOp.axis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = scatterUpdateOp.axis().getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(axis != nullptr, "Only support constant axis");

    const auto axisContent = axisConst.content();
    if (!axisContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScatterUpdateOp>(
            scatterUpdateOp, scatterUpdateOp.getType(), scatterUpdateOp.input(), scatterUpdateOp.indices(),
            scatterUpdateOp.updates(), nullptr, rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()));
    return mlir::success();
}

}  // namespace

void vpux::IE::ScatterUpdateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                            mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
