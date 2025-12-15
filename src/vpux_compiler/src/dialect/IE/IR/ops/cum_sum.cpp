//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::CumSumOp::verify() {
    if (getAxis() != nullptr) {
        auto axisNumElements = mlir::cast<vpux::NDTypeInterface>(getAxis().getType()).getNumElements();
        if (axisNumElements != 1) {
            return errorAt(*this, "Axis should have only 1 element, while it has {0}", axisNumElements);
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::CumSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::CumSumOpAdaptor cumsum(operands, attrs, prop);
    if (mlir::failed(cumsum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(cumsum.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inType);

    const auto outDesc =
            vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), BoundsRef(inShapeInfo.bounds));
    inferredReturnShapes.emplace_back(inShapeInfo.shape, inType.getElementType(), outDesc);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::CumSumOp> {
public:
    using mlir::OpRewritePattern<IE::CumSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const {
    auto axis = cumsumOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = cumsumOp.getAxis().getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return mlir::failure();
    }

    if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    rewriter.replaceOpWithNewOp<IE::CumSumOp>(cumsumOp, cumsumOp.getType(), cumsumOp.getInput(), nullptr,
                                              rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()),
                                              cumsumOp.getExclusiveAttr(), cumsumOp.getReverseAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::CumSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
