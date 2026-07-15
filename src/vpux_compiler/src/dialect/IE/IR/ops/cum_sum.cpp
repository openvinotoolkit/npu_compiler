//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
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
    auto axisValue = axisContent.getSplatValue<int64_t>();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(cumsumOp.getInput().getType());
    if (axisValue < 0) {
        axisValue += inputType.getRank();
    }

    rewriter.replaceOpWithNewOp<IE::CumSumOp>(cumsumOp, cumsumOp.getType(), cumsumOp.getInput(), nullptr,
                                              rewriter.getI64IntegerAttr(axisValue), cumsumOp.getExclusiveAttr(),
                                              cumsumOp.getReverseAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::CumSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::CumSumOp::fold(FoldAdaptor adaptor) {
    if (!getAxisValue().has_value()) {
        return nullptr;
    }

    auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getOperands()[0]);
    if (attr == nullptr) {
        return nullptr;
    }

    const auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    VPUX_THROW_UNLESS(inputType.hasStaticShape(), "CumSum does not support dynamic shapes");

    int64_t axisVal = getAxisValue().value();
    VPUX_THROW_UNLESS(axisVal >= 0 && axisVal < inputType.getRank(), "CumSum: axis {0} out of range for rank {1}",
                      axisVal, inputType.getRank());

    auto axisAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), axisVal);
    auto exclusiveAttr = mlir::BoolAttr::get(getContext(), getExclusive());
    auto reverseAttr = mlir::BoolAttr::get(getContext(), getReverse());

    return attr.transform().cumSum(axisAttr, exclusiveAttr, reverseAttr).get();
}
