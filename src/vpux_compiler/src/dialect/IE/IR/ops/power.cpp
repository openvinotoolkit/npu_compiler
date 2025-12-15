//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::PowerOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PowerOpAdaptor power(operands, attrs, prop);
    if (mlir::failed(power.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<vpux::NDTypeInterface>(power.getInput1().getType());
    const auto in2Type = mlir::cast<vpux::NDTypeInterface>(power.getInput2().getType());

    auto outShapeInfo = inferEltwiseOutputShapeInfo(ShapeInfo::fromNDType(in1Type), ShapeInfo::fromNDType(in2Type),
                                                    power.getAutoBroadcast(), loc);

    const auto outDesc = vpux::getTensorAttr(ctx, inferOrder(in1Type, in2Type), /*memSpace=*/nullptr,
                                             BoundsRef(outShapeInfo.bounds));
    inferredReturnShapes.emplace_back(outShapeInfo.shape, in1Type.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::PowerOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                         mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::PowerOp::fold(FoldAdaptor /*adaptor*/) {
    auto exponent = IE::getExponentSplatVal(*this);
    if (!exponent.has_value() || !isFloatEqual(exponent.value(), 1.0)) {
        return nullptr;
    }

    return getInput1();
}

//
// FuseSqrtAndPower
//

namespace {

class FuseSqrtAndPower final : public mlir::OpRewritePattern<IE::PowerOp> {
public:
    using mlir::OpRewritePattern<IE::PowerOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PowerOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseSqrtAndPower::matchAndRewrite(IE::PowerOp origOp, mlir::PatternRewriter& rewriter) const {
    auto exponent = getExponentSplatVal(origOp);
    if (!exponent.has_value() || !isFloatEqual(exponent.value(), 2.0)) {
        return mlir::failure();
    }

    auto sqrtInOp = mlir::dyn_cast_or_null<IE::SqrtOp>(origOp.getInput1().getDefiningOp());
    if (sqrtInOp != nullptr && sqrtInOp.getOutput().hasOneUse()) {
        rewriter.replaceOp(origOp, sqrtInOp.getInput());
        return mlir::success();
    }

    return mlir::failure();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::PowerOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseSqrtAndPower>(ctx);
}
