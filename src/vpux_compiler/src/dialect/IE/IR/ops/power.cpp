//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/utils/core/numeric.hpp"

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

    const auto in1Type = power.getInput1().getType().cast<mlir::ShapedType>();
    const auto in2Type = power.getInput2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), power.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.value(), in1Type.getElementType());
    }

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
