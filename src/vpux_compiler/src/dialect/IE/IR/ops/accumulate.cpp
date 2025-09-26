//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AccumulateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AccumulateOpAdaptor accumulate(operands, attrs, prop);
    if (mlir::failed(accumulate.verify(loc))) {
        return mlir::failure();
    }

    const auto lhsType = mlir::cast<mlir::ShapedType>(accumulate.getLhs().getType());
    const auto rhsType = mlir::cast<mlir::ShapedType>(accumulate.getRhs().getType());
    VPUX_THROW_UNLESS(lhsType == rhsType, "Types of IE.Accumulate operands must match: lhs = {0}, rhs = {1}", lhsType,
                      rhsType);

    inferredReturnShapes.emplace_back(lhsType.getShape(), lhsType.getElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::AccumulateOp::verify() {
    const auto lhsType = getLhs().getType();
    const auto rhsType = getRhs().getType();
    if (lhsType != rhsType) {
        return errorAt(getLoc(), "Operation {0}: the type of operands must match. Got lhs = {1}, rhs = {2}",
                       getOperationName(), lhsType, rhsType);
    }

    const auto zeroScalesSet = getLhsScale() == nullptr && getRhsScale() == nullptr;
    const auto bothScalesSet = getLhsScale() != nullptr && getRhsScale() != nullptr;
    const auto supportedScales = zeroScalesSet || bothScalesSet;
    if (!supportedScales) {
        return errorAt(getLoc(), "Operation {0}: must set either both scales or none.", getOperationName());
    }

    return mlir::success();
}

//
// ConvertToAdd
//

namespace {
class ConvertToAdd final : public mlir::OpRewritePattern<IE::AccumulateOp> {
public:
    using mlir::OpRewritePattern<IE::AccumulateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToAdd::matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getLhsScale() != nullptr || origOp.getRhsScale() != nullptr) {
        // If scales are set, we cannot convert to Add
        return mlir::failure();
    }

    const auto broadcastTypeAttr =
            vpux::IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, origOp.getType(), origOp.getLhs(), origOp.getRhs(),
                                           broadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::AccumulateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<ConvertToAdd>(ctx);
}
