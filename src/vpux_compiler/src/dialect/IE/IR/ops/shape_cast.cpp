//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;
using namespace IE;

mlir::LogicalResult vpux::IE::ShapeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ShapeCastOpAdaptor shapeCast(operands, attrs, prop);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());
    const auto inType = mlir::cast<vpux::NDTypeInterface>(shapeCast.getInput().getType());

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(inType), "{0} doesn't support dynamic shapes",
                      IE::ShapeCastOp::getOperationName());

    auto outputElementType = inType.getElementType();

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElementType) &&
        inType.getDimsOrder() == DimsOrder::NHWC) {
        outputElementType = vpux::inferPerAxisQuantizedTypeAfterShapeCast(inType, outShape);
    }

    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnTypes.emplace_back(outShape, outputElementType, outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ShapeCastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (inputType == outputType) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType()) &&
        inputType.getDimsOrder() != DimsOrder::NHWC) {
        return nullptr;
    }
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(outputType.getShape()).get();
    }

    return nullptr;
}

//
// FuseWithShapeCastOrAffineReshape
//

namespace {
class FuseWithShapeCastOrAffineReshape final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    using mlir::OpRewritePattern<IE::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithShapeCastOrAffineReshape::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (!mlir::isa_and_nonnull<IE::ShapeCastOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(prevOp->getOperand(0).getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getResult().getType());
    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto outputDimsOrder = outputType.getDimsOrder();
    if (inputDimsOrder != outputDimsOrder) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(origOp, prevOp->getOperand(0), origOp.getShape());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ShapeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseWithShapeCastOrAffineReshape>(ctx);
}
