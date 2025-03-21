//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::AffineReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AffineReshapeOpAdaptor affineReshape(operands, attrs, prop);
    if (mlir::failed(affineReshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(affineReshape.getShapeValue());
    const auto input = affineReshape.getInput();
    const auto inType = input.getType().cast<mlir::RankedTensorType>();
    const auto ndInType = inType.cast<vpux::NDTypeInterface>();
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outputLayout =
            Const::inferAffineReshapeOutputLayout(inOrder.toPermutation(), affineReshape.getDimMapping());
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    const auto outDesc = vpux::getTensorAttr(ctx, outputLayout.value(), ndInType.getMemSpace());

    const auto elemTypeInferResult = inferElemTypeAffineReshape(affineReshape, ndInType.getElementType());
    if (!elemTypeInferResult.has_value()) {
        inferredReturnShapes.emplace_back(outShape, ndInType.getElementType(), outDesc);
    } else {
        inferredReturnShapes.emplace_back(outShape, elemTypeInferResult.value(), outDesc);
    }

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::AffineReshapeOp::verify() {
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = getOutput().getType().cast<vpux::NDTypeInterface>();

    auto inNumElem = inType.getNumElements();
    auto outNumElem = outType.getNumElements();
    if (inNumElem != outNumElem) {
        return errorAt(*this,
                       "AffineReshape input and output must have the same number of elements. Got: input number '{0}'; "
                       "output number '{1}'",
                       inNumElem, outNumElem);
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::AffineReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    auto inputType = getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = getOutput().getType().cast<vpux::NDTypeInterface>();
    if (inputType == outputType) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto ephemeral = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto attr = static_cast<Const::ContentAttr>(ephemeral);
        const auto inputElemType =
                inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
        const auto outputElemType =
                outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();
        if (inputElemType && outputElemType && isQuantizedDimensionPermutation(inputElemType, outputElemType)) {
            const auto newShape = outputType.getShape();
            return attr.transform().changeShapeAndElemType(newShape, outputElemType).get();
        }
        return attr.transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseAffineReshapes
//

namespace {
class FuseAffineReshapes final : public mlir::OpRewritePattern<IE::AffineReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::AffineReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AffineReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseAffineReshapes::matchAndRewrite(IE::AffineReshapeOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    auto inputType = prevOp.getInput().getType().cast<NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();

    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto outputDimsOrder = outputType.getDimsOrder();

    const auto inputShape = inputType.cast<mlir::ShapedType>().getShape();
    const auto outputShape = outputType.cast<mlir::ShapedType>().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Fusing AffineReshape with any of the above mentioned ops might result in another AffineReshape or not,
    // depending on the resulting input and output shapes.
    // E. g. 1 x 24 x 2 x 2 -> AffineReshape -> 1 x 24 x 4 -> AffineReshape -> 1 x 24 x 4 x 1
    //       mapping: id0 = od0, id1 = od1 and id2 * id3 = od2 * od3 (not an AffineReshape)
    // If the Reshape that replaces the two ops ends up being a valid AffineReshape, then it will be converted by
    // Reshape's canonicalizer.
    // TODO: E#70418 1. support reshape(in: NHWC, out: NHWC) 2. support different in&out order of reshape
    if (inputDimsOrder == outputDimsOrder && inputDimsOrder == DimsOrder::NHWC) {
        const auto reassociationMap = vpux::IE::getReassociationMap(inputShape, outputShape);

        if (mlir::failed(reassociationMap)) {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
                origOp, prevOp.getInput(), getIntArrayOfArray(getContext(), reassociationMap.value()), outputShapeAttr);
        return mlir::success();
    }
    // Reshape's output dim order is limited to NCHW, so the compiler will not fuse the ops in this case
    if (outputDimsOrder != DimsOrder::NCHW) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// FuseWithReshape
//

namespace {
class FuseWithReshape final : public mlir::OpRewritePattern<IE::AffineReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::AffineReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AffineReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::AffineReshapeOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp>(prevOp)) {
        return mlir::failure();
    }
    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Fusing AffineReshape with any of the above mentioned ops might result in another AffineReshape or not,
    // depending on the resulting input and output shapes.
    // If the Reshape that replaces the two ops ends up being a valid AffineReshape, then it will be converted by
    // Reshape's canonicalizer.
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::AffineReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseAffineReshapes>(ctx);
    patterns.add<FuseWithReshape>(ctx);
}
