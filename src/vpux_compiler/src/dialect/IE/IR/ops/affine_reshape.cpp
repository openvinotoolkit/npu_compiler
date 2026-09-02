//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

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
    const auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto ndInType = mlir::cast<vpux::NDTypeInterface>(inType);
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outputLayout =
            Const::inferAffineReshapeOutputLayout(inOrder.toPermutation(), affineReshape.getDimMapping());
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(ndInType), "{0} doesn't support dynamic shapes",
                      IE::AffineReshapeOp::getOperationName());
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
// ShaveCodeGenSupportedOpInterface
//

bool vpux::IE::AffineReshapeOp::shouldJITCompile() {
    return false;
}

bool vpux::IE::AffineReshapeOp::shouldJITCompileToEnableFusion() {
    return vpux::ShaveCodeGen::hasOnlySupportedTypes(*this);
}

//
// verify
//

mlir::LogicalResult vpux::IE::AffineReshapeOp::verify() {
    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

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
    // This op is view-like, which means that if input and output type are equal, it's a no-op.
    auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    if (inputType == outputType) {
        return getInput();
    }

    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput()); attr != nullptr) {
        return attr.transform().affineReshape(getDimMappingAttr(), getShapeValue()).get();
    }

    return nullptr;
}

//
// FuseAffineReshapes
//

namespace {
class FuseAffineReshapes final : public mlir::OpRewritePattern<IE::AffineReshapeOp> {
public:
    FuseAffineReshapes(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::AffineReshapeOp>(ctx, benefit) {
        this->setDebugName("FuseAffineReshapes");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AffineReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseAffineReshapes::matchAndRewrite(IE::AffineReshapeOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(prevOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto inputDimsOrder = inputType.getDimsOrder();
    const auto outputDimsOrder = outputType.getDimsOrder();

    const auto inputShape = mlir::cast<mlir::ShapedType>(inputType).getShape();
    const auto outputShape = mlir::cast<mlir::ShapedType>(outputType).getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    // Fusing AffineReshape with any of the above mentioned ops might result in another AffineReshape or not.
    const auto reassociationMap = vpux::IE::getReassociationMap(inputShape, outputShape);
    if (mlir::succeeded(reassociationMap)) {
        const auto outputLayout = Const::inferAffineReshapeOutputLayout(
                inputDimsOrder.toPermutation(), getIntArrayOfArray(getContext(), reassociationMap.value()));

        if (outputLayout.has_value() && outputLayout.value() == outputDimsOrder) {
            rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(origOp, prevOp.getInput(),
                                                             getIntArrayOfArray(getContext(), reassociationMap.value()),
                                                             outputShapeAttr);
            return mlir::success();
        }
    }

    // Reshape's output dim order is limited to NCHW, so the compiler will not fuse the ops in this case
    if (outputDimsOrder != DimsOrder::NCHW) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// FuseWithReshape
//

namespace {
class FuseWithReshape final : public mlir::OpRewritePattern<IE::AffineReshapeOp> {
public:
    FuseWithReshape(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit = 1)
            : mlir::OpRewritePattern<IE::AffineReshapeOp>(ctx, benefit) {
        this->setDebugName("FuseWithReshape");
    }

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
    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), outputShapeAttr);
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

void vpux::IE::registerAffineReshapeOpRewriters(RewriterRegistry& registry,
                                                ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index) {
    registry.registerRewriter<FuseAffineReshapes>("fuse-affine-reshapes", benefitLevels[index]);
    registry.registerRewriter<FuseWithReshape>("fuse-with-reshape", benefitLevels[index]);
}
