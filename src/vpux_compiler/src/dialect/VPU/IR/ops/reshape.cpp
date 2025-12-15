//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <numeric>

using namespace vpux;

//
// getOutShape
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getOutShape(VPU::ReshapeOpAdaptor reshape, mlir::Location loc) {
    if (reshape.getShape() != nullptr && reshape.getShapeValue().has_value()) {
        return errorAt(loc, "Ambiguous shape representation");
    }
    if (reshape.getShape() == nullptr && !reshape.getShapeValue().has_value()) {
        return errorAt(loc, "Missed shape representation");
    }

    if (reshape.getShapeValue().has_value()) {
        return parseIntArrayAttr<int64_t>(reshape.getShapeValue().value());
    }

    auto shapeValue = reshape.getShape();
    while (auto parentOp = shapeValue.getDefiningOp<VPU::CopyOp>()) {
        shapeValue = parentOp->getOperand(0);
    }
    auto shapeConst = shapeValue.getDefiningOp<Const::DeclareOp>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    const auto shapeContent = shapeConst.getContent();
    auto shapeVec = to_small_vector(shapeContent.getValues<int64_t>());

    const auto specialZero = reshape.getSpecialZero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape =
                to_small_vector(mlir::cast<vpux::NDTypeInterface>(reshape.getInput().getType()).getShape().raw());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i >= inShape.size()) {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }

                v = inShape[i];
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

mlir::LogicalResult vpux::VPU::ReshapeOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReshapeOpAdaptor reshape(operands, attrs, prop);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(reshape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(reshape.getInput().getType());

    const auto typeComponents = TypeComponents()
                                        .setShape(ShapeRef(outShape.value()))
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape->size()));
    auto outType = inType.changeTypeComponents(typeComponents);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::ReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(vpux::getShape(getOutput())).get();
    }

    return nullptr;
}

//
// ConvertToShapeCast
//

namespace {

class ConvertToShapeCast final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToShapeCast::matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    if (!inputType.getDimsOrder().isIdentity() || inputType.getRank() != outputType.getRank()) {
        return mlir::failure();
    }

    auto hasSpecialZero = origOp.getSpecialZero();
    auto shapeValueAttr = origOp.getShapeValueAttr();
    if (shapeValueAttr == nullptr || hasSpecialZero) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(origOp, origOp.getInput(), origOp.getShapeValueAttr());
    return mlir::success();
}

}  // namespace

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr || !prevOp->hasOneUse()) {
        return mlir::failure();
    }
    if (!mlir::isa<VPU::SqueezeOp, VPU::UnsqueezeOp, VPU::ReshapeOp, VPU::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    auto newOp =
            rewriter.replaceOpWithNewOp<VPU::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    extendOpLoc(newOp, "fused_with_other");

    return mlir::success();
}

}  // namespace

//
// ConvertToAffineReshape
//

namespace {

class ConvertToAffineReshape final : public mlir::OpRewritePattern<VPU::ReshapeOp> {
public:
    using mlir::OpRewritePattern<VPU::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToAffineReshape::matchAndRewrite(VPU::ReshapeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto outputShape = outputType.getShape();
    const auto outShapeAttr = getIntArrayAttr(getContext(), outputShape);

    const auto inShape = inputType.getShape();
    const auto reassociationMap = vpux::IE::getReassociationMap(inShape, outputShape);
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    // If no valid output layout can be inferred, don't replace with AffineReshape
    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outputLayout = Const::inferAffineReshapeOutputLayout(
            inOrder.toPermutation(), getIntArrayOfArray(getContext(), reassociationMap.value()));
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(
            origOp, origOp.getInput(), getIntArrayOfArray(getContext(), reassociationMap.value()), outShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseReshapes>(ctx);
    patterns.add<ConvertToShapeCast>(ctx);
    patterns.add<ConvertToAffineReshape>(ctx);
}
