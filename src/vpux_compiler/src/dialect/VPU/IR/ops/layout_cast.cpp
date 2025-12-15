//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LayoutCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LayoutCastOpAdaptor overrideLayout(operands, attrs, prop);
    if (mlir::failed(overrideLayout.verify(loc))) {
        return mlir::failure();
    }

    const auto outAffineMap = overrideLayout.getDstOrder();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(overrideLayout.getInput().getType());
    const auto outType = inType.changeDimsOrder(DimsOrder::fromAffineMap(outAffineMap));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LayoutCastOp::verify() {
    const auto outAffineMap = getDstOrder();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    if (inType.getRank() != outAffineMap.getNumDims()) {
        return errorAt(*this, "Cannot apply {0} map to {1}.", outAffineMap, inType.getShape());
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::LayoutCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    auto operands = adaptor.getOperands();
    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        auto dstOrder = DimsOrder::fromAffineMap(getDstOrder());
        return static_cast<Const::ContentAttr>(cst).transform().layoutCast(dstOrder).get();
    }

    return nullptr;
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::LayoutCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto ctx = getContext();
    const auto srcType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto srcOrder = srcType.getDimsOrder();
    const auto dstOrder = dstType.getDimsOrder();
    const auto memPerm = getPermutationFromOrders(srcOrder, dstOrder, ctx);

    auto castedOutputDistribution =
            applyPermutationOnDistributionInfo(inType, distribution, memPerm, srcType.getDimsOrder(),
                                               dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (mlir::failed(castedOutputDistribution)) {
        return mlir::failure();
    }

    const auto typeComponents = TypeComponents()
                                        .setShape(dstType.getShape())
                                        .setDimsOrder(dstType.getDimsOrder())
                                        .setElementType(dstType.getElementType());
    return std::make_pair(mlir::cast<mlir::Type>(dstType.changeTypeComponents(typeComponents)),
                          castedOutputDistribution.value());
}

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::LayoutCastOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    SmallVector<TileInfo> inputTiles;
    const auto inputShape = getShape(getInput());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile LayoutCast operation at '{0}', which has operands with different rank",
                      this->getLoc());
    inputTiles.push_back(outputTile);
    return TilingInfo{inputTiles};
}

void vpux::VPU::LayoutCastOp::adjustAttrs(const TilingInfo&, const TileInfo&, ShapeRef) {
    // Do nothing
}

bool vpux::VPU::LayoutCastOp::isVFSupported() {
    return false;
}

//
// FuseLayoutCasts
//

namespace {
class FuseLayoutCasts final : public mlir::OpRewritePattern<VPU::LayoutCastOp> {
public:
    using mlir::OpRewritePattern<VPU::LayoutCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::LayoutCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseLayoutCasts::matchAndRewrite(VPU::LayoutCastOp origOp, mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> VPU.LayoutCast type2 -> VPU.LayoutCast type3 -> Output type3
    // into
    // Input type1 -> VPU.LayoutCast type3 -> Output type3
    auto producerOp = origOp.getInput().getDefiningOp<VPU::LayoutCastOp>();
    if (producerOp == nullptr || !producerOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::LayoutCastOp>(origOp, origOp.getOutput().getType(), producerOp.getInput(),
                                                   origOp.getDstOrderAttr());

    return mlir::success();
}

//
// FuseLayoutCastsWithShapeCast
//

class FuseLayoutCastsWithShapeCast final : public mlir::OpRewritePattern<VPU::LayoutCastOp> {
public:
    using mlir::OpRewritePattern<VPU::LayoutCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::LayoutCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseLayoutCastsWithShapeCast::matchAndRewrite(VPU::LayoutCastOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> VPU.LayoutCast type2 -> VPU.ShapeCast -> VPU.LayoutCast type1 -> Output type1
    // into
    // Input type1 -> VPU.ShapeCast -> Output type1
    auto shapeCastOp = origOp.getInput().getDefiningOp<VPU::ShapeCastOp>();
    if (shapeCastOp == nullptr || !shapeCastOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto layoutCastOp = shapeCastOp.getInput().getDefiningOp<VPU::LayoutCastOp>();
    if (layoutCastOp == nullptr || !layoutCastOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }
    auto firstLayoutCastInputDimOrder =
            mlir::cast<vpux::NDTypeInterface>(layoutCastOp.getInput().getType()).getDimsOrder();
    ;
    auto firstLayoutCastOutputDimOrder =
            mlir::cast<vpux::NDTypeInterface>(layoutCastOp.getOutput().getType()).getDimsOrder();
    ;
    auto currentLayoutCastInputDimOrder = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getDimsOrder();
    auto currentLayoutCastOutputDimOrder =
            mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getDimsOrder();

    if (firstLayoutCastInputDimOrder != currentLayoutCastOutputDimOrder ||
        firstLayoutCastOutputDimOrder != currentLayoutCastInputDimOrder) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::ShapeCastOp>(origOp, layoutCastOp.getInput(), shapeCastOp.getShape());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::LayoutCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseLayoutCasts>(ctx);
    patterns.add<FuseLayoutCastsWithShapeCast>(ctx);
}
