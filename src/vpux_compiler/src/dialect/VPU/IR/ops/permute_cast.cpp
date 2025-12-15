//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>

using namespace vpux;

mlir::LogicalResult vpux::VPU::PermuteCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PermuteCastOpAdaptor permuteCast(operands, attrs, prop);
    if (mlir::failed(permuteCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(permuteCast.getInput());
    const auto inShape = getShape(permuteCast.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (!isTrivialPermute(inMemShape, permuteCast.getMemPerm())) {
        return errorAt(loc, "Operation represents non trivial permutation");
    }

    VPU::inferPermuteReturnTypes(permuteCast.getInput(), permuteCast.getMemPerm(), permuteCast.getDstOrder(),
                                 inferredReturnTypes);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::PermuteCastOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    auto loc = getOperation()->getLoc();
    auto outputShapedType = llvm::cast<mlir::ShapedType>(getOutput().getType());
    auto reversedDstPermutation = DimsOrder::fromAffineMap(mlir::inversePermutation(getDstOrder()));

    const auto srcType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());

    const auto inOrder = srcType.getDimsOrder();

    const auto srcPermutedOrder = applyPermutation(inOrder, DimsOrder::fromAffineMap(getMemPerm()));
    const auto perm = applyPermutation(srcPermutedOrder, reversedDstPermutation);

    auto reversed = to_small_vector(perm.toPermutation() | transformed([](Dim dim) {
                                        return checked_cast<int64_t>(dim.ind());
                                    }));

    for (int64_t dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (!outputShapedType.isDynamicDim(dim)) {
            // Static dim: Return IntegerAttr.
            shapes.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        } else {
            // Dynamic dim: Return Value.
            mlir::OpFoldResult sourceDim = builder.createOrFold<mlir::tensor::DimOp>(loc, getInput(), reversed[dim]);
            shapes.push_back(sourceDim);
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::PermuteCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto srcType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    auto castedOutputDistribution =
            applyPermutationOnDistributionInfo(inType, distribution, getMemPerm(), srcType.getDimsOrder(),
                                               dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (mlir::failed(castedOutputDistribution)) {
        return mlir::failure();
    };
    const auto typeComponents = TypeComponents()
                                        .setShape(dstType.getShape())
                                        .setDimsOrder(dstType.getDimsOrder())
                                        .setElementType(dstType.getElementType())
                                        .setMemSpace(inType.getMemSpace());
    return std::make_pair(mlir::cast<mlir::Type>(dstType.changeTypeComponents(typeComponents)),
                          castedOutputDistribution.value());
}

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::PermuteCastOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    SmallVector<TileInfo> inputTiles;
    const auto inputShape = getShape(getInput());
    log.trace("PermuteCastOp backInferTileInfo: inputShape={0}, outputTile={1}", inputShape, outputTile);
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile PermuteCast operation at '{0}', which has operands with different rank",
                      this->getLoc());
    auto inputTile = outputTile;
    const auto srcType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    const auto arrInMemOrder = dstType.getDimsOrder().toMemoryOrder(inputTile.shape);
    const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, mlir::inversePermutation(getMemPerm()));
    inputTile.shape = Shape(srcType.getDimsOrder().toLogicalOrder(arrPermutedInMemOrder).raw());

    log.trace("PermuteCastOp backInferTileInfo: inputShape={0}, outputTile={1}, inputTile={2}", inputShape, outputTile,
              inputTile);

    inputTiles.push_back(inputTile);
    return TilingInfo{inputTiles};
}

void vpux::VPU::PermuteCastOp::adjustAttrs(const TilingInfo&, const TileInfo&, ShapeRef) {
    // Do nothing
}

bool vpux::VPU::PermuteCastOp::isVFSupported() {
    // E-163016 remove is VFSupported flag when scf and current algorithm is aligned
    return false;
}

namespace {

//
// PropagatePermuteCast
//

class PropagatePermuteCast final : public mlir::OpRewritePattern<VPU::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<VPU::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult PropagatePermuteCast::matchAndRewrite(VPU::PermuteCastOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto checkShapeChanged = [](mlir::Operation* op) -> bool {
        return getShape(op->getOperand(0)) == getShape(op->getResult(0));
    };

    if (!checkShapeChanged(origOp)) {
        return mlir::failure();
    }

    auto middleOp = origOp.getInput().getDefiningOp();
    if (middleOp == nullptr || !middleOp->hasOneUse()) {
        return mlir::failure();
    }

    // Some other ops can be added if needed
    if (!mlir::isa<VPU::ExpandOp>(middleOp)) {
        return mlir::failure();
    }

    auto prePermuteCastOp = middleOp->getOperand(0).getDefiningOp<VPU::PermuteCastOp>();
    if (prePermuteCastOp == nullptr || !prePermuteCastOp->hasOneUse() || !checkShapeChanged(prePermuteCastOp)) {
        return mlir::failure();
    }

    auto srcInOrder = DimsOrder::fromValue(prePermuteCastOp.getInput());
    auto srcOutOrder = DimsOrder::fromValue(prePermuteCastOp.getOutput());
    auto dstInOrder = DimsOrder::fromValue(origOp.getInput());
    auto dstOutOrder = DimsOrder::fromValue(origOp.getOutput());

    if (srcInOrder != dstOutOrder || srcOutOrder != dstInOrder) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(middleOp->getOperand(0), prePermuteCastOp.getInput());
    auto newOp = rewriter.clone(*middleOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// MergeParallelPermuteCast
//

class MergeParallelPermuteCast final : public mlir::OpRewritePattern<VPU::PermuteCastOp> {
public:
    using mlir::OpRewritePattern<VPU::PermuteCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult MergeParallelPermuteCast::matchAndRewrite(VPU::PermuteCastOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto input = origOp.getInput();

    if (input.hasOneUse()) {
        return mlir::failure();
    }

    SmallVector<VPU::PermuteCastOp> permuteCastOps;
    for (auto user : input.getUsers()) {
        if (user == origOp.getOperation()) {
            continue;
        }

        auto permuteCastOp = mlir::dyn_cast_or_null<VPU::PermuteCastOp>(user);
        if (permuteCastOp == nullptr) {
            continue;
        }

        if (origOp.getMemPermAttr() == permuteCastOp.getMemPermAttr() &&
            origOp.getDstOrderAttr() == permuteCastOp.getDstOrderAttr()) {
            permuteCastOps.push_back(permuteCastOp);
        }
    }

    if (permuteCastOps.empty()) {
        return mlir::failure();
    }

    for (auto permuteCastOp : permuteCastOps) {
        rewriter.replaceOp(permuteCastOp, origOp.getOutput());
    }

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::PermuteCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<PropagatePermuteCast>(ctx);
    patterns.add<MergeParallelPermuteCast>(ctx);
}

mlir::OpFoldResult vpux::VPU::PermuteCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType() && getMemPerm().isIdentity()) {
        return getInput();
    }

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        // PermuteCastOp ensures that it is always a trivial permutation. That's why we can just add MemPermuteAttr
        // which will not perform any data movements.
        return attr.transform()
                .memPermute(DimsOrder::fromAffineMap(getDstOrder()), DimsOrder::fromAffineMap(getMemPerm()))
                .get();
    }

    return nullptr;
}
