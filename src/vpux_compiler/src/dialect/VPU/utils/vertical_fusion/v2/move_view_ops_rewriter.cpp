//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/move_view_ops_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

// An experimental number to help get the threshold of tiling size.
// The purpose is to avoid fuse view like op with tiling limitations.
constexpr static double LIMITED_TILING_DIM_MAX_RATIO = 0.5;

namespace vpux::VPU::VF::v2 {
mlir::LogicalResult MoveViewOpsRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto isOpWeightsFromVFOperandIndex = [](mlir::Operation* op, size_t operandIdx) -> bool {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr) {
            return false;
        }
        if (auto opWeights = llvm::cast_if_present<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            return opWeights.getArgNumber() == operandIdx;
        }
        return false;
    };

    auto isStrategyCompatible = [this](mlir::Operation* userOp, size_t operandIdx,
                                       VPU::TilingViewLikeOpInterface viewOp,
                                       ArrayRef<int64_t> tilingStrategy) -> bool {
        // Check if the parent op is a VerticalFusionOp and it has a compatible tiling strategy
        auto parentOp = mlir::dyn_cast_or_null<VPU::VerticalFusionOp>(viewOp->getOperand(0).getDefiningOp());
        if (parentOp == nullptr) {
            return false;
        }
        auto parentTilingStrategy = parseIntArrayAttr<int64_t>(parentOp.getTilingStrategy());
        auto parentTilingDims = getNonOneDim(ShapeRef(parentTilingStrategy));
        if (parentTilingDims.empty() || !viewOp.isSupportedTilingDim(parentTilingDims)) {
            return false;
        }

        // Check if the user operation has a compatible tiling strategy
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(userOp);
        if (tilingBuilderOp == nullptr) {
            return false;
        }
        auto outTiles = fillDividedTiles(userOp, ShapeRef(tilingStrategy), getShape(userOp->getResult(0)));
        if (mlir::failed(outTiles)) {
            return false;
        }

        auto inTiles = tilingBuilderOp.backInferTileInfo(outTiles.value().front(), _log).tiles;
        auto tilingDims = getNonOneDim(inTiles[operandIdx].axis);
        if (!viewOp.isSupportedTilingDim(tilingDims)) {
            return false;
        }

        auto inShape = getShape(viewOp->getOperand(0));
        for (auto dim : tilingDims) {
            if (viewOp.isSupportedTilingDimWithRestrictions(dim)) {
                // if the tiling number is too large, it may prevent the fusion of the vf ops selecting this dim in
                // later pass, since the merged new VF might increase tiling size on this dim.
                if (tilingStrategy[dim.ind()] >= static_cast<int64_t>(inShape[dim] * LIMITED_TILING_DIM_MAX_RATIO)) {
                    return false;
                }
            }
        }

        // If the user operation is a clustered operation, check its multi-cluster strategy is compatible or not
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(userOp);
        if (clusteredOp == nullptr) {
            return true;
        }
        auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value()) {
            return true;
        }
        auto operand = clusteredOp->getOperand(operandIdx);
        auto inputType = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        auto outType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
        auto numClusters = VPU::getOptimalNumClusters(userOp, outType.getShape(), strategy.value());

        SmallVector<int64_t> activationTensorNumTiles;
        if (mlir::isa<VPU::SWOpInterface>(userOp)) {
            activationTensorNumTiles =
                    VPU::getSWInputTensorNumTiles(clusteredOp, numClusters, strategy.value(), operand, inputType);
        } else {
            activationTensorNumTiles =
                    getActivationTensorNumTiles(clusteredOp, numClusters, strategy.value(), inputType);
        }
        auto mcTilingDims = getNonOneDim(ShapeRef(activationTensorNumTiles));
        return viewOp.isSupportedTilingDim(mcTilingDims);
    };

    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());

    for (auto vfOperand : vfOp->getOperands() | indexed) {
        auto parentOp = vfOperand.value().getDefiningOp<VPU::TilingViewLikeOpInterface>();

        // E-163016 remove is VFSupported flag when scf and current algorithm is aligned
        if (parentOp == nullptr || !VPU::isPureViewOp(parentOp) || !parentOp.isVFSupported()) {
            continue;
        }

        // Skip moving view ops for scenarios below:
        // 1. Exclude weights moving for non-SOC tiling. As only under SOC case, the producer op for weights (if exists)
        // need to be tiled and merged into VF
        // 2. Exclude view op with partial tiling dim support when it's parent and user's tiling dim and multicluster
        // strategy tiling dim is not compatible
        auto hasIncompatibleUses =
                llvm::any_of(vfOp.getBody()->getArgument(vfOperand.index()).getUses(), [&](auto& use) {
                    auto user = use.getOwner();
                    auto operandId = use.getOperandNumber();
                    if (isOpWeightsFromVFOperandIndex(user, vfOperand.index()) &&
                        (tilingStrategy[Dims4D::Act::C.ind()] == 1)) {
                        return true;
                    }
                    return VPU::onlySupportPartialTilingDims(parentOp) &&
                           !isStrategyCompatible(user, operandId, parentOp, tilingStrategy);
                });
        if (hasIncompatibleUses) {
            continue;
        }

        if (llvm::all_of(parentOp->getOperands(), [](auto value) {
                return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                       mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
            })) {
            continue;
        }

        auto newVFOp = fuseOpsInBlock(rewriter, vfOp, parentOp);
        rewriter.replaceOp(vfOp, newVFOp.getResult(0));
        return mlir::success();
    }
    return mlir::failure();
}
}  // namespace vpux::VPU::VF::v2
