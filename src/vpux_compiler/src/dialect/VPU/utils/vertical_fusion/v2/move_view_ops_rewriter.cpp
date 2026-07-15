//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/move_view_ops_rewriter.hpp"
#include <queue>
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/hash_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

// An experimental number to help get the threshold of tiling size.
// The purpose is to avoid fuse view like op with tiling limitations.
constexpr static double LIMITED_TILING_DIM_MAX_RATIO = 0.5;

namespace vpux::VPU::VF::v2 {
mlir::LogicalResult MoveViewOpsRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (vfOp.getIsManualConfigured()) {
        return mlir::failure();
    }

    auto isOpWeightsFromVFOperandIndex = [](mlir::Operation* op, size_t operandIdx) -> bool {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr) {
            return false;
        }
        if (auto opWeights = llvm::dyn_cast_if_present<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            return opWeights.getArgNumber() == operandIdx;
        }
        return false;
    };

    auto isParentVFOpCompatible = [](VPU::VerticalFusionOp parentVFOp,
                                     SmallVector<VPU::TilingViewLikeOpInterface> viewOpChain) -> bool {
        if (!parentVFOp) {
            return false;
        }
        auto parentTilingStrategy = parseIntArrayAttr<int64_t>(parentVFOp.getTilingStrategy());
        auto parentTilingDims = getNonOneDim(ShapeRef(parentTilingStrategy));
        // parentTilingDims are in the view op's input space, but isSupportedTilingDim expects
        // output space dims. Use backInferTilingDim to generically find which output dims
        // correspond to the parent's tiling dims, without op-specific special casing.
        for (auto viewOp : viewOpChain | reversed) {
            // TODO: parentTilingDims is empty, For view ops without restricted tiling dims. It should be fine to move
            // it into VF. But currently we keep it outside, due to affinereshape return false on restrict tiling dim.
            if (parentTilingDims.empty()) {
                return false;
            }
            auto outShape = getShape(viewOp->getResult(0));
            SmallVector<Dim> mappedDims;
            for (size_t i = 0; i < outShape.size(); ++i) {
                Dim outDim(i);
                if (!viewOp.isSupportedTilingDim({outDim})) {
                    continue;
                }
                auto inDim = viewOp.backInferTilingDim(outDim);
                if (llvm::is_contained(parentTilingDims, inDim)) {
                    mappedDims.push_back(outDim);
                }
            }
            if (mappedDims.empty() || !viewOp.isSupportedTilingDim(mappedDims)) {
                return false;
            }
            if (mlir::failed(viewOp.inferTilingStrategy(parentTilingStrategy))) {
                return false;
            }
            parentTilingStrategy = viewOp.inferTilingStrategy(parentTilingStrategy).value();
            parentTilingDims = getNonOneDim(ShapeRef(parentTilingStrategy));
        }
        return true;
    };
    using WorkItem = std::tuple<mlir::Operation*, TileInfo, DimArr>;
    // Back-infer input tile info and MC dims from a given op and its output tile.
    // Returns failure if the output tile is not supported by a view-like op.
    auto backInferOpTileInfo = [](const WorkItem& workItem, bool hasMCStrategy,
                                  Logger log) -> mlir::FailureOr<std::pair<SmallVector<TileInfo>, DimArr>> {
        const auto& [op, outTile, outMCDims] = workItem;
        SmallVector<TileInfo> inTiles;
        DimArr inMCDims;
        if (auto builderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op)) {
            inTiles = builderOp.backInferTileInfo(outTile, log).tiles;
        } else if (auto viewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(op)) {
            if (!viewLikeOp.isSupportedOutTile(outTile)) {
                return mlir::failure();
            }
            inTiles = viewLikeOp.backInferTileInfo(outTile, log).tiles;
            if (hasMCStrategy) {
                for (auto dim : outMCDims) {
                    if (!viewLikeOp.isSupportedTilingDim({dim})) {
                        return mlir::failure();
                    }
                    inMCDims.push_back(viewLikeOp.backInferTilingDim(dim));
                }
            }
        }
        return std::make_pair(inTiles, inMCDims);
    };

    auto backUserTileInfo = [&backInferOpTileInfo](const vpux::OutputTiling& lastOutTiles, mlir::Operation* lastInnerOp,
                                                   mlir::Operation* userOp, size_t operandIdx, bool hasMCStrategy,
                                                   Logger _log) -> std::optional<WorkItem> {
        std::queue<WorkItem> workQueue;
        llvm::DenseSet<mlir::Operation*> visited;
        DimArr userOpMCDims;

        TileInfo userOpTile = lastOutTiles.front();
        if (hasMCStrategy) {
            auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(lastInnerOp);
            if (!clusteredOp) {
                return std::nullopt;
            }
            int64_t numClusters = 0;
            auto mcStrategy = clusteredOp.getMultiClusterStrategy().value();
            auto lastOutType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
            numClusters = VPU::getOptimalNumClusters(clusteredOp, lastOutType.getShape(), mcStrategy);
            SmallVector<int64_t> outputNumTiles =
                    getOutputTensorNumTiles(clusteredOp, numClusters, mcStrategy, lastOutType);
            userOpMCDims = getNonOneDim(ShapeRef(outputNumTiles));
        }
        visited.insert(lastInnerOp);
        workQueue.emplace(std::make_tuple(lastInnerOp, userOpTile, userOpMCDims));

        while (!workQueue.empty()) {
            const auto currentItem = workQueue.front();
            const auto& [currentOp, currentOutTile, currentOutMCDims] = currentItem;
            workQueue.pop();
            auto inTileInfoOrFailure = backInferOpTileInfo(currentItem, hasMCStrategy, _log);
            if (mlir::failed(inTileInfoOrFailure)) {
                return std::nullopt;
            }
            auto [inTiles, inMCDims] = inTileInfoOrFailure.value();
            if (currentOp == userOp) {
                // Reached the target viewlike op; extract tiling dims from the back-inferred tile.
                if (inTiles.empty() || operandIdx >= inTiles.size()) {
                    return std::nullopt;
                }
                userOpTile = inTiles[operandIdx];
                userOpMCDims = mlir::isa<VPU::TilingBuilderOpInterface>(currentOp) ? currentOutMCDims : inMCDims;
                return std::make_tuple(currentOp, userOpTile, userOpMCDims);
            }
            for (size_t i = 0; i < currentOp->getNumOperands() && i < inTiles.size(); ++i) {
                auto* defOp = currentOp->getOperand(i).getDefiningOp();
                if (defOp == nullptr || visited.count(defOp)) {
                    continue;
                }
                visited.insert(defOp);
                workQueue.emplace(std::make_tuple(defOp, inTiles[i], inMCDims));
            }
        }
        return std::nullopt;
    };

    auto isUserStrategyCompatible = [](mlir::Operation* userOp, VPU::TilingViewLikeOpInterface viewOp,
                                       const WorkItem& userTileInfo, bool hasMCStrategy,
                                       std::optional<DimArr> nextTilingDims) -> bool {
        auto [currentUserOp, outTile, outMCDims] = userTileInfo;
        if (currentUserOp != userOp) {
            return false;
        }
        auto tilingDims = getNonOneDim(outTile.axis);
        if (!viewOp.isSupportedTilingDim(tilingDims)) {
            return false;
        }

        // For dims with tiling restrictions (split-outer, merge), check the tiling is not too
        // aggressive. Use actual tiling number from back-inferred tiles and compare against the
        // view op's INPUT shape at the corresponding mapped dim (not output shape, which can be
        // much larger after reshape and mask the true aggressiveness of tiling).
        auto viewInShape = getShape(viewOp->getOperand(0));
        for (auto dim : tilingDims) {
            if (viewOp.isSupportedTilingDimWithRestrictions(dim)) {
                auto inDim = viewOp.backInferTilingDim(dim);
                if (outTile.axis[dim] >= static_cast<int64_t>(viewInShape[inDim] * LIMITED_TILING_DIM_MAX_RATIO)) {
                    return false;
                }
            }
        }

        if (hasMCStrategy && !viewOp.isSupportedTilingDim(outMCDims)) {
            return false;
        }

        // If the user operation has the same op type, check its tiling strategy restrictions as well. There is a
        // possibility that the two ops will be fused into a new VF op in a later pass, so need to make sure the tiling
        // strategy is compatible with the view op
        // e.g. viewOp->viewOp->VFOp0{computeOp}->VFOp1{computeOp}, merge VFOp0 and VFOp1 in mergeVF pass, ensure all
        // viewOp are compatible
        if (nextTilingDims.has_value()) {
            if (!viewOp.isSupportedTilingDim(nextTilingDims.value())) {
                return false;
            }
        }

        return true;
    };

    auto isSizeChangedOpCompatible = [](VPU::TilingViewLikeOpInterface viewOp, bool hasMCStrategy,
                                        const WorkItem& userTileInfo) {
        auto outMCDims = std::get<2>(userTileInfo);
        if (hasMCStrategy) {
            auto getShapeTotalSize = [](vpux::NDTypeInterface type) {
                if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
                    auto ndType = mlir::cast<vpux::NDTypeInterface>(sparseType.getData());
                    return ndType.getShape().totalSize();
                }
                return type.getShape().totalSize();
            };
            auto inType = mlir::cast<vpux::NDTypeInterface>(viewOp->getOperand(0).getType());
            auto outType = mlir::cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType());
            auto isSizeChangedOnly = getShapeTotalSize(inType) != getShapeTotalSize(outType) &&
                                     inType.getElemTypeSize() == outType.getElemTypeSize() &&
                                     inType.getDimsOrder() == outType.getDimsOrder();
            if (isSizeChangedOnly) {
                // get inShape Tiling Dims
                auto changedDim = IE::getDiffInOutSizeDims(inType.getShape(), outType.getShape());
                DimArr tilingDims;
                for (auto dim : outMCDims) {
                    auto tilingDim = viewOp.backInferTilingDim(dim);
                    tilingDims.push_back(tilingDim);
                }
                // If any tiling dim overlaps with the changed dim (in input space), reject the move.
                auto isChangedDimTiled = llvm::any_of(tilingDims, [&](auto dim) {
                    return llvm::is_contained(changedDim, dim);
                });
                if (isChangedDimTiled) {
                    return false;
                }
            }
        }
        return true;
    };

    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());

    auto* lastInnerOp = vfOp.getBody()->getTerminator()->getPrevNode();
    auto lastTilingBuilderOp = mlir::dyn_cast_or_null<VPU::TilingBuilderOpInterface>(lastInnerOp);
    if (lastTilingBuilderOp == nullptr) {
        return mlir::failure();
    }
    auto lastOutTiles = fillDividedTiles(lastInnerOp, ShapeRef(tilingStrategy), getShape(lastInnerOp->getResult(0)));
    if (mlir::failed(lastOutTiles)) {
        return mlir::failure();
    }
    auto lastClusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(lastInnerOp);
    const bool hasMCStrategy = lastClusteredOp != nullptr && lastClusteredOp.getMultiClusterStrategy().has_value();

    for (auto vfOperand : vfOp->getOperands() | indexed) {
        auto parentOp = vfOperand.value().getDefiningOp<VPU::TilingViewLikeOpInterface>();
        // E-163016 remove is VFSupported flag when scf and current algorithm is aligned
        if (parentOp == nullptr || !VPU::isPureViewOp(parentOp) || !parentOp.isVFSupported()) {
            continue;
        }
        if (llvm::all_of(parentOp->getOperands(), [](auto value) {
                return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                       mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
            })) {
            continue;
        }

        // Walk the full view-like op chain to find if a parent VFOp exists
        SmallVector<VPU::TilingViewLikeOpInterface> viewLikeOpChain;
        mlir::Operation* rootOp = parentOp.getOperation();
        while (mlir::isa_and_nonnull<VPU::TilingViewLikeOpInterface>(rootOp)) {
            auto viewLikeOp = mlir::cast<VPU::TilingViewLikeOpInterface>(rootOp);
            // E-163016 remove is VFSupported flag when scf and current algorithm is aligned
            if (!VPU::isPureViewOp(viewLikeOp) || !viewLikeOp.isVFSupported()) {
                break;
            }
            viewLikeOpChain.push_back(viewLikeOp);
            rootOp = viewLikeOp->getOperand(0).getDefiningOp();
        }
        auto hasSparseViewOp = llvm::any_of(viewLikeOpChain, [](auto viewLikeOp) {
            return mlir::isa<VPU::SparseTensorType>(viewLikeOp->getResult(0).getType());
        });
        if (!mlir::isa_and_nonnull<VPU::VerticalFusionOp>(rootOp) && !hasSparseViewOp) {
            // VF{SparseViewOp -> Op -> Op} outperforms SparseViewOp -> VF{Op -> Op} empirically.
            // Allow moving view ops with sparse view op even without a parent VFOp.
            continue;
        }
        // Truncate the chain at the first multi-use op
        for (size_t i = 0; i < viewLikeOpChain.size(); ++i) {
            if (!viewLikeOpChain[i]->hasOneUse()) {
                viewLikeOpChain.resize(i + 1);
                break;
            }
        }
        // Check if vfOperand parentVFOp has compatible tiling strategy with all its users and all viewlike ops in
        // between, if any. If any of them has incompatible tiling strategy, the view op will not be moved into user
        // VFOp.
        if (viewLikeOpChain.empty()) {
            continue;
        }
        auto parentVFOp =
                mlir::dyn_cast_or_null<VPU::VerticalFusionOp>(viewLikeOpChain.back()->getOperand(0).getDefiningOp());
        const auto isParentStrategyCompatible = isParentVFOpCompatible(parentVFOp, viewLikeOpChain);
        bool isValid = true;
        for (auto& use : vfOp.getBody()->getArgument(vfOperand.index()).getUses()) {
            if (!isValid) {
                break;
            }
            auto user = use.getOwner();
            auto operandId = use.getOperandNumber();
            if (isOpWeightsFromVFOperandIndex(user, vfOperand.index()) && (tilingStrategy[Dims4D::Act::C.ind()] == 1)) {
                isValid = false;
                break;
            }
            // check viewLikeOp chain
            auto userTileInfo =
                    backUserTileInfo(lastOutTiles.value(), lastInnerOp, user, operandId, hasMCStrategy, _log);
            if (!userTileInfo.has_value()) {
                isValid = false;
                break;
            }

            auto getNextTilingDims = [&]() -> std::optional<DimArr> {
                if (vfOp->hasOneUse()) {
                    if (auto nextVFOp = mlir::dyn_cast<VPU::VerticalFusionOp>(*vfOp->user_begin())) {
                        auto nextInnerOp = nextVFOp.getFirstInnerTaskOp();
                        auto areSameTypeOps = VPU::hashOperationForTilingExcludingAttr(nextInnerOp, "ppe") ==
                                              VPU::hashOperationForTilingExcludingAttr(user, "ppe");
                        if (areSameTypeOps) {
                            auto nextTilingStrategy = parseIntArrayAttr<int64_t>(nextVFOp.getTilingStrategy());
                            return getNonOneDim(ShapeRef(nextTilingStrategy));
                        }
                    }
                }
                return std::nullopt;
            };

            mlir::Operation* currentUserOp = user;
            auto nextTilingDims = getNextTilingDims();
            for (auto viewOpIt : viewLikeOpChain | indexed) {
                auto currentViewOp = viewOpIt.value();
                // check tiling strategy compatibility for each viewlike op in the chain. If any viewlike op has tiling
                // restriction, not moved into user VFOp.
                if (VPU::onlySupportPartialTilingDims(currentViewOp)) {
                    if (!isParentStrategyCompatible) {
                        isValid = false;
                        break;
                    }
                    if (!isUserStrategyCompatible(currentUserOp, currentViewOp, userTileInfo.value(), hasMCStrategy,
                                                  nextTilingDims)) {
                        isValid = false;
                        break;
                    }
                }

                // For Size changed ops, the dim which is changed should not be the user's multi-cluster tiling dim,
                // otherwise it may cause the spilling DMA between view op's parent and user op. And when the dim is not
                // the multi-cluster tiling dim, there is an copy optimization to only keep CMX2CMX copy between those
                // two ops.
                if (!isSizeChangedOpCompatible(currentViewOp, hasMCStrategy, userTileInfo.value())) {
                    isValid = false;
                    break;
                }
                if (nextTilingDims.has_value()) {
                    DimArr tilingDims;
                    for (auto dim : nextTilingDims.value()) {
                        tilingDims.push_back(currentViewOp.backInferTilingDim(dim));
                    }
                    nextTilingDims = tilingDims;
                }
                const auto hasNextViewOp = viewOpIt.index() + 1 < viewLikeOpChain.size();
                if (!hasNextViewOp) {
                    break;
                }

                // Back-infer tile info through currentViewOp for the next level in the chain
                WorkItem viewOpItem =
                        std::make_tuple(static_cast<mlir::Operation*>(currentViewOp), std::get<1>(userTileInfo.value()),
                                        std::get<2>(userTileInfo.value()));
                auto inTileInfoOrFailure = backInferOpTileInfo(viewOpItem, hasMCStrategy, _log);
                if (mlir::failed(inTileInfoOrFailure)) {
                    isValid = false;
                    break;
                }
                auto [inTiles, inMCDims] = inTileInfoOrFailure.value();
                if (!inTiles.empty()) {
                    userTileInfo = std::make_tuple(static_cast<mlir::Operation*>(currentViewOp), inTiles[0], inMCDims);
                }
                currentUserOp = currentViewOp;
            }
        }
        if (!isValid) {
            continue;
        }
        SmallVector<mlir::Operation*> viewOpChain;
        viewOpChain.reserve(viewLikeOpChain.size());
        for (auto viewOp : viewLikeOpChain) {
            viewOpChain.push_back(viewOp.getOperation());
        }
        auto newVFOp = fuseSingleViewOpsChainInBlock(rewriter, vfOp, viewOpChain);
        rewriter.replaceOp(vfOp, newVFOp.getResult(0));
        return mlir::success();
    }
    return mlir::failure();
}
}  // namespace vpux::VPU::VF::v2
