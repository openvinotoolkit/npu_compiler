//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"

#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "vpux/utils/logger/logger.hpp"

#include <cmath>

using namespace vpux;

namespace {
/*
    Correct offsets and sizes of slice operations based on remainder tiling information.

    When tiling creates uneven tile sizes (due to dimension bounds not dividing evenly),
    correctness requires adjusting slice offsets/sizes for remainder tiles. This function
    replaces the raw induction variable (which increments uniformly) with computed
    values that account for non-uniform tile boundaries.

    For each dimension in remainders:
        - remainders[dim].first = mainStepBound (boundary between uniform and remainder tiles)
        - remainders[dim].second = remainderStep (step size for remainder tiles)

    Algorithm:
        1. tileIndex = iv floorDiv loopStep  (which tile are we in?)
        2. clampedTileIndex = min(tileIndex, mainTilesCount)  (cap to prevent overflow)
        3. adjustedOffset = min(loopStep * tileIndex,
                               remainderStep * tileIndex + (loopStep - remainderStep) * mainTilesCount)
           (Blends uniform and remainder offsets based on tile index)
        4. adjustedSize = min(loopStep,
                             remainderStep + (mainTilesCount - clampedTileIndex) * (loopStep - remainderStep))
           (Adjusts tile size near boundaries)

    All uses of the raw induction variable in slice offsets are replaced with adjustedOffset;
    all uses of the old size affine.min are replaced with adjustedSize. Uses in nested blocks
    are updated via dominance-based rewriting.
*/
void correctOffsetAndSizeByRemainder(mlir::RewriterBase& builder, mlir::OffsetSizeAndStrideOpInterface slice,
                                     const std::unordered_map<Dim, std::pair<int64_t, int64_t>>& remainders) {
    if (remainders.empty()) {
        return;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);

    auto offsets = slice.getMixedOffsets();
    auto sizes = slice.getMixedSizes();

    // helper to reduce cost of building dominance info multiple times in the same loop
    struct LoopDominanceCache final {
        std::optional<mlir::DominanceInfo> dominanceInfo;
        mlir::Operation* loop = nullptr;

        mlir::DominanceInfo& getDominanceInfoForLoop(mlir::Operation* currentLoopOp) {
            if (!dominanceInfo.has_value() || loop != currentLoopOp) {
                dominanceInfo.emplace(currentLoopOp);
                loop = currentLoopOp;
            }
            return dominanceInfo.value();
        }
    };

    // Cache DominanceInfo and rebuild it when the current loop changes.
    LoopDominanceCache loopDominanceCache;

    for (auto& [dim, data] : remainders) {
        if (!slice.isDynamicOffset(dim.ind())) {
            continue;
        }

        // In nested multi-dim tiling the inner insert_slice may have a static size
        // for an outer loop's dimension (the step is a constant). Skip correction
        // for that dim — it will be handled by the outer loop's own insert_slice.
        if (!slice.isDynamicSize(dim.ind())) {
            continue;
        }

        auto dimOffset = offsets[dim.ind()];
        auto blockArgOffset = mlir::dyn_cast<mlir::BlockArgument>(mlir::cast<mlir::Value>(dimOffset));

        auto affineMin = mlir::cast<mlir::Value>(sizes[dim.ind()]).getDefiningOp<mlir::affine::AffineMinOp>();

        if (blockArgOffset == nullptr || affineMin == nullptr) {
            continue;
        }

        auto loopOp = mlir::dyn_cast<mlir::LoopLikeOpInterface>(blockArgOffset.getOwner()->getParentOp());
        if (loopOp == nullptr) {
            continue;
        }

        VPUX_THROW_WHEN(!loopOp.getLoopInductionVars().has_value() || loopOp.getLoopInductionVars()->size() != 1,
                        "The loop {0} has incorrect induction variables", loopOp);
        VPUX_THROW_WHEN(!loopOp.getLoopSteps().has_value() || loopOp.getLoopSteps()->size() != 1,
                        "The loop {0} has incorrect steps", loopOp);

        VPUX_THROW_WHEN(loopOp.getLoopRegions().empty(), "The loop {0} has no regions", loopOp);

        auto inductionVar = loopOp.getLoopInductionVars()->front();

        if (inductionVar != blockArgOffset) {
            continue;
        }

        auto loopStep = mlir::cast<mlir::Value>(loopOp.getLoopSteps()->front());
        auto loopStepConst = loopStep.getDefiningOp<mlir::arith::ConstantIndexOp>();
        VPUX_THROW_WHEN(loopStepConst == nullptr,
                        "Expected constant loop step for remainder-based offset/size correction in loop {0}", loopOp);

        builder.setInsertionPointToStart(&loopOp.getLoopRegions().front()->front());

        const auto loopStepValue = loopStepConst.value();
        VPUX_THROW_WHEN(loopStepValue <= 0, "Loop step must be a positive, non-zero value, got '{0}' for loop '{1}'",
                        loopStepValue, loopOp);
        const auto mainStepBoundValue = data.first;
        VPUX_THROW_WHEN(mainStepBoundValue % loopStepValue != 0,
                        "Main step bound '{0}' must be divisible by loop step '{1}' for loop '{2}'", mainStepBoundValue,
                        loopStepValue, loopOp);
        const auto remainderStepValue = data.second;
        const auto mainTilesCount = mainStepBoundValue / loopStepValue;
        auto loopStepExpr = getAffineConstantExpr(loopStepValue, builder.getContext());
        auto remainderStepExpr = getAffineConstantExpr(remainderStepValue, builder.getContext());
        auto mainTilesExpr = getAffineConstantExpr(mainTilesCount, builder.getContext());

        mlir::AffineExpr d0;
        bindDims(builder.getContext(), d0);

        // tileIndex = iv floordiv loopStep
        auto tileIndexMap = mlir::AffineMap::get(1, 0, {d0.floorDiv(loopStepValue)}, builder.getContext());
        auto tileIndex = mlir::affine::makeComposedAffineApply(builder, appendLoc(loopOp.getLoc(), "tile_index"),
                                                               tileIndexMap, {inductionVar});

        // offset = min(loopStep * tileIndex,
        //              remainderStep * tileIndex + (loopStep - remainderStep) * mainTilesCount)
        auto adjustedOffsetMap = mlir::AffineMap::get(
                1, 0, {loopStepExpr * d0, remainderStepExpr * d0 + (loopStepExpr - remainderStepExpr) * mainTilesExpr},
                builder.getContext());
        auto newOffset = builder.create<mlir::affine::AffineMinOp>(appendLoc(loopOp.getLoc(), "adjusted_offset"),
                                                                   adjustedOffsetMap, mlir::ValueRange{tileIndex});

        // clampedTileIndex = min(tileIndex, mainTilesCount)
        auto clampedTileIndexMap = mlir::AffineMap::get(1, 0, {d0, mainTilesExpr}, builder.getContext());
        auto clampedTileIndex = builder.create<mlir::affine::AffineMinOp>(
                appendLoc(loopOp.getLoc(), "tile_index_cap"), clampedTileIndexMap, mlir::ValueRange{tileIndex});

        // size = min(loopStep,
        //            remainderStep + (mainTilesCount - clampedTileIndex) * (loopStep - remainderStep))
        auto adjustedSizeMap = mlir::AffineMap::get(
                1, 0, {loopStepExpr, remainderStepExpr + (mainTilesExpr - d0) * (loopStepExpr - remainderStepExpr)},
                builder.getContext());
        auto newSize = builder.create<mlir::affine::AffineMinOp>(appendLoc(loopOp.getLoc(), "adjusted_size"),
                                                                 adjustedSizeMap, mlir::ValueRange{clampedTileIndex});

        auto& dominanceInfo = loopDominanceCache.getDominanceInfoForLoop(loopOp.getOperation());

        blockArgOffset.replaceUsesWithIf(newOffset.getResult(), [&](mlir::OpOperand& opOperand) {
            auto* user = opOperand.getOwner();
            if (user == tileIndex.getOperation() || user == newOffset.getOperation() ||
                user == newSize.getOperation()) {
                return false;
            }
            return dominanceInfo.properlyDominates(newOffset.getResult(), user);
        });
        auto oldSize = affineMin.getResult();
        oldSize.replaceUsesWithIf(newSize.getResult(), [&](mlir::OpOperand& opOperand) {
            auto* user = opOperand.getOwner();
            if (user == newSize.getOperation()) {
                return false;
            }
            return dominanceInfo.properlyDominates(newSize.getResult(), user);
        });
    }
}

// Listener to track which original producers have been tiled.
// Uses operation name + location as fingerprint to uniquely identify operations.
// Note: We only track that an original op was tiled (in a set), not store pointer to tiled op,
// because tiled op pointers can become invalid during subsequent loop restructuring.
class TiledOpsTrackingListener : public mlir::RewriterBase::ForwardingListener {
public:
    explicit TiledOpsTrackingListener(llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap,
                                      mlir::OpBuilder::Listener* previousListener = nullptr,
                                      Logger& log = Logger::global())
            : ForwardingListener(previousListener), skipConnectionMap(skipConnectionMap), log(log) {
    }

    void expectProducerFusion(mlir::Operation* originalProducer) {
        pendingProducers.emplace_back(originalProducer, originalProducer->getName(), originalProducer->getLoc());
        log.debug("Expecting producer {0} to be tiled for fusion", originalProducer->getName());
    }

    void notifyOperationInserted(mlir::Operation* op, mlir::OpBuilder::InsertPoint insertPoint) override {
        if (!mlir::isa<mlir::TilingInterface>(op) || op->getNumResults() == 0) {
            ForwardingListener::notifyOperationInserted(op, insertPoint);
            return;
        }
        // Track newly inserted TilingInterface operations that match pending producers.
        // Find the first pending producer with matching operation name and location (FIFO order).
        auto it = std::find_if(pendingProducers.begin(), pendingProducers.end(),
                               [&](const std::tuple<mlir::Operation*, mlir::OperationName, mlir::Location>& entry) {
                                   return std::get<1>(entry) == op->getName() && std::get<2>(entry) == op->getLoc();
                               });
        if (it == pendingProducers.end()) {
            ForwardingListener::notifyOperationInserted(op, insertPoint);
            return;
        }
        // Check whether this newly inserted tiled op corresponds to the preselected
        // 'biggest user' branch of any skip-connection source.
        // If yes, mark biggestUserTiled=true so fusion control can later allow fusing
        // the shared skip-source producer only after the largest branch is materialized.
        auto originalProducer = std::get<0>(*it);
        auto skipConnectionIter = llvm::find_if(skipConnectionMap, [&](const auto& entry) {
            return entry.second.biggestUserOp == originalProducer;
        });
        if (skipConnectionIter != skipConnectionMap.end()) {
            skipConnectionIter->second.biggestUserTiled = true;
            log.debug(" Biggest user of skip connection op {0} has been tiled: {1}",
                      skipConnectionIter->first->getName(), op->getName());
            if (!skipConnectionMap.contains(originalProducer)) {
                ForwardingListener::notifyOperationInserted(op, insertPoint);
                return;
            }
        }
        // Track fusion progress for skip-connection source producers:
        // store the currently materialized tiled result (`tiledValue`) of the original producer.
        // This value is later used by deferred replacement logic to rewire ExtractSlice users
        // from smaller branches to slices derived from the largest-branch tiled tensor.
        // Each time the skip-source operation is updated/tiled/fused, this callback is triggered
        // again because we intentionally keep that producer in `pendingProducers` (do not erase it
        // in this path). This lets us keep `tiledValue` synchronized with the latest materialized
        // result.
        if (skipConnectionMap.contains(originalProducer)) {
            skipConnectionMap.find(originalProducer)->getSecond().tiledValue = op->getResult(0);
            log.debug("Producer of skip connection has been tiled: {0}", op->getName());
        } else {
            // Clear pendingProducers only when the tiled operation is not a skip-connection producer.
            // We need to keep tracking operations that originate skip connections whenever they are
            // tiled or updated.
            // scf::tileConsumerAndFuseProducersUsingSCF may update this operation after tiling/fusion,
            // so we keep tracking it until the transformation finishes to keep tiledValue up to date.
            pendingProducers.erase(it);
        }

        ForwardingListener::notifyOperationInserted(op, insertPoint);
    }

private:
    // Queue of (original operation, operation name, location) tuples - maintains insertion order
    SmallVector<std::tuple<mlir::Operation*, mlir::OperationName, mlir::Location>, 8> pendingProducers;
    llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap;
    Logger log;
};

llvm::SetVector<mlir::Operation*> collectTiledAndFusedOps(mlir::Operation* op,
                                                          const VPU::MergeConfiguration& mergeConfig) {
    SmallVector<mlir::Operation*> worklist;
    llvm::SetVector<mlir::Operation*> producers;
    worklist.push_back(op);
    producers.insert(op);
    while (!worklist.empty()) {
        mlir::Operation* current = worklist.pop_back_val();
        for (mlir::OpOperand& operand : current->getOpOperands()) {
            mlir::Operation* producer = operand.get().getDefiningOp();
            const auto checkProducersUsers = [&](auto* user) {
                return !producers.contains(user);
            };
            if (!mlir::isa_and_nonnull<mlir::TilingInterface>(producer) || producers.contains(producer)) {
                continue;
            }
            // Use the producer result actually consumed by this operand, not a hardcoded result 0.
            // For multi-result producers the consumer may read a result other than 0; planFusion must
            // reason about the distributed type of the consumed result. Equivalent to getOpResult(0)
            // for single-result producers.
            auto producerResult = mlir::cast<mlir::OpResult>(operand.get());
            auto plan = vpux::VPU::planFusion(operand, producerResult, producers, mergeConfig);
            if (!plan.canFuse) {
                continue;
            }
            if (llvm::any_of(producer->getUsers(), checkProducersUsers)) {
                continue;
            }
            // A strategy adjustment is validated against only the consumer through which the BFS
            // encountered this producer. If the producer has multiple users (skip-connection
            // topology), the adjusted strategy may be incompatible with the other consumers.
            // Suppress the adjustment for multi-user producers; accept only if the current
            // strategy is already compatible (plan has no adjustment).
            if (plan.newProducerStrategy.has_value() && !producer->hasOneUse()) {
                continue;
            }
            // All checks passed — fusion is committed, so apply any planned MC strategy adjustment
            // immediately. This keeps the producer's updated strategy visible to subsequent BFS steps
            // that process the producers of `producer`, matching the original eager behavior. Mutation
            // happens only after every precondition succeeds, so no speculative change is introduced.
            vpux::VPU::applyFusionPlan(plan);
            worklist.push_back(producer);
            producers.insert(producer);
        }
    }
    return producers;
}

// A group encloses a long skip connection when one of its operations has two or more distinct user
// operations that are all part of the group (the fan-out is fully contained by the group). The
// default (non-scf) VF partition keeps a skip source in a separate group from the operation that
// merges the skip back in, so a group never encloses such a fan-out. Detecting this lets the
// cost-based growth stop at the same boundaries as the default partition.
bool groupEnclosesLongSkip(const llvm::SetVector<mlir::Operation*>& groupOps) {
    for (auto* op : groupOps) {
        llvm::SmallPtrSet<mlir::Operation*, 4> inGroupUsers;
        for (auto* user : op->getUsers()) {
            if (!groupOps.contains(user)) {
                continue;
            }
            inGroupUsers.insert(user);
            if (inGroupUsers.size() > 1) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

/*
  Get tile size for static shape operations based on strategy
  If size of operations > 1 it means that tile size is computed for VF
  where specifics of every operation should be taken into consideration
  The list of operation doesn't guarantee the order of operations so that lastOperation might be specify separately
  For each tile dimension the maximum size from tiles from fillDividedTiles is taken
*/
SmallVector<mlir::OpFoldResult> vpux::VPU::staticTileSizeComputation(
        mlir::OpBuilder& builder, ArrayRef<mlir::Operation*> operations, mlir::Operation* lastOperation,
        ShapeRef strategy, ShapeRef outputShape, std::unordered_map<Dim, std::pair<int64_t, int64_t>>& remainders) {
    if (operations.empty()) {
        return {};
    }

    if (lastOperation == nullptr) {
        lastOperation = operations.back();
    } else if (!llvm::is_contained(operations, lastOperation)) {
        return {};
    }

    const auto dynOperationAlignment = [&](mlir::Operation* op) {
        return op == lastOperation;
    };

    // disable alignment for performance optimizations
    const auto tiles = fillDividedTiles(lastOperation, operations, strategy, outputShape, dynOperationAlignment,
                                        /*enableSWOptimizationAlignment=*/false);

    if (mlir::failed(tiles) || tiles.value().empty()) {
        return {};
    }

    auto tilingDims = getSCFTilingOrderedDims(lastOperation, strategy);

    if (tilingDims.empty()) {
        lastOperation->removeAttr(tilingStrategy);
        return {};
    }
    std::unordered_map<Dim, int64_t> sizes;

    // if we have uneven distribution for tensor's size among tiles,
    // take the first value and remainder will be adjusted after tiling
    auto& firstTile = tiles.value().front();
    for (auto dim : tilingDims) {
        sizes[dim] = firstTile.shape[dim];
    }

    for (auto dim : tilingDims) {
        auto tileRemainderNumber = 0;
        auto offsetTile = 0;
        auto remainderSize = 0;
        for (auto tile : tiles.value() | reversed) {
            if (tile.shape[dim] == sizes[dim]) {
                break;
            }

            if (offsetTile == 0 || offsetTile != tile.offsets[dim]) {
                ++tileRemainderNumber;
            }

            remainderSize = tile.shape[dim];
            offsetTile = tile.offsets[dim];
        }
        if (tileRemainderNumber > 1 && offsetTile != 0) {
            remainders[dim] = std::make_pair(offsetTile, remainderSize);
        }
    }

    SmallVector<mlir::OpFoldResult> tileSizes;
    tileSizes.reserve(tilingDims.size());

    const auto tileSizeCondition = [&](auto index) -> mlir::OpFoldResult {
        return builder.getIndexAttr(sizes[tilingDims[index]]);
    };

    llvm::transform(llvm::seq<size_t>(0, tilingDims.size()), std::back_inserter(tileSizes), tileSizeCondition);

    return tileSizes;
}

/*
  Get tile size for dynamic shape operations based on strategy
  If size of operations > 1 it means that tile size is computed for VF
  where specifics of every operation should be taken into consideration
  If type of last operation is BoundedTensorType then bounds are used for static tile size computation
  The list of operation doesn't guarantee the order of operations so that lastOperation might be specify separately
  If not, tileSize is computed based on formula (shape value) / divisor + alignment - 1) / alignment
  where divisor is taken from strategy and alignment is taken from operation attribute if exists or set to 1
*/
SmallVector<mlir::OpFoldResult> vpux::VPU::dynamicTileSizeComputation(mlir::OpBuilder& builder,
                                                                      ArrayRef<mlir::Operation*> operations,
                                                                      mlir::Operation* lastOperation, ShapeRef strategy,
                                                                      bool useBoundedType) {
    if (operations.empty()) {
        return {};
    }

    if (lastOperation == nullptr) {
        lastOperation = operations.back();
    } else if (!llvm::is_contained(operations, lastOperation)) {
        return {};
    }

    auto outputType = mlir::cast<mlir::ShapedType>(lastOperation->getResult(0).getType());

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType);
        boundedType != nullptr && useBoundedType) {
        auto bounds = to_small_vector(boundedType.getBounds());
        std::unordered_map<Dim, std::pair<int64_t, int64_t>> emptyRemainders;
        return staticTileSizeComputation(builder, operations, lastOperation, strategy, ShapeRef(bounds),
                                         emptyRemainders);
    }

    auto outputShape = outputType.getShape();

    SmallVector<mlir::OpFoldResult> tileSizes;

    auto tilingDims = getSCFTilingOrderedDims(lastOperation, strategy);
    tileSizes.reserve(tilingDims.size());

    for (auto tileDim : tilingDims) {
        VPUX_THROW_WHEN(!outputType.isDynamicDim(tileDim.ind()), "Tiled axis {0} must be dynamic", tileDim);

        auto loc = lastOperation->getLoc();

        auto shapeValue = getDimValue(builder, lastOperation, tileDim.ind());

        const auto alignments = vpux::getAlignment(lastOperation, strategy, ShapeRef(outputShape));
        const auto divisor = strategy[tileDim];
        const auto alignment = alignments[tileDim.ind()];

        mlir::OpFoldResult tileSize;
        mlir::AffineExpr d0;
        bindDims(builder.getContext(), d0);
        auto tileSizeMap = mlir::AffineMap::get(1, 0, {(d0.ceilDiv(divisor) + alignment - 1).floorDiv(alignment)},
                                                builder.getContext());
        tileSize = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "tileSize"), tileSizeMap,
                                                               {shapeValue});
        tileSizes.emplace_back(tileSize);
    }

    return tileSizes;
}

mlir::LogicalResult vpux::VPU::applySCFTiling(mlir::Operation* operation, mlir::RewriterBase& builder) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }
    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

    mlir::scf::SCFTilingOptions tilingOptions;
    std::unordered_map<Dim, std::pair<int64_t, int64_t>> remainders;

    // Runs synchronously within a single applySCFTiling call; cachedTileSizes is only reused here.
    // TODO: Revisit if tiling is ever invoked concurrently.
    SmallVector<mlir::OpFoldResult> cachedTileSizes;
    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        if (getShape(operation->getResult(0)).isDynamic()) {
            return dynamicTileSizeComputation(builder, {operation}, nullptr, strategy);
        }
        return staticTileSizeComputation(builder, {operation}, nullptr, strategy, getShape(operation->getResult(0)),
                                         remainders);
    };

    cachedTileSizes = tileSizeComputationFnc(builder, operation);
    if (cachedTileSizes.empty() && VPU::hasDynamicDimAlignment(operation)) {
        VPU::removeDynamicDimAlignment(operation);
        cachedTileSizes = tileSizeComputationFnc(builder, operation);
    }

    if (cachedTileSizes.empty()) {
        return mlir::failure();
    }

    // Capture by value; no shared state. Tile sizes are computed eagerly—no speedup from lazy evaluation inside
    // tileUsingSCF.
    tilingOptions.setTileSizes(cachedTileSizes);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->loops.empty()) {
        return mlir::failure();
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    // tileUsingSCF drops the output order in the ForOp and terminator. This adds it back.
    const auto numResults = operation->getNumResults();
    SmallVector<mlir::Type> outputTypes;
    outputTypes.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
        outputTypes.push_back(operation->getResult(i).getType());
    }

    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forOp = mlir::cast<mlir::scf::ForOp>(loop.getOperation());
        for (unsigned i = 0; i < numResults; ++i) {
            forOp.getResult(i).setType(outputTypes[i]);
        }

        auto* terminator = forOp.getBody()->getTerminator();
        if (terminator == nullptr) {
            return;
        }

        for (auto [idx, operand] : llvm::enumerate(terminator->getOperands())) {
            if (idx >= numResults) {
                break;
            }
            operand.setType(outputTypes[idx]);

            auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp());

            if (insertSlice == nullptr) {
                // outer loop has no insertSlice op, modify init args
                if (idx < static_cast<unsigned>(forOp.getNumRegionIterArgs())) {
                    forOp.getInitArgs()[idx].setType(outputTypes[idx]);
                }
                continue;
            }

            insertSlice.getDestMutable().get().setType(outputTypes[idx]);
            if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                auto argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
                forOp.getInitArgs()[argIndex].setType(outputTypes[idx]);
            }
            // insert_slice is in innermost loop
            correctOffsetAndSizeByRemainder(builder, insertSlice, remainders);
        }
    });

    builder.replaceOp(operation, tilingResult->replacements);

    return mlir::success();
}

SmallVector<mlir::Operation*> vpux::VPU::applySCFVerticalFusion(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                                const MergeConfiguration& mergeConfig, Logger log) {
    auto tilingInterfaceOp = mlir::cast<mlir::TilingInterface>(operation);

    mlir::scf::SCFTilingOptions tilingOptions;

    auto allOpsToFuse = collectTiledAndFusedOps(operation, mergeConfig);
    if (allOpsToFuse.size() == 1) {
        return {};
    }

    // Peels the current input boundary of the working set: operations whose operands are all
    // external to the set (block arguments, constants, or values produced outside the set). Only
    // these true source operations can be removed without disconnecting the remaining cone, which
    // must stay a single-output sub-cone anchored at `operation` for the downstream tiling and cost
    // analyses (both walk up from a single output).
    //
    // VFConfig::getInputs() is deliberately not used here: it reports operations by
    // VerticalFusionOpInterface membership and can misclassify a middle operation of the chain as
    // an input when the intermediate producers (for example NCE.Permute or DepthToSpace) do not
    // implement that interface. Removing such a middle operation would split the cone into a
    // disconnected multi-output group that the tiling storage cannot describe.
    //
    // Returns true when at least one operation was removed. A single-output cone always has a
    // source distinct from the anchor while more than one operation remains, so progress is
    // guaranteed; the caller still stops on a false return as a defensive loop-termination bound.
    const auto popInputOperation = [&]() -> bool {
        SmallVector<mlir::Operation*> sources;
        for (auto* op : allOpsToFuse) {
            if (op == operation) {
                continue;
            }
            const bool allOperandsExternal = llvm::all_of(op->getOperands(), [&](mlir::Value operand) {
                auto* producer = operand.getDefiningOp();
                return producer == nullptr || !allOpsToFuse.contains(producer);
            });
            if (allOperandsExternal) {
                sources.push_back(op);
            }
        }
        bool removedAny = false;
        for (auto* op : sources) {
            removedAny |= allOpsToFuse.remove(op);
        }
        return removedAny;
    };

    std::optional<VF::v2::VFCase> bestVFCaseOpt;

    // Cost-based candidates collected while shrinking from the maximal fused cone toward the anchor,
    // ordered from the largest group (front) to the smallest group nearest the anchor (back).
    SmallVector<VF::v2::VFCase, 4> costBasedCandidates;
    VF::v2::VFCacheAnalysis cache(operation);

    while (allOpsToFuse.size() > 1) {
        VF::v2::VFConfig config(allOpsToFuse, cache);
        // Stop once only view-like operations remain: they cannot be tiled/fused on their own.
        const auto nonViewLikeOpCount = llvm::count_if(allOpsToFuse, [](mlir::Operation* op) {
            return !mlir::isa<VPU::TilingViewLikeOpInterface>(op);
        });
        if (nonViewLikeOpCount == 0) {
            break;
        }

        DimArr allowedDims = getAllowedDims(allOpsToFuse.getArrayRef(), log);
        if (allowedDims.empty()) {
            if (!popInputOperation()) {
                break;
            }
            continue;
        }

        auto vfCase = mergeConfig.splitGetter(config, allowedDims);

        if (VPU::hasDynamicDimAlignment(operation) && !vfCase.isInitialized()) {
            VPU::removeDynamicDimAlignment(operation);
            vfCase = mergeConfig.splitGetter(config, allowedDims);
        }

        // Use the merge policy from the configuration to decide whether to proceed
        if (vfCase.isInitialized() && mergeConfig.mergeDecision(vfCase)) {
            if (mergeConfig.selectBetterCase == nullptr) {
                // Greedy configuration: prioritize merging by keeping the first (largest)
                // valid candidate found while shrinking from the input side.
                bestVFCaseOpt = std::move(vfCase);
                break;
            }

            // Cost-based configuration: record every valid candidate; the fusion boundary is
            // selected after the full nested-subset sequence has been enumerated.
            costBasedCandidates.push_back(std::move(vfCase));
        }

        if (!popInputOperation()) {
            break;
        }
    }

    if (mergeConfig.selectBetterCase != nullptr) {
        // Bottom-up growth over the enumerated candidates. Starting from the smallest valid group
        // nearest the anchor (back of the list), extend the group toward the inputs only while
        // fusing the additional operations stays profitable (selectBetterCase), and stop at the
        // first non-profitable boundary. Operations left outside the group are fused at a later
        // anchor of the reverse walk. This mirrors the incremental pairwise merge of the default
        // (non-scf) VF path and avoids over-merging a large group whose best-split cost hides the
        // true per-operation cost.
        for (int idx = costBasedCandidates.size(); idx-- > 0;) {
            auto& candidate = costBasedCandidates[idx];
            if (!bestVFCaseOpt.has_value()) {
                bestVFCaseOpt = std::move(candidate);
                continue;
            }
            // Stop before a group encloses a long skip connection. The default partition keeps a
            // skip source separate from the operation that merges the skip, so growing past this
            // boundary would over-merge relative to the default VF grouping.
            if (groupEnclosesLongSkip(candidate.getConfig().getVFOperations())) {
                break;
            }
            if (!mergeConfig.selectBetterCase(bestVFCaseOpt.value(), candidate)) {
                break;
            }
            bestVFCaseOpt = std::move(candidate);
        }
    }

    if (!bestVFCaseOpt.has_value()) {
        return {};
    }

    auto bestVFCase = std::move(bestVFCaseOpt.value());

    if (!bestVFCase.isInitialized()) {
        return {};
    }

    allOpsToFuse = bestVFCase.getConfig().getVFOperations();

    auto outputs = bestVFCase.getConfig().getOutputs();
    if (outputs.empty()) {
        return {};
    }

    const auto& tilingStorage = bestVFCase.getTilingStorage();

    llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement> skipConnectionMap =
            analyzeSkipConnectionsForTiling(allOpsToFuse, tilingStorage, log);

    std::optional<mlir::Attribute> strategy;
    if (operation->hasAttr(tilingStrategy)) {
        strategy = operation->getAttr(tilingStrategy);
    }

    operation->setAttr(tilingStrategy, bestVFCase.getTiling());
    std::unordered_map<Dim, std::pair<int64_t, int64_t>> remainders;

    auto* lastOp = outputs.back();
    auto outputType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType());

    const auto vfTileSizeComputationFn = [&](mlir::OpBuilder& builder,
                                             mlir::Operation* operation) -> SmallVector<mlir::OpFoldResult> {
        auto strategy = Shape(parseIntArrayAttr<int64_t>(bestVFCase.getTiling()));

        if (outputType.getShape().isStatic()) {
            return staticTileSizeComputation(builder, allOpsToFuse.getArrayRef(), operation, strategy,
                                             getShape(operation->getResult(0)), remainders);
        }

        return dynamicTileSizeComputation(builder, allOpsToFuse.getArrayRef(), operation, strategy);
    };

    tilingOptions.setTileSizeComputationFunction(vfTileSizeComputationFn);

    mlir::scf::SCFTileAndFuseOptions tilingAndFuseOptions;
    tilingAndFuseOptions.setTilingOptions(std::move(tilingOptions));

    bool hasSkipConnection = false;

    auto* previousListener = builder.getListener();
    TiledOpsTrackingListener listener(skipConnectionMap, previousListener, log);
    builder.setListener(&listener);

    mlir::scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
            [&](mlir::tensor::ExtractSliceOp sliceOp, mlir::OpResult originalProducer,
                bool) -> std::optional<mlir::scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (!allOpsToFuse.contains(originalProducer.getOwner())) {
            return std::nullopt;
        }
        auto skipConnectionIter = skipConnectionMap.find(originalProducer.getOwner());
        if (skipConnectionIter == skipConnectionMap.end()) {
            originalProducer.getOwner()->setAttr(tilingStrategy, bestVFCase.getTiling());
            listener.expectProducerFusion(originalProducer.getOwner());
            return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{};
        }
        log.debug("Attempting to fuse producer of skip connection: {0} at loc: {1}",
                  originalProducer.getOwner()->getName(), originalProducer.getOwner()->getLoc());
        auto& deferredReplacement = skipConnectionIter->getSecond();

        if (deferredReplacement.tiledValue != nullptr) {
            log.debug("Operation which originates skip connection has been tiled already. Save current sliceOp "
                      "for future replacement.\n");
            deferredReplacement.relatedExtractSlices.insert(sliceOp);
            return std::nullopt;
        }

        if (!deferredReplacement.biggestUserTiled && !deferredReplacement.allUsersWithTheSameTileSize) {
            log.debug("Biggest user not tiled yet, cannot fuse producer. Skipping fusion for this producer");
            deferredReplacement.relatedExtractSlices.insert(sliceOp);
            return std::nullopt;
        }

        log.debug("Biggest User tiled (or all users have the same tile size), allowing fusion");
        deferredReplacement.biggestTileExtractSlice = sliceOp;

        originalProducer.getOwner()->setAttr(tilingStrategy, bestVFCase.getTiling());
        listener.expectProducerFusion(originalProducer.getOwner());
        hasSkipConnection = true;
        return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{};
    };
    tilingAndFuseOptions.setFusionControlFn(std::move(controlFn));
    builder.setInsertionPoint(operation);

    auto tiledResults =
            mlir::scf::tileConsumerAndFuseProducersUsingSCF(builder, tilingInterfaceOp, tilingAndFuseOptions);

    if (mlir::failed(tiledResults) || tiledResults->replacements.empty() || tiledResults->loops.empty() ||
        tiledResults->fusedProducers.empty()) {
        if (strategy.has_value()) {
            operation->setAttr(tilingStrategy, strategy.value());
        }
        builder.setListener(previousListener);
        return {};
    }

    applyDeferredSliceReplacements(builder, skipConnectionMap, log);

    for (auto result : operation->getResults()) {
        tiledResults->replacements[result].setType(result.getType());
    }

    llvm::for_each(tiledResults->loops, [&](mlir::LoopLikeOpInterface loopOperation) {
        auto loop = mlir::cast<mlir::scf::ForOp>(loopOperation);

        auto* terminator = loop.getBody()->getTerminator();
        if (terminator == nullptr) {
            return;
        }
        llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
            operand.setType(loop.getResult(0).getType());

            auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp());
            if (insertSlice == nullptr) {
                loop.getInitArgs().back().setType(operand.getType());
                return;
            }
            insertSlice.getDestMutable().get().setType(loop.getResult(0).getType());
            if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                auto argIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
                loop.getInitArgs()[argIndex].setType(operand.getType());
            }
            correctOffsetAndSizeByRemainder(builder, insertSlice, remainders);
        });

        if (hasSkipConnection) {
            vpux::VPU::reorderOperations(
                    to_small_vector(loop.getBody()->without_terminator() | transformed([](mlir::Operation& op) {
                                        return &op;
                                    })));
        }
    });

    for (mlir::OpResult res : operation->getResults()) {
        if (auto replacement = tiledResults->replacements.lookup(res)) {
            builder.replaceAllUsesWith(res, replacement);
        }
    }

    if (operation->use_empty()) {
        builder.eraseOp(operation);
    }

    builder.setListener(previousListener);

    return to_small_vector(tiledResults->fusedProducers);
}
