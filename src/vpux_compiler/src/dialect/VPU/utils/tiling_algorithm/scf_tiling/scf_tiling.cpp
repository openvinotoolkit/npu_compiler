//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"

#include "vpux/compiler/dialect/VPU/utils/reorder_ir_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dominance.h"

#include <deque>

using namespace vpux;

SmallVector<mlir::OpFoldResult> vpux::VPU::staticTileSizeComputation(mlir::OpBuilder& builder,
                                                                     mlir::Operation* operation, ShapeRef strategy,
                                                                     ShapeRef outputShape) {
    const auto tiles = fillDividedTiles(operation, strategy, outputShape);

    if (mlir::failed(tiles)) {
        return {};
    }

    auto tilingDims = getSCFTilingOrderedDims(operation, strategy);
    std::unordered_map<Dim, int64_t> sizes;

    for (auto& tile : tiles.value()) {
        for (auto dim : tilingDims) {
            sizes[dim] = std::max(tile.shape[dim], sizes[dim]);
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

SmallVector<mlir::OpFoldResult> vpux::VPU::dynamicTileSizeComputation(mlir::OpBuilder& builder,
                                                                      mlir::Operation* operation, ShapeRef strategy) {
    auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType)) {
        auto bounds = to_small_vector(boundedType.getBounds());
        return staticTileSizeComputation(builder, operation, strategy, ShapeRef(bounds));
    }

    auto outputShape = outputType.getShape();

    SmallVector<mlir::OpFoldResult> tileSizes;

    auto tilingDims = getSCFTilingOrderedDims(operation, strategy);
    tileSizes.reserve(tilingDims.size());

    for (auto tileDim : tilingDims) {
        VPUX_THROW_WHEN(!outputType.isDynamicDim(tileDim.ind()), "Tiled axis {0} must be dynamic", tileDim);

        auto loc = operation->getLoc();

        auto shapeValue = getDimValue(builder, operation, tileDim.ind());

        auto optAlignment = vpux::getAlignment(operation, strategy, ShapeRef(outputShape));
        const auto divisor = strategy[tileDim];
        const auto alignment = optAlignment.has_value() ? optAlignment.value()[tileDim.ind()] : 1;

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

    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        if (getShape(operation->getResult(0)).isDynamic()) {
            return dynamicTileSizeComputation(builder, operation, strategy);
        }

        return staticTileSizeComputation(builder, operation, strategy, getShape(operation->getResult(0)));
    };

    tilingOptions.setTileSizeComputationFunction(tileSizeComputationFnc);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->replacements.empty() ||
        tilingResult->replacements.size() != operation->getNumResults() || tilingResult->loops.empty()) {
        return mlir::failure();
    }

    for (auto [result, loopOutput] : llvm::zip(operation->getResults(), tilingResult->replacements)) {
        loopOutput.setType(result.getType());
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        auto forOp = mlir::cast<mlir::scf::ForOp>(loop.getOperation());

        auto* terminator = forOp.getBody()->getTerminator();
        if (terminator != nullptr) {
            llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
                operand.setType(forOp.getResult(0).getType());

                if (auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp())) {
                    insertSlice.getDestMutable().get().setType(forOp.getResult(0).getType());
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                        auto argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
                        forOp.getInitArgs()[argIndex].setType(operand.getType());
                    }
                } else {
                    // outer loop has no insertSlice op, modify init args by setting order to the last one
                    forOp.getInitArgs().back().setType(operand.getType());
                }
            });
        }
    });

    builder.replaceOp(operation, tilingResult->replacements);

    return mlir::success();
}

// copied from llvm, the logic is adjusted before llvm 20 update
static std::tuple<mlir::OpResult, std::optional<mlir::OpOperand*>> getUntiledProducerFromSliceSource(
        mlir::OpOperand* source, ArrayRef<mlir::LoopLikeOpInterface> loops) {
    std::optional<mlir::OpOperand*> destinationIterArg;
    auto loopIt = loops.rbegin();
    while (auto iterArg = mlir::dyn_cast<mlir::BlockArgument>(source->get())) {
        auto loop = *loopIt;
        if (iterArg.getOwner()->getParentOp() != loop) {
            break;
        }
        source = loop.getTiedLoopInit(iterArg);
        loopIt++;
    }
    if (loopIt == loops.rend()) {
        destinationIterArg = source;
    }
    return {mlir::dyn_cast<mlir::OpResult>(source->get()), destinationIterArg};
}

/// Implementation of tile consumer and fuse producer greedily.
mlir::FailureOr<mlir::scf::SCFTileAndFuseResult> tileConsumerAndFuseProducers(
        mlir::RewriterBase& rewriter, mlir::TilingInterface consumer, const mlir::scf::SCFTileAndFuseOptions& options,
        std::unordered_map<mlir::Operation*, mlir::Value>& tiledOpsLink) {
    // This transformation is only valid for ops that return values (i.e. not
    // valid to use with operations that have memref operands).
    if (!consumer->getNumResults()) {
        return rewriter.notifyMatchFailure(consumer, "invalid pattern for op with no results");
    }

    // 1. First tile the consumer.
    mlir::SetVector<mlir::Operation*> fusedProducers, tiledAndFusedOps;
    llvm::SmallDenseMap<mlir::Value, size_t> origProducerToLoopResultNum;

    auto tilingResult = tileUsingSCF(rewriter, consumer, options.tilingOptions);

    if (failed(tilingResult)) {
        return rewriter.notifyMatchFailure(consumer, "failed to tile consumer");
    }
    for (auto* tiledOp : tilingResult->tiledOps) {
        tiledAndFusedOps.insert(tiledOp);
    }

    // If there are no loops generated, fusion is immaterial.
    auto& loops = tilingResult->loops;
    if (loops.empty()) {
        DenseMap<mlir::Value, mlir::Value> replacements;
        for (auto [origVal, replacement] : llvm::zip_equal(consumer->getResults(), tilingResult->replacements)) {
            replacements[origVal] = replacement;
        }
        return mlir::scf::SCFTileAndFuseResult{std::move(fusedProducers), std::move(tiledAndFusedOps), loops,
                                               replacements};
    }

    // To keep track of replacements for now just record the map from the original
    // untiled value to the result number of the for loop. Since the loop gets
    // potentially replaced during fusion, keeping the value directly wont work.
    DenseMap<mlir::Value, size_t> origValToResultNumber;
    for (auto [index, result] : llvm::enumerate(consumer->getResults())) {
        origValToResultNumber[result] = index;
    }

    std::function<void(mlir::Operation*, std::deque<mlir::tensor::ExtractSliceOp>&)> addCandidateSlices =
            [&](mlir::Operation* fusedOp, std::deque<mlir::tensor::ExtractSliceOp>& candidates) {
                for (mlir::Value operand : fusedOp->getOperands()) {
                    if (auto sliceOp = operand.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
                        if (candidates.empty() || llvm::find(candidates, sliceOp) == candidates.end()) {
                            candidates.push_back(sliceOp);
                        }
                    } else if (auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(operand.getDefiningOp())) {
                        addCandidateSlices(padOp, candidates);
                    }
                }
            };

    std::deque<mlir::tensor::ExtractSliceOp> candidates;
    addCandidateSlices(tiledAndFusedOps.back(), candidates);
    mlir::OpBuilder::InsertionGuard g(rewriter);
    while (!candidates.empty()) {
        // Traverse the slices in BFS fashion.
        mlir::tensor::ExtractSliceOp candidateSliceOp = candidates.front();
        candidates.pop_front();

        // Find the original producer of the slice.
        auto [fusableProducer, destinationInitArg] =
                getUntiledProducerFromSliceSource(&candidateSliceOp.getSourceMutable(), loops);
        if (!fusableProducer) {
            continue;
        }

        auto [fuseSlice, yieldReplacement] =
                options.fusionControlFn(candidateSliceOp, fusableProducer, destinationInitArg.has_value());
        if (!fuseSlice) {
            continue;
        }

        // The operands of the fused producer might themselved be slices of
        // values produced by operations that implement the `TilingInterface`.
        // Add these operations to the worklist.
        auto fusedResult = mlir::scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, loops);
        if (!fusedResult) {
            continue;
        }

        if (mlir::Operation* tiledAndFusedOp = fusedResult->tiledAndFusedProducer.getDefiningOp()) {
            fusedProducers.insert(fusedResult->origProducer.getDefiningOp());
            tiledAndFusedOps.insert(tiledAndFusedOp);
            tiledOpsLink[fusedResult->origProducer.getDefiningOp()] = fusedResult->tiledAndFusedProducer;
            addCandidateSlices(tiledAndFusedOp, candidates);
        }
    }

    DenseMap<mlir::Value, mlir::Value> replacements;
    for (auto [origVal, resultNumber] : origValToResultNumber) {
        replacements[origVal] = loops.front()->getResult(resultNumber);
    }

    return mlir::scf::SCFTileAndFuseResult{std::move(fusedProducers), std::move(tiledAndFusedOps), loops, replacements};
}

llvm::SetVector<mlir::Operation*> collectTiledAndFusedOps(mlir::Operation* op) {
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
            if (!mlir::isa_and_nonnull<mlir::TilingInterface>(producer) || producers.contains(producer) ||
                !vpux::VPU::checkFusion(operand, producer->getOpResult(0)) ||
                llvm::any_of(producer->getUsers(), checkProducersUsers)) {
                continue;
            }
            worklist.push_back(producer);
            producers.insert(producer);
        }
    }
    return producers;
}

VPU::VF::v2::VFSplit getVFSplit(vpux::NDTypeInterface outputType, mlir::Operation* outputOperation,
                                DimArrRef allowedDims, VPU::VF::v2::VFConfig& config) {
    auto shapeType = mlir::cast<mlir::ShapedType>(outputType);
    if (!shapeType.hasStaticShape()) {
        VPU::VF::v2::VFSplit vfSplit;
        auto countDynDims = shapeType.getNumDynamicDims();
        // allowedDims to memoryOrder
        auto outputOrder = outputType.getDimsOrder();
        auto allowedDimsInMemoryOrder = allowedDims.vec();
        llvm::sort(allowedDimsInMemoryOrder, [&](const Dim& d1, const Dim& d2) {
            return outputOrder.dimPos(d1) < outputOrder.dimPos(d2);
        });
        for (auto dim : allowedDimsInMemoryOrder) {
            if (!shapeType.isDynamicDim(dim.ind())) {
                continue;
            }

            VPUX_THROW_WHEN(dim == Dims4D::Act::C, "Dynamic channels are not supported");
            if (countDynDims == 1) {
                vfSplit[dim] = std::nullopt;
                break;
            }

            vfSplit[dim] = vpux::VPU::getTilingLimit(dim, config.getVFOperations().getArrayRef(), true);
            --countDynDims;
        }

        return vfSplit;
    }

    VPU::SiblingOpsAnalysis siblingAnalisys(outputOperation);
    auto outputShape = outputType.getShape().toValues();
    if (auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(outputOperation)) {
        if (clusterOp.getMultiClusterStrategy().has_value()) {
            outputType = clusterOp.getDistributedTypeForOpResult(
                    outputOperation->getResult(0), clusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false);

            auto distribution = VPU::DistributionInfo::getClassFromAttr(
                    mlir::cast<VPU::DistributedTensorType>(outputType).getDistribution());
            if (distribution.getMemoryShapes().empty()) {
                auto optMemoryShapes = VPU::getPerClusterMemoryShapes(outputShape, distribution);
                if (optMemoryShapes.has_value()) {
                    outputShape = Shape(optMemoryShapes.value().front());
                }
            } else {
                outputShape = Shape(distribution.getMemoryShapes().front());
            }
        }
    }

    const auto compareDims = [&](auto dimLeft, auto dimRight) {
        return outputShape[dimLeft] < outputShape[dimRight];
    };
    auto maxDim = std::max_element(allowedDims.begin(), allowedDims.end(), compareDims);

    if (maxDim == allowedDims.end()) {
        return {};
    }

    return {{*maxDim, std::nullopt}};
}

mlir::FailureOr<SmallVector<mlir::Operation*>> vpux::VPU::applySCFVerticalFusion(mlir::Operation* operation,
                                                                                 mlir::RewriterBase& builder,
                                                                                 Logger log) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }

    auto tilingInterfaceOp = mlir::cast<mlir::TilingInterface>(operation);

    const auto strategy = operation->getAttr(tilingStrategy);
    mlir::scf::SCFTilingOptions tilingOptions;

    // calculate tile size based on VF restrictions
    auto allOpsToFuse = collectTiledAndFusedOps(operation);

    if (allOpsToFuse.size() == 1) {
        return mlir::failure();
    }

    VF::v2::VFConfig config(allOpsToFuse);
    DimArr allowedDims = getAllowedDims(allOpsToFuse.getArrayRef(), log);
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    auto outputs = config.getOutputs();
    if (outputs.empty()) {
        return mlir::failure();
    }

    auto* lastOp = outputs.back();
    auto outputType = mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType());

    auto vfSplit = getVFSplit(outputType, lastOp, allowedDims, config);
    if (vfSplit.empty()) {
        return mlir::failure();
    }
    VPU::VF::v2::VFCase bestVFCase(config, vfSplit);
    const auto getMinTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) {
        if (split.size() > 1) {
            return MINIMUM_LENGTH_TILING;
        }

        const auto getDimValue = [&dim](auto* oper) -> int64_t {
            return oper->hasAttr(tilingStrategy) ? Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(
                                                           oper->getAttr(tilingStrategy))))[dim]
                                                 : 1;
        };

        std::set<int64_t> minTilesSet;
        llvm::copy(config.getVFOperations() | transformed(getDimValue), std::inserter(minTilesSet, minTilesSet.end()));
        return *minTilesSet.rbegin();
    };
    const auto getMaxTiles = [&](auto dim, const VPU::VF::v2::VFSplit& split) -> int64_t {
        if (mlir::ShapedType::isDynamic(outputType.getShape()[dim])) {
            return mlir::ShapedType::kDynamic;
        }
        auto maxTiles = getTilingLimit(dim, config.getVFOperations().getArrayRef());
        if (split.size() > 1) {
            // 2D tiling
            auto otherDimSum = VPU::VF::v2::getVFTilesLen(split);
            maxTiles = divUp(maxTiles, otherDimSum);
        }
        return maxTiles;
    };

    bestVFCase = VPU::VF::v2::getVFCaseWithTiling(config, vfSplit.rbegin()->first, vfSplit, getMinTiles, getMaxTiles,
                                                  log, VPU::VF::v2::getSchedulingScenarios(config, log));

    if (!bestVFCase.isInitialized()) {
        return mlir::failure();
    }
    operation->setAttr(tilingStrategy, bestVFCase.getTiling());

    // calculate tile size for VF:
    // 1. allOpsToFuse contains operations to build VF
    // 2. get all allowed dimensions for these operations to tile
    // 3. choose the dimension which tiles the largest dimension
    // 4. get optimal tiling number for vertical fusion
    // 5. calculate tile size based on computed tiling number
    const auto vfTileSizeComputationFn = [&](mlir::OpBuilder& builder,
                                             mlir::Operation* operation) -> SmallVector<mlir::OpFoldResult> {
        auto strategy = Shape(parseIntArrayAttr<int64_t>(bestVFCase.getTiling()));

        if (mlir::cast<vpux::NDTypeInterface>(lastOp->getResult(0).getType()).getShape().isStatic()) {
            return staticTileSizeComputation(builder, operation, strategy, getShape(operation->getResult(0)));
        }

        return dynamicTileSizeComputation(builder, operation, strategy);
    };

    tilingOptions.setTileSizeComputationFunction(vfTileSizeComputationFn);

    mlir::scf::SCFTileAndFuseOptions tilingAndFuseOptions;
    tilingAndFuseOptions.setTilingOptions(std::move(tilingOptions));

    // linked original ops to fused result operations
    std::unordered_map<mlir::Operation*, mlir::Value> tiledOpsLink;
    // check if VF has loop to substitute slice with existing operation
    bool hasVFLoop = false;

    mlir::scf::SCFTileAndFuseOptions::ControlFnTy controlFn = [&](mlir::tensor::ExtractSliceOp sliceOp,
                                                                  mlir::OpResult originalProducer, bool) {
        auto isFusable = allOpsToFuse.contains(originalProducer.getOwner());
        if (isFusable) {
            if (tiledOpsLink.count(originalProducer.getOwner()) > 0) {
                mlir::OpBuilder::InsertionGuard g(builder);
                builder.setInsertionPoint(sliceOp);
                // substitute with existing operation
                auto tiledValue = tiledOpsLink[originalProducer.getOwner()];
                builder.replaceAllUsesWith(sliceOp->getResults(), tiledValue);
                builder.eraseOp(sliceOp);
                hasVFLoop = true;

                return std::make_tuple(false, false);
            }
            originalProducer.getOwner()->setAttr(tilingStrategy, bestVFCase.getTiling());
        }
        return std::make_tuple(isFusable, false);
    };
    tilingAndFuseOptions.setFusionControlFn(std::move(controlFn));

    builder.setInsertionPoint(operation);
    auto tiledResults = tileConsumerAndFuseProducers(builder, tilingInterfaceOp, tilingAndFuseOptions, tiledOpsLink);

    if (mlir::failed(tiledResults) || tiledResults->replacements.empty() || tiledResults->loops.empty() ||
        tiledResults->fusedProducers.empty()) {
        operation->setAttr(tilingStrategy, strategy);
        return mlir::failure();
    }

    // propagate result type with order and bounds attributes to operations
    // created in SCF functions.
    for (auto result : operation->getResults()) {
        tiledResults->replacements[result].setType(result.getType());

        // in case the shape is dynamic, reifyResultShapes functions may add tensor.dim operations
        // to the parent of the function. in case the parent is fused to the loop
        // and original operation is supposed to be removed from the IR, such users should be reassigned
        // to the inputs of VF
        if (mlir::cast<vpux::NDTypeInterface>(result.getType()).getShape().isDynamic()) {
            for (auto operand : operation->getOperands()) {
                auto* parentOp = operand.getDefiningOp();
                if (tiledResults->fusedProducers.contains(parentOp) && !parentOp->hasOneUse()) {
                    for (auto& use : llvm::make_early_inc_range(parentOp->getUses())) {
                        if (use.getOwner() == operation) {
                            continue;
                        }
                        if (auto dimTensor = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner())) {
                            dimTensor.setOperand(use.getOperandNumber(), config.getInputs().front()->getOperand(0));
                        }
                    }
                }
            }
        }
    }

    // E-162999 rewrite to update order attribute for output types more elegantly
    llvm::for_each(tiledResults->loops, [&](mlir::LoopLikeOpInterface loopOperation) {
        auto loop = mlir::cast<mlir::scf::ForOp>(loopOperation);

        auto* terminator = loop.getBody()->getTerminator();
        if (terminator != nullptr) {
            llvm::for_each(terminator->getOperands(), [&](mlir::Value operand) {
                operand.setType(loop.getResult(0).getType());

                if (auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operand.getDefiningOp())) {
                    insertSlice.getDestMutable().get().setType(loop.getResult(0).getType());
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertSlice.getDest())) {
                        auto argIndex = blockArg.getArgNumber() - loop.getNumInductionVars();
                        loop.getInitArgs()[argIndex].setType(operand.getType());
                    }
                    if (hasVFLoop) {
                        // reorder operations in the loop body to ensure that operation in the loop
                        // is in order after resolving circle dependencies
                        vpux::VPU::reorderOperations(to_small_vector(loop.getBody()->without_terminator() |
                                                                     transformed([](mlir::Operation& op) {
                                                                         return &op;
                                                                     })));
                    }

                } else {
                    // outer loop has no insertSlice op, modify init args by setting order to the last one
                    loop.getInitArgs().back().setType(operand.getType());
                }
            });
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

    return to_small_vector(tiledResults->fusedProducers);
}

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
