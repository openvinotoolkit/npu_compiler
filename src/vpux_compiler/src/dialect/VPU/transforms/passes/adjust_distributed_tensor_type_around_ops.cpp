//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTDISTRIBUTEDTENSORAROUNDOPS
#define GEN_PASS_DEF_ADJUSTDISTRIBUTEDTENSORAROUNDOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// DistributedInputTypeRewriter
//
class DistributedInputTypeRewriter final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    DistributedInputTypeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx), _log(log) {
        this->setDebugName("DistributedInputTypeRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface, mlir::PatternRewriter& rewriter) const final;

    bool fitIntoCMX(VPU::NCEOpInterface origOp, VPU::DistributedTensorType newInType) const;

private:
    Logger _log;
};

mlir::LogicalResult DistributedInputTypeRewriter::matchAndRewrite(VPU::NCEOpInterface origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp.getLoc());
    /*
       Convert subgraph below when percluster memory shape of DistributedType1 is included in percluster memory shape
       of DistributedType0

            DistributedType0       DistributedType0
                  |                       |
                Copy                    Copy
                  |                       |
                Copy            =>      Copy
                  |                       |
            DistributedType1       DistributedType0
                  |                       |
                 NCE                     NCE
    */
    if (auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(origOp.getOperation())) {
        if (eltwiseOp.getIsInplace().value_or(false)) {
            _log.trace("Skip for inplace case since the change will affect the output type");
            return mlir::failure();
        }
    }

    auto input = origOp->getOperand(0);
    auto distributedInType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(input.getType());
    if (distributedInType == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Input is not distributed tensor type at '{0}'", origOp->getLoc());
    }
    auto inMode = distributedInType.getDistribution().getMode().getValue();
    if (inMode != VPU::DistributionMode::OVERLAPPED) {
        return matchFailed(_log, rewriter, origOp, "Input distributed tensor type is not OVERLAPPED at '{0}'",
                           origOp->getLoc());
    }

    auto inCopy = input.getDefiningOp<VPU::CopyOp>();
    if (inCopy == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Input is not from copy op at '{0}'", origOp->getLoc());
    }

    const auto tilingScheme = vpux::parseIntArrayAttr<int64_t>(distributedInType.getDistribution().getNumTiles());

    auto parentCopy = inCopy.getInput().getDefiningOp<VPU::CopyOp>();
    if (parentCopy == nullptr) {
        return matchFailed(_log, rewriter, inCopy, "parent is not copy op at '{0}'", inCopy->getLoc());
    }
    auto parentDistributedInType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(parentCopy.getInput().getType());
    if (parentDistributedInType == nullptr) {
        return matchFailed(_log, rewriter, parentCopy,
                           "Input type of parent copy op is not distributed tensor type at '{0}'",
                           parentCopy->getLoc());
    }

    if (mlir::succeeded(VPU::isDistributedCastCompatible(parentDistributedInType, distributedInType))) {
        return matchFailed(_log, rewriter, inCopy, "Copy op types are compatible for optimization at '{0}'",
                           inCopy->getLoc());
    }
    auto parentMode = parentDistributedInType.getDistribution().getMode().getValue();
    if (parentMode != inMode) {
        return matchFailed(_log, rewriter, parentCopy, "Input distributed tensor type is not OVERLAPPED at '{0}'",
                           parentCopy->getLoc());
    }

    const auto parentTilingScheme =
            vpux::parseIntArrayAttr<int64_t>(parentDistributedInType.getDistribution().getNumTiles());

    if (tilingScheme != parentTilingScheme) {
        return matchFailed(_log, rewriter, origOp, "Tiling Scheme are different for {0} output and {1} input",
                           inCopy->getLoc(), parentCopy->getLoc());
    }

    auto perClusterMemShapes = distributedInType.getPerClusterMemoryShapes();
    auto perClusterMemShapeOffsets = distributedInType.getPerClusterMemoryShapeOffsets();
    auto parentPerClusterMemShapes = parentDistributedInType.getPerClusterMemoryShapes();
    auto parentPerClusterMemShapeOffsets = parentDistributedInType.getPerClusterMemoryShapeOffsets();

    // Check if the memory shapes are included in parent's memory shapes
    for (auto idx : irange(perClusterMemShapes.size())) {
        const auto currentMemShape = to_small_vector(perClusterMemShapes[idx]);
        const auto parentMemShape = to_small_vector(parentPerClusterMemShapes[idx]);
        const auto currentMemShapeOffset = to_small_vector(perClusterMemShapeOffsets[idx]);
        const auto parentMemShapeOffset = to_small_vector(parentPerClusterMemShapeOffsets[idx]);

        for (size_t dim = 0; dim < perClusterMemShapes.front().size(); dim++) {
            if (tilingScheme[dim] != 1) {
                if (currentMemShapeOffset[dim] < parentMemShapeOffset[dim] ||
                    currentMemShapeOffset[dim] + currentMemShape[dim] >
                            parentMemShapeOffset[dim] + parentMemShape[dim]) {
                    _log.trace("Memory shape {0} is not included in parent memory shape {1} at '{2}'", currentMemShape,
                               parentMemShape, origOp->getLoc());
                    return mlir::failure();
                }
            } else {
                if (currentMemShapeOffset[dim] != parentMemShapeOffset[dim] ||
                    currentMemShape[dim] != parentMemShape[dim]) {
                    _log.trace("Memory shape {0} is not included in parent memory shape {1} at '{2}'", currentMemShape,
                               parentMemShape, origOp->getLoc());
                    return mlir::failure();
                }
            }
        }
    }
    if (!fitIntoCMX(origOp, parentDistributedInType)) {
        _log.trace("Can not fit into cmx with new input type");
        return mlir::failure();
    }

    _log.trace("Update distributed type {0} to {1} at '{2}'", distributedInType, parentDistributedInType,
               origOp->getLoc());

    rewriter.startOpModification(inCopy);
    inCopy.getResult().setType(parentDistributedInType);
    rewriter.finalizeOpModification(inCopy);

    rewriter.startOpModification(origOp);
    input.setType(parentDistributedInType);
    rewriter.finalizeOpModification(origOp);

    return mlir::success();
}

bool DistributedInputTypeRewriter::fitIntoCMX(VPU::NCEOpInterface origOp, VPU::DistributedTensorType newInType) const {
    return llvm::TypeSwitch<mlir::Operation*, bool>(origOp.getOperation())
            .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEDepthConvolutionOp>([&](auto convOp) {
                auto filterType = convOp.getFilter().getType();
                auto outputType = convOp.getOutput().getType();
                return convOp.fitIntoCMX(newInType, filterType, outputType);
            })
            .Case<VPU::NCEInterpolateOp, VPU::NCEMatMulOp>([&](auto op) {
                auto filterType = op.getWeights().getType();
                auto outputType = op.getOutput().getType();
                return op.fitIntoCMX(newInType, filterType, outputType);
            })
            .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp, VPU::NCEPermuteOp>([&](auto op) {
                auto outputType = op.getOutput().getType();
                return op.fitIntoCMX(newInType, outputType);
            })
            .Case<VPU::NCEEltwiseOp>([&](auto eltwiseOp) {
                auto input2Type = eltwiseOp.getInput2().getType();
                auto outputType = eltwiseOp.getOutput().getType();
                return eltwiseOp.fitIntoCMX(newInType, input2Type, outputType);
            })
            .Default([&](mlir::Operation* op) {
                _log.trace("Unsupported op type at {0}", op->getLoc());
                return false;
            });
}

//
// HaloAssistedSliceOptimization
//
// When a VPU.Slice sits between two CopyOps, check whether the input distributed
// tensor's per-cluster data range covers the output distributed tensor's per-cluster data
// range (taking the Slice offset into account). If the input cannot cover the output only
// on the H or W dimension, make a minimum increase to the input's per-cluster memory offset and
// memory shape so that it covers the output on every cluster. This optimization leverages
// NCE ODU halo capability to reduce intermediate copies.
//
// Currently restricted to ops that share the same VF loop tile index
// (vf_loop_tile_index attribute). Cross-VF-tile and non-VF subgraphs are skipped.
// TODO: E#211950 — lift the VF-only restriction once the cross-VF-tile performance regression is root-caused and fixed.
//
//    CopyOp (parentCopy)                   CopyOp (parentCopy) [updated input type]
//          |                                      |
//        Slice                =>                Slice
//          |                                      |
//    CopyOp (userCopy)                     CopyOp (userCopy)
//
class HaloAssistedSliceOptimization final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    HaloAssistedSliceOptimization(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::SliceOp>(ctx), _log(log) {
        this->setDebugName("HaloAssistedSliceOptimization");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult HaloAssistedSliceOptimization::matchAndRewrite(VPU::SliceOp sliceOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), sliceOp->getName(), sliceOp.getLoc());

    // Check that the parent op is CopyOp
    auto parentCopy = sliceOp.getInput().getDefiningOp<VPU::CopyOp>();
    if (parentCopy == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Parent is not CopyOp at '{0}'", sliceOp.getLoc());
    }

    // Check that the slice has exactly one user and it is CopyOp
    if (!sliceOp.getResult().hasOneUse()) {
        return matchFailed(_log, rewriter, sliceOp, "Slice does not have exactly one user at '{0}'", sliceOp.getLoc());
    }
    auto userCopy = mlir::dyn_cast<VPU::CopyOp>(*sliceOp.getResult().getUsers().begin());
    if (userCopy == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "User is not CopyOp at '{0}'", sliceOp.getLoc());
    }

    // Temporary scope guard: apply halo-assisted optimization only inside one VF region tile.
    // Compare vf_loop_tile_index between parentCopy producer and all userCopy consumers.
    auto parentProducerOp = parentCopy.getInput().getDefiningOp();
    if (parentProducerOp == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Parent CopyOp input is not defined by an op at '{0}'",
                           sliceOp.getLoc());
    }

    auto parentVFLoopTileIndex = parentProducerOp->getAttr(VF_LOOP_TILE_INDEX_ATTR_NAME);
    if (parentVFLoopTileIndex == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Parent producer op does not have '{0}' attr at '{1}'",
                           VF_LOOP_TILE_INDEX_ATTR_NAME, parentProducerOp->getLoc());
    }

    if (userCopy.getResult().use_empty()) {
        return matchFailed(_log, rewriter, sliceOp, "userCopy has no users at '{0}'", userCopy.getLoc());
    }

    // All consumers must stay in the same VF tile as parentProducerOp.
    // If any consumer is outside the tile, skip to avoid cross-tile rewrites.
    for (auto* userOp : userCopy.getResult().getUsers()) {
        auto userVFLoopTileIndex = userOp->getAttr(VF_LOOP_TILE_INDEX_ATTR_NAME);
        if (userVFLoopTileIndex == nullptr) {
            return matchFailed(_log, rewriter, sliceOp, "userCopy user op does not have '{0}' attr at '{1}'",
                               VF_LOOP_TILE_INDEX_ATTR_NAME, userOp->getLoc());
        }

        if (userVFLoopTileIndex != parentVFLoopTileIndex) {
            return matchFailed(
                    _log, rewriter, sliceOp, "'{0}' mismatch between parent producer '{1}' and user '{2}' at '{3}'",
                    VF_LOOP_TILE_INDEX_ATTR_NAME, parentVFLoopTileIndex, userVFLoopTileIndex, sliceOp.getLoc());
        }
    }

    // Get the input of the parent CopyOp (the distributed input)
    auto inputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(parentCopy.getInput().getType());
    if (inputDistType == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Input of parent CopyOp is not DistributedTensorType at '{0}'",
                           sliceOp.getLoc());
    }

    // Get the output of the user CopyOp (the distributed output)
    auto outputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(userCopy.getOutput().getType());
    if (outputDistType == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Output of user CopyOp is not DistributedTensorType at '{0}'",
                           sliceOp.getLoc());
    }

    // Both must have explicit shapes and offsets
    auto inputDist = inputDistType.getDistribution();
    auto outputDist = outputDistType.getDistribution();
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(inputDist) ||
        !VPU::isDistributedAttrWithExplicitShapesAndOffsets(outputDist)) {
        return matchFailed(_log, rewriter, sliceOp,
                           "Input or output distribution does not have explicit shapes and offsets at '{0}'",
                           sliceOp.getLoc());
    }

    // DUPLICATED output means every cluster holds the full slice, so there is no per-cluster
    // locality to exploit. Expanding the input per-cluster memory to cover the destination
    // range would bloat every cluster's footprint to nearly the whole tensor.
    // (SEGMENTED|DUPLICATED is a SOK output mode and does not appear as an NCE activation
    // input in this pattern, but the check handles it defensively as well.)
    const auto outputMode = outputDist.getMode().getValue();
    if (bitEnumContainsAny(outputMode, VPU::DistributionMode::DUPLICATED)) {
        return matchFailed(_log, rewriter, sliceOp,
                           "Output distributed type has DUPLICATED mode at '{0}'; "
                           "halo-assisted slice optimization is not applicable",
                           sliceOp.getLoc());
    }

    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    auto inputPerClusterMemShapes = inputDistType.getPerClusterMemoryShapes();
    auto inputPerClusterMemOffsets = inputDistType.getPerClusterMemoryShapeOffsets();
    auto outputPerClusterMemShapes = outputDistType.getPerClusterMemoryShapes();
    auto outputPerClusterMemOffsets = outputDistType.getPerClusterMemoryShapeOffsets();

    const auto numClusters = inputPerClusterMemShapes.size();
    if (numClusters != outputPerClusterMemShapes.size()) {
        return matchFailed(_log, rewriter, sliceOp, "Cluster count mismatch at '{0}'", sliceOp.getLoc());
    }

    const auto rank = inputDistType.getShape().size();
    if (rank != 4) {
        return matchFailed(_log, rewriter, sliceOp, "Expected 4D tensor, got rank {0} at '{1}'", rank,
                           sliceOp.getLoc());
    }

    const auto hDim = Dims4D::Act::H.ind();
    const auto wDim = Dims4D::Act::W.ind();

    // Verify that the slice output range (mapped back to the producer's coordinate space via
    // sliceOffsets) actually overlaps with every cluster's compute range on H and W.
    // If a cluster has no overlap with the slice on an expandable dimension, that cluster
    // contributes nothing to this slice. Expanding its memory to cover the slice range would
    // be wasteful and is the root cause of the inflated memory shapes observed at the OVERLAPPED output.
    auto inputPerClusterComputeShapes = inputDistType.getPerClusterComputeShapes();
    auto inputPerClusterComputeOffsets = inputDistType.getPerClusterComputeShapeOffsets();
    for (auto clusterIdx : irange(numClusters)) {
        const auto inComputeOffset = to_small_vector(inputPerClusterComputeOffsets[clusterIdx]);
        const auto inComputeShape = to_small_vector(inputPerClusterComputeShapes[clusterIdx]);
        const auto outMemOffset = to_small_vector(outputPerClusterMemOffsets[clusterIdx]);
        const auto outMemShape = to_small_vector(outputPerClusterMemShapes[clusterIdx]);

        for (const auto expandDim : {hDim, wDim}) {
            const auto adjustedOutStart = outMemOffset[expandDim] + sliceOffsets[expandDim];
            const auto adjustedOutEnd = adjustedOutStart + outMemShape[expandDim];
            const auto inComputeStart = inComputeOffset[expandDim];
            const auto inComputeEnd = inComputeStart + inComputeShape[expandDim];

            if (adjustedOutEnd <= inComputeStart || adjustedOutStart >= inComputeEnd) {
                return matchFailed(_log, rewriter, sliceOp,
                                   "Slice output dim-{0} range [{1}, {2}) does not overlap with cluster {3} compute "
                                   "dim-{0} range [{4}, {5}) at '{6}'; skipping to avoid unnecessary memory expansion",
                                   expandDim, adjustedOutStart, adjustedOutEnd, clusterIdx, inComputeStart,
                                   inComputeEnd, sliceOp.getLoc());
            }
        }
    }

    bool needsAdjustment = false;
    bool nonHWDimMismatch = false;

    // First pass: detect whether adjustment is needed and whether it's only on H or W
    for (auto clusterIdx : irange(numClusters)) {
        const auto inMemOffset = to_small_vector(inputPerClusterMemOffsets[clusterIdx]);
        const auto inMemShape = to_small_vector(inputPerClusterMemShapes[clusterIdx]);
        const auto outMemOffset = to_small_vector(outputPerClusterMemOffsets[clusterIdx]);
        const auto outMemShape = to_small_vector(outputPerClusterMemShapes[clusterIdx]);

        for (size_t dim = 0; dim < inMemShape.size(); dim++) {
            // The output is in the sliced coordinate space. To compare with input,
            // we need to shift the output offset by the slice offset.
            const auto adjustedOutStart = outMemOffset[dim] + sliceOffsets[dim];
            const auto adjustedOutEnd = adjustedOutStart + outMemShape[dim];
            const auto inStart = inMemOffset[dim];
            const auto inEnd = inStart + inMemShape[dim];

            if (adjustedOutStart >= inStart && adjustedOutEnd <= inEnd) {
                continue;
            }

            // There's a coverage gap on this dimension
            if (dim == static_cast<size_t>(hDim) || dim == static_cast<size_t>(wDim)) {
                needsAdjustment = true;
            } else {
                nonHWDimMismatch = true;
            }
        }
    }

    if (!needsAdjustment || nonHWDimMismatch) {
        return matchFailed(_log, rewriter, sliceOp, "No H/W-only adjustment needed or non-H/W mismatch found at '{0}'",
                           sliceOp.getLoc());
    }

    // Only NCE ops can adjust output memory shape via ODU halo, so the producer must be an NCE op
    auto nceOp = parentCopy.getInput().getDefiningOp<VPU::NCEOpInterface>();
    if (nceOp == nullptr) {
        return matchFailed(_log, rewriter, sliceOp, "Producer of input distributed type is not an NCE op at '{0}'",
                           sliceOp.getLoc());
    }

    // Second pass: compute the minimum expansion on H for each cluster
    auto newMemShapes = SmallVector<SmallVector<int64_t>>();
    auto newMemOffsets = SmallVector<SmallVector<int64_t>>();
    newMemShapes.reserve(numClusters);
    newMemOffsets.reserve(numClusters);

    for (auto clusterIdx : irange(numClusters)) {
        auto inMemOffset = to_small_vector(inputPerClusterMemOffsets[clusterIdx]);
        auto inMemShape = to_small_vector(inputPerClusterMemShapes[clusterIdx]);
        const auto outMemOffset = to_small_vector(outputPerClusterMemOffsets[clusterIdx]);
        const auto outMemShape = to_small_vector(outputPerClusterMemShapes[clusterIdx]);

        for (const auto expandDim : {hDim, wDim}) {
            const auto adjustedOutStart = outMemOffset[expandDim] + sliceOffsets[expandDim];
            const auto adjustedOutEnd = adjustedOutStart + outMemShape[expandDim];
            auto inStart = inMemOffset[expandDim];
            auto inEnd = inStart + inMemShape[expandDim];

            // Expand the input range to cover the output range on this dimension
            if (adjustedOutStart < inStart) {
                inStart = adjustedOutStart;
            }
            if (adjustedOutEnd > inEnd) {
                inEnd = adjustedOutEnd;
            }

            inMemOffset[expandDim] = inStart;
            inMemShape[expandDim] = inEnd - inStart;
        }

        newMemOffsets.push_back(inMemOffset);
        newMemShapes.push_back(inMemShape);
    }

    auto* ctx = sliceOp.getContext();
    const auto newMemShapesAttr = vpux::getIntArrayOfArray(ctx, newMemShapes);
    const auto newMemOffsetsAttr = vpux::getIntArrayOfArray(ctx, newMemOffsets);

    // Compute shapes also need to be read; keep them unchanged
    auto computeShapesAttr = inputDist.getComputeShapes();
    auto computeOffsetsAttr = inputDist.getComputeOffsets();

    auto newDistribution = VPU::DistributionInfoAttr::get(
            ctx, inputDist.getMode(), inputDist.getNumTiles(), inputDist.getKernel(), inputDist.getPads(),
            inputDist.getStrides(), inputDist.getNumClusters(), inputDist.getAlignment(),
            inputDist.getUniformDistributedSegments(), computeShapesAttr, computeOffsetsAttr, newMemShapesAttr,
            newMemOffsetsAttr, inputDist.getEqualMemoryAndComputeView(), inputDist.getMemoryNumTiles());

    // Compute the new overall shape for the input distributed type.
    // The overall shape stays the same; only the per-cluster distribution changes.
    auto newInputDistType =
            VPU::DistributedTensorType::get(ctx, inputDistType.getShape().raw(), inputDistType.getElementType(),
                                            inputDistType.getOrder(), inputDistType.getMemSpace(), newDistribution);

    _log.trace("Adjusting input distributed type from {0} to {1} at '{2}'", inputDistType, newInputDistType,
               sliceOp.getLoc());

    // Verify that other users of parentCopy that are CopyOps only have NCE consumers.
    // NCE ops can consume the adjusted type by adding offset to workload access,
    // but other ops (e.g. Shave kernel) cannot, so the adjustment could break other optimizations.
    for (auto* user : parentCopy.getResult().getUsers()) {
        if (user == sliceOp.getOperation()) {
            continue;
        }
        if (mlir::isa<VPU::ConcatOp>(user)) {
            return matchFailed(_log, rewriter, sliceOp,
                               "parentCopy has a Concat user at '{0}', adjustment could break CMX concat optimization",
                               user->getLoc());
        }
        auto copyUser = mlir::dyn_cast<VPU::CopyOp>(user);
        if (copyUser == nullptr) {
            continue;
        }
        for (auto* copyConsumer : copyUser.getResult().getUsers()) {
            if (!mlir::isa<VPU::NCEOpInterface>(copyConsumer)) {
                return matchFailed(
                        _log, rewriter, sliceOp,
                        "parentCopy has a CopyOp user with non-NCE consumer '{0}' at '{1}', cannot adjust type",
                        copyConsumer->getName(), copyConsumer->getLoc());
            }
        }
    }

    // Check fitIntoCMX for the producer NCE op with the adjusted output type
    auto fitsIntoCMX =
            llvm::TypeSwitch<mlir::Operation*, bool>(nceOp.getOperation())
                    .Case<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEDepthConvolutionOp>(
                            [&](auto convOp) {
                                return convOp.fitIntoCMX(convOp.getInput().getType(), convOp.getFilter().getType(),
                                                         newInputDistType);
                            })
                    .Case<VPU::NCEInterpolateOp, VPU::NCEMatMulOp>([&](auto op) {
                        return op.fitIntoCMX(op.getInput().getType(), op.getWeights().getType(), newInputDistType);
                    })
                    .Case<VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp, VPU::NCEPermuteOp>([&](auto op) {
                        return op.fitIntoCMX(op.getInput().getType(), newInputDistType);
                    })
                    .Case<VPU::NCEEltwiseOp>([&](auto eltwiseOp) {
                        return eltwiseOp.fitIntoCMX(eltwiseOp.getInput1().getType(), eltwiseOp.getInput2().getType(),
                                                    newInputDistType);
                    })
                    .Default([&](mlir::Operation* op) {
                        _log.trace("Unrecognized NCE op type '{0}' at '{1}', cannot verify fitIntoCMX", op->getName(),
                                   op->getLoc());
                        return false;
                    });
    if (!fitsIntoCMX) {
        return matchFailed(_log, rewriter, sliceOp, "Adjusted distributed type does not fit into CMX at '{0}'",
                           sliceOp.getLoc());
    }

    // If the value we modified is the output of an inplace NCEEltwiseOp,
    // check whether inplace must be removed and whether non-inplace still fits CMX.
    bool removeEltwiseInplaceAttr = false;
    if (auto eltwiseOp = parentCopy.getInput().getDefiningOp<VPU::NCEEltwiseOp>()) {
        if (eltwiseOp.getIsInplace().value_or(false)) {
            const auto input1NDType = mlir::cast<vpux::NDTypeInterface>(eltwiseOp.getInput1().getType());
            const auto input2NDType = mlir::cast<vpux::NDTypeInterface>(eltwiseOp.getInput2().getType());
            const auto newOutputNDType = mlir::cast<vpux::NDTypeInterface>(newInputDistType);

            auto isInputDistributionCompatibleWithOutput = [&](mlir::Value input) {
                auto inputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(input.getType());
                if (inputDistType == nullptr) {
                    return false;
                }

                auto inputDistribution = VPU::DistributionInfo::getClassFromAttr(inputDistType.getDistribution());
                auto outputDistribution = VPU::DistributionInfo::getClassFromAttr(newInputDistType.getDistribution());

                return mlir::succeeded(VPU::areDistributionsCompatible(
                        mlir::cast<vpux::NDTypeInterface>(inputDistType), inputDistribution,
                        mlir::cast<vpux::NDTypeInterface>(newInputDistType), outputDistribution,
                        /*allowDifferentPerClusterMemoryView=*/false));
            };

            const auto outputTotalSize = newOutputNDType.getTotalAllocSize().count();
            const auto requiresInplaceRemovalBySize = input1NDType.getTotalAllocSize().count() < outputTotalSize ||
                                                      input2NDType.getTotalAllocSize().count() < outputTotalSize;

            const auto input1Compatible = isInputDistributionCompatibleWithOutput(eltwiseOp.getInput1());
            const auto input2Compatible = isInputDistributionCompatibleWithOutput(eltwiseOp.getInput2());
            const auto requiresInplaceRemovalByDistribution = !input1Compatible && !input2Compatible;

            if (requiresInplaceRemovalBySize || requiresInplaceRemovalByDistribution) {
                removeEltwiseInplaceAttr = true;

                const auto origIsInplaceAttr = eltwiseOp.getIsInplaceAttr();

                rewriter.startOpModification(eltwiseOp);
                eltwiseOp.removeIsInplaceAttr();
                rewriter.finalizeOpModification(eltwiseOp);

                const auto fitIntoCMXWithoutInplace = eltwiseOp.fitIntoCMX(input1NDType, input2NDType, newOutputNDType);

                rewriter.startOpModification(eltwiseOp);
                if (origIsInplaceAttr != nullptr) {
                    eltwiseOp.setIsInplaceAttr(origIsInplaceAttr);
                }
                rewriter.finalizeOpModification(eltwiseOp);

                if (!fitIntoCMXWithoutInplace) {
                    return matchFailed(
                            _log, rewriter, sliceOp,
                            "Removing inplace for eltwise exceeds CMX; skip HaloAssistedSliceOptimization at '{0}'",
                            sliceOp.getLoc());
                }
            }
        }
    }

    // Update the parent CopyOp's input type
    rewriter.startOpModification(parentCopy);
    parentCopy.getInput().setType(newInputDistType);
    rewriter.finalizeOpModification(parentCopy);

    if (removeEltwiseInplaceAttr) {
        auto eltwiseOp = parentCopy.getInput().getDefiningOp<VPU::NCEEltwiseOp>();
        rewriter.startOpModification(eltwiseOp);
        eltwiseOp.removeIsInplaceAttr();
        rewriter.finalizeOpModification(eltwiseOp);
    }

    return mlir::success();
}

//
// DynamicDequantWeightsTypeRewriter
//
class DynamicDequantWeightsTypeRewriter final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    DynamicDequantWeightsTypeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        this->setDebugName("DynamicDequantWeightsTypeRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

    VPU::DistributedTensorType createAlignedType(VPU::DistributedTensorType baseType, ArrayRef<int64_t> numTiles,
                                                 ArrayRef<int64_t> alignment, int64_t numClusters,
                                                 bool uniformDistributedSegments) const;
    std::optional<int64_t> getMemPermuteInputAxis(VPU::MemPermuteOp memPermuteOp, int64_t outputAxis) const;

private:
    Logger _log;
};

VPU::DistributedTensorType DynamicDequantWeightsTypeRewriter::createAlignedType(VPU::DistributedTensorType baseType,
                                                                                ArrayRef<int64_t> numTiles,
                                                                                ArrayRef<int64_t> alignment,
                                                                                int64_t numClusters,
                                                                                bool uniformDistributedSegments) const {
    const auto distribution =
            VPU::getNonOverlappedDistributedNative(baseType.getShape(), VPU::DistributionMode::SEGMENTED, numTiles,
                                                   numClusters, alignment, uniformDistributedSegments);
    const auto distributionAttr = VPU::DistributionInfo::getAttrFromClass(baseType.getContext(), distribution);

    return mlir::cast<VPU::DistributedTensorType>(
            baseType.changeShapeForExplicitDistribution(baseType.getShape(), distributionAttr));
}

std::optional<int64_t> DynamicDequantWeightsTypeRewriter::getMemPermuteInputAxis(VPU::MemPermuteOp memPermuteOp,
                                                                                 int64_t outputAxis) const {
    const auto memPerm = memPermuteOp.getMemPerm();
    if (outputAxis >= static_cast<int64_t>(memPerm.getNumResults())) {
        return std::nullopt;
    }

    const auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(memPerm.getResult(outputAxis));
    if (dimExpr == nullptr) {
        return std::nullopt;
    }

    return dimExpr.getPosition();
}

mlir::LogicalResult DynamicDequantWeightsTypeRewriter::matchAndRewrite(VPU::NCEConvolutionOp convOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), convOp->getName(), convOp.getLoc());

    auto filterCopy = convOp.getFilter().getDefiningOp<VPU::CopyOp>();
    if (filterCopy == nullptr) {
        return matchFailed(_log, rewriter, convOp, "Filter is not from CopyOp at '{0}'", convOp.getLoc());
    }

    auto filterDistType = mlir::dyn_cast<VPU::DistributedTensorType>(filterCopy.getResult().getType());
    if (filterDistType == nullptr) {
        return matchFailed(_log, rewriter, filterCopy, "Filter copy output is not distributed at '{0}'",
                           filterCopy.getLoc());
    }

    const auto filterDistribution = VPU::DistributionInfo::getClassFromAttr(filterDistType.getDistribution());
    if (filterDistribution.getDistributionMode() != VPU::DistributionMode::SEGMENTED) {
        return matchFailed(_log, rewriter, filterCopy, "Filter distribution is not SEGMENTED at '{0}'",
                           filterCopy.getLoc());
    }

    const auto filterNumTiles = SmallVector<int64_t>(filterDistribution.getNumTiles());
    const auto filterAlignment = SmallVector<int64_t>(filterDistribution.getAlignment());
    if (filterNumTiles.empty() || filterAlignment.empty()) {
        return matchFailed(_log, rewriter, filterCopy, "Filter distribution has no tiling or alignment at '{0}'",
                           filterCopy.getLoc());
    }

    const auto filterAxis = VPU::getDistributedTilingAxis(filterNumTiles);
    if (filterAxis < 0 || filterAxis >= static_cast<int64_t>(filterAlignment.size())) {
        return matchFailed(_log, rewriter, filterCopy, "Filter tiling axis is invalid at '{0}'", filterCopy.getLoc());
    }

    auto permuteCast = filterCopy.getInput().getDefiningOp<VPU::PermuteCastOp>();
    if (permuteCast == nullptr) {
        return matchFailed(_log, rewriter, filterCopy, "Filter copy input is not PermuteCastOp at '{0}'",
                           filterCopy.getLoc());
    }

    auto affineReshape = permuteCast.getInput().getDefiningOp<VPU::AffineReshapeOp>();
    if (affineReshape == nullptr) {
        return matchFailed(_log, rewriter, permuteCast, "PermuteCast input is not AffineReshapeOp at '{0}'",
                           permuteCast.getLoc());
    }

    auto dynamicDequantToDenseCopy = affineReshape.getInput().getDefiningOp<VPU::CopyOp>();
    if (dynamicDequantToDenseCopy == nullptr) {
        return matchFailed(_log, rewriter, affineReshape, "AffineReshape input is not CopyOp at '{0}'",
                           affineReshape.getLoc());
    }

    auto dynamicDequant = dynamicDequantToDenseCopy.getInput().getDefiningOp<VPU::DynamicDequantizeOp>();
    if (dynamicDequant == nullptr) {
        return matchFailed(_log, rewriter, dynamicDequantToDenseCopy,
                           "AffineReshape input copy source is not DynamicDequantizeOp at '{0}'",
                           dynamicDequantToDenseCopy.getLoc());
    }

    auto dequantInputCopy = dynamicDequant.getInput().getDefiningOp<VPU::CopyOp>();
    auto memPermuteToDenseCopy =
            dequantInputCopy != nullptr ? dequantInputCopy.getInput().getDefiningOp<VPU::CopyOp>() : nullptr;
    auto memPermute = memPermuteToDenseCopy != nullptr
                              ? memPermuteToDenseCopy.getInput().getDefiningOp<VPU::MemPermuteOp>()
                              : nullptr;
    if (dequantInputCopy == nullptr || memPermuteToDenseCopy == nullptr || memPermute == nullptr) {
        return matchFailed(_log, rewriter, dynamicDequant,
                           "DynamicDequantize input is not fed by MemPermute through CopyOps at '{0}'",
                           dynamicDequant.getLoc());
    }

    auto memPermuteInputCopy = memPermute.getInput().getDefiningOp<VPU::CopyOp>();
    if (memPermuteInputCopy == nullptr) {
        return matchFailed(_log, rewriter, memPermute, "MemPermute input is not CopyOp at '{0}'", memPermute.getLoc());
    }

    auto dequantInputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(dynamicDequant.getInput().getType());
    auto dequantScaleDistType = mlir::dyn_cast<VPU::DistributedTensorType>(dynamicDequant.getScale().getType());

    auto dequantOutputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(dynamicDequant.getOutput().getType());
    auto memPermuteInputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(memPermute.getInput().getType());
    auto memPermuteOutputDistType = mlir::dyn_cast<VPU::DistributedTensorType>(memPermute.getOutput().getType());

    const auto isSegmentedDistType = [](VPU::DistributedTensorType type) {
        return type != nullptr && type.getDistribution().getMode().getValue() == VPU::DistributionMode::SEGMENTED;
    };

    if (!isSegmentedDistType(dequantInputDistType) || !isSegmentedDistType(dequantScaleDistType) ||
        !isSegmentedDistType(dequantOutputDistType) || !isSegmentedDistType(memPermuteInputDistType) ||
        !isSegmentedDistType(memPermuteOutputDistType)) {
        return matchFailed(_log, rewriter, dynamicDequant,
                           "Expected SEGMENTED distributed types in weights chain at '{0}'", dynamicDequant.getLoc());
    }
    VPU::DistributedTensorType dequantZeroPointDistType = nullptr;
    if (dynamicDequant.getZp() != nullptr) {
        dequantZeroPointDistType = mlir::dyn_cast<VPU::DistributedTensorType>(dynamicDequant.getZp().getType());
        if (!isSegmentedDistType(dequantZeroPointDistType)) {
            return matchFailed(_log, rewriter, dynamicDequant,
                               "Expected SEGMENTED distributed type for DynamicDequantize zero point at '{0}'",
                               dynamicDequant.getLoc());
        }
    }

    const auto permuteCastInputNumTiles = permuteCast.backInferTilingStrategy(filterNumTiles);
    const auto dequantNumTiles = affineReshape.backInferTilingStrategy(permuteCastInputNumTiles);
    const auto dequantAxis = VPU::getDistributedTilingAxis(dequantNumTiles);
    if (dequantAxis < 0 || dequantAxis >= static_cast<int64_t>(dequantNumTiles.size())) {
        return matchFailed(_log, rewriter, dynamicDequant, "Cannot infer DynamicDequantize tiling axis at '{0}'",
                           dynamicDequant.getLoc());
    }

    SmallVector<int64_t> dequantAlignment(dequantNumTiles.size(), 1);
    dequantAlignment[dequantAxis] = filterAlignment[filterAxis];

    SmallVector<int64_t> memPermuteInputNumTiles(dequantNumTiles.size(), 1);
    SmallVector<int64_t> memPermuteInputAlignment(dequantNumTiles.size(), 1);
    const auto memPermuteInputAxis = getMemPermuteInputAxis(memPermute, dequantAxis);
    if (!memPermuteInputAxis.has_value() ||
        memPermuteInputAxis.value() >= static_cast<int64_t>(dequantNumTiles.size())) {
        return matchFailed(_log, rewriter, memPermute, "Cannot infer MemPermute input tiling axis at '{0}'",
                           memPermute.getLoc());
    }
    memPermuteInputNumTiles[memPermuteInputAxis.value()] = dequantNumTiles[dequantAxis];
    memPermuteInputAlignment[memPermuteInputAxis.value()] = dequantAlignment[dequantAxis];

    const auto newDequantInputType =
            createAlignedType(dequantInputDistType, dequantNumTiles, dequantAlignment,
                              filterDistribution.getNumClusters(), filterDistribution.hasUniformDistributedSegments());
    const auto newDequantScaleType =
            createAlignedType(dequantScaleDistType, dequantNumTiles, dequantAlignment,
                              filterDistribution.getNumClusters(), filterDistribution.hasUniformDistributedSegments());
    const auto newDequantOutputType =
            createAlignedType(dequantOutputDistType, dequantNumTiles, dequantAlignment,
                              filterDistribution.getNumClusters(), filterDistribution.hasUniformDistributedSegments());
    VPU::DistributedTensorType newDequantZeroPointType = nullptr;
    if (dequantZeroPointDistType != nullptr) {
        newDequantZeroPointType = createAlignedType(dequantZeroPointDistType, dequantNumTiles, dequantAlignment,
                                                    filterDistribution.getNumClusters(),
                                                    filterDistribution.hasUniformDistributedSegments());
    }
    const auto newMemPermuteInputType =
            createAlignedType(memPermuteInputDistType, memPermuteInputNumTiles, memPermuteInputAlignment,
                              filterDistribution.getNumClusters(), filterDistribution.hasUniformDistributedSegments());
    const auto newMemPermuteOutputType =
            createAlignedType(memPermuteOutputDistType, dequantNumTiles, dequantAlignment,
                              filterDistribution.getNumClusters(), filterDistribution.hasUniformDistributedSegments());
    const auto affineReshapeOutputOrder =
            mlir::cast<vpux::NDTypeInterface>(affineReshape.getOutput().getType()).getDimsOrder();
    const auto newAffineReshapeOutputType =
            mlir::cast<VPU::DistributedTensorType>(filterDistType.changeDimsOrder(affineReshapeOutputOrder));
    const auto newPermuteCastOutputType = filterDistType;

    if (dequantInputDistType == newDequantInputType && dequantScaleDistType == newDequantScaleType &&
        dequantOutputDistType == newDequantOutputType && memPermuteInputDistType == newMemPermuteInputType &&
        (dequantZeroPointDistType == nullptr || dequantZeroPointDistType == newDequantZeroPointType) &&
        memPermuteOutputDistType == newMemPermuteOutputType &&
        dynamicDequantToDenseCopy.getResult().getType() == newDequantOutputType &&
        affineReshape.getOutput().getType() == newAffineReshapeOutputType &&
        permuteCast.getOutput().getType() == newPermuteCastOutputType) {
        return matchFailed(_log, rewriter, dynamicDequant, "Distributed types are already aligned at '{0}'",
                           dynamicDequant.getLoc());
    }

    auto dequantScaleCopy = dynamicDequant.getScale().getDefiningOp<VPU::CopyOp>();
    if (dequantScaleCopy == nullptr) {
        return matchFailed(_log, rewriter, dynamicDequant, "DynamicDequantize scale is not from CopyOp at '{0}'",
                           dynamicDequant.getLoc());
    }

    VPU::CopyOp dequantZeroPointCopy = nullptr;
    if (dynamicDequant.getZp() != nullptr) {
        dequantZeroPointCopy = dynamicDequant.getZp().getDefiningOp<VPU::CopyOp>();
        if (dequantZeroPointCopy == nullptr) {
            return matchFailed(_log, rewriter, dynamicDequant,
                               "DynamicDequantize zero point is not from CopyOp at '{0}'", dynamicDequant.getLoc());
        }
    }

    if (!memPermute.fitIntoCMX(
                SmallVector<vpux::NDTypeInterface>{mlir::cast<vpux::NDTypeInterface>(newMemPermuteInputType),
                                                   mlir::cast<vpux::NDTypeInterface>(newMemPermuteOutputType)})) {
        return matchFailed(_log, rewriter, memPermute, "Aligned MemPermute types do not fit into CMX at '{0}'",
                           memPermute.getLoc());
    }
    SmallVector<vpux::NDTypeInterface> dequantBuffers{mlir::cast<vpux::NDTypeInterface>(newDequantInputType),
                                                      mlir::cast<vpux::NDTypeInterface>(newDequantScaleType),
                                                      mlir::cast<vpux::NDTypeInterface>(newDequantOutputType)};
    if (dequantZeroPointDistType != nullptr) {
        dequantBuffers.push_back(mlir::cast<vpux::NDTypeInterface>(newDequantZeroPointType));
    }
    if (!dynamicDequant.fitIntoCMX(dequantBuffers)) {
        return matchFailed(_log, rewriter, dynamicDequant,
                           "Aligned DynamicDequantize types do not fit into CMX at '{0}'", dynamicDequant.getLoc());
    }

    _log.trace("Align dynamic dequantized weights distribution around '{0}' using filter distribution '{1}'",
               convOp.getLoc(), filterDistType);

    rewriter.startOpModification(memPermuteInputCopy);
    memPermuteInputCopy.getResult().setType(newMemPermuteInputType);
    rewriter.finalizeOpModification(memPermuteInputCopy);

    rewriter.startOpModification(memPermute);
    memPermute.getOutput().setType(newMemPermuteOutputType);
    rewriter.finalizeOpModification(memPermute);

    rewriter.startOpModification(dequantInputCopy);
    dequantInputCopy.getResult().setType(newDequantInputType);
    rewriter.finalizeOpModification(dequantInputCopy);

    rewriter.startOpModification(dequantScaleCopy);
    dequantScaleCopy.getResult().setType(newDequantScaleType);
    rewriter.finalizeOpModification(dequantScaleCopy);

    if (dequantZeroPointCopy != nullptr) {
        rewriter.startOpModification(dequantZeroPointCopy);
        dequantZeroPointCopy.getResult().setType(newDequantZeroPointType);
        rewriter.finalizeOpModification(dequantZeroPointCopy);
    }

    rewriter.startOpModification(dynamicDequant);
    dynamicDequant.getOutput().setType(newDequantOutputType);
    rewriter.finalizeOpModification(dynamicDequant);

    rewriter.startOpModification(dynamicDequantToDenseCopy);
    dynamicDequantToDenseCopy.setOutMemSpaceAttr(newDequantOutputType.getMemSpace());
    dynamicDequantToDenseCopy.getResult().setType(newDequantOutputType);
    rewriter.finalizeOpModification(dynamicDequantToDenseCopy);

    rewriter.startOpModification(affineReshape);
    affineReshape.getOutput().setType(newAffineReshapeOutputType);
    rewriter.finalizeOpModification(affineReshape);

    rewriter.startOpModification(permuteCast);
    permuteCast.getOutput().setType(newPermuteCastOutputType);
    rewriter.finalizeOpModification(permuteCast);

    return mlir::success();
}

//
// AdjustDistributedTensorAroundOpsPass
//

class AdjustDistributedTensorAroundOpsPass final :
        public VPU::impl::AdjustDistributedTensorAroundOpsBase<AdjustDistributedTensorAroundOpsPass> {
public:
    explicit AdjustDistributedTensorAroundOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
};

void AdjustDistributedTensorAroundOpsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    // MTL lacks halo hardware support; LNL has halo hardware but exhibits performance scaling
    // issues that cause full-model regression when this optimization is applied.
    // TODO: E#211948 — remove guard once regression is root-caused and fixed.
    if (VPU::isHaloAssistedSliceOptimizationSupported(func)) {
        mlir::RewritePatternSet slicePatterns(&ctx);
        slicePatterns.add<HaloAssistedSliceOptimization>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(slicePatterns));
    }

    mlir::RewritePatternSet dequantWeightPatterns(&ctx);
    dequantWeightPatterns.add<DynamicDequantWeightsTypeRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(dequantWeightPatterns));

    // DistributedInputTypeRewriter runs after HaloAssistedSliceOptimization so that the
    // per-cluster distributed types produced by the halo rewrite are already settled before
    // the input-widening rewriter runs. Running them in the opposite order could widen the
    // NCE input type first and then miss HaloAssistedSliceOptimization opportunities created
    // by that widening.
    mlir::RewritePatternSet inputPatterns(&ctx);
    inputPatterns.add<DistributedInputTypeRewriter>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(inputPatterns));
}

}  // namespace

//
// createAdjustDistributedTensorAroundOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustDistributedTensorAroundOpsPass(Logger log) {
    return std::make_unique<AdjustDistributedTensorAroundOpsPass>(log);
}
