//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_multicluster_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>

using namespace vpux;
using namespace VPU;

namespace vpux::VPU {

VPU::MultiClusterStrategy getMulticlusteringStrategy(mlir::Operation* computeOp, const int64_t outputTilingAxis) {
    if (outputTilingAxis == Dims4D::Act::H.ind()) {
        return mlir::isa<VPU::InterpolateOp>(computeOp) ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                                                        : VPU::MultiClusterStrategy::SplitOverHeight;
    }

    if (outputTilingAxis == Dims4D::Act::W.ind()) {
        return VPU::MultiClusterStrategy::SplitOverWidth;
    }

    if (outputTilingAxis == Dims4D::Act::C.ind()) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if (outputTilingAxis == Dims4D::Act::N.ind()) {
        return VPU::MultiClusterStrategy::SplitOverBatch;
    }

    VPUX_THROW("Unsupported tiling axis {0} for multiclustering", outputTilingAxis);
}

SmallVector<int64_t> resolveShapeFromDistribution(ArrayRef<int64_t> origShape,
                                                  const VPU::DistributionInfo& distribution) {
    auto resolved = SmallVector<int64_t>(origShape);
    const auto& computeShapes = distribution.getComputeShapes();
    const auto& computeOffsets = distribution.getComputeOffsets();
    if (computeShapes.empty()) {
        return resolved;
    }
    for (size_t dim = 0; dim < resolved.size(); ++dim) {
        if (resolved[dim] != mlir::ShapedType::kDynamic) {
            continue;
        }
        int64_t maxExtent = 0;
        for (size_t c = 0; c < computeShapes.size(); ++c) {
            VPUX_THROW_WHEN(dim >= computeShapes[c].size() || dim >= computeOffsets[c].size(),
                            "Rank mismatch in resolveShapeFromDistribution: origShape rank {0} exceeds "
                            "per-cluster shape rank {1} for cluster {2}",
                            resolved.size(), computeShapes[c].size(), c);
            maxExtent = std::max(maxExtent, computeOffsets[c][dim] + computeShapes[c][dim]);
        }
        resolved[dim] = maxExtent;
    }
    return resolved;
}

SmallVector<int64_t> computeNumTilesForDistribution(VPU::OpChainAnalysis& analysis, mlir::OpResult output) {
    auto parallelInsertSlice = findParallelInsertSlice(output);
    VPUX_THROW_WHEN(parallelInsertSlice == nullptr, "Cannot find parallel_insert_slice for result {0} of op at {1}",
                    output.getResultNumber(), output.getOwner()->getLoc());
    auto forallOp = parallelInsertSlice->getParentOfType<mlir::scf::ForallOp>();
    VPUX_THROW_WHEN(forallOp == nullptr, "Expected scf.forall parent for multiclustered op {0}", parallelInsertSlice);

    ValueRangeMap emptyMap;
    auto mixedOffsets = parallelInsertSlice.getMixedOffsets();
    auto numTiles = SmallVector<int64_t>(mixedOffsets.size(), 1);

    for (auto idx : irange(mixedOffsets.size())) {
        if (!parallelInsertSlice.isDynamicOffset(idx)) {
            continue;
        }

        if (forallOp && !VPU::isDependentOnForallIv(mixedOffsets[idx], forallOp)) {
            continue;
        }

        auto offsetsFolded =
                analysis.getOpFoldResultValue(mixedOffsets[idx], emptyMap, OpChainAnalysis::MODE::ALL_VALUES);
        if (!offsetsFolded) {
            continue;
        }
        numTiles[idx] = static_cast<int64_t>(offsetsFolded.value().size());
    }

    return numTiles;
}

void fillInDistribution(VPU::OpChainAnalysis& analysis, mlir::OffsetSizeAndStrideOpInterface offsetSizeOp,
                        NDTypeInterface type, const int64_t numClusters, VPU::DistributionInfo& distribution) {
    // Used below to skip dimensions whose dynamic offset/size don't actually vary per cluster.
    auto forallOp = offsetSizeOp->getParentOfType<mlir::scf::ForallOp>();
    VPUX_THROW_WHEN(forallOp == nullptr, "Expected scf.forall parent for multiclustered op {0}", offsetSizeOp);

    auto offsets = SmallVector<SmallVector<int64_t>>(numClusters,
                                                     SmallVector<int64_t>(type.getRank(), static_cast<int64_t>(0)));

    const auto shape = SmallVector<int64_t>(type.getShape().raw());
    auto sizes = SmallVector<SmallVector<int64_t>>(numClusters, shape);

    const auto isValidVecSize = [&](ArrayRef<int64_t> arr) {
        return arr.size() == 1 || arr.size() == static_cast<size_t>(numClusters);
    };

    ValueRangeMap emptyMap;
    auto mixedOffsets = offsetSizeOp.getMixedOffsets();
    auto mixedSizes = offsetSizeOp.getMixedSizes();
    auto numTiles = SmallVector<int64_t>(type.getRank(), 1);
    for (auto idx : irange(mixedOffsets.size())) {
        if (!offsetSizeOp.isDynamicOffset(idx) && !offsetSizeOp.isDynamicSize(idx)) {
            continue;
        }

        // Skip if neither offset nor size at this dim depends on the forall IVs —
        // the value is uniform across all clusters and doesn't contribute to distribution.
        if (forallOp && !VPU::isDependentOnForallIv(mixedOffsets[idx], forallOp) &&
            !VPU::isDependentOnForallIv(mixedSizes[idx], forallOp)) {
            continue;
        }

        auto offsetsFolded =
                analysis.getOpFoldResultValue(mixedOffsets[idx], emptyMap, OpChainAnalysis::MODE::ALL_VALUES);
        auto sizesFolded = analysis.getOpFoldResultValue(mixedSizes[idx], emptyMap, OpChainAnalysis::MODE::ALL_VALUES);
        VPUX_THROW_WHEN(!offsetsFolded || !sizesFolded, "Failed to extract offsets and sizes for distribution.");

        const auto invalidOffsetsSizes = !isValidVecSize(offsetsFolded.value()) || !isValidVecSize(sizesFolded.value());
        VPUX_THROW_WHEN(
                invalidOffsetsSizes,
                "Invalid number of offsets or sizes for distribution. Expected 1 or numClusters ({0}), got {1} and {2}",
                numClusters, offsetsFolded.value().size(), sizesFolded.value().size());

        for (int64_t clusterIdx = 0; clusterIdx < numClusters; ++clusterIdx) {
            offsets[clusterIdx][idx] =
                    offsetsFolded.value().size() == 1 ? offsetsFolded.value()[0] : offsetsFolded.value()[clusterIdx];
            sizes[clusterIdx][idx] =
                    sizesFolded.value().size() == 1 ? sizesFolded.value()[0] : sizesFolded.value()[clusterIdx];
        }

        numTiles[idx] = static_cast<int64_t>(offsetsFolded.value().size());
    }

    distribution.setComputeShapes(sizes);
    distribution.setComputeOffsets(offsets);
    distribution.setMemoryShapes(sizes);
    distribution.setMemoryOffsets(offsets);
    distribution.setNumClusters(numClusters);
    distribution.setNumTiles(numTiles);
}

mlir::tensor::ParallelInsertSliceOp findParallelInsertSlice(mlir::OpResult output) {
    if (!output.hasOneUse()) {
        return nullptr;
    }

    mlir::Operation* current = *(output.user_begin());
    while (!mlir::isa_and_present<mlir::tensor::ParallelInsertSliceOp>(current)) {
        if (!mlir::isa_and_present<mlir::tensor::CastOp, VPU::CopyOp>(current)) {
            return nullptr;
        }

        if (current->getNumResults() != 1) {
            return nullptr;
        }

        mlir::Value currentOutput = current->getResult(0);
        if (!currentOutput.hasOneUse()) {
            return nullptr;
        }
        current = *(currentOutput.user_begin());
    }

    return mlir::dyn_cast_if_present<mlir::tensor::ParallelInsertSliceOp>(current);
}

VPU::MultiClusterStrategy inferMulticlusterStrategy(VPU::OpChainAnalysis& analysis, mlir::Operation* computeOp,
                                                    mlir::OpResult output) {
    const auto numTiles = computeNumTilesForDistribution(analysis, output);
    const auto tilingAxes = VPU::getNonOneDimInds(numTiles);
    VPUX_THROW_WHEN(tilingAxes.size() != 1, "Currently only supporting strategies with single multiclustering axis");

    const auto strategy = getMulticlusteringStrategy(computeOp, tilingAxes.front());
    return strategy;
}

VPU::DistributedTensorType getOutputDistributedType(VPU::OpChainAnalysis& analysis, mlir::Operation* computeOp,
                                                    mlir::OpResult output, const VPU::MultiClusterStrategy& strategy,
                                                    mlir::IntegerAttr numClustersAttr, mlir::MLIRContext* ctx) {
    const auto memSpaceCMX = IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));

    auto parallelInsertSlice = findParallelInsertSlice(output);
    VPUX_THROW_WHEN(parallelInsertSlice == nullptr, "Cannot find parallel_insert_slice for result {0} of op at {1}",
                    output.getResultNumber(), computeOp->getLoc());

    auto outputType = mlir::cast<NDTypeInterface>(parallelInsertSlice.getDestType());

    VPU::DistributionInfo distribution;
    fillInDistribution(analysis, parallelInsertSlice, outputType, numClustersAttr.getInt(), distribution);

    // For SOK + NCEOpInterface, getOutputTensorDistributionMode will return SEGMENTED|DUPLICATED due to the presence
    // of only one op inside scf.forall. However, broadcasting is not supported until E#193460 is done, so the correct
    // mode is SEGMENTED, to fit the per cluster offsets/sizes computed above.
    const auto mode = strategy == VPU::MultiClusterStrategy::SplitOverKernel && mlir::isa<NCEOpInterface>(computeOp)
                              ? VPU::DistributionMode::SEGMENTED
                              : VPU::getOutputTensorDistributionMode(mlir::cast<VPU::ClusteredOpInterface>(computeOp),
                                                                     strategy, outputType);
    distribution.setDistributionMode(mode);

    // Resolve dynamic dims from compute_shapes/compute_offsets populated by fillInDistribution.
    // Placed after setDistributionMode so the distribution object is fully configured
    // before any further use.
    const auto resolvedShape = resolveShapeFromDistribution(outputType.getShape().raw(), distribution);

    if (VPU::bitEnumContainsAny(mode, VPU::DistributionMode::SEGMENTED)) {
        const auto alignment =
                vpux::getAlignment(computeOp, ShapeRef(distribution.getNumTiles()), ShapeRef(resolvedShape));

        const auto distributionAxis = VPU::getDistributedTilingAxis(distribution.getNumTiles());
        if (alignment[distributionAxis] != 1) {
            distribution.setAlignment(alignment);
        }
    }

    auto distributionAttr = VPU::DistributionInfo::getAttrFromClass(ctx, distribution);

    auto orderAttr = mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(ctx));
    return VPU::DistributedTensorType::get(ctx, resolvedShape, outputType.getElementType(), orderAttr, memSpaceCMX,
                                           distributionAttr);
}

}  // namespace vpux::VPU
