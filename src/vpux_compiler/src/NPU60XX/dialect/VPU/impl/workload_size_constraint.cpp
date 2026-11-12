//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/impl/workload_size_constraint.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

using namespace vpux;

// Errata E#18039459346
// For specific cases of DW operations with 1x1 kernel, non sparse input data,
// and low spatial height and width compute <= 16 values, we get wrong results
// unless we further split the task workloads.
bool VPU::arch60xx::WorkloadSizeConstraint::doesDWOperationNeedWorkloadSplit(mlir::Operation* op,
                                                                             ShapeRef inputShapeOverride) const {
    if (!isDepthwiseOp(op)) {
        return false;
    }
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(op);
    const auto kernelSize = nceOpInterface.getKernelSizeVal();
    const auto inputTensorType = mlir::cast<vpux::NDTypeInterface>(nceOpInterface->getOperand(0).getType());
    if (mlir::isa<vpux::VPU::SparseTensorType>(inputTensorType)) {
        return false;
    }

    if (kernelSize[Dims4D::Kernel::X.ind()] != 1 || kernelSize[Dims4D::Kernel::Y.ind()] != 1) {
        return false;
    }

    // Gather individual shapes across clusters. `inputShapeOverride`, when given, replaces the
    // op's own IR shape for the single-cluster case — needed by callers (e.g. SCF flows using
    // dynamic/bounded tensor types) where operand(0)'s IR type may still report
    // mlir::ShapedType::kDynamic for dims that are only resolved outside the IR.
    auto inputDistributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(inputTensorType);
    SmallVector<Shape> inputShapes;
    if (inputDistributedTensorType != nullptr) {
        inputShapes = inputDistributedTensorType.getPerClusterComputeShapes();
    } else {
        inputShapes = {inputShapeOverride.empty() ? Shape(inputTensorType.getShape()) : Shape(inputShapeOverride)};
    }
    // Errata E#18039459346
    constexpr auto nceDpuWorkloadSize = 16;

    // Treat dynamic H/W as potentially small (<=16) when evaluating the errata condition.
    // This stays conservative without forcing a split when the other spatial dim is known > 16.
    const auto isDimProblematic = [](int64_t dim) {
        return dim == mlir::ShapedType::kDynamic || dim <= nceDpuWorkloadSize;
    };
    const auto isProblematicShape = [&](ShapeRef shape) {
        return isDimProblematic(shape[Dims4D::Act::H]) && isDimProblematic(shape[Dims4D::Act::W]);
    };

    return llvm::any_of(inputShapes, isProblematicShape);
}

bool VPU::arch60xx::WorkloadSizeConstraint::checkDWOperationWorkloadLimit(mlir::Operation* op,
                                                                          const OutputTiling& tiles) const {
    if (!isDepthwiseOp(op)) {
        return false;
    }

    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(op);
    const auto kernelSize = nceOpInterface.getKernelSizeVal();
    if (kernelSize[Dims4D::Kernel::X.ind()] != 1 || kernelSize[Dims4D::Kernel::Y.ind()] != 1) {
        return false;
    }

    const auto inputTensorType = mlir::cast<vpux::NDTypeInterface>(nceOpInterface->getOperand(0).getType());
    if (mlir::isa<vpux::VPU::SparseTensorType>(inputTensorType)) {
        return false;
    }

    // Cache multi-cluster information outside the lambda to avoid redundant checks
    const bool hasMultiClusterStrategy = op->hasAttr(VPU::multiClusterStrategy);
    VPU::ClusteredOpInterface clusteredOp = nullptr;
    VPU::MultiClusterStrategy strategy = VPU::MultiClusterStrategy::Clustering;

    if (hasMultiClusterStrategy) {
        clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
        if (clusteredOp != nullptr) {
            strategy = mlir::cast<VPU::MultiClusterStrategyAttr>(op->getAttr(VPU::multiClusterStrategy)).getValue();
        }
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(nceOpInterface->getResult(0).getType());

    constexpr auto maxChannelSplit = 32;
    constexpr auto maxDimensionSize = 16;
    constexpr auto workloadMaxSize = 64;

    auto isProblematicShape = [&](auto tensorHeight, auto tensorWidth, auto tensorChannel) -> bool {
        return (tensorHeight <= maxDimensionSize && tensorWidth <= maxDimensionSize &&
                tensorChannel / maxChannelSplit >= workloadMaxSize);
    };

    auto hasProblematicWorkload = [&](const TileInfo& tile) -> bool {
        if (hasMultiClusterStrategy && clusteredOp != nullptr) {
            const auto outputTileType = outputType.extractDenseTile(tile.offsets, tile.shape);

            const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, tile.shape, strategy);

            const auto distributedType =
                    VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, strategy);

            const auto distributedNDType = mlir::cast<vpux::NDTypeInterface>(distributedType);
            SmallVector<Shape> perClusterShapes = {Shape(distributedNDType.getShape())};

            auto distributedTensorType = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(distributedType);
            if (distributedTensorType != nullptr) {
                perClusterShapes = distributedTensorType.getPerClusterComputeShapes();
            }

            for (const auto& clusterShape : perClusterShapes) {
                const auto tensorHeight = clusterShape[Dims4D::Act::H];
                const auto tensorWidth = clusterShape[Dims4D::Act::W];
                const auto tensorChannel = clusterShape[Dims4D::Act::C];

                if (isProblematicShape(tensorHeight, tensorWidth, tensorChannel)) {
                    return true;
                }
            }
            return false;
        }

        const auto tensorHeight = tile.shape[Dims4D::Act::H];
        const auto tensorWidth = tile.shape[Dims4D::Act::W];
        const auto tensorChannel = tile.shape[Dims4D::Act::C];

        return isProblematicShape(tensorHeight, tensorWidth, tensorChannel);
    };

    return llvm::any_of(tiles, hasProblematicWorkload);
}

// We need to limit the workload channels split to just {16, 32}, based on what's
// already deemed as supported
SmallVector<int64_t> VPU::arch60xx::WorkloadSizeConstraint::getChannelsSupportedBySmallSpatialComputeDwOp(
        ArrayRef<int64_t> workloadsChannels) const {
    auto supportedChannels = SmallVector<int64_t>({32, 16});
    llvm::erase_if(supportedChannels, [&](auto channel) {
        if (std::find(workloadsChannels.begin(), workloadsChannels.end(), channel) == workloadsChannels.end()) {
            return true;
        }
        return false;
    });
    return supportedChannels;
}
