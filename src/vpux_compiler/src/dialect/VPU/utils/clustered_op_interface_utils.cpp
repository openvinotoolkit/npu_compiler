//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

int64_t VPU::getNumTiles(mlir::Operation* op) {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    return tileOp.getCount();
}

bool VPU::isOperationSplitOverHeightCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = getNumTiles(op);
    const auto minimumOutputHeightForSOH = numTiles;

    auto isUniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp);

    const Shape outputShape(outputTile.shape.empty() ? getBoundedShape(clusteredOp->getResult(0)) : outputTile.shape);
    auto heightCompatibleCheck = [&](ShapeRef outputShape) {
        const auto OH = outputShape[Dims4D::Act::H];
        auto numClustersForSOH = VPU::getNumberOfClustersForSpatialDim(outputShape[Dims4D::Act::H], numTiles,
                                                                       isUniformDistributedSegments);
        // Each cluster should be used. When it is just with 3 or 2 clusters, there is an accuracy issue.
        // TODO: Find the root cause for this accuracy regression, E#41297
        auto isSOHCompatible = (OH >= minimumOutputHeightForSOH && numClustersForSOH == numTiles);
        return isSOHCompatible;
    };

    auto isSOHCompatible = heightCompatibleCheck(outputShape);
    if (!isSOHCompatible) {
        return false;
    }
    auto siblingsOpsAnalysis = SiblingOpsAnalysis(op);
    auto offset = ShapeRef(outputTile.offsets);
    if (!offset.empty()) {
        // Tiling has impact if:
        // - For NCE ops: the height axis is actually being tiled (i.e., more than 1 tile along H)
        // - For non-NCE ops: always assume tiling has impact
        auto tilingHasImpact = mlir::isa<VPU::NCEOpInterface>(op) ? (outputTile.axis[Dims4D::Act::H] > 1) : true;
        if (!tilingHasImpact) {
            return isSOHCompatible;
        }
        const auto numClusters = vpux::VPU::getOptimalNumClusters(clusteredOp, outputShape,
                                                                  clusteredOp.getMultiClusterStrategy().value());
        {
            const auto outputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getResult(0).getType());
            const auto outputTileType = outputType.extractDenseTile(offset, outputShape);
            auto distributions =
                    VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters, siblingsOpsAnalysis);
            auto distribution =
                    mlir::isa<VPU::SparseTensorType>(outputTileType)
                            ? distributions.at(mlir::cast<vpux::VPU::SparseTensorType>(outputTileType).getData())
                            : distributions.at(outputTileType);
            if (distribution.getMemoryShapes().empty()) {
                auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(outputShape, distribution);
                if (!optionalPerClusterMemoryShapes.has_value()) {
                    return false;
                }
            }
        }

        if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op)) {
            const auto inputTiles = tilingOp.backInferTileInfo(outputTile, vpux::Logger::global()).tiles;
            if (inputTiles.empty()) {
                return false;
            }
            const auto& inputTile = inputTiles[0];

            if (!heightCompatibleCheck(inputTile.shape)) {
                return false;
            }

            const auto inputType = mlir::cast<vpux::NDTypeInterface>(clusteredOp->getOperand(0).getType());
            const auto inputTileType = inputType.extractDenseTile(inputTile.offsets, inputTile.shape);

            auto distributions = VPU::getActivationDistributionAttrFromOp(
                    clusteredOp, clusteredOp->getOperand(0), inputTileType, numClusters, siblingsOpsAnalysis);
            auto distribution =
                    mlir::isa<VPU::SparseTensorType>(inputTileType)
                            ? distributions.at(mlir::cast<vpux::VPU::SparseTensorType>(inputTileType).getData())
                            : distributions.at(inputTileType);
            if (distribution.getMemoryShapes().empty()) {
                auto optionalPerClusterMemoryShapes =
                        VPU::getPerClusterMemoryShapes(inputTileType.getShape(), distribution);
                if (!optionalPerClusterMemoryShapes.has_value()) {
                    return false;
                }
            }
        }
    }

    return isSOHCompatible;
}

bool VPU::isOperationSplitOverWidthCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef /*offset*/,
                                              ShapeRef /*axis*/) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = getNumTiles(op);
    const auto minimumOutputWidthForSOW = numTiles;

    const auto arch = config::getArch(clusteredOp);
    if (outputShape == ShapeRef()) {
        outputShape = getBoundedShape(clusteredOp->getResult(0));
    }

    auto widthCompatibleCheck = [&](ShapeRef outputShape) {
        const auto OW = outputShape[Dims4D::Act::W];
        auto numClustersForSOW = getNumberOfClustersForSpatialDim(outputShape[Dims4D::Act::W], numTiles, true);
        // Each cluster should be used. When it is just with 3 or 2 clusters, there is an accuracy issue.
        // TODO: Find the root cause for this accuracy regression, E#41297
        auto isSOWCompatible = (OW >= minimumOutputWidthForSOW && numClustersForSOW == numTiles);
        return isSOWCompatible;
    };

    auto isSOWCompatible = widthCompatibleCheck(outputShape);
    if (arch >= config::ArchKind::NPU40XX) {
        // For NPU40XX+, W segmented output needs to have explicit halo regions defined.
        // Thus the applicability of SOW on the current operation is tightly dependent
        // if the consumer operations can be SOW themselves.
        // If that's not the case and not all consumers are SOW compatible, we can't represent
        // the output OVERLAP DistributedTensor in an correct manner.
        for (auto consumer : clusteredOp->getResult(0).getUsers()) {
            if (auto clusteredConsumer = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumer)) {
                isSOWCompatible &= widthCompatibleCheck(getBoundedShape(clusteredConsumer->getResult(0)));
            }
        }
    }

    return isSOWCompatible;
}

bool VPU::isOperationSplitOverKernelCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef /*offset*/,
                                               ShapeRef /*axis*/) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = getNumTiles(op);

    if (outputShape == ShapeRef()) {
        outputShape = getShape(clusteredOp->getResult(0));
    }
    const auto OC = outputShape[Dims4D::Act::C];

    if (mlir::isa<NCEOpInterface>(op) && VPU::canAutopadOutput(op)) {
        return false;
    }

    // Sparse Eltwise consuming SOK activations leads to the storage element size different than the number of input
    // channels, which is not a validated scenario
    if (mlir::isa<vpux::VPU::SparseTensorType>(clusteredOp->getResult(0).getType())) {
        const auto hasEltwiseUser = llvm::any_of(clusteredOp->getResult(0).getUsers(), [](mlir::Operation* userOp) {
            return mlir::isa<VPU::NCEEltwiseOp>(userOp);
        });
        if (hasEltwiseUser) {
            return false;
        }
    }
    // Channel alignment is specific for NCE DPU operations and CMX CONCAT
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());

    auto minChannelSize = (nceOp != nullptr) ? VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT * numTiles : numTiles;
    if (OC < minChannelSize) {
        return false;
    }

    if (nceOp == nullptr) {
        return true;
    }

    // SOK will split the weights over output channels. If the weights are sparse, it is necessary to make sure that
    // no split will have only sparse values inside, since that would lead to zero-sized weights
    auto weights = nceOp.getWeightsOperand();
    if (weights != nullptr && mlir::isa<vpux::VPU::SparseTensorType>(weights.getType())) {
        if (const auto sparsityCompression =
                    mlir::cast<vpux::VPU::SparseTensorType>(weights.getType()).getSparsityCompression()) {
            // Create a new type with the new number of output channels
            // If the element type is quantized per-axis, it is replaced with a per-tensor type to avoid the
            // incompatibility between the number of elements per axis and the number of scales & zero-points
            const auto origType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
            auto newShape = Shape(origType.getShape().raw());
            newShape[Dims4D::Filter::OC] = OC;
            auto elemType = origType.getElementType();
            if (auto qElemType = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(elemType)) {
                elemType = mlir::quant::QuantileQuantizedType::get(
                        qElemType.getFlags(), qElemType.getStorageType(), qElemType.getQuantileType(),
                        qElemType.getExpressedType(), qElemType.getQuantiles(), /*scale=*/1.0,
                        /*zeroPoint=*/0, qElemType.getStorageTypeMin(), qElemType.getStorageTypeMax());
            } else if (auto qElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
                elemType = mlir::quant::UniformQuantizedType::get(
                        qElemType.getFlags(), qElemType.getStorageType(), qElemType.getExpressedType(), /*scale=*/1.0,
                        /*zeroPoint=*/0, qElemType.getStorageTypeMin(), qElemType.getStorageTypeMax());
            }
            const auto newType = origType.changeShapeElemType(newShape, elemType);

            // Create a distributed type in order to determine the channel split over clusters
            const auto filterType = VPU::getDistributedFilterTypeFromOp(nceOp, newType, numTiles,
                                                                        VPU::MultiClusterStrategy::SplitOverKernel);
            const auto filterDistType =
                    mlir::cast<vpux::VPU::DistributedTensorType>(filterType.getDistributedTypes().front());
            const auto computeOffsets = filterDistType.getPerClusterComputeShapeOffsets();
            if (!computeOffsets.empty()) {
                int64_t startOC = computeOffsets[0][Dims4D::Filter::OC];
                for (size_t i = 1; i < computeOffsets.size(); ++i) {
                    const int64_t sizeOC = computeOffsets[i][Dims4D::Filter::OC] - startOC;
                    const auto numElems = sparsityCompression.getNumElemsInRange(startOC, sizeOC);
                    if (numElems == 0) {
                        return false;
                    }
                    startOC += sizeOC;
                }
                const auto remainingOC = OC - startOC;
                const auto numElems = sparsityCompression.getNumElemsInRange(startOC, remainingOC);
                if (numElems == 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool VPU::isOperationSplitOverBatchCompatible(mlir::Operation* op, ShapeRef outputShape) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    if (outputShape == ShapeRef()) {
        outputShape = getBoundedShape(clusteredOp->getResult(0));
    }

    // Currently, SOB supported with condition batch being less or equal to number tiles used
    const auto B = outputShape[Dims4D::Act::N];
    const auto numTiles = getNumTiles(op);

    return (B > 1) && (B <= numTiles);
}

bool VPU::isOperationSplitOverGroupCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();
    const auto minimumOutputGroupsForSOG = numTiles;

    auto outputShape = ShapeRef(outputTile.shape);
    if (outputShape == ShapeRef()) {
        outputShape = getBoundedShape(clusteredOp->getResult(0));
    }

    auto groupCompatibleCheck = [&](ShapeRef outputShape) {
        // #E125517 Layer still can be SOG compatible if rank is less than 5, but it is not supported yet
        if (outputShape.size() != DimsGroups5D::Act::numDims) {
            return false;
        }
        const auto groups = outputShape[DimsGroups5D::Act::G];
        return groups >= minimumOutputGroupsForSOG;
    };

    return groupCompatibleCheck(outputShape);
}

bool VPU::checkMCRestrictions(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (config::getAvailableExecutor(module, VPU::ExecutorKind::SHAVE_ACT) == nullptr) {
        return false;
    }

    auto inputShape = getBoundedShape(op->getOperand(0));
    auto outputShape = getBoundedShape(op->getResult(0));
    return !((inputShape.front() > VPU::SINGLE_BATCH && isSingleBatchRequired(op)) ||
             inputShape.size() != VPU::RANK_REQUIRED_FOR_TILING || outputShape.size() != VPU::RANK_REQUIRED_FOR_TILING);
}

bool VPU::doesLayerFitIntoCMX(mlir::Operation* op, VPU::MultiClusterStrategy strategy,
                              SiblingOpsAnalysis& siblingsAnalysis, Byte reservedMem) {
    if (op == nullptr) {
        return false;
    }
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    auto numClusters = getOptimalNumClusters(op, outputType.getShape(), strategy);
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);

    SmallVector<Byte> buffersSize{};
    for (auto input : op->getOperands()) {
        buffersSize.push_back(VPU::getTotalAllocSizeWithDistribution(
                input.getType(), getActivationDistributionAttrFromOp(clusteredOp, input, input.getType(), numClusters,
                                                                     strategy, siblingsAnalysis)));
    }
    for (auto result : op->getResults()) {
        buffersSize.push_back(VPU::getTotalAllocSizeWithDistribution(
                result.getType(), getOutputDistributionAttrFromOp(clusteredOp, result.getType(), numClusters, strategy,
                                                                  siblingsAnalysis)));
    }

    auto totalAvailableCMXSize =
            reservedMem.count() == 0 ? getTotalCMXSize(op).count() : getTotalCMXFragmentationAwareSize(op).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}
