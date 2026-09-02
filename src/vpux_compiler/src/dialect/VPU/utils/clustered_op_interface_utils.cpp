//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/clustered_op_interface_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/dilated_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

bool VPU::isOperationSplitOverHeightCompatible(mlir::Operation* op, const vpux::TileInfo& outputTile) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
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
    // For NCE ops with active ODU transform, also verify the pre-ODU output height is sufficient for SOH.
    if (mlir::isa<VPU::NCEOpInterface>(op)) {
        const auto scales = VPU::getODUScaling(op);
        if (!scales.empty()) {
            const auto preShapeResult = VPU::invertODUScaling(scales, SmallVector<int64_t>(outputShape.raw()));
            if (mlir::failed(preShapeResult)) {
                return false;
            }
            const auto& preShape = preShapeResult.value();
            if (!heightCompatibleCheck(ShapeRef(preShape))) {
                return false;
            }
        }
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

bool VPU::isNCEOpSplitOverHeightCompatible(mlir::Operation* op, mlir::Value input, ShapeRef defaultOutputShape,
                                           const vpux::TileInfo& oriOutputTile, bool checkAlignment) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    auto outputShape = ShapeRef(oriOutputTile.shape);
    auto offset = ShapeRef(oriOutputTile.offsets);
    auto axis = ShapeRef(oriOutputTile.axis);
    if (outputShape.empty()) {
        outputShape = defaultOutputShape;
    }
    vpux::TileInfo outputTile{outputShape, offset, axis, oriOutputTile.isCompletedTile};
    if (!VPU::isOperationSplitOverHeightCompatible(op, outputTile)) {
        return false;
    }

    // Replicate here the logic used in cost model to discard early per cluster tiles that cannot be inverted to pre-ODU
    // representation.
    // When the distributedTypes are not created, generate distributedTypes from strategy.
    auto strategy = MultiClusterStrategy::SplitOverHeight;
    const auto offsets = Shape(outputShape.size(), 0);
    auto outType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    if (mlir::isa<VPU::SparseTensorType, VPUIP::SparseBufferType>(outType)) {
        outType = mlir::cast<vpux::NDTypeInterface>(getEffectiveSparseOutputType(outType));
    }
    outType = outType.extractDenseTile(offsets, outputShape);

    // Adjust numClusters based on the tiled output shape
    // When tiling occurs, the tile shape may be smaller than the full output shape.
    // For strategies that distribute along a specific dimension, numClusters should be
    // limited by the size of that dimension in the tile.
    // For example: SOB splits over batch, SOH splits over height, SOK splits over channels
    auto numClusters = VPU::getOptimalNumClusters(op, outputShape, strategy);
    auto outputDistributedType = mlir::cast<VPU::DistributedTensorType>(
            getDistributedOutputTypeFromOp(clusteredOp, outType, numClusters, strategy));

    if (outputDistributedType != nullptr &&
        outputDistributedType.getDistribution().getMode().getValue() != VPU::DistributionMode::DUPLICATED) {
        auto outputPerClusterShapes = outputDistributedType.getPerClusterComputeShapes();
        auto outputPerClusterOffsets = outputDistributedType.getPerClusterComputeShapeOffsets();
        auto numTiles = vpux::parseIntArrayAttr<int64_t>(outputDistributedType.getDistribution().getNumTiles());
        auto numTilesShape = Shape(numTiles);

        for (auto index : irange(outputPerClusterShapes.size())) {
            TileInfo perClusterOutputTile(outputPerClusterShapes[index], outputPerClusterOffsets[index], numTilesShape);
            if (mlir::failed(VPU::invertODUScaling(getODUScaling(op), perClusterOutputTile))) {
                return false;
            }
        }
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Op {0} does not implement TilingBuilderOpInterface", op->getLoc());
    Shape inputShape = getBoundedShape(input).toValues();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    if (outputShape != defaultOutputShape) {
        VPUX_THROW_WHEN(offset.empty() || axis.empty(),
                        "Offsets and axis must have value when create TileInfo. Loc: {0}", op->getLoc());
        outputTile.isCompletedTile = true;
        auto computerShape = tilingBuilder.backInferTileInfo(outputTile, Logger::global());
        inputShape = computerShape.tiles.front().shape;
        auto inputOffset = computerShape.tiles.front().offsets;
        inputType = inputType.extractDenseTile(inputOffset, inputShape);
    }

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    return isSOHSupportedByDPU(inputType, inputShape, numTiles, checkAlignment, config::getArch(op));
}

bool VPU::isOperationSplitOverWidthCompatible(mlir::Operation* op, ShapeRef outputShape, ShapeRef /*offset*/,
                                              ShapeRef /*axis*/) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
    const auto minimumOutputWidthForSOW = numTiles;

    const auto arch = config::getArch(clusteredOp);
    if (outputShape == ShapeRef()) {
        outputShape = getBoundedShape(clusteredOp->getResult(0));
    }

    // For elementwise SW operations that do not support cycle cost calculation,
    // SOW (Split Over Width) is only allowed if the output width is sufficiently large.
    // This is because accurate cost modeling is not available for these ops, so clustering is preferred
    // for small widths to avoid potential performance issues.
    auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op);
    if (swOp != nullptr && op->hasTrait<VPU::EltwiseOp>() && !swOp.supportCycleCostCalculation()) {
        // The threshold of 128 for MIN_WIDTH_FOR_SOW was chosen based on empirical benchmarking results.
        // Below this threshold, clustering yields better performance.
        // This value may be adjusted in the future if performance data change.
        constexpr int64_t MIN_WIDTH_FOR_SOW = 128;
        const bool isTensorEffectively1D = outputShape[Dims4D::Act::W] == outputShape.totalSize();
        if (isTensorEffectively1D && outputShape[Dims4D::Act::W] < MIN_WIDTH_FOR_SOW) {
            // Clustering is a preferred strategy
            return false;
        }
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
    // For NCE ops with active ODU transform, also verify the pre-ODU output width is sufficient for SOW.
    if (mlir::isa<VPU::NCEOpInterface>(op)) {
        const auto scales = VPU::getODUScaling(op);
        if (!scales.empty()) {
            const auto preShapeResult = VPU::invertODUScaling(scales, SmallVector<int64_t>(outputShape.raw()));
            if (mlir::failed(preShapeResult)) {
                return false;
            }
            const auto preShape = preShapeResult.value();
            if (!widthCompatibleCheck(ShapeRef(preShape))) {
                return false;
            }
        }
    }
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
    auto clusteredOp = mlir::dyn_cast_if_present<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
    VPUX_THROW_WHEN(numTiles <= 0, "Number of tiles must be greater than zero");

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

    if (auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
        const auto ocAlignment = alignedOp.getOutputChannelAlignment();
        if (ocAlignment > VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT && OC / numTiles < ocAlignment) {
            return false;
        }
    }

    // For NCE ops with active ODU transform, also verify the pre-ODU channel count is sufficient for SOK.
    const auto scales = VPU::getODUScaling(op);
    if (!scales.empty()) {
        const auto preShapeResult = VPU::invertODUScaling(scales, SmallVector<int64_t>(outputShape.raw()));
        if (mlir::failed(preShapeResult)) {
            return false;
        }
        const auto preShape = ShapeRef(preShapeResult.value());
        VPUX_THROW_WHEN(preShape.size() != outputShape.size(),
                        "Pre-ODU shape rank does not match output shape rank for op {0}", op->getLoc());
        if (preShape[Dims4D::Act::C] < minChannelSize) {
            return false;
        }
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
            if (auto qElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
                if (auto quantileStorage = mlir::dyn_cast<vpux::type::QuantileType>(qElemType.getStorageType())) {
                    elemType = mlir::quant::UniformQuantizedType::get(
                            qElemType.getFlags(), quantileStorage, qElemType.getExpressedType(), /*scale=*/1.0,
                            /*zeroPoint=*/0, qElemType.getStorageTypeMin(), qElemType.getStorageTypeMax());
                } else {
                    elemType = mlir::quant::UniformQuantizedType::get(qElemType.getFlags(), qElemType.getStorageType(),
                                                                      qElemType.getExpressedType(), /*scale=*/1.0,
                                                                      /*zeroPoint=*/0, qElemType.getStorageTypeMin(),
                                                                      qElemType.getStorageTypeMax());
                }
            }
            const auto newType = origType.changeShapeElemType(newShape, elemType);

            // Determine the channel split over clusters using the lightweight native distribution
            // instead of materializing a distributed tensor type.
            // SEP depthwise convolution uses an explicit, potentially non-uniform per-cluster
            // channel split for SplitOverKernel. The lightweight getFilterDistributionAttrFromOp
            // does not model that split, so the distributed filter type is materialized only for
            // this case to preserve the exact per-cluster offsets; all other cases keep the
            // lightweight native path.
            SmallVector<Shape> computeOffsets;
            const auto nceOperation = nceOp.getOperation();
            if (VPU::isSEPDWConv(nceOperation)) {
                const auto distributedFilterType = VPU::getDistributedFilterTypeFromOp(
                        nceOp, newType, numTiles, VPU::MultiClusterStrategy::SplitOverKernel);
                const auto distributedDataType = mlir::cast<vpux::VPU::DistributedTensorType>(
                        distributedFilterType.getDistributedTypes().front());
                computeOffsets = distributedDataType.getPerClusterComputeShapeOffsets();
            } else {
                // For sparse types, getFilterDistributionAttrFromOp keys the map by the inner data
                // type (getData()), not by the sparse wrapper. For non-sparse types it uses the type
                // itself. filterDataType resolves to the correct map key in both cases, so it is used
                // directly for both the lookup and the offset computation below.
                vpux::NDTypeInterface filterDataType;
                if (const auto newSparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(newType)) {
                    filterDataType = mlir::cast<vpux::NDTypeInterface>(newSparseType.getData());
                } else {
                    filterDataType = mlir::cast<vpux::NDTypeInterface>(newType);
                }
                const auto filterDistributions = VPU::getFilterDistributionAttrFromOp(
                        nceOp, newType, numTiles, VPU::MultiClusterStrategy::SplitOverKernel);
                const auto filterDistributionIt = filterDistributions.find(mlir::Type(filterDataType));
                VPUX_THROW_WHEN(filterDistributionIt == filterDistributions.end(),
                                "Missing filter distribution for data type '{0}' in op '{1}'",
                                mlir::Type(filterDataType), *nceOperation);
                const auto& filterDistribution = filterDistributionIt->second;
                computeOffsets = VPU::getPerClusterComputeShapeOffsets(filterDataType.getShape(), filterDistribution);
            }
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
    const auto numTiles = config::getNumOfTiles(op);

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
        if (mlir::isa<VPU::NCEMatMulOp>(op)) {
            return (groups > 1);
        }

        return groups >= minimumOutputGroupsForSOG;
    };

    return groupCompatibleCheck(outputShape);
}

bool VPU::checkMCRestrictions(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (config::getAvailableExecutor(module, config::ExecutorKind::SHAVE_ACT) == nullptr) {
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
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);

    SmallVector<Byte> buffersSize{};
    for (auto input : op->getOperands()) {
        TensorDistributionMap distAttr;
        if (nceOp != nullptr && isWeightsLikeOperand(nceOp, input)) {
            distAttr = getFilterDistributionAttrFromOp(nceOp, mlir::cast<vpux::NDTypeInterface>(input.getType()),
                                                       numClusters, strategy);
        } else {
            distAttr = getActivationDistributionAttrFromOp(clusteredOp, input, input.getType(), numClusters, strategy,
                                                           siblingsAnalysis);
        }
        buffersSize.push_back(VPU::getTotalAllocSizeWithDistribution(input.getType(), distAttr));
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

bool VPU::isEltwiseSWOpSplitOverHeightCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment) {
    if (!VPU::checkMCRestrictions(op)) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
    const auto isUniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(op);

    const auto outShape = outputShape.empty() ? getBoundedShape(op->getResult(0)) : outputShape;
    const auto OH = outShape[Dims4D::Act::H];
    const auto numClustersForSOH = VPU::getNumberOfClustersForSpatialDim(OH, numTiles, isUniformDistributedSegments);

    const auto hSlice = OH / numClustersForSOH;
    const auto hRemainder = OH % numClustersForSOH;
    if (hSlice % alignment != 0 || hRemainder % alignment != 0) {
        return false;
    }

    return OH >= numTiles && numClustersForSOH == numTiles;
}

bool VPU::isEltwiseSWOpSplitOverWidthCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment) {
    if (!VPU::checkMCRestrictions(op)) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
    const auto isUniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(op);

    const auto outShape = outputShape.empty() ? getBoundedShape(op->getResult(0)) : outputShape;
    const auto OW = outShape[Dims4D::Act::W];

    // The threshold of 128 for MIN_WIDTH_FOR_SOW was chosen based on empirical benchmarking results.
    // Below this threshold, clustering yields better performance.
    // This value may be adjusted in the future if performance data change.
    constexpr int64_t MIN_WIDTH_FOR_SOW = 128;
    const bool isTensorEffectively1D = OW == outShape.totalSize();
    auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op);
    if (swOp != nullptr && !swOp.supportCycleCostCalculation() && isTensorEffectively1D && OW < MIN_WIDTH_FOR_SOW) {
        // Clustering is a preferred strategy
        return false;
    }

    const auto numClustersForSOW = VPU::getNumberOfClustersForSpatialDim(OW, numTiles, isUniformDistributedSegments);
    const auto wSlice = OW / numClustersForSOW;
    const auto wRemainder = OW % numClustersForSOW;
    if (wSlice % alignment != 0 || wRemainder % alignment != 0) {
        return false;
    }

    return OW >= numTiles && numClustersForSOW == numTiles;
}

bool VPU::isEltwiseSWOpSplitOverKernelCompatible(mlir::Operation* op, vpux::ShapeRef outputShape, int64_t alignment) {
    if (!VPU::checkMCRestrictions(op)) {
        return false;
    }

    const auto numTiles = config::getNumOfTiles(op);
    VPUX_THROW_WHEN(numTiles == 0, "Number of tiles must be greater than zero.");

    const auto outShape = outputShape.empty() ? getBoundedShape(op->getResult(0)) : outputShape;
    const auto outK = outShape[Dims4D::Act::C];
    const auto kSlice = outK / numTiles;
    const auto kRemainder = outK % numTiles;
    if (kSlice % alignment != 0 || kRemainder % alignment != 0) {
        return false;
    }
    return outK >= numTiles;
}
