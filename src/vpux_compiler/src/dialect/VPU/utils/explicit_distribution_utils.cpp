//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <optional>

using namespace vpux;

VPU::OverlapDistributionParams VPU::getExplicitOverlapParamsForSWOpInput(VPU::SWOpInterface swOp, ShapeRef outShape,
                                                                         ArrayRef<int64_t> numTiles,
                                                                         ArrayRef<int64_t> alignment,
                                                                         const vpux::TileInfo& origOutTile) {
    VPUX_THROW_WHEN(swOp == nullptr, "Cannot get SW DistributionInfoAttr, is not a SW op");
    VPUX_THROW_WHEN(swOp->getNumResults() != 1, "More than one result for Sw op: {0}", swOp);

    std::optional<size_t> overlappedInputIdx = std::nullopt;
    const auto strategy = VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
    const auto operands = swOp->getOperands();

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Sw op {0} is not a ClusteredOp", swOp->getLoc());
    for (const auto& [index, operand] : operands | indexed) {
        const auto ndTypeInterface = mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType());
        if (getSWInputTensorDistributionMode(clusteredOp, strategy, operand, ndTypeInterface) ==
            VPU::DistributionMode::OVERLAPPED) {
            VPUX_THROW_WHEN(overlappedInputIdx.has_value(), "More than one OVERLAPPED input for Sw op: {0}", swOp);
            overlappedInputIdx = index;
        }
    }
    VPUX_THROW_UNLESS(overlappedInputIdx.has_value(), "Sw op {0} has no OVERLAPPED inputs", swOp);

    std::optional<ArrayRef<int64_t>> alignmentValue = std::nullopt;
    if (!alignment.empty()) {
        alignmentValue = alignment;
    }

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Cannot cast op to TilingBuilderOpInterface at {0}", swOp.getLoc());

    std::optional<InputTiling> origInputsTileInfo = std::nullopt;
    if (origOutTile != vpux::TileInfo(ShapeRef())) {
        origInputsTileInfo = tilingBuilder.backInferTileInfo(origOutTile, Logger::global());
    }

    const auto tiles = fillDividedTiles(ShapeRef(numTiles), outShape, alignmentValue);
    VPUX_THROW_WHEN(mlir::failed(tiles), "Incorrect tiles at {0}", swOp.getLoc());
    const auto& outTiles = tiles.value();

    // When we multicluster a slice of the original op, we need to locate
    // the multicluster slice relative to the full tensor. Otherwise, the
    // back-infer algorithm might fail due to not also getting the updated
    // attributes for the original op slice.

    // At first, we take the multicluster slice (subTile) and add to it the
    // original offsets of the tile to it (if we do have tile information).
    // E.g. orig output tensor 1x512x256x122
    //       -> we're looking to multicluster slice:
    //     origOutTile = off[0, 0, 120, 10] sz[1, 512, 100, 50]
    //       -> for 2 clusters we get on W:
    //     cl0: off[0, 0, 120, 10] sz[1, 512, 100, 25]
    //     cl1: off[0, 0, 120, 35] sz[1, 512, 100, 25]
    auto getOutTileInFullOutput = [&](const TileInfo& subTile) -> TileInfo {
        if (origOutTile == vpux::TileInfo(ShapeRef())) {
            return subTile;
        }

        auto adjustedTile = subTile;
        for (auto idx : irange(subTile.offsets.size())) {
            adjustedTile.offsets[Dim(idx)] += origOutTile.offsets[Dim(idx)];
            adjustedTile.axis[Dim(idx)] *= origOutTile.axis[Dim(idx)];
        }

        return adjustedTile;
    };

    // After back-inferring, we use the generated slice in the full tensor
    // and the back-inferred input of origOutTile to represent the input slice
    // relative to the tile we got for origOp
    // E.g. orig back-inferred input tensor
    //          off [0, 30, 15, 0] sz[1, 13, 26, 40]
    //       -> for 2 clusters, we got the following slices in full tensor:
    //     cl0: off[0, 30, 15, 0] sz[1, 7, 26, 40]
    //     cl1: off[0, 37, 15, 0] sz[1, 6, 26, 40]
    //       -> after adjusting offsets relative to the tile slice:
    //     cl0: off[0, 0, 0, 0] sz[1, 7, 26, 40]
    //     cl1: off[0, 7, 0, 0] sz[1, 6, 26, 40]
    auto getClusterTileFromTileInFullTensor = [&](InputTiling& inTiles) {
        if (!origInputsTileInfo.has_value()) {
            return;
        }

        const auto& overlappedInputOrigOffests = origInputsTileInfo.value().tiles[overlappedInputIdx.value()].offsets;
        for (auto idx : irange(overlappedInputOrigOffests.size())) {
            inTiles.tiles[overlappedInputIdx.value()].offsets[Dim(idx)] -= overlappedInputOrigOffests[Dim(idx)];
        }
    };

    SmallVector<InputTiling> inputTiles;
    for (const auto& outTile : outTiles) {
        auto outTileInFullTensor = getOutTileInFullOutput(outTile);
        auto inputTiling = tilingBuilder.backInferTileInfo(outTileInFullTensor, Logger::global());
        VPUX_THROW_UNLESS(inputTiling.tiles.size() == operands.size(),
                          "Unexpected input operands size: expected {0}, but got {1}", operands.size(),
                          inputTiling.tiles.size());

        getClusterTileFromTileInFullTensor(inputTiling);
        inputTiles.push_back(inputTiling);
    }

    SmallVector<SmallVector<int64_t>> inputPerClusterShape;
    SmallVector<SmallVector<int64_t>> inputPerClusterOffset;
    for (auto i : irange(outTiles.size())) {
        inputPerClusterShape.push_back(to_small_vector(inputTiles[i].tiles[overlappedInputIdx.value()].shape));
        inputPerClusterOffset.push_back(to_small_vector(inputTiles[i].tiles[overlappedInputIdx.value()].offsets));
    }

    return OverlapDistributionParams(inputPerClusterShape, inputPerClusterOffset, inputPerClusterShape,
                                     inputPerClusterOffset);
}

VPU::DistributionInfoAttr VPU::getSWExplicitDistributionInfoAttr(
        VPU::SWOpInterface swOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    VPUX_THROW_WHEN(swOp == nullptr, "Cannot get SW DistributionInfoAttr, is not a SW op");
    auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
    auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};

    return vpux::VPU::DistributionInfo::getAttrFromClass(
            swOp.getContext(),
            getSWExplicitDistributionInfo(swOp, shape, distributionMode, numTilesArr, numClusters.getInt(),
                                          alignmentArr, uniformDistributedSegments != nullptr, overlapParams));
}

VPU::DistributionInfo VPU::getSWExplicitDistributionInfo(VPU::SWOpInterface swOp, ShapeRef shape,
                                                         VPU::DistributionMode distributionMode,
                                                         ArrayRef<int64_t> numTiles, const int64_t numClusters,
                                                         ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                         const vpux::VPU::OverlapDistributionParams& overlapParams) {
    VPUX_THROW_WHEN(swOp == nullptr, "Cannot get SW DistributedTensor, is not a SW op");

    if (distributionMode != VPU::DistributionMode::OVERLAPPED) {
        return getNonOverlappedDistributedNative(shape, distributionMode, numTiles, numClusters, alignment,
                                                 uniformDistributedSegments);
    }

    if (overlapParams.hasNonnullComputeAndMemoryShapesOffsets()) {
        return VPU::DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                     uniformDistributedSegments, overlapParams.getComputeShapes(),
                                     overlapParams.getComputeOffsets(), overlapParams.getMemoryShapes(),
                                     overlapParams.getComputeOffsets(), {});
    }

    const auto untiledOverlapParams =
            getExplicitOverlapParamsForSWOpInput(swOp, getShape(swOp->getResult(0)), numTiles, alignment);

    return VPU::DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                 uniformDistributedSegments, untiledOverlapParams.getComputeShapes(),
                                 untiledOverlapParams.getComputeOffsets(), untiledOverlapParams.getMemoryShapes(),
                                 untiledOverlapParams.getComputeOffsets(), {});
}

VPU::DistributionInfoAttr VPU::getNCEExplicitDistributionInfoAttr(
        VPU::NCEOpInterface nceOp, ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    VPUX_THROW_WHEN(nceOp == nullptr, "Cannot get HW DistributionInfoAttr, is not a HW op");
    auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
    auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};

    return vpux::VPU::DistributionInfo::getAttrFromClass(
            nceOp.getContext(),
            getNCEExplicitDistributionInfo(nceOp, shape, distributionMode, numTilesArr, numClusters.getInt(),
                                           alignmentArr, uniformDistributedSegments ? true : false, overlapParams));
}

VPU::DistributionInfo VPU::getNCEExplicitDistributionInfo(VPU::NCEOpInterface nceOp, ShapeRef shape,
                                                          VPU::DistributionMode distributionMode,
                                                          ArrayRef<int64_t> numTiles, const int64_t numClusters,
                                                          ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                          const vpux::VPU::OverlapDistributionParams& overlapParams) {
    VPUX_THROW_WHEN(nceOp == nullptr, "Cannot get HW DistributionInfo, is not a HW op");

    if (distributionMode == DistributionMode::OVERLAPPED || overlapParams.hasNonnullComputeAndMemoryShapesOffsets()) {
        VPUX_THROW_WHEN(overlapParams.getMemoryShapes().empty() || overlapParams.getMemoryOffsets().empty() ||
                                overlapParams.getComputeShapes().empty() || overlapParams.getComputeOffsets().empty(),
                        "memoryShapes, memoryOffsets, computeShapes, computeOffsets cannot be empty.");

        return DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                uniformDistributedSegments, overlapParams.getComputeShapes(),
                                overlapParams.getComputeOffsets(), overlapParams.getMemoryShapes(),
                                overlapParams.getMemoryOffsets(), {});
    }

    auto distributedTensor = DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                              uniformDistributedSegments, {}, {}, {}, {}, {});

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensor);

    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensor);
    auto perClusterMemoryShapes = optionalClusterMemoryShapes.value();
    auto perClusterMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensor);
    auto perClusterComputeShapes = VPU::getPerClusterComputeShapes(shape, distributedTensor);
    auto perClusterComputeOffsets = VPU::getPerClusterComputeShapeOffsets(shape, distributedTensor);

    distributedTensor.setMemoryShapes(arrayOfArrayFromShape(perClusterMemoryShapes));
    distributedTensor.setMemoryOffsets(arrayOfArrayFromShape(perClusterMemoryOffsets));
    distributedTensor.setComputeShapes(arrayOfArrayFromShape(perClusterComputeShapes));
    distributedTensor.setComputeOffsets(arrayOfArrayFromShape(perClusterComputeOffsets));

    return distributedTensor;
}

VPU::DistributionInfoAttr VPU::getConcatExplicitDistributedAttr(
        ShapeRef shape, VPU::DistributionMode distributionMode, mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
        mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams, mlir::MLIRContext* ctx) {
    auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
    auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};

    return vpux::VPU::DistributionInfo::getAttrFromClass(
            ctx,
            getConcatExplicitDistributedNative(shape, distributionMode, numTilesArr, numClusters.getInt(), alignmentArr,
                                               uniformDistributedSegments ? true : false, overlapParams));
}

VPU::DistributionInfo VPU::getConcatExplicitDistributedNative(
        ShapeRef shape, VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles, int64_t numClusters,
        ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    if (distributionMode == DistributionMode::OVERLAPPED) {
        VPUX_THROW_WHEN(overlapParams.getMemoryShapes().empty() || overlapParams.getMemoryOffsets().empty(),
                        "memoryShapes and memoryOffsets cannot be empty.");

        return VPU::DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                     uniformDistributedSegments, overlapParams.getMemoryShapes(),
                                     overlapParams.getMemoryOffsets(), overlapParams.getMemoryShapes(),
                                     overlapParams.getMemoryOffsets(), {});
    }

    auto distributedTensor = VPU::DistributionInfo(distributionMode, numTiles, {}, {}, {}, numClusters, alignment,
                                                   uniformDistributedSegments, {}, {}, {}, {}, {});

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensor);
    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensor);
    auto perClusterMemoryShapes = optionalClusterMemoryShapes.value();
    auto perClusterMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensor);

    distributedTensor.setMemoryShapes(arrayOfArrayFromShape(perClusterMemoryShapes));
    distributedTensor.setMemoryOffsets(arrayOfArrayFromShape(perClusterMemoryOffsets));
    distributedTensor.setComputeShapes(arrayOfArrayFromShape(perClusterMemoryShapes));
    distributedTensor.setComputeOffsets(arrayOfArrayFromShape(perClusterMemoryOffsets));

    return distributedTensor;
}

VPU::DistributionInfoAttr vpux::VPU::getConcatExplicitDistributedAttrForNewShape(
        VPU::DistributionInfoAttr originDistribution, vpux::ShapeRef newShape, mlir::MLIRContext* ctx) {
    auto distribution = VPU::DistributionInfo::getClassFromAttr(originDistribution);
    return VPU::DistributionInfo::getAttrFromClass(
            ctx, getConcatExplicitDistributedNativeForNewShape(distribution, newShape));
}

VPU::DistributionInfo VPU::getConcatExplicitDistributedNativeForNewShape(
        const VPU::DistributionInfo& originDistribution, vpux::ShapeRef newShape) {
    // For non-overlapped mode, use already existing methods that compute per cluster shapes/methods
    if (originDistribution.getDistributionMode() != VPU::DistributionMode::OVERLAPPED) {
        return VPU::getConcatExplicitDistributedNative(
                newShape, originDistribution.getDistributionMode(), originDistribution.getNumTiles(),
                originDistribution.getNumClusters(), originDistribution.getAlignment(),
                originDistribution.hasUniformDistributedSegments(), VPU::OverlapDistributionParams());
    }

    const auto numTiles = originDistribution.getNumTiles();
    auto memoryShapes = originDistribution.getMemoryShapes();
    auto newMemoryShapes = SmallVector<SmallVector<int64_t>>{};

    // For overlapped mode, on the clustering dim, the shapes are taken from the initial distribution, while the rest of
    // the dims will take values from the new shape; this works as long as the concat axis != clustering axis, which is
    // a prerequisite of Distributed Concat
    for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
        newMemoryShapes.push_back(memoryShapes[cluster]);
        for (size_t dim = 0; dim < numTiles.size(); dim++) {
            if (numTiles[dim] == 1) {
                newMemoryShapes[cluster][dim] = newShape[Dim(dim)];
            }
        }
    }

    auto newDistribution = originDistribution;
    newDistribution.setMemoryShapes(newMemoryShapes);
    newDistribution.setMemoryShapes(newMemoryShapes);
    newDistribution.setComputeShapes(newMemoryShapes);
    newDistribution.setComputeOffsets(originDistribution.getMemoryOffsets());

    return newDistribution;
}

/// @param distributionWithProperAlignment The original alignment may need be updated to get valid perClusterShapesAttr
/// for slice ops.
/// E.g., C=64, T=4, Alignment=16, then perClusterShape is [16, 16, 16, 16]. For sliceShape C = 32,
/// perClusterShape should be [8, 8, 8, 8], thus original alignment must be changed
VPU::DistributionInfoAttr VPU::getExplicitDistrAttrForSliceLikeOps(
        VPU::DistributionInfoAttr distributionWithProperAlignment, ArrayRef<int64_t> sliceShape,
        ArrayRef<int64_t> originShape, mlir::MLIRContext* ctx) {
    auto distribution = VPU::DistributionInfo::getClassFromAttr(distributionWithProperAlignment);

    return VPU::DistributionInfo::getAttrFromClass(
            ctx, getExplicitDistrNativeForSliceLikeOps(distribution, sliceShape, originShape));
}

VPU::DistributionInfo VPU::getExplicitDistrNativeForSliceLikeOps(
        const VPU::DistributionInfo& distributionWithProperAlignment, ArrayRef<int64_t> sliceShape,
        ArrayRef<int64_t> originShape) {
    const auto mode = distributionWithProperAlignment.getDistributionMode();

    // Explicit DistributedAttr can be inferred for Slice in SEGMENTED case or in any case that has full tensor
    // in all cluster (i.e. if mode contains DUPLICATED or SEGMENTED).
    VPUX_THROW_WHEN(
            (mode != VPU::DistributionMode::SEGMENTED) && (mode != VPU::DistributionMode::OVERLAPPED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) &&
                    !VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED),
            "Cannot apply Slice-like Op on input with explicit memory/compute shapes/offsets with DistributionMode {0}",
            mode);

    // For Overlapped, if slice axis is clustering axis, per cluster shapes/offsets need to be computed taking into
    // consideration Slice/Subview's neighbour ops, which cannot be done with information available here; the calling
    // pass should fill the correct information in this scenario
    VPUX_THROW_WHEN(
            mode == VPU::DistributionMode::OVERLAPPED &&
                    VPU::isSegmentedOverlappedAxisSameAsSliceAxis(distributionWithProperAlignment.getNumTiles(),
                                                                  originShape, sliceShape),
            "Overlapped clustering axis is the same as Slice/Subview axis; cannot infer per cluster shapes/offsets "
            "without compute op information");

    const auto getDistribution = [&](ArrayRef<SmallVector<int64_t>> perClusterShapesAttr,
                                     ArrayRef<SmallVector<int64_t>> perClusterOffsetsAttr) -> VPU::DistributionInfo {
        // Slice/SubviewOp is not a "compute" op, so compute shapes/offsets have no reason to be different
        // from memory shapes/offsets
        auto newDistribution = distributionWithProperAlignment;
        newDistribution.setMemoryShapes(perClusterShapesAttr);
        newDistribution.setMemoryOffsets(perClusterOffsetsAttr);
        newDistribution.setComputeShapes(perClusterShapesAttr);
        newDistribution.setComputeOffsets(perClusterOffsetsAttr);

        return newDistribution;
    };

    if (mode == VPU::DistributionMode::OVERLAPPED) {
        auto memoryShapes = distributionWithProperAlignment.getMemoryShapes();
        auto newMemoryShapes = SmallVector<SmallVector<int64_t>>{};

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            newMemoryShapes.push_back(memoryShapes[cluster]);
            for (size_t dim = 0; dim < originShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != originShape[dim]) {
                    newMemoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }

        return getDistribution(newMemoryShapes, distributionWithProperAlignment.getMemoryOffsets());
    }

    const auto memoryShapes = VPU::getPerClusterMemoryShapes(ShapeRef(sliceShape), distributionWithProperAlignment);
    VPUX_THROW_WHEN(
            !memoryShapes.has_value(),
            "Cannot compute memory shapes for the shape of Slice/Subview's output; shape = {0}, distribution ={1}",
            sliceShape, distributionWithProperAlignment);

    auto perClusterShapes = arrayOfArrayFromShape(memoryShapes.value());
    auto perClusterOffsets = arrayOfArrayFromShape(
            VPU::getPerClusterMemoryShapeOffsets(ShapeRef(sliceShape), distributionWithProperAlignment));

    return getDistribution(perClusterShapes, perClusterOffsets);
}

/**
 * @brief  Get Explicit DistAttr by provided explicit shapes. The function is used to get the last slice of a segmented
 distributed type. E.g., C=128, T=6, Alignment=16, then perClusterShape is [32, 32, 16, 16, 16, 16]. For sliceShape
 C=80, with offset=48, perClusterShape should be [24, 24, 8, 8, 8, 8], which is unable to be infered by the original
 dist attr.
 * @param distribution the src distribution of slice like op
 * @param sliceOutputShape The output shape of the slice like op
 * @param explicitShapes The expected output shapes on all clusters
 */
VPU::DistributionInfoAttr vpux::VPU::getSegmentedExplicitDistrAttrForSliceLikeOps(
        VPU::DistributionInfoAttr distributionAttr, ArrayRef<int64_t> sliceOutputShape, mlir::ArrayAttr explicitShapes,
        mlir::MLIRContext* ctx) {
    auto explicitShapesArr =
            explicitShapes ? parseIntArrayOfArrayAttr<int64_t>(explicitShapes) : SmallVector<SmallVector<int64_t>>{};
    auto distribution = VPU::DistributionInfo::getClassFromAttr(distributionAttr);
    return VPU::DistributionInfo::getAttrFromClass(
            ctx, getSegmentedExplicitDistrNativeForSliceLikeOps(distribution, sliceOutputShape, explicitShapesArr));
}

VPU::DistributionInfo vpux::VPU::getSegmentedExplicitDistrNativeForSliceLikeOps(
        const VPU::DistributionInfo& distribution, ArrayRef<int64_t> sliceOutputShape,
        ArrayRef<SmallVector<int64_t>> explicitShapes) {
    const auto mode = distribution.getDistributionMode();
    // Explicit DistributedAttr can be inferred in any case that has full tensor
    // in all cluster (i.e. if mode contains DUPLICATED or MULTICASTED).
    VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED,
                      "Cannot get SEGMENTED explicit distribution for Slice-like op with DistributionMode {0}",
                      distribution.getDistributionMode());

    auto hasSameDimSize = llvm::all_of(explicitShapes, [&](const auto& shape) {
        return shape.size() == sliceOutputShape.size();
    });
    VPUX_THROW_UNLESS(hasSameDimSize, "Explicit shapes have different dim num: shapes {0}, slice output {1}",
                      explicitShapes, sliceOutputShape);

    int64_t sliceDim = -1;
    for (auto i : irange(sliceOutputShape.size())) {
        auto sliceOnCurrentDim = llvm::all_of(explicitShapes, [&](const auto& shape) {
            return shape[i] != sliceOutputShape[i];
        });
        if (sliceOnCurrentDim) {
            VPUX_THROW_UNLESS(sliceDim == -1, "Only support explicit shapes on single dim");
            auto sumDimSize = std::accumulate(explicitShapes.begin(), explicitShapes.end(), 0,
                                              [&](const int64_t sum, const auto& shape) {
                                                  return sum + shape[i];
                                              });
            VPUX_THROW_UNLESS(sumDimSize == sliceOutputShape[i], "explicit shapes {0} don't match shape {1}",
                              explicitShapes, sliceOutputShape);
            sliceDim = i;
        }
    }
    VPUX_THROW_WHEN(sliceDim == -1, "All explicit shapes have same shape. Explicit shapes are not necessary");

    SmallVector<SmallVector<int64_t>> offsets;
    int64_t offsetVal = 0;
    for (auto& shape : explicitShapes) {
        SmallVector<int64_t> offset(sliceOutputShape.size(), 0);
        offset[sliceDim] = offsetVal;
        offsets.emplace_back(std::move(offset));
        offsetVal += shape[sliceDim];
    }

    // Create DistributionInfoAttr with provided shapes and offsets. Since alignment is unnecessary, remove it from the
    // new attr
    auto newDistribution = distribution;
    newDistribution.setMemoryShapes(explicitShapes);
    newDistribution.setComputeShapes(explicitShapes);
    newDistribution.setMemoryOffsets(offsets);
    newDistribution.setComputeOffsets(offsets);
    newDistribution.setAlignment(SmallVector<int64_t>{});

    return newDistribution;
}

VPU::DistributionInfoAttr vpux::VPU::getNonOverlappedDistributedAttr(
        ShapeRef shape, VPU::DistributionModeAttr distrModeAttr, mlir::ArrayAttr numTiles,
        mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment, mlir::UnitAttr uniformDistributedSegments,
        mlir::MLIRContext* ctx) {
    VPUX_THROW_WHEN(distrModeAttr.getValue() == VPU::DistributionMode::OVERLAPPED,
                    "getNonOverlappedDistributedAttr: distribution mode is OVERLAPPED");
    auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
    auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};
    return vpux::VPU::DistributionInfo::getAttrFromClass(
            ctx, getNonOverlappedDistributedNative(shape, distrModeAttr.getValue(), numTilesArr, numClusters.getInt(),
                                                   alignmentArr, uniformDistributedSegments ? true : false));
}

VPU::DistributionInfo vpux::VPU::getNonOverlappedDistributedNative(ShapeRef shape, VPU::DistributionMode distrMode,
                                                                   ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                                   ArrayRef<int64_t> alignment,
                                                                   bool uniformDistributedSegments) {
    VPUX_THROW_WHEN(distrMode == VPU::DistributionMode::OVERLAPPED,
                    "getNonOverlappedDistributedNative: distribution mode is OVERLAPPED");

    auto distributedTensor = VPU::DistributionInfo(distrMode, numTiles, {}, {}, {}, numClusters, alignment,
                                                   uniformDistributedSegments, {}, {}, {}, {}, {});

    auto optionalClusterMemoryShapes = VPU::getPerClusterMemoryShapes(shape, distributedTensor);

    VPUX_THROW_UNLESS(optionalClusterMemoryShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distributedTensor);

    auto perClusterMemoryShapes = optionalClusterMemoryShapes.value();
    auto perClusterMemoryOffsets = VPU::getPerClusterMemoryShapeOffsets(shape, distributedTensor);
    auto perClusterComputeShapes = VPU::getPerClusterComputeShapes(shape, distributedTensor);
    auto perClusterComputeOffsets = VPU::getPerClusterComputeShapeOffsets(shape, distributedTensor);

    distributedTensor.setMemoryShapes(VPU::arrayOfArrayFromShape(perClusterMemoryShapes));
    distributedTensor.setMemoryOffsets(VPU::arrayOfArrayFromShape(perClusterMemoryOffsets));
    distributedTensor.setComputeShapes(VPU::arrayOfArrayFromShape(perClusterComputeShapes));
    distributedTensor.setComputeOffsets(VPU::arrayOfArrayFromShape(perClusterComputeOffsets));

    return distributedTensor;
}

NDTypeInterface vpux::VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                              mlir::Type elemType) {
    auto distributedBuff = mlir::dyn_cast<VPUIP::DistributedBufferType>(buff);
    VPUX_THROW_WHEN(distributedBuff == nullptr,
                    "changeShapeElemTypeForNonOverlappedDistributedBuffers: buff is not DistributedBufferType = {0}",
                    buff);

    auto distribution = distributedBuff.getDistribution();
    VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                    "DistributedBuffer has mode different from DUPLICATED after unrolling");
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedBuff.getDistribution())) {
        auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                shape, distribution.getMode(), nullptr, distribution.getNumClusters(), nullptr,
                distribution.getUniformDistributedSegments(), distributedBuff.getContext());
        return distributedBuff.changeShapeElemTypeForExplicitDistribution(shape, elemType, newDistribution);
    }

    return distributedBuff.changeShapeElemType(shape, elemType);
};

VPU::DistributionInfoAttr vpux::VPU::getExplicitDistrAttrForSparseData(VPU::DistributionInfoAttr denseDataDistribution,
                                                                       ShapeRef dataShape, VPU::SEAttr seAttr,
                                                                       mlir::MLIRContext* ctx) {
    if (seAttr == nullptr) {
        return denseDataDistribution;
    }

    auto getDataShapesOffsets =
            [&](mlir::ArrayAttr denseDataShapesAttr,
                mlir::ArrayAttr denseDataOffsetsAttr) -> std::pair<mlir::ArrayAttr, mlir::ArrayAttr> {
        const auto denseDataShapes = parseIntArrayOfArrayAttr<int64_t>(denseDataShapesAttr);
        const auto denseDataOffsets = parseIntArrayOfArrayAttr<int64_t>(denseDataOffsetsAttr);
        const auto clusterNum = denseDataShapes.size();
        auto dataShapesVec = SmallVector<Shape>(clusterNum, Shape(denseDataShapes[0]));
        auto dataOffsetsVec = SmallVector<Shape>(clusterNum, Shape(denseDataOffsets[0]));

        for (size_t clusterIdx = 0; clusterIdx < clusterNum; ++clusterIdx) {
            const auto denseDataShape = Shape(denseDataShapes[clusterIdx]);
            const auto denseDataOffset = Shape(denseDataOffsets[clusterIdx]);

            seAttr.extractTile(denseDataOffset, denseDataShape, dataShape, dataOffsetsVec[clusterIdx],
                               dataShapesVec[clusterIdx]);
        }

        return {getIntArrayOfArray(ctx, dataShapesVec), getIntArrayOfArray(ctx, dataOffsetsVec)};
    };

    const auto computeView =
            getDataShapesOffsets(denseDataDistribution.getComputeShapes(), denseDataDistribution.getComputeOffsets());
    const auto memoryView =
            getDataShapesOffsets(denseDataDistribution.getMemoryShapes(), denseDataDistribution.getMemoryOffsets());

    return VPU::DistributionInfoAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                          nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                          /*alignment*/ nullptr, denseDataDistribution.getUniformDistributedSegments(),
                                          computeView.first, computeView.second, memoryView.first, memoryView.second,
                                          denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributionInfoAttr vpux::VPU::getExplicitDistrAttrForSparsityMap(VPU::DistributionInfoAttr denseDataDistribution,
                                                                        ShapeRef sparsityMapShape,
                                                                        mlir::UnitAttr isWeights,
                                                                        mlir::MLIRContext* ctx) {
    if (isWeights == nullptr) {
        return denseDataDistribution;
    }

    auto isValidDistributionForWeights = [&]() -> bool {
        if (denseDataDistribution.getNumTiles() == nullptr) {
            return true;
        }

        const auto numTiles = parseIntArrayAttr<int64_t>(denseDataDistribution.getNumTiles());
        if (numTiles.size() == 4 && numTiles[Dims4D::Act::C.ind()] == 1 && numTiles[Dims4D::Act::H.ind()] == 1 &&
            numTiles[Dims4D::Act::W.ind()] == 1) {
            return true;
        }

        return false;
    };

    VPUX_THROW_WHEN(!isValidDistributionForWeights(),
                    "Weights should be segmented only over OC dim, distributed attr = {0}", denseDataDistribution);

    auto getWeightsShapes = [&](mlir::ArrayAttr shapesAttr) -> mlir::ArrayAttr {
        auto shapesVec = parseIntArrayOfArrayAttr<int64_t>(shapesAttr);

        for (auto& shapes : shapesVec) {
            shapes[Dims4D::Filter::IC.ind()] = sparsityMapShape[Dims4D::Filter::IC];
            shapes[Dims4D::Filter::KY.ind()] = sparsityMapShape[Dims4D::Filter::KY];
            shapes[Dims4D::Filter::KX.ind()] = sparsityMapShape[Dims4D::Filter::KX];
        }

        return getIntArrayOfArray(ctx, shapesVec);
    };

    return VPU::DistributionInfoAttr::get(
            ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(), nullptr, nullptr, nullptr,
            denseDataDistribution.getNumClusters(), denseDataDistribution.getAlignment(),
            denseDataDistribution.getUniformDistributedSegments(),
            getWeightsShapes(denseDataDistribution.getComputeShapes()), denseDataDistribution.getComputeOffsets(),
            getWeightsShapes(denseDataDistribution.getMemoryShapes()), denseDataDistribution.getMemoryOffsets(),
            denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributionInfoAttr vpux::VPU::getExplicitDistrAttrForSETable(VPU::DistributionInfoAttr denseDataDistribution,
                                                                    const size_t seSize, mlir::MLIRContext* ctx) {
    auto getSETableShapesOffsets = [&](mlir::ArrayAttr shapesOffsetsAttr,
                                       const bool isOffset = false) -> mlir::ArrayAttr {
        auto shapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(shapesOffsetsAttr);
        int64_t idx = 0;
        for (auto& shapesOffsets : shapesOffsetsVec) {
            // In cases where tensor is SEGMENTED over C, SETable depth per cluster must be adjusted
            if (seSize == 0) {
                if (VPU ::isSegmentedOverC(denseDataDistribution)) {
                    // SeSize is zero when multi seSizes are used for DWConv, which means for each cluster, the SE table
                    // depth is 1;
                    shapesOffsets[Dims4D::Act::C.ind()] = isOffset ? idx : 1;
                    idx++;
                } else {
                    shapesOffsets[Dims4D::Act::C.ind()] = isOffset ? 0 : static_cast<int64_t>(shapesOffsetsVec.size());
                }
            } else {
                shapesOffsets[Dims4D::Act::C.ind()] =
                        isOffset ? shapesOffsets[Dims4D::Act::C.ind()] / static_cast<int64_t>(seSize)
                                 : divUp(shapesOffsets[Dims4D::Act::C.ind()], static_cast<int64_t>(seSize));
            }
        }
        return getIntArrayOfArray(ctx, shapesOffsetsVec);
    };

    auto seTableAlignmentAttr = denseDataDistribution.getAlignment();
    if (seTableAlignmentAttr != nullptr) {
        auto seTableAlignment = parseIntArrayAttr<int64_t>(seTableAlignmentAttr);
        seTableAlignment[Dims4D::Act::C.ind()] = 1;
        seTableAlignmentAttr = getIntArrayAttr(ctx, seTableAlignment);
    }

    return VPU::DistributionInfoAttr::get(ctx, denseDataDistribution.getMode(), denseDataDistribution.getNumTiles(),
                                          nullptr, nullptr, nullptr, denseDataDistribution.getNumClusters(),
                                          seTableAlignmentAttr, denseDataDistribution.getUniformDistributedSegments(),
                                          getSETableShapesOffsets(denseDataDistribution.getComputeShapes()),
                                          getSETableShapesOffsets(denseDataDistribution.getComputeOffsets(), true),
                                          getSETableShapesOffsets(denseDataDistribution.getMemoryShapes()),
                                          getSETableShapesOffsets(denseDataDistribution.getMemoryOffsets(), true),
                                          denseDataDistribution.getEqualMemoryAndComputeView());
}

VPU::DistributionInfoAttr VPU::getExplicitDistrAttrForActualDataFromSparseType(mlir::Type origType) {
    VPUX_THROW_WHEN(!mlir::isa<VPU::DistributedTypeInterface>(origType),
                    "getExplicitDistrAttrForActualDataFromSparseType: type is not distributed");

    auto ctx = origType.getContext();

    auto getDistribution = [](mlir::Type componentType) -> DistributionInfoAttr {
        if (auto distributedTensor = mlir::dyn_cast<vpux::VPU::DistributedTensorType>(componentType)) {
            return distributedTensor.getDistribution();
        } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(componentType)) {
            return distributedBuffer.getDistribution();
        }

        VPUX_THROW("Sparse type's component is not distributed, component type = {0}", componentType);
    };

    auto patchDistributionChannels = [&](mlir::ArrayAttr data, mlir::ArrayAttr seTable) -> mlir::ArrayAttr {
        const auto dataShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(data);
        auto actualShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(seTable);

        std::transform(dataShapesOffsetsVec.begin(), dataShapesOffsetsVec.end(), actualShapesOffsetsVec.begin(),
                       actualShapesOffsetsVec.begin(),
                       [](const SmallVector<int64_t>& dataShapesOffsets, SmallVector<int64_t> actualShapesOffsets) {
                           actualShapesOffsets[Dims4D::Act::C.ind()] = dataShapesOffsets[Dims4D::Act::C.ind()];
                           return actualShapesOffsets;
                       });

        return getIntArrayOfArray(ctx, actualShapesOffsetsVec);
    };

    mlir::Type dataType;
    mlir::Type seTableType;
    if (auto sparseTensorType = mlir::dyn_cast<VPU::SparseTensorType>(origType)) {
        dataType = sparseTensorType.getData();
        seTableType = sparseTensorType.getStorageElementTable();
    } else if (auto sparseBufferType = mlir::dyn_cast<VPUIP::SparseBufferType>(origType)) {
        dataType = sparseBufferType.getData();
        seTableType = sparseBufferType.getStorageElementTable();
    } else {
        VPUX_THROW("Expected sparse type. Got {0}", origType);
    }

    const auto dataDistribution = getDistribution(dataType);

    VPUX_THROW_WHEN(!isDistributedAttrWithExplicitShapesAndOffsets(dataDistribution),
                    "Distribution for SparseType is not explicit, data distribution = {0}", dataDistribution);

    if (seTableType == nullptr) {
        return dataDistribution;
    }

    auto seTableDistribution = getDistribution(seTableType);
    mlir::ArrayAttr computeShapes =
            patchDistributionChannels(dataDistribution.getComputeShapes(), seTableDistribution.getComputeShapes());
    mlir::ArrayAttr computeOffsets =
            patchDistributionChannels(dataDistribution.getComputeOffsets(), seTableDistribution.getComputeOffsets());
    mlir::ArrayAttr memoryShapes =
            patchDistributionChannels(dataDistribution.getMemoryShapes(), seTableDistribution.getMemoryShapes());
    mlir::ArrayAttr memoryOffsets =
            patchDistributionChannels(dataDistribution.getMemoryOffsets(), seTableDistribution.getMemoryOffsets());

    return DistributionInfoAttr::get(ctx, seTableDistribution.getMode(), seTableDistribution.getNumTiles(), nullptr,
                                     nullptr, nullptr, seTableDistribution.getNumClusters(),
                                     seTableDistribution.getAlignment(),
                                     seTableDistribution.getUniformDistributedSegments(), computeShapes, computeOffsets,
                                     memoryShapes, memoryOffsets, seTableDistribution.getEqualMemoryAndComputeView());
}
