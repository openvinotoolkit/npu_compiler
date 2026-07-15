//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

//
// build
//

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::SliceOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SliceOpAdaptor sliceOp(operands, attrs, prop);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = mlir::dyn_cast<vpux::NDTypeInterface>(sliceOp.getInput().getType());
    if (origType == nullptr) {
        return errorAt(loc, "VPU::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    auto inferExplicitDistributedAttr = [&](VPU::DistributionInfoAttr origDistribution,
                                            ArrayRef<int64_t> inShape) -> VPU::DistributionInfoAttr {
        if (origDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, sliceShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, sliceShape, inShape,
                                                            origType.getElementType(), ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Slice Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }
        const auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);
        const auto zeroOffsets =
                SmallVector<SmallVector<int64_t>>(memoryShapes.size(), SmallVector<int64_t>(inShape.size(), 0));
        const auto perClusterOffsetsAttr = vpux::getIntArrayOfArray(ctx, zeroOffsets);

        return VPU::DistributionInfoAttr::get(
                ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(), perClusterShapesAttr,
                perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                origDistribution.getEqualMemoryAndComputeView(), origDistribution.getMemoryNumTiles());
    };

    const auto distributedIn = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(origType);
    VPU::DistributionInfoAttr possibleDistribution =
            distributedIn != nullptr && distributedIn.containsDistributedTypes()
                    ? mlir::cast<vpux::VPU::DistributedTensorType>(distributedIn.getDistributedTypes().front())
                              .getDistribution()
                    : nullptr;

    if (possibleDistribution != nullptr && VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
        if (auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(distributedIn)) {
            possibleDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistribution =
                VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(), ShapeRef(sliceShape), possibleDistribution);

        const auto sliceDistributedAttr = inferExplicitDistributedAttr(newDistribution, origType.getShape().raw());

        const auto newType = distributedIn.extractDenseTileForExplicitDistribution(
                ShapeRef(sliceOffsets), ShapeRef(sliceShape), sliceDistributedAttr);
        inferredTypes.emplace_back(newType);
    } else {
        const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
        inferredTypes.emplace_back(newType);
    }

    return mlir::success();
}

mlir::LogicalResult VPU::SliceOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                    mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    auto loc = getOperation()->getLoc();
    auto outputShapedType = llvm::cast<mlir::ShapedType>(getOutput().getType());
    for (int64_t dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (!outputShapedType.isDynamicDim(dim)) {
            // Static dim: Return IntegerAttr.
            shapes.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        } else {
            // Dynamic dim: Return Value.
            mlir::OpFoldResult ofr = builder.createOrFold<mlir::tensor::DimOp>(loc, getInput(), dim);
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, ofr));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult VPU::SliceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto origContent = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::SliceOp::verify() {
    const auto loc = getLoc();

    const auto inShape = getBoundedShape(getInput());
    const auto outShape = getBoundedShape(getOutput());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(getStaticOffsets());
    if (inShape.size() != outShape.size()) {
        return errorAt(loc, "Input shape '{0}' and output shape '{1}' must have the same rank", inShape, outShape);
    }
    if (inShape.size() != sliceOffsets.size()) {
        return errorAt(loc, "Input shape '{0}' and slice offsets '{1}' must have the same rank", inShape, sliceOffsets);
    }

    const auto sliceDims = IE::getDiffInOutSizeDims(inShape, outShape);

    for (auto idx : irange(inShape.size())) {
        auto dim = Dim(idx);
        auto isSliceDim = llvm::find(sliceDims, dim) != sliceDims.end();
        if (isSliceDim) {
            if (outShape[dim] + sliceOffsets[idx] > inShape[dim]) {
                return errorAt(loc,
                               "Slice offset '{0}' and output shape '{1}' exceed input shape '{2}' on dimension '{3}'",
                               sliceOffsets[idx], outShape[dim], inShape[dim], dim);
            }
        } else {
            if (inShape[dim] != outShape[dim]) {
                return errorAt(loc, "Input shape '{0}' and output shape '{1}' must match on non-sliced dimensions",
                               inShape, outShape);
            }
            if (sliceOffsets[idx] != 0) {
                return errorAt(loc, "Slice offset '{0}' for non-sliced dimension '{1}' must be zero", sliceOffsets[idx],
                               dim);
            }
        }
    }
    return mlir::success();
}

//
// ComposeSlice
//

namespace {

class ComposeSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto producerSliceOp = origOp.getInput().getDefiningOp<VPU::SliceOp>();
        if (producerSliceOp == nullptr) {
            return mlir::failure();
        }

        auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.getStaticOffsets());
        const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
        for (auto i : irange(finalOffsets.size())) {
            finalOffsets[i] += secondOffsets[i];
        }

        const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
        const auto finalShapeAttr = origOp.getStaticSizes();
        rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, producerSliceOp.getInput(), finalOffsetsAttr, finalShapeAttr);

        return mlir::success();
    }
};

// Remove redundant pairs of Expand->Slice operations which negate each other's effects For example:
//
// Case 1. Only expand at the end
// [1, 16, 1, 1]
//   -> Expand {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} -> [16, 16, 1, 1]
//   -> Slice  {offsets =    [0, 0, 0, 0], sizes =    [1, 16, 1, 1]} -> [1, 16, 1, 1]
//
// Case 2. Expand on both sides
// [1, 16, 1, 1]
//   -> Expand {pads_begin = [4, 0, 0, 0], pads_end = [15, 0, 0, 0]} -> [20, 16, 1, 1]
//   -> Slice  {offsets =    [4, 0, 0, 0], sizes =    [1, 16, 1, 1]} -> [1, 16, 1, 1]
class RemoveRedundantExpandSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final {
        auto expandOp = sliceOp.getInput().getDefiningOp<VPU::ExpandOp>();
        if (expandOp == nullptr) {
            return mlir::failure();
        }

        const auto origInputShape = getShape(expandOp.getInput());
        const auto origOutputShape = getShape(sliceOp.getOutput());
        if (origInputShape != origOutputShape) {
            return mlir::failure();
        }

        const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        for (size_t i = 0; i < origInputShape.size(); ++i) {
            if (sliceOffsets[i] - expandPadsBegin[i] != 0) {
                return mlir::failure();
            }
        }

        rewriter.replaceAllOpUsesWith(sliceOp, expandOp.getInput());

        return mlir::success();
    }
};

}  // namespace

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::SliceOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto inputShape = getShape(getInput());
    const auto outputShape = getShape(getOutput());
    const auto offsets = parseIntArrayAttr<int64_t>(getStaticOffsets());

    log.trace("SliceOp backInferTileInfo: inputShape={0}, outputTile={1}", inputShape, outputTile);
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile Slice operation at '{0}', which has operands with different rank", this->getLoc());

    // Example: input[32] -> slice(offset=8, size=16) -> output[16]
    //   Not tiled:               inputTile {shape=32, offset=0}  (full input dim)
    //   Tiled {shape=8, offset=0}: inputTile {shape=8,  offset=8}  (shifted by slice offset)
    //   Tiled {shape=8, offset=8}: inputTile {shape=8,  offset=16} (shifted by slice offset)
    TileInfo inputTile(inputShape.size());
    inputTile.axis = outputTile.axis;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        const auto isDimTiled = outputTile.shape[Dim(i)] != outputShape[Dim(i)];
        if (!isDimTiled) {
            // Not tiled: request the full input dimension regardless of whether the dim is sliced.
            // The Slice op will apply its original offset/size to extract the correct region.
            inputTile.shape[Dim(i)] = inputShape[Dim(i)];
            inputTile.offsets[Dim(i)] = 0;
        } else {
            // Tiled: request only the needed portion from the original input.
            // Shift output tile offset by the slice offset so that SubView extracts the
            // correct region directly. For non-sliced dims, offsets[i] is always 0.
            inputTile.shape[Dim(i)] = outputTile.shape[Dim(i)];
            inputTile.offsets[Dim(i)] = outputTile.offsets[Dim(i)] + offsets[i];
        }
    }

    log.trace("SliceOp backInferTileInfo: inputTile={0}", inputTile);
    return TilingInfo{{std::move(inputTile)}};
}

void vpux::VPU::SliceOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile, ShapeRef) {
    const auto outputShape = getShape(getOutput());
    const auto offsets = parseIntArrayAttr<int64_t>(getStaticOffsets());
    SmallVector<int64_t> tiledOffsets(outputTile.shape.size(), 0);
    for (size_t i = 0; i < outputTile.shape.size(); ++i) {
        const auto isDimTiled = outputTile.shape[Dim(i)] != outputShape[Dim(i)];
        // When tiled: SubView already positioned the input correctly, so offset becomes 0.
        // When not tiled: preserve the original slice offset (0 for non-sliced dims).
        tiledOffsets[i] = isDimTiled ? 0 : offsets[i];
    }
    setStaticOffsetsAttr(getIntArrayAttr(getContext(), tiledOffsets));
    setStaticSizesAttr(getIntArrayAttr(getContext(), outputTile.shape));
}

bool vpux::VPU::SliceOp::isSupportedTilingDim(DimArrRef tilingDims) {
    if (tilingDims.empty()) {
        return true;
    }

    const auto srcShape = getShape(getInput());
    const auto dstShape = getShape(getOutput());
    return llvm::none_of(tilingDims, [&](Dim dim) {
        return srcShape[dim] != dstShape[dim];
    });
}

bool vpux::VPU::SliceOp::isVFSupported() {
    if (!getInput().hasOneUse() || !getOutput().hasOneUse()) {
        return false;
    }
    auto parentOp = getInput().getDefiningOp();
    if (parentOp != nullptr && !VPU::opHasAccurateCost(parentOp)) {
        return false;
    }
    auto userOp = *getOutput().user_begin();
    return VPU::opHasAccurateCost(userOp);
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::SliceOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }

    const auto srcShape = inType.getShape();
    const auto dstType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
    const auto outShape = dstType.getShape();

    // Identify the dims that are reduced by the slice operation
    const auto sliceDims = IE::getDiffInOutSizeDims(srcShape, outShape);

    const auto mode = distribution.getDistributionMode();
    const bool isSegmented = VPU::bitEnumContainsAny(mode, VPU::DistributionMode::SEGMENTED);
    const bool isOverlapped = VPU::bitEnumContainsAny(mode, VPU::DistributionMode::OVERLAPPED);
    const bool isDuplicated = VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED);
    const bool isMulticasted = VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);

    // Guard: the slice must not reduce the memory-visible tiling axis.
    // For compound modes the memory tiling axis may differ from num_tiles (the compute axis).
    if (isSegmented && isOverlapped) {
        // SEGMENTED|OVERLAPPED: memory layout is OVERLAPPED, described by memory_num_tiles.
        const auto memNumTiles = distribution.getMemoryNumTiles();
        VPUX_THROW_UNLESS(memNumTiles.has_value(), "SEGMENTED|OVERLAPPED distribution requires memory_num_tiles");
        const auto memTilingDim = Dim(getDistributedTilingAxis(memNumTiles.value()));
        if (llvm::find(sliceDims, memTilingDim) != sliceDims.end()) {
            return mlir::failure();
        }
    } else if ((isSegmented || isOverlapped) && !isDuplicated && !isMulticasted) {
        // Pure SEGMENTED or pure OVERLAPPED: num_tiles defines the single tiling axis.
        // SEGMENTED|DUPLICATED / SEGMENTED|MULTICASTED are excluded: every cluster holds a full
        // copy in memory, so no tiling axis constrains propagation.
        const auto tilingDim = Dim(getDistributedTilingAxis(distribution.getNumTiles()));
        if (llvm::find(sliceDims, tilingDim) != sliceDims.end()) {
            return mlir::failure();
        }
    }

    const auto typeComponents = TypeComponents().setShape(outShape);
    const bool hasExplicit = VPU::isDistributionWithExplicitShapesAndOffsets(distribution);

    // Adjust alignment for the slice operation: scale down or drop alignment when the aligned
    // dimension is among the sliced dimensions, matching the logic in SliceOp::inferReturnTypes.
    auto* ctx = getOperation()->getContext();
    const auto distAttr = VPU::DistributionInfo::getAttrFromClass(ctx, distribution);
    const auto adjustedDistAttr = VPU::updateSliceLikeOpsAlignment(ctx, srcShape, outShape, distAttr);
    auto adjustedDistribution = VPU::DistributionInfo::getClassFromAttr(adjustedDistAttr);

    // Helper: reduce per-cluster shapes along the sliced dims to their output size.
    const auto slicePerClusterShapes =
            [&](ArrayRef<SmallVector<int64_t>> perClusterShapes) -> SmallVector<SmallVector<int64_t>> {
        SmallVector<SmallVector<int64_t>> newShapes(perClusterShapes.begin(), perClusterShapes.end());
        for (auto& shape : newShapes) {
            for (auto dim : sliceDims) {
                shape[dim.ind()] = outShape[dim];
            }
        }
        return newShapes;
    };

    // Cast ops operate on the memory view of the distributed tensor. For compound modes with a
    // broadcast-in-memory component (DUPLICATED or MULTICASTED), the SEGMENTED compute aspect is
    // irrelevant to downstream consumers. Reduce to plain DUPLICATED with no tiling parameters.
    if (isSegmented && (isDuplicated || isMulticasted)) {
        VPU::DistributionInfo outDistribution;
        if (!hasExplicit) {
            outDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED,
                                                    /*numTiles=*/{}, /*kernel=*/{}, /*strides=*/{},
                                                    /*padding=*/std::nullopt, distribution.getNumClusters(),
                                                    /*alignment=*/{}, distribution.hasUniformDistributedSegments(),
                                                    /*computeShapes=*/{}, /*computeOffsets=*/{},
                                                    /*memoryShapes=*/{}, /*memoryOffsets=*/{},
                                                    distribution.hasEqualMemoryAndComputeView(),
                                                    /*memoryNumTiles=*/std::nullopt);
        } else {
            // All clusters hold an identical full copy of the output tensor.
            const int64_t numClusters = distribution.getNumClusters();
            const SmallVector<int64_t> outShapeVec(outShape.begin(), outShape.end());
            const SmallVector<SmallVector<int64_t>> allSameShapes(numClusters, outShapeVec);
            const SmallVector<int64_t> zeroOff(outShape.size(), 0);
            const SmallVector<SmallVector<int64_t>> allZeroOffsets(numClusters, zeroOff);
            outDistribution = VPU::DistributionInfo(VPU::DistributionMode::DUPLICATED,
                                                    /*numTiles=*/{}, /*kernel=*/{}, /*strides=*/{},
                                                    /*padding=*/std::nullopt, numClusters,
                                                    /*alignment=*/{}, distribution.hasUniformDistributedSegments(),
                                                    allSameShapes, allZeroOffsets, allSameShapes, allZeroOffsets,
                                                    distribution.hasEqualMemoryAndComputeView(),
                                                    /*memoryNumTiles=*/std::nullopt);
        }
        return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)),
                              std::move(outDistribution));
    }

    // For SEGMENTED|OVERLAPPED, the memory view is OVERLAPPED. The SEGMENTED compute aspect is
    // dropped and memory_num_tiles is promoted to num_tiles for the pure OVERLAPPED output.
    if (isSegmented && isOverlapped) {
        const auto memNumTiles = distribution.getMemoryNumTiles();  // non-null: validated in guard above
        VPU::DistributionInfo outDistribution;
        if (!hasExplicit) {
            outDistribution = VPU::DistributionInfo(
                    VPU::DistributionMode::OVERLAPPED, memNumTiles.value(), distribution.getKernel(),
                    distribution.getStrides(), distribution.getPadding(), distribution.getNumClusters(),
                    adjustedDistribution.getAlignment(), distribution.hasUniformDistributedSegments(),
                    /*computeShapes=*/{}, /*computeOffsets=*/{},
                    /*memoryShapes=*/{}, /*memoryOffsets=*/{}, distribution.hasEqualMemoryAndComputeView(),
                    /*memoryNumTiles=*/std::nullopt);
        } else {
            // Apply the slice to the OVERLAPPED memory shapes (halos on the tiling dim are preserved;
            // only the sliced dims are narrowed). Compute and memory shapes are stored explicitly.
            const auto newMemShapes = slicePerClusterShapes(distribution.getMemoryShapes());
            outDistribution = VPU::DistributionInfo(
                    VPU::DistributionMode::OVERLAPPED, memNumTiles.value(), distribution.getKernel(),
                    distribution.getStrides(), distribution.getPadding(), distribution.getNumClusters(),
                    adjustedDistribution.getAlignment(), distribution.hasUniformDistributedSegments(), newMemShapes,
                    distribution.getMemoryOffsets(), newMemShapes, distribution.getMemoryOffsets(),
                    distribution.hasEqualMemoryAndComputeView(),
                    /*memoryNumTiles=*/std::nullopt);
        }
        return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)),
                              std::move(outDistribution));
    }

    // Non-compound mode
    if (!hasExplicit) {
        return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)),
                              std::move(adjustedDistribution));
    }

    adjustedDistribution.setMemoryShapes(slicePerClusterShapes(distribution.getMemoryShapes()));
    adjustedDistribution.setComputeShapes(slicePerClusterShapes(distribution.getComputeShapes()));
    return std::make_pair(mlir::cast<mlir::Type>(inType.changeTypeComponents(typeComponents)),
                          std::move(adjustedDistribution));
}

//
// getCanonicalizationPatterns
//

void vpux::VPU::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSlice>(ctx);
    results.add<RemoveRedundantExpandSlice>(ctx);
}
