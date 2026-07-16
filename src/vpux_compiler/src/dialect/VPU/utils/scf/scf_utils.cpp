//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

#include <iterator>
#include <numeric>

using namespace vpux;
using namespace VPU;

int64_t vpux::VPU::getTilePositionShift(vpux::Dim dim) {
    // Point encoding packs one TilePosition (2 bits) per dimension into a single integer:
    // bits [7:6] = N,  bits [5:4] = C,  bits [3:2] = H,  bits [1:0] = W
    constexpr std::array<int64_t, 4> shifts = {6, 4, 2, 0};
    VPUX_THROW_WHEN(dim.ind() < 0 || static_cast<size_t>(dim.ind()) >= shifts.size(),
                    "NCHW point/range encoding only supports N/C/H/W (dim indices 0-3), got dim {0}", dim.ind());
    return shifts[dim.ind()];
}

mlir::Value vpux::VPU::encodePointPosition(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::Value> values,
                                           ArrayRef<mlir::Value> shiftAmounts) {
    VPUX_THROW_WHEN(values.empty(), "encodePointPosition requires at least one value");
    VPUX_THROW_WHEN(values.size() != shiftAmounts.size(),
                    "encodePointPosition: values and shiftAmounts must have the same size ({0} vs {1})", values.size(),
                    shiftAmounts.size());
    auto index = builder.create<mlir::arith::ShLIOp>(loc, values[0], shiftAmounts[0]).getResult();
    for (auto [val, shiftAmt] : llvm::zip(values.drop_front(), shiftAmounts.drop_front())) {
        auto shifted = builder.create<mlir::arith::ShLIOp>(loc, val, shiftAmt).getResult();
        index = builder.create<mlir::arith::OrIOp>(loc, index, shifted).getResult();
    }
    return index;
}

int64_t vpux::VPU::getEncodedPointPosition(ArrayRef<std::pair<vpux::Dim, TilePosition>> dimPositions) {
    int64_t encoded = 0;
    for (auto [dim, pos] : dimPositions) {
        encoded |= (static_cast<int64_t>(pos) << getTilePositionShift(dim));
    }
    return encoded;
}

TilePosition vpux::VPU::getDimPosition(int64_t pointEncoded, vpux::Dim dim) {
    const int64_t mask = (1LL << NUMBITS) - 1;
    return static_cast<TilePosition>((pointEncoded >> getTilePositionShift(dim)) & mask);
}

int64_t vpux::VPU::setDimPosition(int64_t pointEncoded, vpux::Dim dim, TilePosition pos) {
    const int64_t mask = (1LL << NUMBITS) - 1;
    auto shift = getTilePositionShift(dim);
    pointEncoded &= ~(mask << shift);
    pointEncoded |= (static_cast<int64_t>(pos) << shift);
    return pointEncoded;
}

SmallVector<TilePosition> vpux::VPU::decodePointPosition(int64_t encoded, ArrayRef<vpux::Dim> dims) {
    SmallVector<TilePosition> positions;
    positions.reserve(dims.size());
    const int64_t mask = (1LL << NUMBITS) - 1;
    for (auto dim : dims) {
        positions.push_back(static_cast<TilePosition>((encoded >> getTilePositionShift(dim)) & mask));
    }
    return positions;
}

int64_t vpux::VPU::encodeRangePosition(vpux::Dim dim, int64_t currentValue, TilePosition start, TilePosition end) {
    auto baseShift = getTilePositionShift(dim) * 2;
    currentValue |= (static_cast<int64_t>(start) << baseShift);
    currentValue |= (static_cast<int64_t>(end) << (baseShift + static_cast<int64_t>(NUMBITS)));
    return currentValue;
}

std::pair<TilePosition, TilePosition> vpux::VPU::decodeRangePosition(int64_t rangeEncoded, vpux::Dim dim) {
    const int64_t mask = (1LL << NUMBITS) - 1;
    auto baseShift = getTilePositionShift(dim) * 2;
    auto start = static_cast<TilePosition>((rangeEncoded >> baseShift) & mask);
    auto end = static_cast<TilePosition>((rangeEncoded >> (baseShift + static_cast<int64_t>(NUMBITS))) & mask);
    return {start, end};
}

mlir::LogicalResult vpux::VPU::getResultTileBounds(mlir::Operation* operation, unsigned resultNumber,
                                                   DimArrRef tilingDims, ArrayRef<mlir::OpFoldResult> sizes,
                                                   Bounds& resultBounds) {
    if (tilingDims.empty() || sizes.empty()) {
        return mlir::failure();
    }

    auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(resultNumber).getType());
    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(outputType);
    if (boundedType == nullptr) {
        return mlir::failure();
    }

    resultBounds = boundedType.getBounds().toValues();

    mlir::tensor::ExtractSliceOp extractSliceOp = nullptr;
    Logger log = Logger::global().nest("result_tile_bounds");
    for (auto* user : operation->getUsers()) {
        if (!mlir::isa<mlir::tensor::ExtractSliceOp>(user)) {
            continue;
        }
        if (extractSliceOp != nullptr) {
            log.error("Multiple extract slices in users of op {0}", operation);
            return mlir::failure();
        }
        extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(user);
    }

    if (extractSliceOp == nullptr) {
        OpChainAnalysis affineUtils;

        for (auto dim : tilingDims) {
            llvm::DenseMap<mlir::Value, SmallVector<int64_t>> valueMap;
            auto sizeValue = affineUtils.getOpFoldResultValue(sizes[dim.ind()], valueMap);

            if (!sizeValue.has_value() || sizeValue.value().empty()) {
                continue;
            }
            resultBounds[dim] = sizeValue.value().front();
        }
        return mlir::success();
    }

    auto boundedSliceType = mlir::dyn_cast<Core::BoundedTensorType>(extractSliceOp.getType());
    if (boundedSliceType == nullptr) {
        return mlir::failure();
    }

    resultBounds = boundedSliceType.getBounds().toValues();
    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::getDimValue(mlir::OpBuilder& builder, mlir::Operation* operation, int64_t dim) {
    const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());

    mlir::ReifiedRankedShapedTypeDims resultShape;
    if (mlir::failed(reifyResultShapes(builder, operation, resultShape))) {
        return builder.getIndexAttr(outputType.getDimSize(dim));
    }

    return resultShape[0][dim];
}

mlir::Value vpux::VPU::generateTile(mlir::Location loc, mlir::OpBuilder& builder, mlir::Value origInput,
                                    const SCFTileInfo& inputTileInfo, SmallVector<mlir::Operation*>& generatedSlices) {
    auto origType = mlir::cast<vpux::NDTypeInterface>(origInput.getType());

    auto staticNewShape = mlir::getConstantIntValues(inputTileInfo.shape);
    if (origType.getShape().isStatic() && staticNewShape.has_value() &&
        llvm::equal(origType.getShape().raw(), staticNewShape.value())) {
        return origInput;
    }

    SmallVector<mlir::OpFoldResult> defaultStrides(inputTileInfo.offsets.size(), builder.getIndexAttr(1));

    auto extractTile = builder.create<mlir::tensor::ExtractSliceOp>(
            appendLoc(loc, "extractSlice"), origInput, inputTileInfo.offsets, inputTileInfo.shape, defaultStrides);

    auto newShape = getShape(extractTile.getResult());

    /*For cases where we cannot evenly divide tensor with static shapes into pieces of equal size, SCF tiling
     * generates dynamic shapes for extract and pad from static shapes. For such cases setting the bounds here
     * based on the original input shape. For other cases bounds are inferred from available input bounds*/

    auto newType = origType.changeShape(ShapeRef(newShape));
    if (newShape.isDynamic()) {
        if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
            // `changeBounds` expects non-empty bounds and will assert otherwise.
            // Keep current bounds when tile bounds were not inferred.
            if (!inputTileInfo.bounds.raw().empty()) {
                newType = boundedType.changeBounds(inputTileInfo.bounds);
            }
        } else {
            auto origInputShape = to_small_vector(getShape(origInput));
            SmallVector<int64_t, 4> boundsValue(newShape.begin(), newShape.end());
            for (size_t ind = 0, sz = boundsValue.size(); ind < sz; ++ind) {
                if (boundsValue[ind] == mlir::ShapedType::kDynamic) {
                    boundsValue[ind] = origInputShape[ind];
                }
            }
            auto newBounds = vpux::BoundsRef(ArrayRef<int64_t>(boundsValue.data(), boundsValue.size()));
            newType = Core::BoundedTensorType::get(newType, newBounds);
        }
    }

    // by default output type loses NPU-specific attributes so we have to set it manually
    extractTile->getResult(0).setType(newType);

    generatedSlices.emplace_back(extractTile);

    return extractTile;
}

mlir::Type vpux::VPU::extractResultType(mlir::Type origType, SCFShapeRef newShape, BoundsRef bounds) {
    auto ndTensorType = mlir::cast<vpux::NDTypeInterface>(origType);
    auto origElemType = ndTensorType.getElementType();

    VPUX_THROW_WHEN(mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(origElemType),
                    "Per axis quantized types are not supported in scf");

    const auto tensorDesc =
            vpux::getTensorAttr(origElemType.getContext(), ndTensorType.getDimsOrder(), ndTensorType.getMemSpace(),
                                mlir::isa<Core::BoundedTensorType>(origType) ? bounds : BoundsRef{});

    SmallVector<mlir::Value> dynamicDims;  // unused cause for shape static dims are enough
    SmallVector<int64_t> staticDims;
    mlir::dispatchIndexOpFoldResults(newShape, dynamicDims, staticDims);
    return mlir::RankedTensorType::get(staticDims, origElemType, tensorDesc);
}

SCFTileInfo vpux::VPU::getWeightsTableSCFTile(mlir::Type origWeightsTableType, mlir::OpBuilder& builder,
                                              const SCFTileInfo& outputTile) {
    auto origWeightsTableShape = mlir::cast<mlir::ShapedType>(origWeightsTableType).getShape();

    SCFTileInfo weightsTableTile(origWeightsTableShape, builder);
    weightsTableTile.offsets[0] = outputTile.offsets[Dims4D::Act::C.ind()];
    weightsTableTile.shape[0] = outputTile.shape[Dims4D::Act::C.ind()];
    return weightsTableTile;
}

mlir::AffineMap vpux::VPU::getAlignValUpMap(mlir::OpBuilder& builder, int64_t alignment) {
    mlir::AffineExpr dimC;
    bindDims(builder.getContext(), dimC);
    auto alignmentExpr = builder.getAffineConstantExpr(alignment);
    auto oneExpr = builder.getAffineConstantExpr(1);

    // Expression: ((dimC + alignment - 1) floordiv alignment) * alignment
    mlir::AffineExpr alignedExpr = ((dimC + alignmentExpr - oneExpr).floorDiv(alignmentExpr)) * alignmentExpr;
    return mlir::AffineMap::get(1, 0, {alignedExpr}, builder.getContext());
}

std::pair<std::optional<mlir::Range>, std::optional<int64_t>> vpux::VPU::solutionForOutputRange(
        mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile, Dim dim, const int64_t kernel,
        const int64_t stride, mlir::OpFoldResult origInputSize, int64_t origOutputSize,
        const std::pair<int64_t, int64_t>& origPadding, mlir::OpFoldResult& padBefore, mlir::OpFoldResult& padAfter) {
    auto zero = builder.getIndexAttr(0);
    auto one = builder.getIndexAttr(1);
    mlir::Range inputRange = {zero, zero, one};
    mlir::Range outputRange = {outputTile.offsets[dim.ind()], outputTile.shape[dim.ind()], one};

    // define dimensions (d0, d1, ...) as variables which are represented by loop dim identifier
    // and symbols (s0, s1, ...) which are either known constants or known attributes of operation
    mlir::AffineExpr s0, s1, d0, d1, d2;
    bindDims(builder.getContext(), d0, d1, d2);
    bindSymbols(builder.getContext(), s0, s1);

    mlir::AffineExpr requiredInputSizeExpr = (d0 - 1) * stride + kernel;
    auto requiredInputSizeMap = mlir::AffineMap::get(1, 0, {requiredInputSizeExpr}, builder.getContext());

    std::optional<int64_t> dimBound;
    if (!outputTile.bounds.raw().empty()) {
        auto outputTileBound = builder.getIntegerAttr(builder.getIndexType(), outputTile.bounds[dim]);
        SmallVector<mlir::Attribute> resultsAttrs;
        if (requiredInputSizeMap.constantFold({outputTileBound}, resultsAttrs).succeeded()) {
            if (auto result = mlir::dyn_cast<mlir::IntegerAttr>(resultsAttrs.front())) {
                dimBound = result.getInt() - origPadding.first - origPadding.second;
            }
        }
    }

    if (mlir::isConstantIntValue(outputTile.axis[dim.ind()], 1)) {
        return {std::nullopt, dimBound};
    }

    const auto hasPadBefore = origPadding.first != 0;
    const auto hasPadAfter = origPadding.second != 0;

    // input offset is based on output tile offset and operation's parameters
    // current calculation is
    // offset: max((output offset) * stride - padding, 0).
    // size: (output size - 1) * stride + kernel - padding
    // if operation has padding, the median tile size will be corrected later if needed
    if (!hasPadBefore && stride == 1) {
        inputRange.offset = outputRange.offset;
    } else {
        mlir::AffineExpr offsetExpr = d0 * stride - origPadding.first;
        auto offsetMap = mlir::AffineMap::get(1, 1, {offsetExpr, s0}, builder.getContext());
        inputRange.offset = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "inputOffset"), offsetMap,
                                                                      {outputRange.offset, zero});

        auto maxDiffMap = mlir::AffineMap::get(1, 2, {s0 - offsetExpr, s1}, builder.getContext());
        auto maxDiffValue = mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                                                                      {outputRange.offset, zero, zero});
        auto padBeforeMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padBefore = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingBefore"), padBeforeMap,
                                                              {maxDiffValue, builder.getIndexAttr(origPadding.first)});
    }

    if (hasPadAfter) {
        auto maxDiffMap = mlir::AffineMap::get(2, 2, {d1 + requiredInputSizeExpr - s0, s1}, builder.getContext());
        auto maxDiffValue =
                mlir::affine::makeComposedFoldedAffineMax(builder, appendLoc(loc, "maxDiff"), maxDiffMap,
                                                          {outputRange.size, inputRange.offset, origInputSize, zero});
        auto padAfterMap = mlir::AffineMap::get(0, 2, {s0, s1}, builder.getContext());
        padAfter = mlir::affine::makeComposedFoldedAffineMin(builder, appendLoc(loc, "paddingAfter"), padAfterMap,
                                                             {maxDiffValue, builder.getIndexAttr(origPadding.second)});
    }

    const auto numTiles = mlir::getConstantIntValue(outputTile.axis[dim.ind()]);
    const bool hasTwoTiles = numTiles.has_value() && numTiles.value() == 2;
    const bool outputSizeDivisible = numTiles.has_value() && !mlir::ShapedType::isDynamic(origOutputSize) &&
                                     origOutputSize % numTiles.value() == 0;
    const bool symmetricPadding = origPadding.first == origPadding.second;
    if (hasTwoTiles && outputSizeDivisible && symmetricPadding) {
        auto sizeMap = mlir::AffineMap::get(1, 1, {requiredInputSizeExpr - s0}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, builder.getIndexAttr(origPadding.first)});
    } else {
        auto sizeMap = mlir::AffineMap::get(3, 0, {requiredInputSizeExpr - d1 - d2}, builder.getContext());
        inputRange.size = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputSize"), sizeMap,
                {outputRange.size, hasPadBefore ? padBefore : zero, hasPadAfter ? padAfter : zero});
    }

    return {inputRange, dimBound};
}

std::optional<vpux::Dim> vpux::VPU::getSinglePaddedExpandDim(mlir::Operation* op) {
    auto expandOp = mlir::dyn_cast_or_null<VPU::ExpandOp>(op);
    if (expandOp == nullptr) {
        return std::nullopt;
    }
    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    if (padsBegin.size() != padsEnd.size() || padsBegin.empty()) {
        return std::nullopt;
    }
    // Only end-padding is supported (matches all real VPU.Expand workloads:
    // channel alignment to 16, H/W canonicalization). `pads_begin != 0` would
    // require offset arithmetic not covered by the SCF Expand tiling model.
    for (auto v : padsBegin) {
        if (v != 0) {
            return std::nullopt;
        }
    }
    std::optional<size_t> paddedIdx;
    for (size_t i = 0; i < padsEnd.size(); ++i) {
        if (padsEnd[i] == 0) {
            continue;
        }
        if (padsEnd[i] < 0) {
            return std::nullopt;
        }
        if (paddedIdx.has_value()) {
            return std::nullopt;
        }
        paddedIdx = i;
    }
    if (!paddedIdx.has_value()) {
        // Zero-pad Expand is a no-op; defer to canonical folding and generic handling.
        return std::nullopt;
    }
    // Padded dim must be statically shaped: callers emit it as a constant extent.
    const auto inputShape = getShape(expandOp.getInput());
    if (*paddedIdx >= inputShape.size() || inputShape[vpux::Dim(*paddedIdx)] == mlir::ShapedType::kDynamic) {
        return std::nullopt;
    }
    return vpux::Dim(*paddedIdx);
}

bool vpux::VPU::checkFusion(mlir::OpOperand& consumer, mlir::OpResult producerCandidate,
                            const llvm::SetVector<mlir::Operation*>& producers, const MergeConfiguration& mergeConfig) {
    // TODO E-172888 rewrite unified code for checking compatibility with current VF

    auto* producer = producerCandidate.getOwner();

    if (!mlir::isa<mlir::TilingInterface>(producer)) {
        return false;
    }

    if (VPU::isPureViewOp(producer)) {
        return mergeConfig.viewLikePolicy(producer);
    }

    // so far no checks for consumer with view like ops
    // MC strategies alignment should be brought here and E-218728 should be resolved.
    if (VPU::isPureViewOp(consumer.getOwner())) {
        return true;
    }

    // Single-dim end-padded Expand can be fused as an SCF VF producer, but isn't
    // in `VPU::isPureViewOp` (which is used by other code paths — cost model,
    // copy optimization, strategy assignment — that need Expand's non-view behavior).
    // Accept it here explicitly instead of widening `isPureViewOp`.
    if (getSinglePaddedExpandDim(producer).has_value()) {
        return true;
    }

    auto producerShape = getShape(producerCandidate);
    auto alignment = getAlignment(producer, producerShape, producerShape);
    const auto opAddsComputationalCost = [](auto* operation) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation);

        return nceOp != nullptr && llvm::any_of(nceOp.getKernelSizeVal(), [](int64_t k) {
                   return k > 1;
               });
    };
    if (llvm::any_of(producers, opAddsComputationalCost)) {
        if (alignment[Dims4D::Act::W.ind()] > 1 || alignment[Dims4D::Act::H.ind()] > 1) {
            return false;
        }
    }

    const auto hasMCStategy = [](mlir::Operation* operation) {
        auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    auto consumerHasStrategy = hasMCStategy(consumer.getOwner());
    auto producerHasStrategy = hasMCStategy(producer);

    if (!consumerHasStrategy && !producerHasStrategy) {
        return true;
    }

    if (consumerHasStrategy ^ producerHasStrategy) {
        return false;
    }

    auto producerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(producerCandidate.getOwner());
    auto consumerClusterOp = mlir::cast<VPU::ClusteredOpInterface>(consumer.getOwner());

    VPU::SiblingOpsAnalysis siblingAnalisys(consumer.getOwner());

    auto consumerDistrType = mlir::cast<VPU::DistributedTensorType>(
            consumerClusterOp.getDistributedTypeForOpOperand(consumer, false, siblingAnalisys));
    auto producerDistrType = mlir::cast<VPU::DistributedTensorType>(producerClusterOp.getDistributedTypeForOpResult(
            producerCandidate, producerClusterOp.getMultiClusterStrategy().value(), siblingAnalisys, false));

    return areDistributionAttrsCompatible(producerDistrType, consumerDistrType, true).succeeded();
}

bool vpux::VPU::isNceOpWithPadAttr(mlir::Operation* op) {
    return (mlir::isa<mlir::TilingInterface>(op) && mlir::isa<VPU::NCEOpInterface>(op) && op->hasAttr("pad"));
}

/**
 * @brief Converts tensor.extract_slice operation to VPU.Slice operation
 *
 * This function replaces a tensor.extract_slice operation with a VPU.Slice operation when
 * all offsets, sizes, and strides are constant values and all strides equal 1.
 *
 * @param extractSliceOp The extract_slice operation to convert
 * @param builder OpBuilder for creating new operations
 * @param mapper IRMapping for operand lookup
 * @return The newly created VPU.SliceOp or nullptr if conversion not applicable
 */
static mlir::Operation* convertStaticExtractSliceToVPUSlice(mlir::tensor::ExtractSliceOp extractSliceOp,
                                                            mlir::OpBuilder& builder, mlir::IRMapping& mapper) {
    // Helper lambda to extract constant values from OpFoldResults
    auto extractConstantValues = [&](ArrayRef<mlir::OpFoldResult> results) -> std::optional<SmallVector<int64_t>> {
        SmallVector<int64_t> values;
        for (auto result : results) {
            auto intValue = mlir::getConstantIntValue(result);
            if (!intValue.has_value()) {
                return std::nullopt;
            }
            values.push_back(intValue.value());
        }
        return values;
    };

    // Try to extract constant offsets and sizes
    auto cstOffsets = extractConstantValues(extractSliceOp.getMixedOffsets());
    if (!cstOffsets.has_value()) {
        return nullptr;
    }

    auto cstSizes = extractConstantValues(extractSliceOp.getMixedSizes());
    if (!cstSizes.has_value()) {
        return nullptr;
    }

    bool allStridesOne = llvm::all_of(extractSliceOp.getMixedStrides(), [](auto stride) {
        auto intValue = mlir::getConstantIntValue(stride);
        return intValue.has_value() && intValue.value() == 1;
    });

    if (!allStridesOne) {
        return nullptr;
    }

    // Replace with VPU.Slice using static result type
    auto sliceOp = builder.create<VPU::SliceOp>(
            extractSliceOp.getLoc(), mapper.lookupOrDefault(extractSliceOp.getSource()),
            builder.getI64ArrayAttr(cstOffsets.value()), builder.getI64ArrayAttr(cstSizes.value()));
    return sliceOp;
}

/**
 * @brief Clones an MLIR operation with ability to remap operations
 * based on specific cases.
 *
 * This function takes an existing MLIR operation and creates a clone of it using the provided
 * OpBuilder. It allows for remapping of certain operations based on predefined cases. For example,
 * if the operation is a tensor::CastOp, it is replaced with a VPU::SliceOp with appropriate parameters.
 * For all other operations, a standard clone is performed using the provided IRMapping.
 * @param oldOp The original MLIR operation to be cloned
 * @param builder The OpBuilder used to create the new operation
 * @param mapper An IRMapping used for cloning operations
 * @return A pointer to the newly created MLIR operation
 */
mlir::Operation* cloneOperationMapped(mlir::Operation& oldOp, mlir::OpBuilder& builder, mlir::IRMapping& mapper) {
    return llvm::TypeSwitch<mlir::Operation&, mlir::Operation*>(oldOp)
            .Case<mlir::tensor::ExtractSliceOp>([&](mlir::tensor::ExtractSliceOp extractSliceOp) {
                return convertStaticExtractSliceToVPUSlice(extractSliceOp, builder, mapper);
            })
            .Default([&builder, &mapper](mlir::Operation& defaultOp) -> mlir::Operation* {
                return builder.clone(defaultOp, mapper);
            });
};

/**
 * @brief Clones an existing MLIR function operation with optional type modification
 *
 * This function creates a deep copy of an MLIR function operation, allowing for optional
 * function type changes. It performs a complete clone of the function body, including
 * all blocks, operations, and value mappings. When a new function type is provided,
 * it also performs type inference for VPU dialect operations and updates the function's
 * return types based on the terminator operations.
 *
 * @param originalFunc The source function operation to be cloned
 * @param newName The name for the newly created function
 * @param moduleOp The target module where the new function will be inserted
 * @param newFuncType Optional new function type for the cloned function. If nullptr,
 *                    the original function's type is preserved
 *
 * @return The newly created function operation that is a clone of the original
 *
 * @note The cloned function is positioned immediately after the original function
 *       in the module. When newFuncType is provided, VPU dialect operations undergo
 *       shape inference and the function's return types are updated based on the
 *       actual terminator operand types.
 */
mlir::func::FuncOp vpux::VPU::cloneFuncOp(mlir::func::FuncOp originalFunc, const std::string& newName,
                                          mlir::FunctionType newFuncType) {
    assert(newFuncType != nullptr && "newFuncType should be a valid FunctionType");
    auto moduleOp = vpux::getModuleOp(originalFunc);
    mlir::OpBuilder builder(moduleOp.getContext());
    auto newFunc = builder.create<mlir::func::FuncOp>(takeOpLoc(originalFunc, newName), newName, newFuncType);
    moduleOp.push_back(newFunc);
    newFunc->moveAfter(originalFunc);

    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    for (auto& oldBlock : originalFunc.getBody()) {
        auto* newBlock = newFunc.addEntryBlock();
        for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
            oldToNewMap[oldArg] = newArg;
        }

        builder.setInsertionPointToStart(newBlock);
        for (auto& oldOp : oldBlock.getOperations()) {
            mlir::IRMapping mapper;
            for (auto operand : oldOp.getOperands()) {
                mapper.map(operand, oldToNewMap[operand]);
            }

            if (mlir::isa<mlir::tensor::CastOp>(oldOp)) {
                auto srcOperation = oldOp.getOperand(0).getDefiningOp();
                if (srcOperation != nullptr && mlir::isa<mlir::tensor::ExtractSliceOp, VPU::SliceOp>(srcOperation)) {
                    oldToNewMap[oldOp.getResult(0)] = mapper.lookupOrDefault(oldOp.getOperand(0));
                    continue;
                }
            }

            auto newOp = cloneOperationMapped(oldOp, builder, mapper);
            VPUX_THROW_WHEN(newOp == nullptr, "Cloning operation {0} failed", oldOp.getName());

            if (mlir::isa<VPU::VPUDialect>(newOp->getDialect())) {
                vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
            }

            // Unique to SliceOp as we cannot modify the static size attribute in the inferReturnType interface
            // TODO: will be removed by E#198282
            if (mlir::isa<VPU::SliceOp>(newOp)) {
                auto sliceOp = mlir::cast<VPU::SliceOp>(newOp);
                auto newShape = mlir::cast<mlir::ShapedType>(sliceOp.getOutput().getType()).getShape();
                sliceOp.setStaticSizesAttr(getIntArrayAttr(newOp->getContext(), newShape));
            }

            // Map results only if we didn't already map them (e.g., in extract_slice handling)
            bool alreadyMapped = oldOp.getNumResults() > 0 && oldToNewMap.contains(oldOp.getResult(0));
            if (!alreadyMapped) {
                for (auto [oldResult, newResult] : llvm::zip(oldOp.getResults(), newOp->getResults())) {
                    oldToNewMap[oldResult] = newResult;
                }
            }
        }
    }

    SmallVector<mlir::Type> newReturnTypes;
    for (auto& block : newFunc.getBody()) {
        auto terminatorOp = block.getTerminator();
        for (auto operand : terminatorOp->getOperands()) {
            newReturnTypes.push_back(operand.getType());
        }
    }

    auto modifiedType =
            mlir::FunctionType::get(newFunc.getContext(), newFunc.getFunctionType().getInputs(), newReturnTypes);
    newFunc.setType(modifiedType);
    return newFunc;
}

mlir::RankedTensorType vpux::VPU::removeBoundsAttr(mlir::RankedTensorType type) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    const auto tensorType = vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(),
                                                ndType.getMemSpace(), {}, {});
    return tensorType;
}

// Check if all operands of op are defined before 'beforeOp' and op has no side effects
bool isSafeToMoveBefore(mlir::Operation* op, mlir::Operation* beforeOp) {
    // Check SSA dependencies
    for (auto operand : op->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
            if (definingOp->getBlock() != op->getBlock()) {
                continue;
            }

            if (!definingOp->isBeforeInBlock(beforeOp)) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Moves affine and arithmetic operations early in the execution order within a block
 *
 * This function optimizes the placement of affine dialect and arithmetic dialect operations
 * by moving them as early as possible in the block's execution order, while maintaining
 * correctness. It also handles scf.if operations that return only index types by treating
 * them similarly to affine/arithmetic operations.
 *
 * The function performs a two-phase operation:
 * 1. Identifies all affine/arithmetic operations and scf.if operations that return only
 *    index types as candidates for early movement
 * 2. For each candidate operation, finds the earliest safe position in the block where
 *    it can be moved without violating dependencies
 *
 * @param block The MLIR block within which to perform the optimization
 */
void vpux::VPU::moveAffineArithOpsEarly(mlir::Block& block) {
    SmallVector<mlir::Operation*> affineArithOps;

    // Collect affine/arithmetic operations and qualifying scf.if operations
    for (auto& op : block) {
        if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(op.getDialect())) {
            affineArithOps.push_back(&op);
            continue;
        }

        if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
            // Only move scf.if operations that return only index types
            bool allIndexTypes = llvm::all_of(ifOp.getResultTypes(), [](mlir::Type type) {
                return mlir::isa<mlir::IndexType>(type);
            });
            if (allIndexTypes) {
                affineArithOps.push_back(&op);
            }
        }
    }

    // Move operations to earliest safe positions
    for (auto* op : affineArithOps) {
        mlir::Operation* targetPosition = nullptr;

        // Find the earliest position where this operation can be safely moved
        for (auto& candidate : block) {
            if (&candidate == op) {
                break;  // Don't move past current position
            }

            // Skip other affine/arithmetic operations - we want to move before them
            if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(candidate.getDialect())) {
                continue;
            }

            // Check if it's safe to move before this candidate
            if (isSafeToMoveBefore(op, &candidate)) {
                targetPosition = &candidate;
                break;
            }
        }

        // Only move if we found a valid earlier position
        if (targetPosition != nullptr) {
            op->moveBefore(targetPosition);
        }
    }
}

void vpux::VPU::addCheckForBlockSize(mlir::OpBuilder& builder, mlir::tensor::DimOp dimOp, mlir::Value blockSize,
                                     mlir::func::FuncOp funcOp, llvm::StringRef errorMsg) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto blockSizeDefiningOp = blockSize.getDefiningOp();
    mlir::Location loc = mlir::UnknownLoc::get(builder.getContext());
    if (dimOp->getBlock() == blockSizeDefiningOp->getBlock()) {
        if (dimOp->isBeforeInBlock(blockSizeDefiningOp)) {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        }
    } else {
        mlir::DominanceInfo dom(funcOp);
        if (dom.dominates(blockSizeDefiningOp, dimOp)) {
            builder.setInsertionPointAfter(dimOp);
            loc = appendLoc(dimOp->getLoc(), "tile0_check");
        } else {
            builder.setInsertionPointAfter(blockSizeDefiningOp);
            loc = appendLoc(blockSizeDefiningOp->getLoc(), "tile0_check");
        }
    }

    auto stepGreaterThanBound =
            builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, dimOp, blockSize);
    builder.create<mlir::cf::AssertOp>(takeOpLoc(stepGreaterThanBound, "assert"), stepGreaterThanBound, errorMsg);
}

mlir::LogicalResult vpux::VPU::getTensorDimOpFromIndex(mlir::OpBuilder& builder, mlir::Value tensor, int64_t dimIdx,
                                                       mlir::tensor::DimOp& dimOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    bool tensorIsBlockArg = false;
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        tensorIsBlockArg = true;
        while (blockParentOp != nullptr) {
            if (mlir::isa<mlir::func::FuncOp>(blockParentOp)) {
                break;
            }

            auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockParentOp);
            if (forOp == nullptr) {
                return vpux::errorAt(blockParentOp->getLoc(),
                                     "Expected parent scf.for operation for block argument but got {0}",
                                     blockParentOp->getName());
            }

            unsigned idx = blockArg.getArgNumber() - 1;
            mlir::Value initVal = forOp.getInitArgs()[idx];
            auto definingOp = initVal.getDefiningOp();
            if (definingOp != nullptr) {
                tensor = initVal;
                tensorIsBlockArg = false;
                break;
            }

            blockParentOp = blockParentOp->getParentOp();
        }
    }

    if (tensorIsBlockArg) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor);
        auto blockParentOp = blockArg.getOwner()->getParentOp();
        builder.setInsertionPointToStart(blockArg.getOwner());
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(blockParentOp, "dim_" + std::to_string(dimIdx)), tensor,
                                                    dimIdx);
    } else {
        auto loc = tensor.getDefiningOp();
        builder.setInsertionPointAfter(loc);
        dimOp = builder.create<mlir::tensor::DimOp>(takeOpLoc(loc, "dim_" + std::to_string(dimIdx)), tensor, dimIdx);
    }
    return mlir::success();
}

void replaceUsesOfOpWithDominanceCheck(mlir::Value toReplace, mlir::Value newValue) {
    auto newValueDefiningOp = newValue.getDefiningOp();
    if (newValueDefiningOp == nullptr) {
        return;
    }

    mlir::DominanceInfo dom(newValueDefiningOp);
    for (auto& use : llvm::make_early_inc_range(toReplace.getUses())) {
        mlir::Operation* userOp = use.getOwner();

        if (!dom.properlyDominates(newValue, userOp)) {
            continue;  // Skip users not dominated by the new value
        }

        use.set(newValue);
    }
}

SmallVector<mlir::Value> vpux::VPU::applyIndexBacktracking(mlir::tensor::InsertSliceOp insertSliceOp,
                                                           ArrayRef<size_t> dimsToAdjust) {
    auto parentForOp = insertSliceOp->getParentOfType<mlir::scf::ForOp>();
    assert(parentForOp && "InsertSliceOp not inside scf.for");

    auto getLoopIV = [&](mlir::Value val) {
        mlir::Value inductionVar = nullptr;
        mlir::scf::ForOp parentOp = nullptr;

        if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(val)) {
            auto blockOp = blockArg.getOwner()->getParentOp();
            if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockOp)) {
                inductionVar = val;
                parentOp = forOp;
            }
        }

        if (inductionVar == nullptr) {
            OpChainAnalysis utils;
            auto opChain = utils.collectParentOpsChain(val);

            for (auto op : opChain) {
                for (auto operand : op->getOperands()) {
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
                        auto blockOp = blockArg.getOwner()->getParentOp();
                        if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockOp)) {
                            inductionVar = operand;
                            parentOp = forOp;
                            break;
                        }
                    }
                }
            }
        }

        return std::pair<mlir::Value, mlir::scf::ForOp>{inductionVar, parentOp};
    };

    auto insertionPoint = &(parentForOp.getBody()->front());
    mlir::OpBuilder localBuilder(insertSliceOp);
    localBuilder.setInsertionPointToStart(parentForOp.getBody());

    SmallVector<mlir::OpFoldResult> offsetsToAdjust;
    if (!dimsToAdjust.empty()) {
        for (auto dim : dimsToAdjust) {
            offsetsToAdjust.push_back(insertSliceOp.getMixedOffsets()[dim]);
        }
    } else {
        offsetsToAdjust = insertSliceOp.getMixedOffsets();
    }

    SmallVector<mlir::Value> newOffsets;
    for (auto offset : offsetsToAdjust) {
        auto offsetValue = mlir::dyn_cast_or_null<mlir::Value>(offset);
        if (offsetValue == nullptr) {
            continue;
        }

        if (!mlir::isa<mlir::BlockArgument>(offsetValue) &&
            mlir::isa<mlir::arith::ConstantOp>(offsetValue.getDefiningOp())) {
            continue;
        }

        auto [loopIv, parentOp] = getLoopIV(offsetValue);
        assert(loopIv && "Could not find loop induction variable for offset");

        auto upperBound = parentOp.getUpperBound();
        auto stepSize = parentOp.getStep();
        mlir::AffineExpr s0, s1;
        bindSymbols(localBuilder.getContext(), s0, s1);
        mlir::AffineExpr subExpr = s0 - s1;
        auto affineMap = mlir::AffineMap::get(0, 2, {subExpr}, localBuilder.getContext());
        // Apply the affine map to compute upperBound - stepSize
        auto lastTileOffset = mlir::affine::makeComposedFoldedAffineApply(
                localBuilder, takeOpLoc(insertionPoint, "last_tile_offset"), affineMap, {upperBound, stepSize});
        // take min between loopIv and adjustedOffset using affine exprs
        mlir::AffineExpr d0;
        bindDims(localBuilder.getContext(), d0);
        auto minAffineMap = mlir::AffineMap::get(1, 1, {d0, s0}, localBuilder.getContext());
        auto newOffset = mlir::affine::makeComposedFoldedAffineMin(
                localBuilder, takeOpLoc(insertionPoint, "new_offset"), minAffineMap, {loopIv, lastTileOffset});
        auto newOffsetValue = mlir::getValueOrCreateConstantIndexOp(
                localBuilder, takeOpLoc(insertionPoint, "adjusted_offset_val"), newOffset);

        replaceUsesOfOpWithDominanceCheck(loopIv, newOffsetValue);
        replaceUsesOfOpWithDominanceCheck(offsetValue, newOffsetValue);
        newOffsets.push_back(newOffsetValue);
    }

    return newOffsets;
}

SmallVector<mlir::Value> vpux::VPU::castOutputForInsertion(mlir::OpBuilder& builder, ArrayRef<SCFTileInfo> outputTiles,
                                                           DimArrRef dims, mlir::Operation* origOperation,
                                                           mlir::Operation* tiledOperation) {
    const auto numResults = origOperation->getNumResults();
    VPUX_THROW_UNLESS(outputTiles.size() == numResults, "castOutputForInsertion: expected {0} output tiles but got {1}",
                      numResults, outputTiles.size());

    SmallVector<mlir::Value> castedValues;
    castedValues.reserve(numResults);

    for (auto resultNumber : irange(numResults)) {
        const auto outputShape = getShape(origOperation->getResult(resultNumber));
        const auto& outputTile = outputTiles[resultNumber];

        // Find dynamic dims that are not tiling dims — only those need a cast
        const auto nonTilingDynDims = [&]() -> SmallVector<Dim> {
            SmallVector<Dim> dynDims;
            for (auto idx : irange(outputShape.size())) {
                const auto dim = Dim(idx);
                if (outputShape[dim] != mlir::ShapedType::kDynamic) {
                    continue;
                }
                if (!llvm::is_contained(dims, dim)) {
                    dynDims.push_back(dim);
                }
            }
            return dynDims;
        }();

        mlir::Value resultValue = tiledOperation->getResult(resultNumber);
        if (nonTilingDynDims.empty()) {
            castedValues.push_back(resultValue);
            continue;
        }

        auto tiledOpType = mlir::cast<vpux::NDTypeInterface>(resultValue.getType());
        auto castedOutputShape = Shape(tiledOpType.getShape());
        auto tiledOpBoundedShape = getBoundedShape(tiledOpType);
        for (const auto dynDim : nonTilingDynDims) {
            const auto evaluatedDim = mlir::getConstantIntValue(outputTile.shape[dynDim.ind()]);
            castedOutputShape[dynDim] = evaluatedDim.value_or(tiledOpBoundedShape[dynDim]);
        }

        mlir::Type castedTiledOutputType = tiledOpType.changeShape(castedOutputShape);
        auto castOp = builder.create<mlir::tensor::CastOp>(appendLoc(tiledOperation->getLoc(), "insert_cast"),
                                                           castedTiledOutputType, resultValue);
        castedValues.push_back(castOp.getResult());
    }

    return castedValues;
}

/**
 * @brief For tensors with dynamic dimensions along padded axes, the PadOp might have identical input and output
 * types. However, for static dimensions, the shapes of PadOp source and destination differ along the padded axes.
 * Add a tensor.cast to make the input dynamic when required, as the input tensor on which the NCE operation operates
 * will determine the shape of the tensor.
 */
mlir::Value getInputOperand(mlir::tensor::PadOp padOp, mlir::OpBuilder builder) {
    mlir::RankedTensorType srcType = padOp.getSourceType(), dstType = padOp.getResultType();
    bool isDynamic = false;
    VPUX_THROW_WHEN(srcType == nullptr || dstType == nullptr, "Expected RankedTensorType for PadOp source and result");

    auto newShape = SmallVector<int64_t>{};
    vpux::ShapeRef newBounds = vpux::ShapeRef{};
    transform(enumerate(padOp.getMixedLowPad()), std::back_inserter(newShape),
              [&srcType, &padOp, &isDynamic](auto&& indexedOffset) {
                  auto [idx, lowOffset] = indexedOffset;
                  auto highOffset = padOp.getMixedHighPad()[idx];

                  // Check if both low and high padding are static integer attributes
                  bool lowIsStatic = false, highIsStatic = false;

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(lowOffset)) {
                      lowIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(highOffset)) {
                      highIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  // If either padding is not static, make the dimension dynamic
                  if (!lowIsStatic || !highIsStatic || srcType.getShape()[idx] == mlir::ShapedType::kDynamic) {
                      isDynamic = true;
                      return mlir::ShapedType::kDynamic;
                  }

                  return srcType.getShape()[idx];
              });

    if (isDynamic) {
        newBounds = getBoundedShape(srcType);
    }

    auto dstBounds = vpux::BoundsRef(newBounds);
    const auto inType = mlir::cast<NDTypeInterface>(srcType);
    auto outDesc = vpux::getTensorAttr(builder.getContext(), inType.getDimsOrder(), inType.getMemSpace(), dstBounds);
    auto newDstType = mlir::RankedTensorType::get(newShape, dstType.getElementType(), outDesc);
    if (newDstType != srcType) {
        return builder.create<mlir::tensor::CastOp>(appendLoc(padOp.getLoc(), "cast"), newDstType, padOp.getSource());
    }

    return padOp.getSource();
}

bool isPadInsideSpatiallySegmentedForallLoop(mlir::tensor::PadOp padOp) {
    if (padOp->getParentOfType<mlir::scf::ForallOp>() == nullptr) {
        return false;
    }

    // Expecting scf.forall ... { (tensor.extract_slice -> VPU.Copy ->) tensor.pad -> compute_op } pattern
    // tensor.extract_slice -> VPU.Copy sequence is optional; if it does not exist, the input is not segmented
    // in any way for multiclustering
    auto copyOp = padOp.getSource().getDefiningOp<VPU::CopyOp>();
    if (copyOp == nullptr) {
        return false;
    }

    auto extractSliceOp = copyOp.getInput().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (extractSliceOp == nullptr) {
        return false;
    }

    const auto offsets = extractSliceOp.getMixedOffsets();
    // Check that there is no segmentation along any spatial dimension
    return std::any_of(offsets.begin() + Dims4D::Act::getSpatialDim(0).ind(), offsets.end(),
                       [](mlir::OpFoldResult ofr) {
                           return mlir::isa_and_nonnull<mlir::Value>(ofr);
                       });
}

void vpux::VPU::restorePaddingAttribute(mlir::Operation* region, Logger log) {
    SmallVector<std::pair<mlir::Operation*, mlir::tensor::PadOp>> worklist;
    region->walk([&](mlir::tensor::PadOp padOp) {
        if (isPadInsideSpatiallySegmentedForallLoop(padOp)) {
            return mlir::WalkResult::advance();
        }

        for (auto user : padOp->getUsers()) {
            if (VPU::isNceOpWithPadAttr(user)) {
                worklist.push_back({user, padOp});
            }
        }
        return mlir::WalkResult::advance();
    });

    OpChainAnalysis opChainAnalysis;
    auto getPadAttribute = [&](mlir::tensor::PadOp padOp) {
        auto spatialDims = {Dims4D::Act::W, Dims4D::Act::H};
        llvm::SmallVector<int64_t> padValues;
        for (auto dim : spatialDims) {
            auto lowPad = padOp.getMixedLowPad()[dim.ind()];
            auto highPad = padOp.getMixedHighPad()[dim.ind()];

            ValueRangeMap emptyMap;
            auto lowValue = opChainAnalysis.getOpFoldResultValue(lowPad, emptyMap);
            auto highValue = opChainAnalysis.getOpFoldResultValue(highPad, emptyMap);
            VPUX_THROW_WHEN(!lowValue.has_value() || !highValue.has_value(),
                            "Failed to compute static padding values for {0} operation", padOp->getName());
            padValues.emplace_back(lowValue.value()[0]);
            padValues.emplace_back(highValue.value()[0]);
        }

        return VPU::getPaddingAttr(padOp.getContext(), padValues[0], padValues[1], padValues[2], padValues[3]);
    };

    mlir::IRRewriter rewriter(region->getContext());
    for (auto [nceOp, padOp] : llvm::make_early_inc_range(worklist)) {
        log.trace("Found convolution operation {0} with tensor.pad parent", nceOp->getName());

        rewriter.setInsertionPoint(nceOp);
        auto inputOperand = getInputOperand(padOp, rewriter);
        auto restoredPadAttr = getPadAttribute(padOp);

        mlir::IRMapping mapper;
        mapper.map(nceOp->getOperand(0), inputOperand);
        auto newNceOp = rewriter.clone(*nceOp, mapper);
        newNceOp->setAttr("pad", restoredPadAttr);
        vpux::inferReturnTypes(newNceOp, vpux::InferShapedTypeMode::SHAPE);

        // Replace all uses of the original convolution operation with the new one
        rewriter.replaceOp(nceOp, newNceOp->getResults());

        if (padOp->getUsers().empty()) {
            padOp.erase();
        }
    }
}

bool vpux::VPU::isDependentOnForallIv(mlir::OpFoldResult ofr, mlir::scf::ForallOp forallOp) {
    if (mlir::getConstantIntValue(ofr).has_value()) {
        return false;
    }

    auto value = mlir::dyn_cast_if_present<mlir::Value>(ofr);
    if (value == nullptr) {
        return false;
    }

    auto ivs = forallOp.getInductionVars();
    if (llvm::is_contained(ivs, value)) {
        return true;
    }

    OpChainAnalysis analysis;
    llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE> blockOperands;
    analysis.traverseAndGetBlockArgs(value, blockOperands);

    return llvm::any_of(blockOperands, [&](mlir::Value operand) {
        return llvm::is_contained(ivs, operand);
    });
}

llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement> vpux::VPU::analyzeSkipConnectionsForTiling(
        const llvm::SetVector<mlir::Operation*>& allOpsToFuse, const TilingOperationStorage::UPtr& tilingStorage,
        const Logger& log) {
    // At this stage, we analyze skip connections and record, for each skip-source op, which user branch
    // requires the largest tile. This information is used later in tile+fuse decisions.
    // If fusion reaches the skip-source through:
    //   - the user with the largest tile, we allow fusion as usual;
    //   - a user with a smaller tile, we do not fuse and only remember the slice op in
    //     `futureReplacement` for deferred replacement.
    llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement> skipConnectionMap;
    if (tilingStorage == nullptr) {
        log.warning("Could not find tiling storage for the best VF case. Vertical fusion will be applied without skip "
                    "connection support.\n");
        return skipConnectionMap;
    } else {
        auto allOps = tilingStorage->getAll();

        // Analyze tiling requirements for each skip connection branch
        log.debug("Analyzing skip connections to determine tile requirements for each branch...");

        for (auto op : allOpsToFuse) {
            if (op->hasOneUse()) {
                continue;
            }

            auto allUsesSameOwner = llvm::all_of(op->getUses(), [&](mlir::OpOperand& use) {
                return op->getUses().begin()->getOwner() == use.getOwner();
            });
            if (allUsesSameOwner) {
                continue;
            }
            log.debug("Skip connection SourceOp: {0}", op->getName());

            SmallVector<mlir::Operation*> users;
            users.reserve(static_cast<size_t>(std::distance(op->user_begin(), op->user_end())));
            llvm::copy_if(op->getUsers(), std::back_inserter(users), [&](mlir::Operation* user) {
                return allOpsToFuse.contains(user);
            });

            if (users.size() <= 1) {
                continue;
            }

            // Structure to store tile info for comparison
            std::pair<size_t, Shape> userWithBiggestTiles{0, Shape{}};
            bool allUsersWithTheSameTileSize = true;
            for (size_t userIdx = 0; userIdx < users.size(); ++userIdx) {
                auto user = users[userIdx];
                log.debug("Branch[{0}]: {1}", userIdx, user->getName());

                auto tilingContainer = allOps[user];

                // Find which input of this user comes from skipOp
                // Find which input of this user comes from skipOp
                const auto operandsWithIdx = llvm::enumerate(user->getOperands());
                auto skipInputIt = llvm::find_if(operandsWithIdx, [&](const auto& item) {
                    return item.value().getDefiningOp() == op;
                });

                if (skipInputIt == operandsWithIdx.end()) {
                    log.warning("Could not find input from skip connection!\n");
                    continue;
                }

                const auto inputIdxFromSkipOp = (*skipInputIt).index();

                // NOTE: Here we compare only the very first tile (tile index 0) for each branch.
                // In the general case, each branch has multiple tiles, and a stricter analysis would
                // compare tile-by-tile across branches (0 with 0, 1 with 1, etc.).
                // For now, we use only tile #0 as a representative and infer which branch needs a
                // larger tile from that single comparison, assuming the remaining tiles correlate
                // in size with the first one.
                auto tile0 = tilingContainer.find(0);
                if (tile0 == tilingContainer.end()) {
                    log.warning("Could not find tile 0!\n");
                    continue;
                }
                const auto& inputTiling = tile0->second.first;
                const auto& skipOpInputTiling = inputTiling.tiles[inputIdxFromSkipOp];

                if (userWithBiggestTiles.second.empty()) {
                    userWithBiggestTiles.first = userIdx;
                    userWithBiggestTiles.second = skipOpInputTiling.shape;
                    continue;
                }
                if (skipOpInputTiling.shape.totalSize() == userWithBiggestTiles.second.totalSize()) {
                    continue;
                }
                allUsersWithTheSameTileSize = false;
                if (skipOpInputTiling.shape.totalSize() > userWithBiggestTiles.second.totalSize()) {
                    userWithBiggestTiles.first = userIdx;
                    userWithBiggestTiles.second = skipOpInputTiling.shape;
                }
            }
            // Print which user has the biggest tiles for this skip connection
            log.debug("Branch[{0}] requires the biggest tile size of {1} (total size: {2})", userWithBiggestTiles.first,
                      userWithBiggestTiles.second, userWithBiggestTiles.second.totalSize());

            VPU::PendingSliceReplacement futureReplacement{};
            futureReplacement.allUsersWithTheSameTileSize = allUsersWithTheSameTileSize;
            futureReplacement.biggestUserOp = users[userWithBiggestTiles.first];
            skipConnectionMap[op] = std::move(futureReplacement);
        }
    }
    return skipConnectionMap;
}

void vpux::VPU::applyDeferredSliceReplacements(
        mlir::RewriterBase& builder,
        const llvm::DenseMap<mlir::Operation*, VPU::PendingSliceReplacement>& skipConnectionMap, const Logger& log) {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);

    OpChainAnalysis opChainAnalysis;

    llvm::DenseMap<mlir::Value, SmallVector<int64_t>> valueMap;
    const auto multipleFunc = [&opChainAnalysis, &valueMap](int64_t value0, auto value1) -> int64_t {
        auto intVal1List = opChainAnalysis.getOpFoldResultValue(value1, valueMap);

        VPUX_THROW_WHEN(!intVal1List.has_value() || intVal1List.value().empty(),
                        "Failed to get integer value from OpFoldResult");
        return value0 * intVal1List.value().front();
    };

    for (const auto& mapEntry : skipConnectionMap) {
        const auto& deferredReplacement = mapEntry.second;
        if (!deferredReplacement.biggestUserTiled && !deferredReplacement.allUsersWithTheSameTileSize) {
            continue;
        }
        if (deferredReplacement.tiledValue == nullptr) {
            log.debug("Skip-connection producer was never tiled for op {0}; leaving related ExtractSlices in place",
                      mapEntry.first->getName());
            continue;
        }

        auto tiledValue = deferredReplacement.tiledValue;
        auto biggestSliceOp = deferredReplacement.biggestTileExtractSlice;
        auto biggestSliceOffsets = biggestSliceOp.getMixedOffsets();

        auto biggestSliceSizes = biggestSliceOp.getMixedSizes();
        auto biggestSliceTotalSize =
                std::accumulate(biggestSliceSizes.begin(), biggestSliceSizes.end(), 1LL, multipleFunc);

        builder.setInsertionPointAfterValue(tiledValue);
        for (auto sliceToReplace : deferredReplacement.relatedExtractSlices) {
            auto currentSliceSizes = sliceToReplace.getMixedSizes();
            auto currentSliceTotalSize =
                    std::accumulate(currentSliceSizes.begin(), currentSliceSizes.end(), 1LL, multipleFunc);

            if (biggestSliceTotalSize == currentSliceTotalSize) {
                log.debug("Current slice has the same total size as biggest slice, replacing with biggest slice result "
                          "directly");
                builder.replaceOp(sliceToReplace, tiledValue);
                continue;
            }
            VPUX_THROW_UNLESS(biggestSliceTotalSize > currentSliceTotalSize,
                              "Biggest slice total size should be greater than current slice total size to handle skip "
                              "connection properly");

            auto currentSliceOffsets = sliceToReplace.getMixedOffsets();

            SmallVector<mlir::OpFoldResult> adjustedOffsets;
            adjustedOffsets.reserve(currentSliceOffsets.size());

            for (size_t i = 0; i < currentSliceOffsets.size(); ++i) {
                mlir::AffineExpr d0, d1;
                bindDims(builder.getContext(), d0, d1);
                auto offsetMap = mlir::AffineMap::get(2, 0, {d0 - d1}, builder.getContext());
                auto adjustedOffset = mlir::affine::makeComposedFoldedAffineApply(
                        builder, sliceToReplace.getLoc(), offsetMap, {currentSliceOffsets[i], biggestSliceOffsets[i]});
                adjustedOffsets.push_back(adjustedOffset);
            }

            auto newSliceOp = builder.create<mlir::tensor::ExtractSliceOp>(
                    sliceToReplace.getLoc(), tiledValue, adjustedOffsets, sliceToReplace.getMixedSizes(),
                    sliceToReplace.getMixedStrides());

            newSliceOp->setAttr(SKIP_CONNECTION_SLICE_MARKER_ATTR_NAME, builder.getUnitAttr());
            builder.replaceAllUsesWith(sliceToReplace.getResult(), newSliceOp.getResult());
            builder.eraseOp(sliceToReplace);
        }
    }
}
