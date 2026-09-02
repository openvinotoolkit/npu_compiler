//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_interpolate_helpers.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Support/LLVM.h>

#include <optional>

namespace vpux::VPU {

//
// SCFTilingCommonModelOp
//

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingCommonModelOp : public mlir::TilingInterface::ExternalModel<ConcreteModel, ConcreteOp> {
protected:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        return static_cast<const ConcreteModel*>(this)->backInferSCFTileInfo(operation, builder, outputTile);
    }

    // Return per-result output tiles for the given iteration offsets/sizes.
    // Builds each result's tile from getResultTilePosition and computes bounds
    // for dynamic tensors. Concrete models can provide their own
    // getOutputSCFTiling to customize output tile computation;
    mlir::FailureOr<SmallVector<SCFTileInfo>> getOutputSCFTiling(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                                 ArrayRef<mlir::OpFoldResult> offsets,
                                                                 ArrayRef<mlir::OpFoldResult> sizes) const {
        const auto numResults = operation->getNumResults();
        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);

        SmallVector<SmallVector<mlir::OpFoldResult>> allResultOffsets;
        SmallVector<SmallVector<mlir::OpFoldResult>> allResultSizes;
        SmallVector<Bounds> allResultBounds;
        allResultOffsets.reserve(numResults);
        allResultSizes.reserve(numResults);
        allResultBounds.reserve(numResults);

        for (auto resultNumber : irange(numResults)) {
            allResultOffsets.emplace_back();
            allResultSizes.emplace_back();
            if (mlir::failed(getResultTilePosition(operation, builder, resultNumber, offsets, sizes,
                                                   allResultOffsets.back(), allResultSizes.back()))) {
                return mlir::failure();
            }
            allResultBounds.emplace_back();
        }

        if (IE::hasDynamicTensors(operation)) {
            for (auto resultNumber : irange(numResults)) {
                if (mlir::failed(getResultTileBounds(operation, resultNumber, tilingDims, allResultSizes[resultNumber],
                                                     allResultBounds[resultNumber]))) {
                    return mlir::failure();
                }
            }
        }

        SmallVector<SCFTileInfo> allOutputTiles;
        allOutputTiles.reserve(numResults);
        for (auto resultNumber : irange(numResults)) {
            auto& resOffsets = allResultOffsets[resultNumber];
            SmallVector<mlir::OpFoldResult> axis(resOffsets.size(), builder.getIndexAttr(1));
            for (auto tileDim : tilingDims) {
                const auto tileDimIdx = checked_cast<size_t>(tileDim.ind());
                if (tileDimIdx >= axis.size()) {
                    return mlir::failure();
                }
                axis[tileDimIdx] = builder.getIndexAttr(strategy[tileDim]);
            }
            allOutputTiles.emplace_back(allResultSizes[resultNumber], resOffsets, axis, allResultBounds[resultNumber]);
        }

        return allOutputTiles;
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, DimArrRef dims,
                                          SmallVector<mlir::Value>& tiledOperands, mlir::Operation* operation) const {
        // Subclass creates the tiled operation; casting is applied uniformly
        // in getTiledImplementation via castOutputForInsertion for all results.
        return static_cast<const ConcreteModel*>(this)->createTiledOperation(
                std::move(opGenerator), std::move(operandsGenerator), builder, inputTiling, outputTile, dims,
                tiledOperands, operation);
    }

    void fillInResultTilePositions(mlir::Operation* operation, mlir::OpBuilder& builder, unsigned resultNumber,
                                   ArrayRef<mlir::OpFoldResult> offsets, ArrayRef<mlir::OpFoldResult> sizes,
                                   SmallVector<mlir::OpFoldResult>& resultOffsets,
                                   SmallVector<mlir::OpFoldResult>& resultSizes) const {
        auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(resultNumber).getType());

        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);
        resultOffsets = SmallVector<mlir::OpFoldResult>(outputType.getRank(), builder.getIndexAttr(0));
        resultSizes.reserve(outputType.getRank());
        resultSizes = mlir::getAsIndexOpFoldResult(builder.getContext(), outputType.getShape());

        for (auto dimIndex : irange(outputType.getRank())) {
            auto tilingDim = std::find(tilingDims.begin(), tilingDims.end(), Dim(dimIndex));
            if (tilingDim != tilingDims.end()) {
                auto index = sizes.size() == resultOffsets.size() ? static_cast<size_t>(tilingDim->ind())
                                                                  : std::distance(tilingDims.begin(), tilingDim);
                resultOffsets[tilingDim->ind()] = offsets[index];
                resultSizes[tilingDim->ind()] = sizes[index];
            } else {
                resultOffsets[dimIndex] = builder.getIndexAttr(0);
                resultSizes[dimIndex] =
                        getAsIndexOpFoldResult(builder.getContext(), getBoundedShape(outputType)[Dim(dimIndex)]);
            }
        }
    }

public:
    SmallVector<mlir::Range> getIterationDomain(mlir::Operation* operation, mlir::OpBuilder& builder) const {
        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);
        auto loc = operation->getLoc();

        const auto tilingRank = static_cast<int64_t>(tilingDims.size());
        SmallVector<mlir::Range> loops(tilingRank);
        mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

        for (auto dim : llvm::seq<int64_t>(0, tilingRank)) {
            loops[dim].offset = zero;
            loops[dim].size = getDimValue(builder, operation, tilingDims[dim].ind());
        }
        return loops;
    }

    SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(mlir::Operation* operation) const {
        // Read tiling strategy from attribute
        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

        // Determine which dimensions are being tiled (same logic used by getIterationDomain)
        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);

        SmallVector<mlir::utils::IteratorType> iterTypes;
        iterTypes.reserve(tilingDims.size());

        // All SCF loops for convolution tiling are parallel loops
        for (size_t i = 0; i < tilingDims.size(); ++i) {
            iterTypes.push_back(mlir::utils::IteratorType::parallel);
        }

        return iterTypes;
    }

    mlir::FailureOr<mlir::TilingResult> getTiledImplementation(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                               ArrayRef<mlir::OpFoldResult> offsets,
                                                               ArrayRef<mlir::OpFoldResult> sizes) const {
        const auto numResults = operation->getNumResults();
        if (numResults == 0) {
            return mlir::failure();
        }

        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);

        // Compute per-result output tiles (tile position + bounds).
        auto allOutputTilesOrFailure =
                static_cast<const ConcreteModel*>(this)->getOutputSCFTiling(operation, builder, offsets, sizes);
        if (mlir::failed(allOutputTilesOrFailure)) {
            return mlir::failure();
        }
        auto allOutputTiles = std::move(*allOutputTilesOrFailure);

        SmallVector<mlir::Operation*> results;
        SmallVector<mlir::Value> resultValues;
        SmallVector<mlir::Operation*> generatedSlices;
        resultValues.reserve(numResults);

        // All results share the same SCF iteration space, so result 0 is used
        // to back-infer the input tiling. Per-result output shape differences
        // (e.g. different result ranks or non-tiling dims) are handled by
        // getOutputSCFTiling, which is the correct override point for that.
        auto inputTiling = backInferSCFTileInfo(operation, builder, allOutputTiles[0]);
        const auto& outputTile = allOutputTiles[0];
        SmallVector<mlir::Value> tiledOperands;
        tiledOperands.reserve(operation->getNumOperands());

        OpTilingOperandsFunc createTiledOperands = [&](auto& tiling) {
            tiledOperands.clear();
            llvm::DenseMap<mlir::Value, mlir::Value> sliceMatch;
            for (auto p : operation->getOperands() | indexed) {
                auto origInput = p.value();
                auto inputIdx = p.index();

                if (tiling.tiles.size() <= inputIdx) {
                    tiledOperands.emplace_back(origInput);
                    continue;
                }

                if (sliceMatch.find(origInput) != sliceMatch.end()) {
                    tiledOperands.emplace_back(sliceMatch[origInput]);
                    continue;
                }

                auto inputTileInfo = tiling.tiles[inputIdx];
                auto tiledInput = generateTile(operation->getLoc(), builder, origInput, inputTileInfo, generatedSlices);
                sliceMatch[origInput] = tiledInput;

                tiledOperands.emplace_back(tiledInput);
            }
        };

        OpGeneratorFunc generatorFunc = [&]() {
            SmallVector<mlir::Type> resultDenseTiles;
            resultDenseTiles.reserve(numResults);
            for (auto resultNumber : irange(numResults)) {
                resultDenseTiles.emplace_back(extractResultType(operation->getResult(resultNumber).getType(),
                                                                allOutputTiles[resultNumber].shape,
                                                                allOutputTiles[resultNumber].bounds));
            }

            auto* tiledOp = mlir::cloneWithoutRegions(builder, operation, resultDenseTiles, tiledOperands);
            // inferReturnTypes adjusts the result types to match the tiled input shapes,
            // including bounds for dynamic tensors. This works for multi-result ops too
            // since TopK and LSTMGates have correct inferReturnTypes implementations.
            vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::SHAPE);
            tiledOp->removeAttr(tilingStrategy);
            return tiledOp;
        };

        auto* resultOp = createTiledOperation(std::move(generatorFunc), std::move(createTiledOperands), builder,
                                              inputTiling, outputTile, tilingDims, tiledOperands, operation);

        results.emplace_back(resultOp);
        // Cast each result for insertion — handles both single and multi-result ops uniformly.
        // For results with no non-tiling dynamic dims, the original value is returned as-is.
        resultValues = castOutputForInsertion(builder, allOutputTiles, tilingDims, operation, resultOp);

        return mlir::TilingResult{std::move(results), std::move(resultValues), std::move(generatedSlices)};
    }

    mlir::FailureOr<mlir::TilingResult> generateResultTileValue(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                                unsigned resultNumber,
                                                                mlir::ArrayRef<mlir::OpFoldResult> offsets,
                                                                mlir::ArrayRef<mlir::OpFoldResult> sizes) const {
        auto tilingResult = getTiledImplementation(operation, builder, offsets, sizes);
        if (mlir::failed(tilingResult)) {
            return mlir::failure();
        }
        if (resultNumber >= tilingResult->tiledValues.size()) {
            return mlir::failure();
        }
        tilingResult->tiledValues = {tilingResult->tiledValues[resultNumber]};
        return tilingResult;
    }

    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        return static_cast<const ConcreteModel*>(this)->getResultTilePosition(operation, builder, resultNumber, offsets,
                                                                              sizes, resultOffsets, resultSizes);
    }
};

//
// SCFExpandTilingModelOp
//
// SCF tiling model for `VPU.Expand` with single-dim end-padding
// (`pads_begin == 0`, exactly one `pads_end[d] > 0`). The padded dim must stay
// untiled: input tile keeps `d` at full unpadded extent, other dims pass through
// 1:1. Returns failure on unsupported cases so the SCF VF driver leaves it
// untiled.
//
class SCFExpandTilingModelOp : public mlir::TilingInterface::ExternalModel<SCFExpandTilingModelOp, VPU::ExpandOp> {
    mlir::LogicalResult checkPaddedDimIsUntiled(mlir::Operation* op, unsigned resultNumber,
                                                ArrayRef<mlir::OpFoldResult> offsets,
                                                ArrayRef<mlir::OpFoldResult> sizes) const {
        const auto paddedDim = VPU::getSinglePaddedExpandDim(op);
        if (!paddedDim.has_value()) {
            return mlir::failure();
        }

        const auto pIdx = checked_cast<size_t>(paddedDim->ind());
        if (pIdx >= sizes.size() || pIdx >= offsets.size()) {
            return mlir::failure();
        }

        const auto outputShape = getShape(op->getResult(resultNumber));
        const auto tileSize = mlir::getConstantIntValue(sizes[pIdx]);
        const auto tileOffset = mlir::getConstantIntValue(offsets[pIdx]);
        if (!tileSize.has_value() || !tileOffset.has_value() || tileSize.value() != outputShape[*paddedDim] ||
            tileOffset.value() != 0) {
            return mlir::failure();
        }

        return mlir::success();
    }

public:
    SmallVector<mlir::Range> getIterationDomain(mlir::Operation*, mlir::OpBuilder&) const {
        return {};
    }

    mlir::FailureOr<mlir::TilingResult> getTiledImplementation(mlir::Operation*, mlir::OpBuilder&,
                                                               ArrayRef<mlir::OpFoldResult>,
                                                               ArrayRef<mlir::OpFoldResult>) const {
        return mlir::failure();
    }

    mlir::FailureOr<mlir::TilingResult> generateResultTileValue(mlir::Operation* op, mlir::OpBuilder& builder,
                                                                unsigned resultNumber,
                                                                ArrayRef<mlir::OpFoldResult> offsets,
                                                                ArrayRef<mlir::OpFoldResult> sizes) const {
        if (mlir::failed(checkPaddedDimIsUntiled(op, resultNumber, offsets, sizes))) {
            return mlir::failure();
        }

        Bounds resultBounds;
        if (IE::hasDynamicTensors(op) && op->hasAttr(tilingStrategy)) {
            const auto strategy =
                    Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(tilingStrategy))));
            auto tilingDims = getSCFTilingOrderedDims(op, strategy);
            if (mlir::failed(getResultTileBounds(op, resultNumber, tilingDims, sizes, resultBounds))) {
                return mlir::failure();
            }
        }

        auto outputTile = SCFTileInfo(sizes, offsets, SCFShape(offsets.size(), builder.getIndexAttr(1)), resultBounds);
        auto inputTiling = backInferSCFTileInfo(op, builder, outputTile);

        SmallVector<mlir::Value> tiledOperands;
        SmallVector<mlir::Operation*> generatedSlices;
        tiledOperands.reserve(op->getNumOperands());

        for (auto p : op->getOperands() | indexed) {
            auto origInput = p.value();
            auto inputIdx = p.index();

            if (inputTiling.tiles.size() <= inputIdx) {
                tiledOperands.emplace_back(origInput);
                continue;
            }

            auto inputTileInfo = inputTiling.tiles[inputIdx];
            auto tiledInput = generateTile(op->getLoc(), builder, origInput, inputTileInfo, generatedSlices);

            tiledOperands.emplace_back(tiledInput);
        }

        auto resultDenseTile = extractResultType(op->getResult(resultNumber).getType(), sizes, resultBounds);
        auto* tiledOp = mlir::cloneWithoutRegions(builder, op, {resultDenseTile}, tiledOperands);
        vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::SHAPE);
        tiledOp->removeAttr(tilingStrategy);

        return mlir::TilingResult{{tiledOp}, {tiledOp->getResult(resultNumber)}, std::move(generatedSlices)};
    }

    // Mirror of `VPU::ExpandOp::backInferTileInfo` (int64 `TileInfo`); keep in sync.
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* op, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        const auto paddedDim = VPU::getSinglePaddedExpandDim(op);
        VPUX_THROW_UNLESS(paddedDim.has_value(),
                          "SCFExpandTilingModelOp invoked on '{0}' which is not a single-dim end-padded Expand",
                          op->getLoc());

        auto expandOp = mlir::cast<VPU::ExpandOp>(op);
        const auto inputShape = getShape(expandOp.getInput());
        const auto pIdx = static_cast<size_t>(paddedDim->ind());
        VPUX_THROW_UNLESS(pIdx < outputTile.shape.size(),
                          "Padded dim {0} is out of range for VPU.Expand operand rank {1}", pIdx,
                          outputTile.shape.size());

        auto inputTile = outputTile;
        inputTile.shape[pIdx] = builder.getIndexAttr(inputShape[*paddedDim]);
        inputTile.offsets[pIdx] = builder.getIndexAttr(0);
        if (!inputTile.bounds.empty()) {
            inputTile.bounds[*paddedDim] = inputShape[*paddedDim];
        }
        return SCFTilingInfo{{std::move(inputTile)}};
    }
};

template <class ConcreteOp>
class SCFTilingEltwiseLikeModelOp : public SCFTilingCommonModelOp<SCFTilingEltwiseLikeModelOp<ConcreteOp>, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& tiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(tiling);
        return opGenerator();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto loc = operation->getLoc();

        auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(operation);
        const auto inChannelAlignment = alignedOp != nullptr ? alignedOp.getInputChannelAlignment() : 1;
        const auto outChannelAlignment = alignedOp != nullptr ? alignedOp.getOutputChannelAlignment() : 1;

        VPUX_THROW_WHEN(!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1) &&
                                inChannelAlignment != outChannelAlignment,
                        "[EltwiseLike] Dynamic tiling step for Channel dimension is not supported for operation: {0} "
                        "with outputTile: {1}: input channel alignment {2} differs from output channel alignment {3}",
                        operation->getLoc(), outputTile, inChannelAlignment, outChannelAlignment);

        auto alignMap = getAlignValUpMap(builder, inChannelAlignment);

        SmallVector<SCFTileInfo> inputTiles;
        for (auto origInput : operation->getOperands()) {
            const auto inputType = mlir::cast<mlir::ShapedType>(origInput.getType());
            const auto curShape = inputType.getShape();

            auto curTile = outputTile;
            if (!outputTile.bounds.raw().empty()) {
                // same as alignValUp(outputTile.bounds[Dims4D::Act::C], inChannelAlignment);
                // just to have same function for bounds calculation as for shape calculation below
                auto outCBound = builder.getI64IntegerAttr(outputTile.bounds[Dims4D::Act::C]);
                auto inCBound = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputBoundsSize"),
                                                                            alignMap, {outCBound});
                curTile.bounds[Dims4D::Act::C] = mlir::getConstantIntValue(inCBound).value();
            }

            for (auto ind : irange(curShape.size())) {
                if (curShape[ind] == 1) {
                    curTile.shape[ind] = builder.getIndexAttr(1);
                    curTile.offsets[ind] = builder.getIndexAttr(0);
                }
            }

            auto& inChannelsTile = curTile.shape[Dims4D::Act::C.ind()];
            inChannelsTile = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputChSize"),
                                                                         alignMap, {inChannelsTile});

            inputTiles.push_back(curTile);
        }
        return SCFTilingInfo{std::move(inputTiles)};
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingPoolingModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
protected:
    SCFTilingInfo backInferPoolTile(mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile,
                                    SCFShapeRef origInputShape, ArrayRef<int64_t> origOutputShape,
                                    mlir::ArrayAttr kernelSize, mlir::ArrayAttr strides, const PadInfo& origPadding,
                                    std::optional<int64_t> inChannelAlignment) const {
        mlir::AffineMap alignMap;
        if (inChannelAlignment.has_value()) {
            alignMap = getAlignValUpMap(builder, inChannelAlignment.value());
        }

        SCFTileInfo inputTile(origInputShape, builder);
        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];
        inputTile.shape[Dims4D::Act::C.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
        inputTile.offsets[Dims4D::Act::C.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds = outputTile.bounds;
            if (inChannelAlignment.has_value()) {
                // same as alignValUp(outputTile.bounds[Dims4D::Act::C], inChannelAlignment);
                // just to have same function for bounds calculation as for shape calculation below
                auto outCBound = builder.getI64IntegerAttr(outputTile.bounds[Dims4D::Act::C]);
                auto inCBound = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputBoundsSize"),
                                                                            alignMap, {outCBound});
                inputTile.bounds[Dims4D::Act::C] = mlir::getConstantIntValue(inCBound).value();
            }
        }

        auto padMap = origPadding.toPadByDims();

        auto pads = mlir::getAsIndexOpFoldResult(
                builder.getContext(), {origPadding.top, origPadding.left, origPadding.bottom, origPadding.right});

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);
            const auto stride = mlir::cast<mlir::IntegerAttr>(strides[index]).getValue().getSExtValue();
            const auto kernel = mlir::cast<mlir::IntegerAttr>(kernelSize[index]).getValue().getSExtValue();

            auto [inputRange, dimBound] =
                    solutionForOutputRange(loc, builder, outputTile, dim, kernel, stride, origInputShape[dim.ind()],
                                           origOutputShape[dim.ind()], padMap[dim.ind()], pads[index], pads[index + 2]);

            if (inputRange.has_value()) {
                inputTile.offsets[dim.ind()] = inputRange.value().offset;
                inputTile.shape[dim.ind()] = inputRange.value().size;
            }

            if (dimBound.has_value()) {
                inputTile.bounds[dim] = dimBound.value();
            }
        }

        if (inChannelAlignment.has_value()) {
            auto& inChannelsTile = inputTile.shape[Dims4D::Act::C.ind()];
            inChannelsTile = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "inputChSize"),
                                                                         alignMap, {inChannelsTile});
        }

        return {std::move(inputTile), std::move(pads)};
    }

public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        correctPaddedOutput(builder, mlir::cast<ConcreteOp>(operation), resultSizes);
        return mlir::success();
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, DimArrRef dims,
                                          SmallVector<mlir::Value>& tiledOperands, mlir::Operation* operation) const {
        return createTiledPaddedOperation<ConcreteOp>(std::move(opGenerator), std::move(operandsGenerator), builder,
                                                      inputTiling, outputTile, dims, tiledOperands, operation);
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(operation);
        const auto inChannelAlignment = alignedOp != nullptr ? alignedOp.getInputChannelAlignment() : 1;
        const auto outChannelAlignment = alignedOp != nullptr ? alignedOp.getOutputChannelAlignment() : 1;

        VPUX_THROW_WHEN(!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1) &&
                                inChannelAlignment != outChannelAlignment,
                        "[Pool] Dynamic tiling step for Channel dimension is not supported for operation: {0} "
                        "with outputTile: {1}: input channel alignment {2} differs from output channel alignment {3}",
                        operation->getLoc(), outputTile, inChannelAlignment, outChannelAlignment);

        auto poolingOperation = mlir::cast<ConcreteOp>(operation);
        const auto origInputShape =
                mlir::tensor::getMixedSizes(builder, operation->getLoc(), poolingOperation.getInput());

        const auto outputType = mlir::cast<mlir::ShapedType>(poolingOperation.getOutput().getType());
        const auto origOutputShape = outputType.getShape();
        const auto origPadding = toPadInfo(poolingOperation.getPad());

        auto inputTiling = backInferPoolTile(operation->getLoc(), builder, outputTile, origInputShape, origOutputShape,
                                             poolingOperation.getKernelSize(), poolingOperation.getStrides(),
                                             origPadding, inChannelAlignment);

        auto nceOp = mlir::dyn_cast<NCEOpInterface>(operation);
        if (nceOp != nullptr && nceOp.getWeightsTableOperand() != nullptr &&
            !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1)) {
            inputTiling.tiles.emplace_back(
                    getWeightsTableSCFTile(nceOp.getWeightsTableOperand().getType(), builder, outputTile));
        }

        return inputTiling;
    }
};

class SCFMaxPoolOpModel : public SCFTilingPoolingModelOp<SCFMaxPoolOpModel, NCEMaxPoolOp> {};
class SCFAvgPoolOpModel : public SCFTilingPoolingModelOp<SCFAvgPoolOpModel, NCEAveragePoolOp> {};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingConvModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        correctPaddedOutput(builder, mlir::cast<ConcreteOp>(operation), resultSizes);
        return mlir::success();
    }

    Shape getRawFilterShape(mlir::Operation* operation) const {
        auto op = mlir::cast<ConcreteOp>(operation);
        auto mixedRawFilterShape = op.getMixedRawFilterShape();

        // Extract constant values where available; use kDynamic for non-constant dimensions.
        SmallVector<int64_t> filterShape;
        filterShape.reserve(mixedRawFilterShape.size());
        for (const auto& fold : mixedRawFilterShape) {
            if (auto constInt = mlir::getConstantIntValue(fold)) {
                filterShape.push_back(constInt.value());
            } else {
                filterShape.push_back(mlir::ShapedType::kDynamic);
            }
        }
        return Shape(filterShape);
    }

protected:
    SCFTilingInfo backInferConvInputTile(mlir::Location loc, mlir::OpBuilder& builder, const SCFTileInfo& outputTile,
                                         SCFShapeRef origInputShape, ArrayRef<int64_t> origOutputShape,
                                         ArrayRef<int64_t> kernelSize, mlir::ArrayAttr strides,
                                         ShapeRef boundedInputShape, const PadInfo& origPadding) const {
        SCFTileInfo inputTile(origInputShape, builder);
        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];

        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds = Bounds(boundedInputShape.raw());
        }

        auto padMap = origPadding.toPadByDims();
        auto pads = mlir::getAsIndexOpFoldResult(
                builder.getContext(), {origPadding.top, origPadding.left, origPadding.bottom, origPadding.right});
        // spatial dims are ind = 0 -> H, ind = 1 -> W
        // so padding for H -> pads[0] = padBefore = top and pads[2] = padAfter = bottom,
        //  for W -> pads[1] = padBefore = left and pads[3] = padAfter = right
        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);
            const auto stride = mlir::cast<mlir::IntegerAttr>(strides[index]).getValue().getSExtValue();
            const auto kernel = kernelSize[dim.ind()];

            auto [inputRange, dimBound] =
                    solutionForOutputRange(loc, builder, outputTile, dim, kernel, stride, origInputShape[dim.ind()],
                                           origOutputShape[dim.ind()], padMap[dim.ind()], pads[index], pads[index + 2]);

            if (inputRange.has_value()) {
                inputTile.offsets[dim.ind()] = inputRange.value().offset;
                inputTile.shape[dim.ind()] = inputRange.value().size;
            }

            if (dimBound.has_value()) {
                inputTile.bounds[dim] = dimBound.value();
            }
        }

        return SCFTilingInfo(inputTile, pads);
    }

public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto convOperation = mlir::cast<ConcreteOp>(operation);
        auto boundedInputShape = getBoundedShape(convOperation.getInput());
        const auto origInputShape = mlir::tensor::getMixedSizes(builder, operation->getLoc(), convOperation.getInput());

        const auto outputType = mlir::cast<mlir::ShapedType>(convOperation.getOutput().getType());
        const auto origOutputShape = outputType.getShape();
        const auto origFilterShape = getRawFilterShape(convOperation);
        const auto origPadding = toPadInfo(convOperation.getPad());

        SCFTilingInfo tilingInfo = backInferConvInputTile(operation->getLoc(), builder, outputTile, origInputShape,
                                                          origOutputShape, origFilterShape.raw(),
                                                          convOperation.getStrides(), boundedInputShape, origPadding);

        const auto tileOverChannels = !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1);

        if (tileOverChannels) {
            SCFTileInfo filterTile(getBoundedShape(convOperation.getFilter()), builder);

            filterTile.shape[Dims4D::Filter::OC.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
            filterTile.offsets[Dims4D::Filter::OC.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

            tilingInfo.tiles.emplace_back(filterTile);

            auto nceOp = mlir::dyn_cast<NCEOpInterface>(operation);
            if (nceOp != nullptr && nceOp.getWeightsTableOperand() != nullptr) {
                tilingInfo.tiles.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightsTableOperand().getType(), builder, outputTile));
            }
            if (nceOp != nullptr && nceOp.getWeightTableDataPtrOperand() != nullptr) {
                tilingInfo.tiles.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightTableDataPtrOperand().getType(), builder, outputTile));
            }
            if (nceOp != nullptr && nceOp.getWeightTableScaleOperand() != nullptr) {
                tilingInfo.tiles.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightTableScaleOperand().getType(), builder, outputTile));
            }
            if (nceOp != nullptr && nceOp.getWeightTableBiasOperand() != nullptr) {
                tilingInfo.tiles.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightTableBiasOperand().getType(), builder, outputTile));
            }
            if (nceOp != nullptr && nceOp.getWeightZeroPointsOperand() != nullptr) {
                tilingInfo.tiles.emplace_back(
                        getWeightsTableSCFTile(nceOp.getWeightZeroPointsOperand().getType(), builder, outputTile));
            }
        }

        return tilingInfo;
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, DimArrRef dims,
                                          SmallVector<mlir::Value>& tiledOperands, mlir::Operation* operation) const {
        auto generator = opGenerator;
        if (!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1)) {
            generator = [&]() -> mlir::Operation* {
                auto newOperation = mlir::cast<ConcreteOp>(opGenerator());
                auto mixedRawFilterShape = newOperation.getMixedRawFilterShape();
                mixedRawFilterShape[Dims4D::Filter::OC.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
                SmallVector<int64_t> staticValues;
                SmallVector<mlir::Value> dynamicValues;
                mlir::dispatchIndexOpFoldResults(mixedRawFilterShape, dynamicValues, staticValues);
                newOperation.setStaticRawFilterShape(staticValues);
                newOperation.getRawFilterShapeMutable().assign(dynamicValues);
                vpux::inferReturnTypes(newOperation, vpux::InferShapedTypeMode::SHAPE);
                return newOperation.getOperation();
            };
        }
        return createTiledPaddedOperation<ConcreteOp>(std::move(generator), std::move(operandsGenerator), builder,
                                                      inputTiling, outputTile, dims, tiledOperands, operation);
    }
};

class SCFConvOpModel : public SCFTilingConvModelOp<SCFConvOpModel, NCEConvolutionOp> {};

class SCFCompressConvOpModel : public SCFTilingConvModelOp<SCFCompressConvOpModel, NCECompressConvolutionOp> {};

class SCFTilingDepthConvModelOp : public SCFTilingConvModelOp<SCFTilingDepthConvModelOp, NCEDepthConvolutionOp> {
public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto inputConvTiling =
                SCFTilingConvModelOp<SCFTilingDepthConvModelOp, NCEDepthConvolutionOp>::backInferSCFTileInfo(
                        operation, builder, outputTile);

        const auto tileOverChannels = !mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1);

        if (tileOverChannels) {
            auto depthOperation = mlir::cast<VPU::NCEDepthConvolutionOp>(operation);
            const auto origFilterShape = getRawFilterShape(depthOperation);
            const auto origInputShape = getShape(depthOperation.getInput());
            auto& inputTiles = inputConvTiling.tiles[0];

            mlir::AffineExpr d0;
            bindDims(builder.getContext(), d0);

            const auto numOutChannelsPerGroup = origFilterShape[Dims4D::Filter::OC] / origInputShape[Dims4D::Act::C];
            auto loc = operation->getLoc();

            auto groupMap = mlir::AffineMap::get(
                    1, 0, {d0.floorDiv(numOutChannelsPerGroup) * origFilterShape[Dims4D::Filter::IC]},
                    builder.getContext());
            inputTiles.offsets[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, loc, groupMap, {outputTile.offsets[Dims4D::Act::C.ind()]});

            inputTiles.shape[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, loc, groupMap, {outputTile.shape[Dims4D::Act::C.ind()]});
        }

        return inputConvTiling;
    }
};

class SCFTilingPermuteModelOp : public SCFTilingPoolingModelOp<SCFTilingPermuteModelOp, NCEPermuteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& inputTiling, const SCFTileInfo& outputTile,
                                          DimArrRef, SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(inputTiling);

        auto generator = opGenerator;
        auto newChannelValue = mlir::getConstantIntValue(outputTile.shape[Dims4D::Act::C.ind()]);
        if (!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1) && newChannelValue.has_value()) {
            generator = [&]() -> mlir::Operation* {
                auto newOperation = mlir::cast<NCEPermuteOp>(opGenerator());
                newOperation.setExpandedChannels(newChannelValue.value());
                vpux::inferReturnTypes(newOperation, vpux::InferShapedTypeMode::SHAPE);

                return newOperation.getOperation();
            };
        }
        return generator();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto permuteOperation = mlir::cast<NCEOpInterface>(operation);
        auto inputShape = getShape(operation->getOperand(0));
        const auto origInputShape = mlir::getAsIndexOpFoldResult(builder.getContext(), inputShape.raw());
        const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());
        const auto origOutputShape = outputType.getShape();
        auto loc = operation->getLoc();
        const auto kernelSize = getIntArrayAttr(builder.getContext(), permuteOperation.getKernelSizeVal());
        const auto strides = getIntArrayAttr(builder.getContext(), permuteOperation.getStridesVal());

        auto inputTiling = SCFTilingPoolingModelOp<SCFTilingPermuteModelOp, NCEPermuteOp>::backInferPoolTile(
                loc, builder, outputTile, origInputShape, origOutputShape, kernelSize, strides, PadInfo(),
                std::nullopt);

        if (mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1)) {
            inputTiling.tiles[0].shape[Dims4D::Act::C.ind()] = origInputShape[Dims4D::Act::C.ind()];
            auto origInputRawShape = inputShape.raw();
            if (!inputTiling.tiles[0].bounds.raw().empty()) {
                inputTiling.tiles[0].bounds[Dims4D::Act::C] = origInputRawShape[Dims4D::Act::C.ind()];
            }
        } else {
            auto outputTileShape = mlir::getConstantIntValue(outputTile.shape[Dims4D::Act::C.ind()]);
            if (outputTileShape.has_value() && inputShape[Dims4D::Act::C] % outputTileShape.value() == 0) {
                inputTiling.tiles[0].shape[Dims4D::Act::C.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
            } else {
                mlir::AffineExpr d0, d1, s0;
                bindDims(builder.getContext(), d0, d1);
                bindSymbols(builder.getContext(), s0);
                auto reminderMap = mlir::AffineMap::get(2, 1, {s0 - d0, d1}, builder.getContext());
                inputTiling.tiles[0].shape[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineMin(
                        builder, loc, reminderMap,
                        {outputTile.offsets[Dims4D::Act::C.ind()], outputTile.shape[Dims4D::Act::C.ind()],
                         origInputShape[Dims4D::Act::C.ind()]});
            }
        }

        return inputTiling;
    }
};

class SCFDepthToSpaceModelOp : public SCFTilingCommonModelOp<SCFDepthToSpaceModelOp, VPU::DepthToSpaceOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

public:
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        VPUX_THROW_WHEN(!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1),
                        "[DepthToSpace] Dynamic tiling step for Channel dimension is not supported for operation: {0}, "
                        "outputTile: {1}",
                        operation->getLoc(), outputTile);
        auto loc = operation->getLoc();
        auto d2sOp = mlir::cast<VPU::DepthToSpaceOp>(operation);
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getBoundedShape(d2sOp.getInput()).raw());
        const auto blockSize = d2sOp.getBlockSize();

        SCFTileInfo inputTile(origInputShape, builder);
        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];

        int64_t paddedIC = 0;
        int64_t paddedOC = 0;

        auto paddedChannels = d2sOp.getPaddedChannels();
        if (paddedChannels.has_value()) {
            paddedIC = paddedChannels.value().getInput() ? paddedChannels.value().getInput().getInt() : 0;
            paddedOC = paddedChannels.value().getOutput() ? paddedChannels.value().getOutput().getInt() : 0;
        }

        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds = outputTile.bounds;
            inputTile.bounds[Dims4D::Act::C] =
                    (outputTile.bounds[Dims4D::Act::C] - paddedOC) * blockSize * blockSize + paddedIC;
            inputTile.bounds[Dims4D::Act::H] /= blockSize;
            inputTile.bounds[Dims4D::Act::W] /= blockSize;
        }

        mlir::AffineExpr dimC;
        bindDims(builder.getContext(), dimC);

        mlir::AffineExpr outCToInCExpr = (dimC - paddedOC) * blockSize * blockSize + paddedIC;
        auto outCToInCExprMap = mlir::AffineMap::get(1, 0, {outCToInCExpr}, builder.getContext());

        auto& outChShape = outputTile.shape[Dims4D::Act::C.ind()];

        inputTile.offsets[Dims4D::Act::C.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];
        inputTile.shape[Dims4D::Act::C.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                builder, appendLoc(loc, "inputChSize"), outCToInCExprMap, {outChShape});

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);

            mlir::AffineExpr d0;
            bindDims(builder.getContext(), d0);
            mlir::AffineExpr outToInExpr = d0.floorDiv(blockSize);
            auto outToInExprMap = mlir::AffineMap::get(1, 0, {outToInExpr}, builder.getContext());

            auto& outOffset = outputTile.offsets[dim.ind()];
            auto& outShape = outputTile.shape[dim.ind()];

            inputTile.offsets[dim.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "inputSpatialOffset"), outToInExprMap, {outOffset});
            inputTile.shape[dim.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "inputSpatialSize"), outToInExprMap, {outShape});
        }

        return SCFTilingInfo(inputTile);
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& inputTiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(inputTiling);
        return opGenerator();
    }
};

class SCFNCEReduceModelOp : public SCFTilingCommonModelOp<SCFNCEReduceModelOp, VPU::NCEReduceOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* op, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        SCFTileInfo inputTile = outputTile;
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(op->getOperand(0)).raw());
        const auto origBoundedInputShape = getBoundedShape(op->getOperand(0));

        inputTile.offsets[Dims4D::Act::C.ind()] = builder.getIndexAttr(0);
        inputTile.shape[Dims4D::Act::C.ind()] = origInputShape[Dims4D::Act::C.ind()];
        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds[Dims4D::Act::C] = origBoundedInputShape[Dims4D::Act::C];
        }
        return SCFTilingInfo(inputTile);
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& inputTiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(inputTiling);
        return opGenerator();
    }
};

class SCFInterpolateModelOp : public SCFTilingCommonModelOp<SCFInterpolateModelOp, VPU::InterpolateOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    // Main back-inference function
    // Mirrors: backInferInterpolateTile() in tiling.cpp
    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto interpolateOp = mlir::cast<VPU::InterpolateOp>(operation);
        auto loc = operation->getLoc();

        auto interpolateMode = interpolateOp.getAttr().getMode().getValue();
        auto coordMode = interpolateOp.getAttr().getCoordMode().getValue();
        auto nearestMode = interpolateOp.getAttr().getNearestMode().getValue();

        // Always use getBoundedShape to resolve dynamic dims to upper bounds.
        // The attrs may contain kDynamic values if backInferTileInfo (non-SCF path) ran earlier
        // and set them using getShape() on dynamic tensors. Using kDynamic here would cause
        // initialInSize == initialOutSize to be true, skipping the affine back-inference.
        const SmallVector<int64_t> initialInputDims(getBoundedShape(interpolateOp.getInput()).raw());
        const SmallVector<int64_t> initialOutputDims(getBoundedShape(interpolateOp.getOutput()).raw());

        auto currentInputDims = getBoundedShape(interpolateOp.getInput()).raw();
        const auto origInputShape = mlir::getAsIndexOpFoldResult(builder.getContext(), currentInputDims);

        SCFTileInfo inputTile(origInputShape, builder);

        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];

        inputTile.shape[Dims4D::Act::C.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
        inputTile.offsets[Dims4D::Act::C.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds = outputTile.bounds;
        }

        SmallVector<mlir::OpFoldResult> outOffsetBegin;
        SmallVector<mlir::OpFoldResult> outOffsetEnd;
        outOffsetBegin.reserve(outputTile.offsets.size());
        outOffsetEnd.reserve(outputTile.offsets.size());

        for (size_t i = 0; i < outputTile.offsets.size(); ++i) {
            outOffsetBegin.push_back(outputTile.offsets[i]);

            mlir::AffineExpr d0, d1;
            bindDims(builder.getContext(), d0, d1);
            auto endMap = mlir::AffineMap::get(2, 0, {d0 + d1 - 1}, builder.getContext());
            auto endOffset = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "outOffsetEnd"), endMap, {outputTile.offsets[i], outputTile.shape[i]});
            outOffsetEnd.push_back(endOffset);
        }

        auto inOffsetBegin = backInferSCFOffsetForInterpolate(builder, loc, outOffsetBegin, interpolateMode, coordMode,
                                                              nearestMode, initialInputDims, initialOutputDims,
                                                              currentInputDims, /*roundUp=*/false);
        auto inOffsetEnd = backInferSCFOffsetForInterpolate(builder, loc, outOffsetEnd, interpolateMode, coordMode,
                                                            nearestMode, initialInputDims, initialOutputDims,
                                                            currentInputDims, /*roundUp=*/true);

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            const auto dim = Dims4D::Act::getSpatialDim(index);
            const int64_t initialInSize = initialInputDims[dim.ind()];
            const int64_t initialOutSize = initialOutputDims[dim.ind()];

            if (initialInSize == initialOutSize) {
                inputTile.shape[dim.ind()] = outputTile.shape[dim.ind()];
                inputTile.offsets[dim.ind()] = outputTile.offsets[dim.ind()];
                continue;
            }

            inputTile.offsets[dim.ind()] = inOffsetBegin[dim.ind()];

            mlir::AffineExpr d0, d1;
            bindDims(builder.getContext(), d0, d1);
            auto sizeMap = mlir::AffineMap::get(2, 0, {d1 - d0 + 1}, builder.getContext());
            inputTile.shape[dim.ind()] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "inputSize"), sizeMap, {inOffsetBegin[dim.ind()], inOffsetEnd[dim.ind()]});

            // Update bounds if present
            if (!outputTile.bounds.raw().empty()) {
                inputTile.bounds[dim] = (outputTile.bounds[dim] * initialInSize + initialOutSize - 1) / initialOutSize;
            }
        }

        return SCFTilingInfo(inputTile);
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& opBuilder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, DimArrRef, SmallVector<mlir::Value>&,
                                          mlir::Operation* operation) const {
        operandsGenerator(inputTiling);

        auto generator = [&]() -> mlir::Operation* {
            auto newOp = mlir::cast<VPU::InterpolateOp>(opGenerator());
            auto ctx = newOp->getContext();
            mlir::Builder attrBuilder(ctx);
            auto origInterpolate = mlir::cast<VPU::InterpolateOp>(operation);

            const auto numDims = static_cast<int64_t>(outputTile.shape.size());
            const auto i64Type = opBuilder.getI64Type();
            const auto offsetsTensorType = mlir::RankedTensorType::get({numDims}, i64Type);

            SmallVector<int64_t> inputOffsets(numDims, 0);
            SmallVector<int64_t> outputOffsets(numDims, 0);
            bool hasDynamicInputOffset = false;
            bool hasDynamicOutputOffset = false;

            // First pass: classify each offset dimension and fill the initial_*_offset_attr arrays.
            // No ops are created here, so the common fully-static tiling case leaves no offset ops
            // behind. Returns true when the offset is dynamic (not a compile-time constant).
            const auto classifyOffset = [&](mlir::OpFoldResult ofr, SmallVector<int64_t>& staticOffsets,
                                            int64_t dim) -> bool {
                if (auto val = mlir::getConstantIntValue(ofr)) {
                    staticOffsets[dim] = val.value();
                    return false;
                }
                staticOffsets[dim] = mlir::ShapedType::kDynamic;
                return true;
            };
            for (auto i : irange(numDims)) {
                hasDynamicInputOffset |= classifyOffset(inputTiling.tiles[0].offsets[i], inputOffsets, i);
                hasDynamicOutputOffset |= classifyOffset(outputTile.offsets[i], outputOffsets, i);
            }
            newOp.setInitialInputOffsetAttrAttr(attrBuilder.getI64ArrayAttr(inputOffsets));
            newOp.setInitialOutputOffsetAttrAttr(attrBuilder.getI64ArrayAttr(outputOffsets));

            // Save/restore insertion point: materialized offset ops (constants and IndexCast/ExtSI)
            // must be placed before newOp so their results dominate it.
            mlir::OpBuilder::InsertionGuard insertGuard(opBuilder);
            opBuilder.setInsertionPoint(newOp);

            // Second pass, taken only when at least one dimension is dynamic: materialize every
            // per-dim offset as an i64 value (constant dims become arith.constant, dynamic dims are
            // cast to i64). Fully-static offsets omit the dynamic_*_offsets operand and create no
            // offset ops.
            const auto materializeOffsetValues = [&](bool hasDynamic, const auto& offsets) -> SmallVector<mlir::Value> {
                SmallVector<mlir::Value> values;
                if (!hasDynamic) {
                    return values;
                }
                values.reserve(numDims);
                for (auto i : irange(numDims)) {
                    mlir::OpFoldResult ofr = offsets[i];
                    if (auto val = mlir::getConstantIntValue(ofr)) {
                        values.push_back(opBuilder.create<mlir::arith::ConstantIntOp>(newOp.getLoc(), val.value(), 64));
                        continue;
                    }
                    auto offsetVal = mlir::getValueOrCreateConstantIndexOp(opBuilder, newOp.getLoc(), ofr);
                    if (offsetVal.getType().isIndex()) {
                        offsetVal = opBuilder.create<mlir::arith::IndexCastOp>(newOp.getLoc(), i64Type, offsetVal);
                    } else if (!offsetVal.getType().isSignlessInteger(64)) {
                        offsetVal = opBuilder.create<mlir::arith::ExtSIOp>(newOp.getLoc(), i64Type, offsetVal);
                    }
                    values.push_back(offsetVal);
                }
                return values;
            };
            // Materialize both operands' offset values before building any tensor so all offset
            // values precede the packed tensors (preserves the operation ordering).
            auto inputOffsetValues = materializeOffsetValues(hasDynamicInputOffset, inputTiling.tiles[0].offsets);
            auto outputOffsetValues = materializeOffsetValues(hasDynamicOutputOffset, outputTile.offsets);

            // Pack the materialized offset values into a tensor operand for each dynamic operand.
            const auto assignDynamicOffsets = [&](bool hasDynamic, SmallVector<mlir::Value>& values,
                                                  auto mutableAccessor) {
                if (!hasDynamic) {
                    return;
                }
                auto offsetsTensor =
                        opBuilder.create<mlir::tensor::FromElementsOp>(newOp.getLoc(), offsetsTensorType, values);
                mutableAccessor().assign(mlir::ValueRange{offsetsTensor.getResult()});
            };
            assignDynamicOffsets(hasDynamicInputOffset, inputOffsetValues, [&]() {
                return newOp.getDynamicInputOffsetsMutable();
            });
            assignDynamicOffsets(hasDynamicOutputOffset, outputOffsetValues, [&]() {
                return newOp.getDynamicOutputOffsetsMutable();
            });

            SmallVector<double> zeroTileOffset(numDims, 0.0);
            newOp.setTileOffsetAttrAttr(attrBuilder.getF64ArrayAttr(zeroTileOffset));

            SmallVector<int64_t> zeroPads(numDims, 0);
            auto origAttr = newOp.getAttr();

            // Use getBoundedShape to resolve dynamic dims to upper bounds, avoiding kDynamic in attrs
            const SmallVector<int64_t> initialInputDims(getBoundedShape(origInterpolate.getInput()).raw());
            const SmallVector<int64_t> initialOutputDims(getBoundedShape(origInterpolate.getOutput()).raw());
            newOp.setInitialInputDimsAttrAttr(attrBuilder.getI64ArrayAttr(initialInputDims));
            newOp.setInitialOutputDimsAttrAttr(attrBuilder.getI64ArrayAttr(initialOutputDims));

            // The VPU dialect pins Interpolate to SIZES shape_calc_mode and uses useScaleAttr
            // (preserved on the clone) as the sole marker for whether scales_attr is the
            // authoritative coordinate-transform scale. Keep shape_calc_mode at SIZES and recompute
            // a static per-tile sizes_attr so inferReturnTypes sizes each tile; scales_attr and
            // useScaleAttr stay as cloned (mirrors the non-SCF VPU::InterpolateOp::adjustAttrs path).
            // A static sizes_attr is available when every interpolated axis has a compile-time
            // constant tile extent (static tiling) or a known upper bound (dynamic SCF tiling). When
            // a tile extent is a non-constant value without bounds (e.g. cluster tiling via
            // scf.forall), the tile falls back to SCALES mode with the global output/input dim ratio
            // so inferReturnTypes derives each tile's output shape from its input shape.
            auto calcModeAttr = origAttr.getShapeCalcMode();
            auto axesAttrOpt = origInterpolate.getAxesAttr();
            if (axesAttrOpt.has_value()) {
                auto axesValue = parseIntArrayAttr<int64_t>(axesAttrOpt.value());
                SmallVector<int64_t> sizes;
                sizes.reserve(axesValue.size());
                bool staticSizesAvailable = true;
                for (auto axis : axesValue) {
                    const auto axisDim = Dim(axis);
                    if (auto tileSize = mlir::getConstantIntValue(outputTile.shape[axisDim.ind()])) {
                        sizes.push_back(tileSize.value());
                    } else if (!outputTile.bounds.raw().empty()) {
                        sizes.push_back(outputTile.bounds[axisDim]);
                    } else {
                        staticSizesAvailable = false;
                        break;
                    }
                }
                if (staticSizesAvailable) {
                    newOp.setSizesAttrAttr(attrBuilder.getI64ArrayAttr(sizes));
                } else {
                    // If scales are authoritative (SCALES-authored), keep the cloned scales_attr.
                    // Otherwise (SIZES-authored), derive a bounded-dims ratio scale for shape inference.
                    if (!newOp.getUseScaleAttr() || !newOp.getScalesAttr().has_value()) {
                        SmallVector<double> scales(axesValue.size(), 1.0);
                        for (size_t idx = 0; idx < axesValue.size(); ++idx) {
                            const auto axisDim = Dim(axesValue[idx]);
                            const auto inDim = initialInputDims[axisDim.ind()];
                            const auto outDim = initialOutputDims[axisDim.ind()];
                            if (inDim != 0) {
                                scales[idx] = static_cast<double>(outDim) / static_cast<double>(inDim);
                            }
                        }
                        newOp.setScalesAttrAttr(attrBuilder.getF64ArrayAttr(scales));
                    }
                    calcModeAttr = IE::InterpolateCalcModeAttr::get(ctx, IE::InterpolateCalcMode::SCALES);
                }
            }

            // Only the per-tile pads are reset here; the per-tile output sizing is carried by
            // sizes_attr (SIZES mode) or scales_attr (SCALES fallback) set above.
            auto newInterpolateAttr = IE::InterpolateAttr::get(
                    ctx, origAttr.getMode(), calcModeAttr, origAttr.getCoordMode(), origAttr.getNearestMode(),
                    origAttr.getAntialias(), attrBuilder.getI64ArrayAttr(zeroPads),
                    attrBuilder.getI64ArrayAttr(zeroPads), origAttr.getCubeCoeff());
            newOp.setAttrAttr(newInterpolateAttr);

            vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);
            return newOp.getOperation();
        };

        return generator();
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class SCFReduceModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* op, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        const auto axesValue = parseIntArrayAttr<int64_t>(mlir::cast<ConcreteOp>(op).getAxesValue());
        const auto inShape = mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(op->getOperand(0)).raw());
        const auto origBoundedInputShape = getBoundedShape(op->getOperand(0));

        VPUX_THROW_WHEN(!mlir::cast<ConcreteOp>(op).getKeepDims(),
                        "[SW Reduce] Expected reduce op to have keep_dims {0} "
                        "outputTile: {1}",
                        op->getLoc(), outputTile);

        SCFTileInfo inTile = outputTile;
        for (auto axesInd : axesValue) {
            inTile.shape[Dim(axesInd).ind()] = inShape[Dim(axesInd).ind()];
            if (!outputTile.bounds.raw().empty()) {
                inTile.bounds[Dim(axesInd)] = origBoundedInputShape[Dim(axesInd)];
            }
        }

        return SCFTilingInfo(inTile);
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& inputTiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(inputTiling);
        return opGenerator();
    }
};

class SCFReduceLogicalOrModelOp : public SCFReduceModelOp<SCFReduceLogicalOrModelOp, VPU::ReduceLogicalOrOp> {};
class SCFReduceLogicalAndModelOp : public SCFReduceModelOp<SCFReduceLogicalAndModelOp, VPU::ReduceLogicalAndOp> {};
class SCFReduceMeanModelOp : public SCFReduceModelOp<SCFReduceMeanModelOp, VPU::ReduceMeanOp> {};
class SCFReduceSumModelOp : public SCFReduceModelOp<SCFReduceSumModelOp, VPU::ReduceSumOp> {};
class SCFReduceL2ModelOp : public SCFReduceModelOp<SCFReduceL2ModelOp, VPU::ReduceL2Op> {};
class SCFReduceL1ModelOp : public SCFReduceModelOp<SCFReduceL1ModelOp, VPU::ReduceL1Op> {};
class SCFReduceSquareModelOp : public SCFReduceModelOp<SCFReduceSquareModelOp, VPU::ReduceSquareOp> {};
class SCFReduceMinModelOp : public SCFReduceModelOp<SCFReduceMinModelOp, VPU::ReduceMinOp> {};
class SCFReduceMaxModelOp : public SCFReduceModelOp<SCFReduceMaxModelOp, VPU::ReduceMaxOp> {};
class SCFReduceProdModelOp : public SCFReduceModelOp<SCFReduceProdModelOp, VPU::ReduceProdOp> {};

class SCFYuvToRgbModelOp : public SCFTilingCommonModelOp<SCFYuvToRgbModelOp, VPU::YuvToRgbOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& inputTiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(inputTiling);
        return opGenerator();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto loc = operation->getLoc();
        auto ctx = builder.getContext();
        const auto operands = operation->getOperands();
        const auto numInputs = static_cast<int64_t>(operands.size());

        const auto dimN = Dims4D::Act::N.ind();
        const auto dimC = Dims4D::Act::C.ind();
        const auto dimH = Dims4D::Act::H.ind();

        auto inputTiles = SmallVector<SCFTileInfo>();

        // d0 -> floor(d0 / 2) for chroma spatial halving
        auto d0 = mlir::AffineExpr();
        bindDims(ctx, d0);
        auto halfMap = mlir::AffineMap::get(1, 0, {d0.floorDiv(2)}, ctx);

        if (numInputs == 1) {
            // Single-plane NV12 (1 input)
            // Physical: [N, C*3/2, H, 1]
            // Luma and chroma rows stacked vertically — C is NOT tileable.
            // Tile N and H from output; C and W keep original.
            const auto origShape = mlir::tensor::getMixedSizes(builder, loc, operands[0]);
            auto inputTile = SCFTileInfo(origShape, builder);

            inputTile.offsets[dimN] = outputTile.offsets[dimN];
            inputTile.shape[dimN] = outputTile.shape[dimN];

            inputTile.offsets[dimH] = outputTile.offsets[dimH];
            inputTile.shape[dimH] = outputTile.shape[dimH];

            if (!outputTile.bounds.raw().empty()) {
                auto inputBoundedShape = getBoundedShape(operands[0]);
                inputTile.bounds = Bounds(inputBoundedShape.raw());
                inputTile.bounds[Dims4D::Act::N] = outputTile.bounds[Dims4D::Act::N];
                inputTile.bounds[Dims4D::Act::H] = outputTile.bounds[Dims4D::Act::H];
            }

            return SCFTilingInfo{inputTile};
        }

        // Multi-plane (2 inputs = NV12, 3 inputs = I420)

        // Input 0 (Y plane): [N, C, H, 1]
        // Tile N, C, H from output; W stays original (1 vs 3)
        {
            const auto origShape = mlir::tensor::getMixedSizes(builder, loc, operands[0]);
            auto yTile = SCFTileInfo(origShape, builder);

            yTile.offsets[dimN] = outputTile.offsets[dimN];
            yTile.shape[dimN] = outputTile.shape[dimN];

            yTile.offsets[dimC] = outputTile.offsets[dimC];
            yTile.shape[dimC] = outputTile.shape[dimC];

            yTile.offsets[dimH] = outputTile.offsets[dimH];
            yTile.shape[dimH] = outputTile.shape[dimH];

            if (!outputTile.bounds.raw().empty()) {
                auto yBoundedShape = getBoundedShape(operands[0]);
                yTile.bounds = Bounds(yBoundedShape.raw());
                yTile.bounds[Dims4D::Act::N] = outputTile.bounds[Dims4D::Act::N];
                yTile.bounds[Dims4D::Act::C] = outputTile.bounds[Dims4D::Act::C];
                yTile.bounds[Dims4D::Act::H] = outputTile.bounds[Dims4D::Act::H];
            }

            inputTiles.push_back(std::move(yTile));
        }

        // Inputs 1+ (chroma planes): [N, C/2, H/2, W_chroma]
        //   NV12 input 1 (UV): W_chroma = 2
        //   I420 input 1 (U):  W_chroma = 1
        //   I420 input 2 (V):  W_chroma = 1
        // Tile N from output; C and H halved from output; W stays original
        for (auto idx : irange<int64_t>(1, numInputs)) {
            const auto origShape = mlir::tensor::getMixedSizes(builder, loc, operands[idx]);
            auto chromaTile = SCFTileInfo(origShape, builder);

            chromaTile.offsets[dimN] = outputTile.offsets[dimN];
            chromaTile.shape[dimN] = outputTile.shape[dimN];

            chromaTile.offsets[dimC] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "chromaOffsetC"), halfMap, {outputTile.offsets[dimC]});
            chromaTile.shape[dimC] = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "chromaSizeC"),
                                                                                 halfMap, {outputTile.shape[dimC]});

            chromaTile.offsets[dimH] = mlir::affine::makeComposedFoldedAffineApply(
                    builder, appendLoc(loc, "chromaOffsetH"), halfMap, {outputTile.offsets[dimH]});
            chromaTile.shape[dimH] = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "chromaSizeH"),
                                                                                 halfMap, {outputTile.shape[dimH]});

            if (!outputTile.bounds.raw().empty()) {
                auto chromaBoundedShape = getBoundedShape(operands[idx]);
                chromaTile.bounds = Bounds(chromaBoundedShape.raw());
                chromaTile.bounds[Dims4D::Act::N] = outputTile.bounds[Dims4D::Act::N];
                chromaTile.bounds[Dims4D::Act::C] = outputTile.bounds[Dims4D::Act::C] / 2;
                chromaTile.bounds[Dims4D::Act::H] = outputTile.bounds[Dims4D::Act::H] / 2;
            }

            inputTiles.push_back(std::move(chromaTile));
        }

        return SCFTilingInfo{inputTiles};
    }
};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingTopKModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& tiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(tiling);
        return opGenerator();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto topKOp = mlir::cast<ConcreteOp>(operation);
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(operation->getOperand(0)).raw());
        const auto origBoundedInputShape = getBoundedShape(operation->getOperand(0));
        const auto kAxis = Dim(topKOp.getAxis());

        // Start from output tile (inherits bounds for dynamic tensors),
        // then restore the TopK axis to full input size
        SCFTileInfo inputTile = outputTile;
        inputTile.shape[kAxis.ind()] = origInputShape[kAxis.ind()];
        inputTile.offsets[kAxis.ind()] = builder.getIndexAttr(0);
        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds[kAxis] = origBoundedInputShape[kAxis];
        }

        SmallVector<SCFTileInfo> inputTiles;
        inputTiles.push_back(inputTile);

        // k operand (scalar) — pass through untiled
        if (topKOp.getK()) {
            const auto kShape = mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(topKOp.getK()).raw());
            inputTiles.emplace_back(kShape, builder);
        }

        // lineBuffer (auxiliary) — pass through untiled
        if (topKOp.getLineBuffer()) {
            const auto bufShape =
                    mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(topKOp.getLineBuffer()).raw());
            inputTiles.emplace_back(bufShape, builder);
        }

        return SCFTilingInfo{std::move(inputTiles)};
    }
};
class SCFTopKModelOp : public SCFTilingTopKModelOp<SCFTopKModelOp, VPU::TopKOp> {};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingLSTMGatesModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto lstmOp = mlir::cast<ConcreteOp>(operation);

        // Mirror the static backInferTileInfo: tile all operands,
        // restoring the last dimension to each operand's original size.
        // Start from outputTile to inherit bounds for dynamic tensors.
        SmallVector<SCFTileInfo> inputTiles;
        for (const auto& origInput : lstmOp.getInputs()) {
            const auto curShape = getShape(origInput);
            const auto origInputShape = mlir::getAsIndexOpFoldResult(builder.getContext(), curShape.raw());
            const auto origBoundedInputShape = getBoundedShape(origInput);
            const auto lastDim = curShape.size() - 1;

            // Start from outputTile (preserves bounds and dynamic dim handling)
            SCFTileInfo curTile = outputTile;
            // Restore the last dim to the original input size (4*hidden vs hidden)
            curTile.shape[lastDim] = origInputShape[lastDim];
            curTile.offsets[lastDim] = builder.getIndexAttr(0);
            if (!outputTile.bounds.raw().empty()) {
                curTile.bounds[Dim(lastDim)] = origBoundedInputShape[Dim(lastDim)];
            }
            inputTiles.push_back(curTile);
        }

        return SCFTilingInfo{std::move(inputTiles)};
    }
    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& tiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(tiling);
        return opGenerator();
    }
};
class SCFLSTMGatesModelOp : public SCFTilingLSTMGatesModelOp<SCFLSTMGatesModelOp, VPU::LSTMGatesOp> {};

template <typename ConcreteModel, typename ConcreteOp>
class SCFTilingMVN1NormalizeModelOp : public SCFTilingCommonModelOp<ConcreteModel, ConcreteOp> {
public:
    mlir::LogicalResult getResultTilePosition(mlir::Operation* operation, mlir::OpBuilder& builder,
                                              unsigned resultNumber, ArrayRef<mlir::OpFoldResult> offsets,
                                              ArrayRef<mlir::OpFoldResult> sizes,
                                              SmallVector<mlir::OpFoldResult>& resultOffsets,
                                              SmallVector<mlir::OpFoldResult>& resultSizes) const {
        this->fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);
        return mlir::success();
    }

    SCFTilingInfo backInferSCFTileInfo(mlir::Operation* operation, mlir::OpBuilder& builder,
                                       const SCFTileInfo& outputTile) const {
        auto mvnOp = mlir::cast<ConcreteOp>(operation);

        // input (operand 0): tiles like output
        SmallVector<SCFTileInfo> inputTiles;
        inputTiles.push_back(outputTile);

        // meanVar (operand 1): untiled — pass through with full shape
        const auto meanVarShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(mvnOp.getMeanVar()).raw());
        inputTiles.emplace_back(meanVarShape, builder);

        return SCFTilingInfo{std::move(inputTiles)};
    }

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder&, SCFTilingInfo& tiling, const SCFTileInfo&, DimArrRef,
                                          SmallVector<mlir::Value>&, mlir::Operation*) const {
        operandsGenerator(tiling);
        return opGenerator();
    }
};
class SCFMVN1NormalizeModelOp : public SCFTilingMVN1NormalizeModelOp<SCFMVN1NormalizeModelOp, VPU::MVN1NormalizeOp> {};
}  // namespace vpux::VPU
