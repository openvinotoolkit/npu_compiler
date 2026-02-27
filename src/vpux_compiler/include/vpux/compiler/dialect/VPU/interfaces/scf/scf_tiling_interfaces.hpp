//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

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

    mlir::Operation* createTiledOperation(OpGeneratorFunc opGenerator, OpTilingOperandsFunc operandsGenerator,
                                          mlir::OpBuilder& builder, SCFTilingInfo& inputTiling,
                                          const SCFTileInfo& outputTile, DimArrRef dims,
                                          SmallVector<mlir::Value>& tiledOperands, mlir::Operation* operation) const {
        auto generatedOp = static_cast<const ConcreteModel*>(this)->createTiledOperation(
                std::move(opGenerator), std::move(operandsGenerator), builder, inputTiling, outputTile, dims,
                tiledOperands, operation);

        return castOutputForInsertion(builder, outputTile, dims, operation, generatedOp);
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
        SmallVector<mlir::OpFoldResult> resultOffsets;
        SmallVector<mlir::OpFoldResult> resultSizes;

        // E-162801 extend to multiple results
        unsigned resultNumber = 0;
        fillInResultTilePositions(operation, builder, resultNumber, offsets, sizes, resultOffsets, resultSizes);

        const auto strategy =
                Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));
        auto tilingDims = getSCFTilingOrderedDims(operation, strategy);

        SmallVector<mlir::Operation*> results;
        SmallVector<mlir::Value> resultValues;
        SmallVector<mlir::Operation*> generatedSlices;
        results.reserve(resultOffsets.size());
        resultValues.reserve(resultOffsets.size());
        SmallVector<mlir::OpFoldResult> axis(resultOffsets.size(), builder.getIndexAttr(1));
        auto origShape = mlir::getAsIndexOpFoldResult(
                builder.getContext(), mlir::cast<mlir::ShapedType>(operation->getResult(0).getType()).getShape());

        Bounds resultBounds;
        if (IE::hasDynamicTensors(operation)) {
            if (mlir::failed(getResultTileBounds(operation, resultNumber, tilingDims, resultSizes, resultBounds))) {
                return mlir::failure();
            }
        }

        for (auto tileDim : tilingDims) {
            axis[tileDim.ind()] = builder.getIndexAttr(strategy[tileDim]);
        }
        auto outputTile = SCFTileInfo(resultSizes, resultOffsets, axis, resultBounds);
        auto inputTiling = backInferSCFTileInfo(operation, builder, outputTile);
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
            auto resultDenseTile = extractResultType(operation->getResult(0).getType(), resultSizes, resultBounds);
            auto* tiledOp = mlir::cloneWithoutRegions(builder, operation, {resultDenseTile}, tiledOperands);
            vpux::inferReturnTypes(tiledOp, vpux::InferShapedTypeMode::SHAPE);
            tiledOp->removeAttr(tilingStrategy);
            return tiledOp;
        };

        auto* resultOp = createTiledOperation(std::move(generatorFunc), std::move(createTiledOperands), builder,
                                              inputTiling, outputTile, tilingDims, tiledOperands, operation);

        results.emplace_back(resultOp);
        resultValues.emplace_back(resultOp->getResult(0));

        return mlir::TilingResult{std::move(results), std::move(resultValues), std::move(generatedSlices)};
    }

    mlir::FailureOr<mlir::TilingResult> generateResultTileValue(mlir::Operation* operation, mlir::OpBuilder& builder,
                                                                unsigned, mlir::ArrayRef<mlir::OpFoldResult> offsets,
                                                                mlir::ArrayRef<mlir::OpFoldResult> sizes) const {
        return getTiledImplementation(operation, builder, offsets, sizes);
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

        mlir::AffineExpr dimC;
        bindDims(builder.getContext(), dimC);
        auto alignmentExpr = builder.getAffineConstantExpr(inChannelAlignment);
        auto oneExpr = builder.getAffineConstantExpr(1);

        // Expression: ((dimC + alignment - 1) floordiv alignment) * alignment
        mlir::AffineExpr alignedExpr = ((dimC + alignmentExpr - oneExpr).floorDiv(alignmentExpr)) * alignmentExpr;

        auto alignMap = mlir::AffineMap::get(1, 0, {alignedExpr}, builder.getContext());

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
                                    mlir::ArrayAttr kernelSize, mlir::ArrayAttr strides,
                                    const PadInfo& origPadding) const {
        SCFTileInfo inputTile(origInputShape, builder);
        inputTile.shape[Dims4D::Act::N.ind()] = outputTile.shape[Dims4D::Act::N.ind()];
        inputTile.offsets[Dims4D::Act::N.ind()] = outputTile.offsets[Dims4D::Act::N.ind()];
        inputTile.shape[Dims4D::Act::C.ind()] = outputTile.shape[Dims4D::Act::C.ind()];
        inputTile.offsets[Dims4D::Act::C.ind()] = outputTile.offsets[Dims4D::Act::C.ind()];

        if (!outputTile.bounds.raw().empty()) {
            inputTile.bounds = outputTile.bounds;
        }

        auto padMap = origPadding.toPadByDims();

        auto pads = mlir::getAsIndexOpFoldResult(
                builder.getContext(), {origPadding.left, origPadding.top, origPadding.right, origPadding.bottom});

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
        auto poolingOperation = mlir::cast<ConcreteOp>(operation);
        const auto origInputShape =
                mlir::tensor::getMixedSizes(builder, operation->getLoc(), poolingOperation.getInput());

        const auto outputType = mlir::cast<mlir::ShapedType>(poolingOperation.getOutput().getType());
        const auto origOutputShape = outputType.getShape();
        const auto origPadding = toPadInfo(poolingOperation.getPad());

        auto inputTiling =
                backInferPoolTile(operation->getLoc(), builder, outputTile, origInputShape, origOutputShape,
                                  poolingOperation.getKernelSize(), poolingOperation.getStrides(), origPadding);

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
        return Shape(parseIntArrayAttr<int64_t>(mlir::cast<ConcreteOp>(operation).getRawFilterShape()));
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
                builder.getContext(), {origPadding.left, origPadding.top, origPadding.right, origPadding.bottom});
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
        auto newChannelValue = mlir::getConstantIntValue(outputTile.shape[Dims4D::Act::C.ind()]);
        if (!mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1) && newChannelValue.has_value()) {
            generator = [&]() -> mlir::Operation* {
                auto newOperation = mlir::cast<ConcreteOp>(opGenerator());
                auto newRawFilterShape = getRawFilterShape(newOperation);
                newRawFilterShape[Dims4D::Filter::OC] = newChannelValue.value();
                newOperation.setRawFilterShapeAttr(getIntArrayAttr(newOperation->getContext(), newRawFilterShape));
                vpux::inferReturnTypes(newOperation, vpux::InferShapedTypeMode::SHAPE);

                return newOperation.getOperation();
            };
        }
        return createTiledPaddedOperation<ConcreteOp>(std::move(generator), std::move(operandsGenerator), builder,
                                                      inputTiling, outputTile, dims, tiledOperands, operation);
    }
};

class SCFConvOpModel : public SCFTilingConvModelOp<SCFConvOpModel, NCEConvolutionOp> {};

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
        const auto origInputShape =
                mlir::getAsIndexOpFoldResult(builder.getContext(), getShape(operation->getOperand(0)).raw());
        const auto outputType = mlir::cast<mlir::ShapedType>(operation->getResult(0).getType());
        const auto origOutputShape = outputType.getShape();
        auto loc = operation->getLoc();
        const auto kernelSize = getIntArrayAttr(builder.getContext(), permuteOperation.getKernelSizeVal());
        const auto strides = getIntArrayAttr(builder.getContext(), permuteOperation.getStridesVal());

        auto inputTiling = SCFTilingPoolingModelOp<SCFTilingPermuteModelOp, NCEPermuteOp>::backInferPoolTile(
                loc, builder, outputTile, origInputShape, origOutputShape, kernelSize, strides, PadInfo());

        if (mlir::isConstantIntValue(outputTile.axis[Dims4D::Act::C.ind()], 1)) {
            inputTiling.tiles[0].shape[Dims4D::Act::C.ind()] = origInputShape[Dims4D::Act::C.ind()];
            auto origInputRawShape = getShape(operation->getOperand(0)).raw();
            if (!inputTiling.tiles[0].bounds.raw().empty()) {
                inputTiling.tiles[0].bounds[Dims4D::Act::C] = origInputRawShape[Dims4D::Act::C.ind()];
            }
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

}  // namespace vpux::VPU
