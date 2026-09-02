//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dynamic_shape_propagation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <optional>
#include <utility>

using namespace vpux;

mlir::LogicalResult vpux::VPU::InterpolateOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::InterpolateOpAdaptor interpolate(operands, attrs, prop);
    if (mlir::failed(interpolate.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(interpolate.getInput().getType());

    // calcOutputShapes uses getBoundedShape internally for proper output shape computation
    auto outShapeVec = IE::calcOutputShapes(interpolate, loc, Logger::global(), ctx);

    auto [outDesc, outShape] = callOnShapeOf(inputType, [&](const auto& shape) {
        if constexpr (std::is_same_v<std::decay_t<decltype(shape)>, BoundedShape>) {
            // For bounded tensors, the output bounds are the computed output shape
            auto desc =
                    vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), BoundsRef(outShapeVec));
            const auto axesVal = IE::getInterpAxesVal(loc, interpolate.getAxes(), interpolate.getAxesAttr(), inputType);
            auto staticShape = outShapeVec;
            for (const auto& axis : axesVal) {
                if (inputType.getShape()[Dim(axis)] == mlir::ShapedType::kDynamic) {
                    staticShape[axis] = mlir::ShapedType::kDynamic;
                }
                // If input dim is static, keep the computed output dim (already in outShapeVec)
            }
            return std::make_pair(desc, staticShape);
        } else if constexpr (std::is_same_v<std::decay_t<decltype(shape)>, DimsMaskedShape>) {
            auto inDynamicDimsMaskType = mlir::cast<Core::DynamicDimsMaskTensorType>(inputType);
            auto desc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace(), {},
                                            inDynamicDimsMaskType.getDynamicDimsMask());
            return std::make_pair(desc, outShapeVec);
        } else {
            // Static shape case
            auto desc = vpux::getTensorAttr(ctx, inputType.getDimsOrder(), inputType.getMemSpace());
            return std::make_pair(desc, outShapeVec);
        }
    });

    auto outputType = mlir::RankedTensorType::get(outShape, inputType.getElementType(), outDesc);
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::InterpolateOp::verify() {
    const auto checkOffsets = [&](mlir::Value dynamicOffsets, std::optional<mlir::ArrayAttr> staticOffsetsAttr,
                                  const int64_t expectedRank, llvm::StringRef operandName,
                                  llvm::StringRef attrName) -> mlir::LogicalResult {
        if (dynamicOffsets == nullptr) {
            if (!staticOffsetsAttr.has_value()) {
                return mlir::success();
            }

            const auto staticOffsets = parseIntArrayAttr<int64_t>(staticOffsetsAttr.value());
            if (static_cast<int64_t>(staticOffsets.size()) != expectedRank) {
                return errorAt(*this, "'{0}' must have {1} elements, got {2}", attrName, expectedRank,
                               staticOffsets.size());
            }

            if (llvm::any_of(staticOffsets, [](int64_t value) {
                    return value == mlir::ShapedType::kDynamic;
                })) {
                return errorAt(*this, "'{0}' contains dynamic sentinel values but '{1}' is not provided", attrName,
                               operandName);
            }

            return mlir::success();
        }

        auto offsetsType = mlir::dyn_cast<vpux::NDTypeInterface>(dynamicOffsets.getType());
        if (!offsetsType) {
            return errorAt(*this, "'{0}' must be a tensor type", operandName);
        }

        if (offsetsType.getRank() != 1) {
            return errorAt(*this, "'{0}' must be 1D, got rank {1}", operandName, offsetsType.getRank());
        }

        if (!offsetsType.getElementType().isSignlessInteger(64)) {
            return errorAt(*this, "'{0}' must have i64 element type", operandName);
        }

        const auto offsetsShape = offsetsType.getShape();
        if (offsetsShape[Dim(0)] == mlir::ShapedType::kDynamic || offsetsShape[Dim(0)] != expectedRank) {
            return errorAt(*this, "'{0}' must have {1} elements, got {2}", operandName, expectedRank,
                           offsetsShape[Dim(0)]);
        }

        if (!staticOffsetsAttr.has_value()) {
            return errorAt(*this, "'{0}' requires '{1}' with dynamic sentinels to describe dynamic dimensions",
                           operandName, attrName);
        }

        const auto staticOffsets = parseIntArrayAttr<int64_t>(staticOffsetsAttr.value());
        if (static_cast<int64_t>(staticOffsets.size()) != expectedRank) {
            return errorAt(*this, "'{0}' must have {1} elements, got {2}", attrName, expectedRank,
                           staticOffsets.size());
        }

        if (!llvm::any_of(staticOffsets, [](int64_t value) {
                return value == mlir::ShapedType::kDynamic;
            })) {
            return errorAt(*this, "'{0}' is provided, but '{1}' does not contain dynamic sentinel values", operandName,
                           attrName);
        }

        return mlir::success();
    };

    const auto inputRank = static_cast<int64_t>(getShape(getInput()).size());
    if (mlir::failed(checkOffsets(getDynamicInputOffsets(), getInitialInputOffsetAttr(), inputRank,
                                  "dynamic_input_offsets", "initial_input_offset_attr"))) {
        return mlir::failure();
    }

    const auto outputRank = static_cast<int64_t>(getShape(getOutput()).size());
    if (mlir::failed(checkOffsets(getDynamicOutputOffsets(), getInitialOutputOffsetAttr(), outputRank,
                                  "dynamic_output_offsets", "initial_output_offset_attr"))) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::InterpolateOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto outputShape = getShape(getOutput());
    const auto isCompatibleStrategy{[&](auto strategyToCheck, auto dimensionToCheck) {
        return strategy == strategyToCheck && outputShape[dimensionToCheck] >= static_cast<int64_t>(numTiles);
    }};

    if (isCompatibleStrategy(VPU::MultiClusterStrategy::SplitOverHeightOverlapped, Dims4D::Act::H)) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::InterpolateOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

void vpux::VPU::InterpolateOp::build(
        ::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState, ::mlir::Value input,
        /*optional*/ ::mlir::Value sizes, /*optional*/ ::mlir::Value scales, /*optional*/ ::mlir::Value axes,
        /*optional*/ ::mlir::Value coordinates, /*optional*/ ::mlir::Value lambdas,
        /*optional*/ ::mlir::ArrayAttr sizes_attr, /*optional*/ ::mlir::ArrayAttr scales_attr,
        /*optional*/ ::mlir::ArrayAttr axes_attr, /*optional*/ ::mlir::ArrayAttr tile_offset_attr,
        /*optional*/ ::mlir::ArrayAttr initial_input_dims_attr, /*optional*/ ::mlir::ArrayAttr initial_output_dims_attr,
        vpux::IE::InterpolateAttr attr, ::mlir::ArrayAttr outputPadding, ::mlir::ArrayAttr inputPadding) {
    build(odsBuilder, odsState, input, sizes, scales, axes, coordinates, lambdas,
          /*dynamic_input_offsets=*/nullptr, /*dynamic_output_offsets=*/nullptr, sizes_attr, scales_attr, axes_attr,
          tile_offset_attr, initial_input_dims_attr, initial_output_dims_attr, nullptr, nullptr, nullptr, nullptr, attr,
          outputPadding, inputPadding);
}

//
// SWOpInterface
//

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 2 && buffers.size() <= 9,
                      "InterpolateOp can have a maximum of 1 input, 7 optional inputs and 1 output, but the "
                      "number of buffers is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    const auto interpolateMode = getAttr().getMode().getValue();
    const auto coordinates = getCoordinates();
    const auto lambdas = getLambdas();

    // Computing coordinates at compile time is a feature supported only for linear interpolate modes. The ticket
    // for adding support for all interpolate modes is E#132985.
    const auto isLinearInterpolateMode =
            interpolateMode == IE::InterpolateMode::LINEAR || interpolateMode == IE::InterpolateMode::LINEAR_ONNX;

    if (isLinearInterpolateMode && (coordinates == nullptr || lambdas == nullptr)) {
        const auto inputType = mlir::cast<NDTypeInterface>(getInput().getType());
        const auto inOrder = inputType.getDimsOrder();

        const auto axesValue = IE::getInterpAxesVal(getLoc(), getAxes(), getAxesAttr(), inputType);
        const auto innermostAxisResult = IE::getInnermostAxis(getLoc(), inOrder, axesValue);
        VPUX_THROW_WHEN(mlir::failed(innermostAxisResult), "Failed to get the innermost axis");
        const auto innermostAxis = innermostAxisResult.value();

        if (coordinates == nullptr) {
            const auto coordinatesSize = IE::getInterpCoordinatesSize(getOutput(), innermostAxis);
            const auto coordinatesElemSize = 4_Byte;
            buffersSize.push_back(coordinatesSize * coordinatesElemSize);
        }
        if (lambdas == nullptr) {
            const auto lambdasSize = IE::getInterpLambdasSize(getOutput(), innermostAxis);
            const auto lambdasElemSize = 2_Byte;
            buffersSize.push_back(lambdasSize * lambdasElemSize);
        }
    }

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::InterpolateOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::InterpolateOp::supportCycleCostCalculation() {
    return false;
}

InputTiling vpux::VPU::InterpolateOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto axesVal = IE::getInterpAxesVal(getLoc(), getAxes(), getAxesAttr(), inputType);

    auto iShape = getInitialInputDimsAttr().has_value() ? parseIntArrayAttr<int64_t>(getInitialInputDimsAttr().value())
                                                        : to_small_vector(getShape(getInput()));
    auto oShape = getInitialOutputDimsAttr().has_value()
                          ? parseIntArrayAttr<int64_t>(getInitialOutputDimsAttr().value())
                          : to_small_vector(getShape(getOutput()));
    auto initialInputOffsets = getInitialInputOffsetAttr().has_value()
                                       ? parseIntArrayAttr<int64_t>(getInitialInputOffsetAttr().value())
                                       : SmallVector<int64_t>(getShape(getInput()).size(), 0);

    auto initialOutputOffsets = getInitialOutputOffsetAttr().has_value()
                                        ? parseIntArrayAttr<int64_t>(getInitialOutputOffsetAttr().value())
                                        : SmallVector<int64_t>(getShape(getOutput()).size(), 0);

    mlir::Builder builder(*this);
    if (!getInitialInputDimsAttr().has_value()) {
        auto newInitialInputDims = builder.getI64ArrayAttr(iShape);
        setInitialInputDimsAttrAttr(newInitialInputDims);
    }
    if (!getInitialOutputDimsAttr().has_value()) {
        auto newInitialOutputDims = builder.getI64ArrayAttr(oShape);
        setInitialOutputDimsAttrAttr(newInitialOutputDims);
    }

    SmallVector<double> tileOffset(iShape.size(), 0.f);
    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    setTileOffsetAttrAttr(newTileOffset);

    vpux::Scales fwdScales;
    // Compute scale-factors based on full I/O resolution ratio
    SmallVector<double> backwardScale;
    for (size_t i = 0; i < axesVal.size(); i++) {
        backwardScale.push_back(static_cast<double>(iShape[axesVal[i]]) / oShape[axesVal[i]]);
    }

    SmallVector<int64_t> beginPads(iShape.size(), 0);
    SmallVector<int64_t> endPads(iShape.size(), 0);

    mlir::FailureOr<SmallVector<int64_t>> inferedInputTile;
    auto coordMode = getAttr().getCoordMode().getValue();
    auto interpolateMode = getAttr().getMode().getValue();
    auto nearestMode = getAttr().getNearestMode().getValue();
    // shape_calc_mode is uniformly SIZES in the VPU dialect; useScaleAttr alone decides whether
    // scales_attr is the authoritative coordinate-transform scale (1/originalScales) or the
    // back-inference uses the initial_IH/initial_OH dim ratio.
    const bool useScaleAttr = getUseScaleAttr();
    auto currentInputShape = to_small_vector(getShape(getInput()));

    SmallVector<double> originalScalesVec;
    if (auto scalesAttr = getScalesAttr()) {
        auto scalesResult = IE::extractFPVector(getLoc(), nullptr, scalesAttr);
        VPUX_THROW_UNLESS(mlir::succeeded(scalesResult), "InterpolateOp::backInferTileInfo failed to extract scales");
        originalScalesVec = scalesResult.value();
    }

    std::optional<ArrayRef<int64_t>> coordinatesShape;
    std::optional<ArrayRef<int64_t>> lambdasShape;
    if (const auto coordinates = getCoordinates(); coordinates != nullptr) {
        coordinatesShape = getShape(coordinates).raw();
    }
    if (const auto lambdas = getLambdas(); lambdas != nullptr) {
        lambdasShape = getShape(lambdas).raw();
    }

    auto inTiles = vpux::backInferInterpolateTile(
            outputTile, iShape, oShape, initialInputOffsets, initialOutputOffsets, currentInputShape, coordinatesShape,
            lambdasShape, interpolateMode, coordMode, nearestMode, useScaleAttr, originalScalesVec, axesVal, log);
    auto newInputOffset = to_small_vector(inTiles.tiles[0].offsets);

    // Recalculate the backward scale based on the new input/output shape
    for (size_t i = 0; i < axesVal.size(); i++) {
        fwdScales.push_back(static_cast<double>(outputTile.shape[Dim(axesVal[i])]) /
                            inTiles.tiles[0].shape[Dim(axesVal[i])]);
    }

    auto shapeCalcMode = IE::InterpolateCalcMode::SCALES;
    auto forwardInferedShape = IE::inferInterpOutShape(
            getLoc(), axesVal, inTiles.tiles[0].shape, {beginPads}, {endPads}, shapeCalcMode,
            IE::extractIntVector(getLoc(), getSizes(), getSizesAttr().value_or<mlir::ArrayAttr>({})), {fwdScales},
            mlir::Float64Type::get(getContext()), log);

    // TODO: E#36319 we counting only endpads - begin pad might matter for offsets not for dims
    auto shapeArray = to_small_vector(outputTile.shape);
    if (endPads.size() == shapeArray.size()) {
        for (auto shapeOrig : shapeArray | indexed) {
            endPads[shapeOrig.index()] = shapeOrig.value() - forwardInferedShape[shapeOrig.index()];
        }
    }

    VPUX_THROW_WHEN(getSizes() != nullptr, "Interpolate `sizes` input should have been converted to an attribute.");
    VPUX_THROW_WHEN(getScales() != nullptr, "Interpolate `scales` input should have been converted to an attribute.");
    VPUX_THROW_WHEN(getAxes() != nullptr, "Interpolate `axes` input should have been converted to an attribute.");

    inTiles.pads = {0, endPads[2], 0, endPads[3]};
    return inTiles;
}

void vpux::VPU::InterpolateOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outTile) {
    if (!inputTiling.pads.has_value()) {
        return;
    }
    mlir::Builder builder(*this);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto initialInputDims = parseIntArrayAttr<int64_t>(getInitialInputDimsAttrAttr());

    const auto initialInputOffset = builder.getI64ArrayAttr(to_small_vector(inputTiling.tiles[0].offsets));
    const auto initialOutputOffset = builder.getI64ArrayAttr(to_small_vector(outTile.offsets));
    setInitialInputOffsetAttrAttr(initialInputOffset);
    setInitialOutputOffsetAttrAttr(initialOutputOffset);

    const auto numDims = initialInputDims.size();

    SmallVector<double> tileOffset(numDims, 0.f);
    auto newTileOffset = builder.getF64ArrayAttr(tileOffset);
    setTileOffsetAttrAttr(newTileOffset);

    SmallVector<int64_t> endPads(numDims, 0);
    SmallVector<int64_t> beginPads(numDims, 0);

    endPads[2] = inputTiling.pads.value().right;
    endPads[3] = inputTiling.pads.value().bottom;

    auto newEndPads = builder.getI64ArrayAttr(endPads);
    auto newBeginPads = builder.getI64ArrayAttr(beginPads);

    // After tiling, output shape is explicitly determined by tiling strategy.
    // shape_calc_mode is already SIZES (pinned at convert_layers_to_VPU); do not change it.
    // Only refresh the per-tile pad attrs (and sizes_attr below) for this tile's output shape.
    auto newAttrs = IE::InterpolateAttr::get(
            getContext(), getAttr().getMode(), getAttr().getShapeCalcMode(), getAttr().getCoordMode(),
            getAttr().getNearestMode(), getAttr().getAntialias(), newBeginPads, newEndPads, getAttr().getCubeCoeff());

    // Set sizes_attr to the explicit output tile shape (for inferReturnTypes in SIZES mode)
    const auto axesValue = IE::getInterpAxesVal(getLoc(), getAxes(), getAxesAttr(), inputType);
    SmallVector<int64_t> sizes;
    sizes.reserve(axesValue.size());
    for (auto axis : axesValue) {
        sizes.push_back(outTile.shape[Dim(axis)]);
    }
    setSizesAttrAttr(builder.getI64ArrayAttr(sizes));

    // set pads begin + end attrs
    setAttrAttr(newAttrs);
    // DO NOT overwrite scales_attr — it preserves the user's original scale
    // for coordinate transformation in the kernel.
    // scales_attr and useScaleAttr are invariant across tiling and are left untouched.
}

mlir::FailureOr<OutputTiling> vpux::VPU::InterpolateOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::VPU::InterpolateOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());

    // Get the axes that are being interpolated
    const auto axesResult = IE::extractIntVector(getLoc(), getAxes(), getAxesAttrAttr());
    if (mlir::failed(axesResult)) {
        return axesResult;
    }
    const auto axesVal = axesResult.value();

    // In the VPU dialect shape_calc_mode is uniformly SIZES; useScaleAttr marks whether scales_attr
    // is the authoritative forward scale. When it is (and scales are available), reify each dynamic
    // output dim as input * scales_attr. Otherwise (SIZES-authored, or when scales are missing),
    // derive the per-axis forward scale from the global output/input bounded-dim ratio so dynamic
    // dims reify consistently with the tiling path.
    std::optional<mlir::ArrayAttr> scalesAttr = getScalesAttr();
    if (!getUseScaleAttr() || !scalesAttr.has_value()) {
        const auto inputDims = getBoundedShape(getInput());
        const auto outputDims = getBoundedShape(getOutput());
        // Derive the per-axis forward scale only when every interpolated axis has a known
        // (non-kDynamic) bounded input and output dim. getBoundedShape returns the raw shape for
        // unbounded types, where a dynamic dim is kDynamic (-1) and would yield a bogus scale (e.g.
        // -1/-1 = 1.0), reifying incorrect output dims. When any axis is unknown, keep the original
        // scales/scalesAttr and let reification use the provided scales or fail.
        SmallVector<double> derivedScales;
        derivedScales.reserve(axesVal.size());
        bool canDerive = true;
        for (auto axis : axesVal) {
            const auto inDim = inputDims[Dim(axis)];
            const auto outDim = outputDims[Dim(axis)];
            if (inDim == mlir::ShapedType::kDynamic || outDim == mlir::ShapedType::kDynamic) {
                canDerive = false;
                break;
            }
            derivedScales.push_back(inDim != 0 ? static_cast<double>(outDim) / static_cast<double>(inDim) : 1.0);
        }
        if (canDerive) {
            scalesAttr = builder.getF64ArrayAttr(derivedScales);
        }
    }

    return reifyInterpolateResultShape(builder, loc, getInput(), /*scales=*/nullptr, scalesAttr, axesVal,
                                       outputShapedType, reifiedReturnShapes);
}
