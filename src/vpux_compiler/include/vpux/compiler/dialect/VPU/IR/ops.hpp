//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

// E#173010: remove dependency on IE operations for VPU operations
namespace vpux::IE {
class AvgPoolOp;
class AddOp;
class BatchNormInferenceOp;
class ConvolutionOp;
class GroupConvolutionOp;
class InterpolateOp;
class LSTMCellOp;
class LSTMSequenceOp;
class MatMulOp;
class MaxPoolOp;
class MultiplyOp;
class PermuteQuantizeOp;
class SubtractOp;
class TransposedConvolutionOp;
class YuvToRgbOp;
}  // namespace vpux::IE

//
// Generated
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

//
// Operation verifiers
//

namespace vpux {
namespace VPU {

//
// Tiling
//

// Returns a WeightsTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getWeightsTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                             std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origWeightsTable = origOp->getWeightsTable();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have a WeightsTable", *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);
    VPUX_THROW_UNLESS((weightsOutputChannels.has_value() ||
                       origWeightsTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                              origWeightsTableShape[Dim(1)] == 1 && origWeightsTableShape[Dim(2)] == 1 &&
                              origWeightsTableShape[Dim(3)] == VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origWeightsTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the WeightsTable corresponds to its own output channel
    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    weightsTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return weightsTableTile;
}

// Returns a ScaleTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getScaleTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile) {
    const auto origScaleTable = origOp->getWeightTableScale();
    VPUX_THROW_UNLESS(origScaleTable != nullptr, "The operation {0} doesn't have a ScaleTable", *origOp);

    const auto origScaleTableShape = getShape(origScaleTable);
    VPUX_THROW_UNLESS(origScaleTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C] &&
                              origScaleTableShape[Dim(1)] == 1 && origScaleTableShape[Dim(2)] == 1 &&
                              origScaleTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected ScaleTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origScaleTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the ScaleTable corresponds to its own output channel
    TileInfo scaleTableTile(origScaleTableShape);
    scaleTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    scaleTableTile.shape[Dim(0)] = outputTile.shape[Dims4D::Act::C];
    return scaleTableTile;
}

// Returns a BiasTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getBiasTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile) {
    const auto origBiasTable = origOp->getWeightTableBias();
    VPUX_THROW_UNLESS(origBiasTable != nullptr, "The operation {0} doesn't have a BiasTable", *origOp);

    const auto origBiasTableShape = getShape(origBiasTable);
    VPUX_THROW_UNLESS(origBiasTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C] &&
                              origBiasTableShape[Dim(1)] == 1 && origBiasTableShape[Dim(2)] == 1 &&
                              origBiasTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected BiasTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origBiasTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the BiasTable corresponds to its own output channel
    TileInfo biasTableTile(origBiasTableShape);
    biasTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    biasTableTile.shape[Dim(0)] = outputTile.shape[Dims4D::Act::C];
    return biasTableTile;
}

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    VPUX_THROW_UNLESS(inputTiling.pads.has_value(), "Missing tile information for paddings");

    auto newPadAttr = getPaddingAttr(op->getContext(), inputTiling.pads.value());

    op->setPadAttr(newPadAttr);
}

// Adjust rawFilterShape attribute for specific output tile
template <typename ConcreteOp>
void adjustRawFilterShape(ConcreteOp* op, const TileInfo& outputTile) {
    auto newRawFilterShape = Shape(parseIntArrayAttr<int64_t>(op->getRawFilterShape()));

    newRawFilterShape[Dims4D::Filter::OC] = outputTile.shape[Dims4D::Act::C];

    op->setRawFilterShapeAttr(getIntArrayAttr(op->getContext(), newRawFilterShape));
}

//
// Misc
//

mlir::LogicalResult sameLayout(VPU::DistributedTensorType inDistributedType,
                               VPU::DistributedTensorType outDistributedType, LogCb logCb = emptyLogCb);
mlir::LogicalResult sameLayout(VPUIP::DistributedBufferType inDistributedType,
                               VPUIP::DistributedBufferType outDistributedType, LogCb logCb = emptyLogCb);

bool arePerClusterDistributionMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface srcType,
                                                         VPU::DistributionInfo& sourceDistribution,
                                                         vpux::NDTypeInterface targetType,
                                                         VPU::DistributionInfo& targetDistribution);

bool arePerClusterMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface sourceType,
                                             const VPU::DistributionInfo& sourceDistribution,
                                             const VPU::DistributionInfo& targetDistribution);

mlir::LogicalResult areDistributionsCompatible(vpux::NDTypeInterface srcType, VPU::DistributionInfo& sourceAttr,
                                               vpux::NDTypeInterface targetType, VPU::DistributionInfo& targetAttr,
                                               const bool allowDifferentPerClusterMemoryView = false);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult areDistributionAttrsCompatible(T sourceType, T targetType,
                                                   const bool allowDifferentPerClusterMemoryView = false) {
    auto inDistribution = VPU::DistributionInfo::getClassFromAttr(sourceType.getDistribution());
    auto outDistribution = VPU::DistributionInfo::getClassFromAttr(targetType.getDistribution());
    auto inType = mlir::cast<vpux::NDTypeInterface>(sourceType);
    auto outType = mlir::cast<vpux::NDTypeInterface>(targetType);
    return areDistributionsCompatible(inType, inDistribution, outType, outDistribution,
                                      allowDifferentPerClusterMemoryView);
}

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult isDistributedCastCompatible(T inDistributedType, T outDistributedType, LogCb logCb = emptyLogCb) {
    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        logCb(formatv("Mismatch between shapes for input ({0}) and output ({1}).", inDistributedType.getShape(),
                      outDistributedType.getShape()));
        return mlir::failure();
    }

    if (areDistributionElementTypesCompatible(inDistributedType.getElementType(), outDistributedType.getElementType())
                .failed()) {
        logCb(formatv("Mismatch between element types for input ({0}) and output ({1}).",
                      inDistributedType.getElementType(), outDistributedType.getElementType()));
        return mlir::failure();
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        logCb(formatv("Mismatch between memspaces for input ({0}) and output ({1}).", inDistributedType.getMemSpace(),
                      outDistributedType.getMemSpace()));
        return mlir::failure();
    }

    const auto sameLayoutCheck = sameLayout(inDistributedType, outDistributedType, logCb);
    if (sameLayoutCheck.failed()) {
        return mlir::failure();
    }

    auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());
    auto outDistribution = VPU::DistributionInfo::getClassFromAttr(outDistributedType.getDistribution());
    auto inType = mlir::cast<vpux::NDTypeInterface>(inDistributedType);
    auto outType = mlir::cast<vpux::NDTypeInterface>(outDistributedType);
    if (areDistributionsCompatible(inType, inDistribution, outType, outDistribution).failed()) {
        logCb(formatv("Mismatch between distributionAttr for input ({0}) and output ({1}).",
                      inDistributedType.getDistribution(), outDistributedType.getDistribution()));
        return mlir::failure();
    }

    return mlir::success();
}

bool isNCEWithInt4Weights(mlir::Operation* op);
bool isNCEWithSEPActivation(mlir::Operation* op);

std::optional<int64_t> getWeightsChannelsAutopad(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
