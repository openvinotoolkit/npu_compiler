//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/odu_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;
using namespace VPU;

namespace {

constexpr MPEModeGrid CUBOID_MPE_GRIDS[] = {
        {MPEMode::CUBOID_4x16, /*width=*/8, /*height=*/8, /*outputChannels=*/256},
        {MPEMode::CUBOID_8x16, /*width=*/16, /*height=*/8, /*outputChannels=*/128},
        {MPEMode::CUBOID_16x16, /*width=*/16, /*height=*/16, /*outputChannels=*/64}};

constexpr MPEModeGrid CUBOID_MPE_GRIDS_WITH_REDUCED_ACCUMULATORS[] = {
        {MPEMode::CUBOID_4x16, /*width=*/8, /*height=*/8, /*outputChannels=*/128},
        {MPEMode::CUBOID_8x16, /*width=*/16, /*height=*/8, /*outputChannels=*/64},
        {MPEMode::CUBOID_16x16, /*width=*/16, /*height=*/16, /*outputChannels=*/32}};

}  // namespace

ArrayRef<VPU::MPEModeGrid> vpux::VPU::getCuboidMpeGrids() {
    return CUBOID_MPE_GRIDS;
}

ArrayRef<VPU::MPEModeGrid> vpux::VPU::getCuboidMpeGridsWithReducedAccumulators() {
    return CUBOID_MPE_GRIDS_WITH_REDUCED_ACCUMULATORS;
}

VPU::MPEMode vpux::VPU::getMpeModeForConv(ShapeRef shape, ArrayRef<MPEModeGrid> grids) {
    VPUX_THROW_WHEN(grids.empty(), "Expected at least one MPE mode grid");
    std::vector<std::pair<double, VPU::MPEMode>> mpeCosts;
    mpeCosts.reserve(grids.size());

    for (const auto& grid : grids) {
        // Each output-channel grid requires another read of the activations.
        const auto numWeightGrids = std::ceil(static_cast<double>(shape[Dims4D::Act::C]) / grid.outputChannels);
        // Each spatial grid requires another read of the weights.
        const auto numActivationGrids = std::ceil(static_cast<double>(shape[Dims4D::Act::H]) / grid.height) *
                                        std::ceil(static_cast<double>(shape[Dims4D::Act::W]) / grid.width);
        const auto activationReads = numWeightGrids * shape[Dims4D::Act::H] * shape[Dims4D::Act::W];
        const auto weightReads = numActivationGrids * shape[Dims4D::Act::C];
        mpeCosts.emplace_back(activationReads + weightReads, grid.mode);
    }

    // Preserve the existing mode ordering when read costs are equal.
    return std::min_element(mpeCosts.begin(), mpeCosts.end())->second;
}

bool vpux::VPU::isDepthwiseOp(mlir::Operation* op) {
    return mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op);
}

bool VPU::isNCEWithInt4Weights(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }

    auto weights = nceOp.getWeightsOperand();
    if (weights == nullptr) {
        return false;
    }

    auto weightsElemType = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType();
    if (const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(weightsElemType)) {
        return quantizedType.getStorageTypeIntegralWidth() == 4;
    }

    return false;
}

bool VPU::isNCEWithSEPActivation(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_or_null<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return false;
    }
    auto sparseTensorActivation = nceOp->getOperand(0).getDefiningOp<VPU::GroupSparseTensorOp>();
    if (sparseTensorActivation == nullptr) {
        return false;
    }
    return sparseTensorActivation.getStorageElementTable() != nullptr;
}

bool vpux::VPU::isEltwiseNCEPooling(mlir::Operation* op) {
    auto poolingOp = mlir::dyn_cast_or_null<VPU::NCEAveragePoolOp>(op);
    if (poolingOp == nullptr) {
        return false;
    }

    const auto inputType = mlir::cast<NDTypeInterface>(poolingOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(poolingOp.getOutput().getType());
    if (inputType.getDimsOrder() != outputType.getDimsOrder()) {
        return false;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(poolingOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(poolingOp.getStrides());

    const auto paddingAttr = poolingOp.getPad();
    const auto padTop = paddingAttr.getTop().getValue().getSExtValue();
    const auto padBottom = paddingAttr.getBottom().getValue().getSExtValue();
    const auto padLeft = paddingAttr.getLeft().getValue().getSExtValue();
    const auto padRight = paddingAttr.getRight().getValue().getSExtValue();
    const auto noPadding = (padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0);

    const auto isOne = [](const int64_t val) -> bool {
        return val == 1;
    };

    return llvm::all_of(kernelSize, isOne) && llvm::all_of(strides, isOne) && noPadding;
}

SmallVector<vpux::VPU::ODUDimScale> vpux::VPU::getODUScaling(mlir::Operation* op) {
    auto nceOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return {};
    }
    return nceOp.getODUScaling();
}

namespace {

// The op's declared output channel count, taken back into pre-ODU space.
// Inverts through ShapeInfo rather than a plain shape vector: its shape field skips
// ShapedType::kDynamic entries, which ops like DepthToSpace permit for H/W.
mlir::FailureOr<int64_t> preODUOutputChannels(mlir::Operation* op) {
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto preODUShapeInfo = VPU::invertODUScaling(VPU::getODUScaling(op), ShapeInfo::fromNDType(outputType));
    if (mlir::failed(preODUShapeInfo)) {
        return mlir::failure();
    }
    return preODUShapeInfo->shape[Dims4D::Act::C.ind()];
}

}  // namespace

int64_t vpux::VPU::getPreODUOutputChannels(mlir::Operation* op) {
    const auto channels = preODUOutputChannels(op);
    VPUX_THROW_UNLESS(mlir::succeeded(channels), "Failed to invert ODU scaling for '{0}'", *op);
    return channels.value();
}

mlir::FailureOr<vpux::TileInfo> vpux::VPU::fwdInferNceOutputTile(mlir::Operation* op, const vpux::TileInfo& inputTile,
                                                                 vpux::Logger log) {
    auto nceOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(op);
    if (nceOp == nullptr) {
        return mlir::failure();
    }

    const auto outputShape = getShape(nceOp->getResult(0));
    const auto& inShape = inputTile.shape;
    // Forward shape inference is only modelled for the 4D activation layout.
    if (outputShape.size() != 4 || inShape.size() != 4) {
        return mlir::failure();
    }

    // Every output dimension is derived from the input tile, so a dynamic input cannot yield an
    // inferable output.
    if (inShape.isDynamic()) {
        log.trace("Input tile {0} of op at '{1}' is dynamic; forward inference needs a resolved tile", inShape,
                  op->getLoc());
        return mlir::failure();
    }

    const auto kernel = nceOp.getKernelSizeVal();
    const auto strides = nceOp.getStridesVal();
    if (kernel.size() != 2 || strides.size() != 2) {
        return mlir::failure();
    }

    const auto pads = VPU::toPadInfo(nceOp.getPad());
    const auto inputHW = std::make_pair(inShape[Dims4D::Act::H], inShape[Dims4D::Act::W]);
    const auto outputHW = vpux::spatialOutputForInputWindowSize(inputHW, kernel, strides, pads);
    if (!outputHW.has_value()) {
        return mlir::failure();
    }

    // The window formula describes the real convolution/pooling output, i.e. pre-ODU space.
    // An active S2D/D2S transform is re-applied at the end; empty and neutral scales are an
    // identity inside the helper, so ops without a transform need no guard here.
    Shape result(outputShape.size(), static_cast<int64_t>(0));
    result[Dims4D::Act::N] = inShape[Dims4D::Act::N];
    result[Dims4D::Act::H] = outputHW->first;
    result[Dims4D::Act::W] = outputHW->second;

    // Channel-preserving ops (depthwise convolution, both poolings, eltwise) keep the input C
    // Ops with weights take OC from the weights.
    // Anything else falls back to the declared output C, read in pre-ODU space because `result` is
    // scaled back through the transform below. Of the three ops that can carry an S2D/D2S config
    // (NCE.Convolution, NCE.DepthConvolution, NCE.MaxPool) only the convolution reaches this
    // fallback, and only when its own weights OC is unknown.
    if (VPU::isDepthwiseOp(op) || mlir::isa<VPU::NCEEltwiseOp>(op)) {
        result[Dims4D::Act::C] = inShape[Dims4D::Act::C];
    } else {
        std::optional<int64_t> channels;
        if (const auto weights = nceOp.getWeightsOperand()) {
            const auto weightsShape = getShape(weights);
            if (weightsShape.size() == 4 && weightsShape[Dims4D::Filter::OC] != mlir::ShapedType::kDynamic) {
                channels = weightsShape[Dims4D::Filter::OC];
            }
        }
        if (!channels.has_value()) {
            const auto declaredChannels = preODUOutputChannels(op);
            if (mlir::failed(declaredChannels)) {
                log.trace("Output shape {0} of op at '{1}' is not invertible through its ODU transform", outputShape,
                          op->getLoc());
                return mlir::failure();
            }
            channels = declaredChannels.value();
        }
        result[Dims4D::Act::C] = channels.value();
    }

    // C can still be kDynamic, when neither the weights nor the declared output resolve it
    // a convolution with dynamic input channels, for instance. Report that instead of leaking
    // the sentinel; N, H and W come from the (already static) input tile.
    if (result.isDynamic()) {
        log.trace("Could not infer every output dimension of op at '{0}': got {1}", op->getLoc(), result);
        return mlir::failure();
    }

    // `result` is fully static by now, so the plain ShapeRef overload is enough here.
    const auto postODUResult = VPU::applyODUScaling(nceOp.getODUScaling(), ShapeRef(result));
    if (mlir::failed(postODUResult)) {
        log.trace("Pre-ODU shape {0} of op at '{1}' is not exactly scalable through its ODU transform", result,
                  op->getLoc());
        return mlir::failure();
    }
    const Shape postODUShape(postODUResult.value());
    return vpux::TileInfo(postODUShape);
}

mlir::FailureOr<vpux::TileInfo> vpux::VPU::fwdInferOutputTileDefault(mlir::Operation* op,
                                                                     ArrayRef<vpux::TileInfo> inputTiles,
                                                                     vpux::Logger log) {
    if (inputTiles.empty()) {
        return mlir::failure();
    }
    if (mlir::isa<VPU::NCEOpInterface>(op)) {
        return VPU::fwdInferNceOutputTile(op, inputTiles.front(), log);
    }
    if (op->hasTrait<VPU::EltwiseOp>()) {
        // A fully parallel iteration space: the output tile is the input tile.
        const auto& firstTile = inputTiles.front();
        const auto allSameShape = llvm::all_of(inputTiles, [&](const vpux::TileInfo& tile) {
            return tile.shape == firstTile.shape;
        });
        if (allSameShape) {
            return firstTile;
        }

        // Broadcasted eltwise output takes the per-dimension maximum.
        // Only equal-or-one broadcasts are valid; otherwise report failure.
        Shape broadcastShape = firstTile.shape;
        if (broadcastShape.isDynamic()) {
            log.trace("Eltwise op at '{0}' has a dynamic input tile {1}; cannot forward infer its output", op->getLoc(),
                      broadcastShape);
            return mlir::failure();
        }
        for (const auto& tile : inputTiles.drop_front()) {
            if (tile.shape.size() != broadcastShape.size() || tile.shape.isDynamic()) {
                log.trace("Eltwise op at '{0}' has an unusable input tile {1}; cannot forward infer its output",
                          op->getLoc(), tile.shape);
                return mlir::failure();
            }
            for (const auto dim : irange(broadcastShape.size())) {
                auto& acc = broadcastShape[Dim(dim)];
                const auto val = tile.shape[Dim(dim)];
                if (acc != val && acc != 1 && val != 1) {
                    log.trace("Eltwise op at '{0}' has input tiles that do not broadcast: {1} vs {2}", op->getLoc(),
                              firstTile.shape, tile.shape);
                    return mlir::failure();
                }
                acc = std::max(acc, val);
            }
        }
        return vpux::TileInfo(ShapeRef(broadcastShape));
    }
    return mlir::failure();
}
