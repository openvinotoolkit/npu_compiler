//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

bool VPU::areChannelsCompatibleWithIDUAutoPad(int64_t inputChannels, int64_t elemTypeBitWidth) {
    return elemTypeBitWidth >= CHAR_BIT &&
           ((elemTypeBitWidth < FP16_WIDTH && inputChannels < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) ||
            (elemTypeBitWidth >= FP16_WIDTH && inputChannels < WIDTH16_CHANNEL_LIMIT));
}

bool VPU::areChannelsCompatibleWithODUAutoPad(int64_t outputChannels, int64_t elemTypeBitWidth) {
    return outputChannels < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT && elemTypeBitWidth >= CHAR_BIT;
}

bool VPU::inputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    if (type.getRank() != 4) {
        return false;
    }
    const auto inShape = type.getShape();
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    return areChannelsCompatibleWithIDUAutoPad(inShape[Dims4D::Act::C], elemTypeBitWidth);
}

bool VPU::outputCompatibleWithAutoPad(mlir::Operation* op, vpux::NDTypeInterface type) {
    if (type.getRank() != 4) {
        return false;
    }
    // type is assumed to be in post-ODU space; invert an active S2D/D2S transform so the channel
    // count checked below matches what the DPU core actually operates on.
    const auto preODUShapeInfoResult = VPU::invertODUScaling(VPU::getODUScaling(op), ShapeInfo::fromNDType(type));
    VPUX_THROW_UNLESS(mlir::succeeded(preODUShapeInfoResult), "Failed to invert ODU scaling for '{0}'", *op);
    return outputCompatibleWithAutoPad(AutoPadPreS2DD2SInfo{preODUShapeInfoResult->rank(),
                                                            preODUShapeInfoResult->shape[Dims4D::Act::C.ind()],
                                                            type.getElemTypeSize().count()});
}

bool VPU::outputCompatibleWithAutoPad(const AutoPadPreS2DD2SInfo& info) {
    if (info.rank != 4) {
        return false;
    }
    return areChannelsCompatibleWithODUAutoPad(info.channels, info.elemTypeBitWidth);
}

bool VPU::canAutopadInput(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::NCEConvolutionOp>(op)) {
        return false;
    }
    const auto inputType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
    return inputCompatibleWithAutoPad(inputType) && config::hasAutoPaddingIDU(getModuleOp(op));
}

bool VPU::canAutopadOutput(mlir::Operation* op, std::optional<vpux::NDTypeInterface> optType) {
    const auto outputType = optType.value_or(mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()));
    if (outputType.getRank() != 4) {
        return false;
    }
    // Invert an active S2D/D2S transform so the channel count checked below matches what the DPU
    // core actually operates on, then delegate to the pre-ODU-info overload.
    const auto preODUShapeInfoResult = VPU::invertODUScaling(VPU::getODUScaling(op), ShapeInfo::fromNDType(outputType));
    VPUX_THROW_UNLESS(mlir::succeeded(preODUShapeInfoResult), "Failed to invert ODU scaling for '{0}'", *op);
    return canAutopadOutput(
            op, AutoPadPreS2DD2SInfo{preODUShapeInfoResult->rank(), preODUShapeInfoResult->shape[Dims4D::Act::C.ind()],
                                     outputType.getElemTypeSize().count()});
}

bool VPU::canAutopadOutput(mlir::Operation* op, const AutoPadPreS2DD2SInfo& info) {
    return outputCompatibleWithAutoPad(info) && config::hasAutoPaddingODU(getModuleOp(op));
}

bool VPU::canConsumeIDUAutopad(VPU::NCEConvolutionOp nceConvOp, LogCb logCb) {
    const auto inputPaddingAttr = nceConvOp.getInputPaddingAttr();
    if (inputPaddingAttr == nullptr) {
        logCb(formatv("Missing input_padding attribute"));
        return false;
    }
    // When IDU autopad is used, the input channels in the weights still need to be padded to 16, while the input
    // activation has the channels compressed. This leads to a representation for Convolutions where IC<16 in the input
    // and IC=16 in the weights, which is not supported in the compiler at the moment. Therefore, in case weights
    // sparsity is encountered, skip IDU autopad. The benefit from weight sparsity is also expected to be marginal for
    // such cases, so the heuristic for enabling weight sparsity is expected to leave the weights dense. More details in
    // E#177066.
    auto filter = nceConvOp.getFilter();
    if (mlir::isa<VPU::SparseTensorType>(filter.getType())) {
        logCb(formatv("Weights are sparse, so IDU autopad is skipped"));
        return false;
    }
    const auto weightShape = getShape(filter);
    // Skip the case where the weights have been flattened
    if (weightShape[Dims4D::Filter::IC] == 1) {
        logCb(formatv("The weights have been flattened"));
        return false;
    }

    const auto inputType = mlir::cast<NDTypeInterface>(nceConvOp.getInput().getType());
    if (inputType.getRank() != 4) {
        logCb(formatv("Only 4D inputs are currently supported for autopad"));
        return false;
    }

    const auto inputChannels = inputType.getShape()[Dims4D::Act::C];
    const auto inputPadding = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(inputPaddingAttr));
    const auto unpaddedInputChannels = inputChannels - inputPadding[Dims4D::Act::C.ind()];
    if (unpaddedInputChannels < 0) {
        logCb(formatv("Invalid number of unpadded input channels: {0}", unpaddedInputChannels));
        return false;
    }

    const auto elemTypeBitWidth = inputType.getElemTypeSize().count();
    const auto canUseAutopad = inputChannels == VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT &&
                               VPU::areChannelsCompatibleWithIDUAutoPad(unpaddedInputChannels, elemTypeBitWidth);
    if (!canUseAutopad) {
        logCb(formatv("Cannot autopad input"));
        return false;
    };

    return true;
}

bool VPU::canConsumeODUAutopad(VPU::NCEOpInterface nceOp, LogCb logCb) {
    (void)nceOp;
    (void)logCb;
    return true;
}

std::optional<int64_t> VPU::getWeightsChannelsAutopad(mlir::Operation* op,
                                                      std::optional<vpux::NDTypeInterface> optType) {
    // WeightsTable-like operands are always sized in pre-ODU space, so autopad eligibility and the
    // aligned channel count must be evaluated against the pre-ODU (real convolution) output rather
    // than the op's post-ODU result (e.g. after an active S2D/D2S transform).
    const auto outputType = optType.value_or(mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()));
    if (outputType.getRank() != 4) {
        return std::nullopt;
    }

    // Use ShapeInfo rather than a plain shape vector: its shape field skips ShapedType::kDynamic entries
    const auto preODUShapeInfoResult = VPU::invertODUScaling(VPU::getODUScaling(op), ShapeInfo::fromNDType(outputType));
    VPUX_THROW_UNLESS(mlir::succeeded(preODUShapeInfoResult), "Failed to invert ODU scaling for '{0}'", *op);
    const auto preODUChannels = preODUShapeInfoResult->shape[Dims4D::Act::C.ind()];

    return VPU::getWeightsChannelsAutopad(op, AutoPadPreS2DD2SInfo{preODUShapeInfoResult->rank(), preODUChannels,
                                                                   outputType.getElemTypeSize().count()});
}

std::optional<int64_t> VPU::getWeightsChannelsAutopad(mlir::Operation* op, const AutoPadPreS2DD2SInfo& info) {
    if (info.rank != 4) {
        return std::nullopt;
    }
    if (!VPU::canAutopadOutput(op, info)) {
        return std::nullopt;
    }
    return alignValUp(info.channels, vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
}
