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

bool VPU::outputCompatibleWithAutoPad(vpux::NDTypeInterface type) {
    if (type.getRank() != 4) {
        return false;
    }
    const auto outShape = type.getShape();
    const auto elemTypeBitWidth = type.getElemTypeSize().count();
    return areChannelsCompatibleWithODUAutoPad(outShape[Dims4D::Act::C], elemTypeBitWidth);
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
    return outputCompatibleWithAutoPad(outputType) && config::hasAutoPaddingODU(getModuleOp(op));
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

std::optional<int64_t> VPU::getWeightsChannelsAutopad(mlir::Operation* op) {
    if (VPU::canAutopadOutput(op)) {
        const auto outputShape = getShape(op->getResult(0));
        return alignValUp(outputShape[Dims4D::Act::C], vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT);
    }
    return std::nullopt;
}
