//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEEltwiseSupported(mlir::Operation* op, vpux::NDTypeInterface input1Type,
                                      vpux::NDTypeInterface input2Type, vpux::NDTypeInterface outputType,
                                      bool allowDifferentScales, bool allowDifferentZp, bool checkLayout,
                                      bool checkChannelAlignment, LogCb logCb) {
    if (input1Type.getRank() != 4 || input2Type.getRank() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (input1Type.getShape() != input2Type.getShape()) {
        logCb(formatv("Broadcasting is not supported"));
        return false;
    }

    // Output type can differ from input type. In case of quantization this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();

    if (!input1ElemType.isa<mlir::quant::QuantizedType>() && !input2ElemType.isa<mlir::quant::QuantizedType>()) {
        if (!input1ElemType.isa<mlir::Float16Type>() || !input2ElemType.isa<mlir::Float16Type>()) {
            return false;
        }
    } else if (input1ElemType.isa<mlir::quant::UniformQuantizedType>() &&
               input2ElemType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(op);
        if (!isSupportedEltwiseQuantization(input1ElemType, input2ElemType, allowDifferentScales, allowDifferentZp,
                                            eltwiseType, logCb)) {
            return false;
        }
    } else {
        logCb(formatv("Unsupported inputs element types"));
        return false;
    }

    auto arch = getArch(op);
    if (checkChannelAlignment) {
        auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
        auto outputAlignment = iface != nullptr ? iface.getOutputChannelAlignment()
                                                : vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType());
        bool hasAutoPad = VPU::hasAutoPaddingIDU(getModuleOp(op));
        auto inputAlignmentFirst = hasAutoPad && VPU::inputCompatibleWithAutoPad(input1Type)
                                           ? 1
                                           : vpux::VPU::NCEInvariant::getAlignment(input1Type.getElementType());

        auto inputAlignmentSecond = hasAutoPad && VPU::inputCompatibleWithAutoPad(input2Type)
                                            ? 1
                                            : vpux::VPU::NCEInvariant::getAlignment(input2Type.getElementType());

        if (!NCEInvariant::isInputActTypeSupported(input1Type, inputAlignmentFirst, false) ||
            !NCEInvariant::isInputActTypeSupported(input2Type, inputAlignmentSecond, false) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, outputAlignment)) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        if (!NCEInvariant::checkLayouts({input1Type, input2Type}, {outputType}, arch, 2, logCb)) {
            return false;
        }
    }

    return true;
}

VPU::EltwiseType VPU::decodeNceEltwiseType(mlir::Operation* operation) {
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(operation)) {
        return nceEltwise.getOpType();
    } else if (mlir::isa<VPU::DesparsifyOp>(operation)) {
        return VPU::EltwiseType::ADD;
    } else if (mlir::isa<IE::AddOp>(operation)) {
        return VPU::EltwiseType::ADD;
    } else if (mlir::isa<IE::SubtractOp>(operation)) {
        return VPU::EltwiseType::SUBTRACT;
    } else if (mlir::isa<IE::MultiplyOp>(operation)) {
        return VPU::EltwiseType::MULTIPLY;
    }

    VPUX_THROW("Unsupported NCE eltwise type: {0}", operation->getName());
}
