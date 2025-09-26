//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

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

    if (input1Type.getShape()[Dims4D::Act::N] != 1) {
        logCb(formatv("Only Batch size 1 is supported"));
        return false;
    }

    // Output type can differ from input type. In case of quantization this can be different quant scale value.
    // Input types can also differ when both of them are quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();

    if (!mlir::isa<mlir::quant::QuantizedType>(input1ElemType) &&
        !mlir::isa<mlir::quant::QuantizedType>(input2ElemType)) {
        if (!mlir::isa<mlir::Float16Type>(input1ElemType) || !mlir::isa<mlir::Float16Type>(input2ElemType)) {
            return false;
        }
    } else if (mlir::isa<mlir::quant::UniformQuantizedType>(input1ElemType) &&
               mlir::isa<mlir::quant::UniformQuantizedType>(input2ElemType)) {
        const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(op);
        if (!isSupportedEltwiseQuantization(input1ElemType, input2ElemType, allowDifferentScales, allowDifferentZp,
                                            eltwiseType, logCb)) {
            return false;
        }
    } else {
        logCb(formatv("Unsupported inputs element types"));
        return false;
    }

    auto arch = config::getArch(op);
    if (checkChannelAlignment) {
        auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
        auto outputAlignment = iface != nullptr ? iface.getOutputChannelAlignment()
                                                : vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType());
        auto inputAlignmentFirst = VPU::NCEInvariant::getAlignment(input1Type.getElementType());
        auto inputAlignmentSecond = VPU::NCEInvariant::getAlignment(input2Type.getElementType());
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
