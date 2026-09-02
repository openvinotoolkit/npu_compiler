//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEEltwiseSupported(mlir::Operation* op, vpux::NDTypeInterface input1Type,
                                      vpux::NDTypeInterface input2Type, vpux::NDTypeInterface outputType,
                                      bool allowDifferentScales, bool allowDifferentZp, bool checkLayout,
                                      bool checkChannelAlignment, LogCb logCb) {
    if (config::getCompilationMode(op) == config::CompilationMode::ReferenceSW) {
        // We are in reference SW compilation mode
        return false;
    }

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
    // Input types can also differ when both of them are per-tensor quantized. E.g. scale value for Eltwise Multiply
    const auto input1ElemType = input1Type.getElementType();
    const auto input2ElemType = input2Type.getElementType();
    const auto outputElemType = outputType.getElementType();

    const bool anyPerAxisQuantized = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(input1ElemType) ||
                                     mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(input2ElemType) ||
                                     mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);

    if (anyPerAxisQuantized) {
        // Per-axis quantization is only supported via the per-channel WT scale table; it is intended to be used for
        // standalone quantize/dequantize ops mapped to eltwise, ADD being the only supported mode. Two directions are
        // allowed:
        //   - Quantize:   float inputs → per-axis quantized output.
        //   - Dequantize: per-axis quantized inputs → float output.
        const auto eltwiseType = vpux::VPU::decodeNceEltwiseType(op);
        if (eltwiseType != vpux::VPU::EltwiseType::ADD) {
            logCb(formatv("Per-axis quantized eltwise is only supported for ADD, got '{0}'", op->getName()));
            return false;
        }
        if (!VPU::MPEEngineConfig::useNewWeightTableFormat(op, /*isCompressConv=*/false)) {
            logCb(formatv("Per-axis quantized eltwise requires the new WT format"));
            return false;
        }
        if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(input1ElemType) &&
            mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(input2ElemType) &&
            !mlir::isa<mlir::quant::QuantizedType>(outputElemType)) {
            // Dequantize direction: per-axis inputs → float output.
            if (!isSupportedEltwisePerAxisQuantization(input1ElemType, input2ElemType, logCb)) {
                return false;
            }
        } else if (mlir::isa<mlir::Float16Type>(input1ElemType) && mlir::isa<mlir::Float16Type>(input2ElemType) &&
                   mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
            // Quantize direction: f16 inputs → per-axis output.
            if (!isSupportedEltwisePerAxisQuantization(outputElemType, logCb)) {
                return false;
            }
        } else {
            logCb(formatv("Unsupported per-axis eltwise type combination: in1='{0}', in2='{1}', out='{2}'",
                          input1ElemType, input2ElemType, outputElemType));
            return false;
        }
    } else if (!mlir::isa<mlir::quant::QuantizedType>(input1ElemType) &&
               !mlir::isa<mlir::quant::QuantizedType>(input2ElemType)) {
        // Pure float eltwise: both inputs must be f16.
        if (!mlir::isa<mlir::Float16Type>(input1ElemType) || !mlir::isa<mlir::Float16Type>(input2ElemType)) {
            logCb(formatv("Non-quantized eltwise inputs must be f16"));
            return false;
        }
        // Multiply with f16 inputs and a quantized output is produced by QuantizeWithMultiplyRewriter (mixed
        // precision fusion), and the purpose is to have this operation on Shave, not mapped to NCE.
        if (mlir::isa<IE::MultiplyOp>(op) && mlir::isa<mlir::quant::QuantizedType>(outputElemType)) {
            logCb(formatv("Multiply with f16 inputs and quantized output is not supported for NCE eltwise"));
            return false;
        }
    } else if (mlir::isa<mlir::quant::UniformQuantizedType>(input1ElemType) &&
               mlir::isa<mlir::quant::UniformQuantizedType>(input2ElemType)) {
        // Per-tensor quantized eltwise.
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
            !NCEInvariant::isOutputActTypeSupported(op, outputType, outputAlignment)) {
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
    } else if (mlir::isa<IE::PermuteQuantizeOp>(operation)) {
        return VPU::EltwiseType::ADD;
    } else if (mlir::isa<IE::SubtractOp>(operation)) {
        return VPU::EltwiseType::SUBTRACT;
    } else if (mlir::isa<IE::MultiplyOp>(operation)) {
        return VPU::EltwiseType::MULTIPLY;
    }

    VPUX_THROW("Unsupported NCE eltwise type: {0}", operation->getName());
}
