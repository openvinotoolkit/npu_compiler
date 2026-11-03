//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <kernels/inc/common_types.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

#include <array>
#include <cstdint>
#include <optional>

namespace vpux {

sw_params::DataType getDataTypeFromMlirType(mlir::Type t);

namespace details {

using DT = sw_params::DataType;

constexpr uint64_t convertMode(DT input, DT output) {
    return (static_cast<uint64_t>(input) << 32) | static_cast<uint64_t>(output);
}

// FP32 -> FP16 is deliberately absent: since NPU40XX it maps onto DMA, so no SHAVE kernel is compiled for it.
inline constexpr std::array CONVERT_SUPPORTED_MODES = {
        convertMode(DT::NN_FP16, DT::NN_FP32), convertMode(DT::NN_I4, DT::NN_FP16),
        convertMode(DT::NN_I32, DT::NN_I64),   convertMode(DT::NN_I64, DT::NN_FP16),
        convertMode(DT::NN_I64, DT::NN_I32),   convertMode(DT::NN_FP16, DT::NN_I32),
        convertMode(DT::NN_I32, DT::NN_FP16),  convertMode(DT::NN_U8, DT::NN_FP16),
        convertMode(DT::NN_I8, DT::NN_FP16),   convertMode(DT::NN_I32, DT::NN_FP32),
        convertMode(DT::NN_I8, DT::NN_I32),    convertMode(DT::NN_U8, DT::NN_I32),
        convertMode(DT::NN_FP16, DT::NN_U8),   convertMode(DT::NN_U8, DT::NN_FP32),
        convertMode(DT::NN_I8, DT::NN_FP32),   convertMode(DT::NN_I32, DT::NN_U8),
        convertMode(DT::NN_I32, DT::NN_I8),    convertMode(DT::NN_I32, DT::NN_U64),
        convertMode(DT::NN_FP16, DT::NN_I8),   convertMode(DT::NN_FP32, DT::NN_U8),
        convertMode(DT::NN_FP32, DT::NN_I8),   convertMode(DT::NN_FP32, DT::NN_I32),
        convertMode(DT::NN_U16, DT::NN_FP16),  convertMode(DT::NN_FP16, DT::NN_U16),
        convertMode(DT::NN_I16, DT::NN_FP16),  convertMode(DT::NN_FP16, DT::NN_I16),
        convertMode(DT::NN_I8, DT::NN_I64),    convertMode(DT::NN_FP16, DT::NN_I64),
        convertMode(DT::NN_FP32, DT::NN_I64),  convertMode(DT::NN_U8, DT::NN_I64),
        convertMode(DT::NN_I64, DT::NN_I8),    convertMode(DT::NN_I64, DT::NN_FP32),
        convertMode(DT::NN_I64, DT::NN_U8),    convertMode(DT::NN_U8, DT::NN_U32),
        convertMode(DT::NN_I8, DT::NN_U32),    convertMode(DT::NN_FP32, DT::NN_U32),
        convertMode(DT::NN_FP16, DT::NN_U32),  convertMode(DT::NN_I64, DT::NN_U32),
        convertMode(DT::NN_U32, DT::NN_U8),    convertMode(DT::NN_U32, DT::NN_I8),
        convertMode(DT::NN_U32, DT::NN_I32),   convertMode(DT::NN_U32, DT::NN_FP16),
        convertMode(DT::NN_U32, DT::NN_FP32),  convertMode(DT::NN_U32, DT::NN_I64),
        convertMode(DT::NN_U64, DT::NN_U32),   convertMode(DT::NN_U32, DT::NN_U64),
        convertMode(DT::NN_U4, DT::NN_U8),     convertMode(DT::NN_U4, DT::NN_I8),
        convertMode(DT::NN_U4, DT::NN_FP16),   convertMode(DT::NN_I4, DT::NN_I8),
        convertMode(DT::NN_BF16, DT::NN_FP16), convertMode(DT::NN_I64, DT::NN_FP64),
        convertMode(DT::NN_FP64, DT::NN_I64),  convertMode(DT::NN_I8, DT::NN_U8),
        convertMode(DT::NN_U8, DT::NN_I8),     convertMode(DT::NN_FP64, DT::NN_FP16),
        convertMode(DT::NN_FP16, DT::NN_FP64),
};

}  // namespace details

inline bool isConvertConfigSupported(mlir::Type inputType, mlir::Type outputType, config::ArchKind arch) {
    using DT = sw_params::DataType;
    const auto input = getDataTypeFromMlirType(inputType);
    const auto output = getDataTypeFromMlirType(outputType);

    if (input == DT::NN_FP32 && output == DT::NN_FP16) {
        return arch < config::ArchKind::NPU40XX;
    }

    return llvm::is_contained(details::CONVERT_SUPPORTED_MODES, details::convertMode(input, output));
}

// quantDim: reverse mem-dim index of the per-axis quantization axis (as returned by
// computeReverseMemDim), where 0 means the innermost axis; std::nullopt for per-tensor quantization.
inline bool isQuantizeConfigSupported(mlir::Type inputType, mlir::Type outputType, config::ArchKind arch,
                                      std::optional<int64_t> quantDim = std::nullopt) {
    using DT = sw_params::DataType;
    const auto input = getDataTypeFromMlirType(inputType);
    const auto output = getDataTypeFromMlirType(outputType);

    if (input == DT::NN_FP32 && output == DT::NN_U8) {
        return true;
    }
    if (input != DT::NN_FP16) {
        return false;
    }
    if (output == DT::NN_U8 || output == DT::NN_I8 || output == DT::NN_U16 || output == DT::NN_I16) {
        return true;
    }
    if (output == DT::NN_U4 || output == DT::NN_I4) {
        // quantize.cpp's innermost-axis dispatch table (quant_inner) has no U4/I4 entries;
        // only per-tensor or outer-axis per-axis quantization reach the U4/I4 asm kernels.
        if (quantDim.has_value() && quantDim.value() == 0) {
            return false;
        }
        return arch >= config::ArchKind::NPU40XX;
    }
    return false;
}

inline bool isDequantizeConfigSupported(mlir::Type inputType, mlir::Type outputType, config::ArchKind arch) {
    using DT = sw_params::DataType;
    const auto input = getDataTypeFromMlirType(inputType);
    const auto output = getDataTypeFromMlirType(outputType);

    if (input == DT::NN_U8 && output == DT::NN_FP32) {
        return true;
    }
    if (output != DT::NN_FP16) {
        return false;
    }
    if (input == DT::NN_U8 || input == DT::NN_I8 || input == DT::NN_U16 || input == DT::NN_I16) {
        return true;
    }
    if (arch >= config::ArchKind::NPU40XX && (input == DT::NN_I4 || input == DT::NN_U4)) {
        return true;
    }
    return arch >= config::ArchKind::NPU50XX && (input == DT::NN_BF8 || input == DT::NN_HF8);
}

inline bool isDynamicDequantizeConfigSupported(mlir::Type inputType, mlir::Type outputType, config::ArchKind arch,
                                               mlir::Type zeroPointType = {}) {
    using DT = sw_params::DataType;
    const auto input = getDataTypeFromMlirType(inputType);
    const auto output = getDataTypeFromMlirType(outputType);
    const auto hasZeroPoint = static_cast<bool>(zeroPointType);
    const auto zeroPoint = hasZeroPoint ? getDataTypeFromMlirType(zeroPointType) : DT::NN_UNDEFINED;

    if (output == DT::NN_FP16) {
        if (input == DT::NN_I4 || input == DT::NN_I8 || input == DT::NN_U8 || input == DT::NN_NF4) {
            return true;
        }
        // The kernel reads the zero-point buffer through the input's storage type and strides it with the
        // input's element width, so a U16 input with a narrower zero-point would be misread. Only the
        // no-zero-point U16 path exists.
        if (input == DT::NN_U16) {
            return !hasZeroPoint;
        }
        // asm_dequant_dynamic_u2_f16_v0 is the only U2 kernel and it requires a U2 zero-point operand;
        // unlike the other dtypes above, U2 has no generic per-element fallback in the kernel.
        return arch >= config::ArchKind::NPU40XX && input == DT::NN_U2 && zeroPoint == DT::NN_U2;
    }

    if (output == DT::NN_FP32) {
        return input == DT::NN_I4 || input == DT::NN_I8 || input == DT::NN_U8;
    }

    return false;
}

}  // namespace vpux
