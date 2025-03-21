//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <functional>

using namespace vpux;
using namespace vpux::VPUIPDPU;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.cpp.inc>

//
// Custom
//

namespace {

template <typename T>
mlir::LogicalResult verifyPPEBiasAdd(T& op) {
    auto scaleTableExists = (op.getScaleTable() != nullptr);
    auto biasStaticExists = op.getBiasStatic().has_value();

    // scale_table only
    if (scaleTableExists && !biasStaticExists) {
        return ::mlir::success();
    }

    // bias_static only
    if (biasStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0} needs either scale_table or bias_static as parameter",
                   op.getOperationName());
}

template <typename T>
mlir::LogicalResult verifyPPEScaleMult(T& op) {
    auto scaleTableExists = (op.getScaleTable() != nullptr);
    auto scaleStaticExists = op.getScaleStatic().has_value();

    // scale_table only
    if (scaleTableExists && !scaleStaticExists) {
        return ::mlir::success();
    }

    // scale_static only
    if (scaleStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(op.getLoc(), "Operation {0} needs either scale_table or scale_static as parameter",
                   op.getOperationName());
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult IDUEltWiseCfgOp::verify() {
    if (getElopScaleAAttr().getTypeID() != getElopScaleBAttr().getTypeID()) {
        return errorAt(
                getLoc(),
                "Operation {0}: parameters types are invalid, combinations of IntegerAttr and FloatAttr not supported.",
                getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult PPEFpBiasAddOp::verify() {
    return verifyPPEBiasAdd(*this);
}

mlir::LogicalResult PPEFpScalePreluMultOp::verify() {
    return verifyPPEScaleMult(*this);
}

mlir::LogicalResult PPEFpConvertOp::verify() {
    auto convMode = getConvertMode();
    auto clampModeExists = getClampMode().has_value();
    auto ftzModeExists = getFtzMode().has_value();
    auto bf16RoundModeExists = getBf16RoundMode().has_value();

    // Validate the supported conversions with different clamp, ftz and bf16 combinations
    switch (convMode) {
    case PPEFpConvertMode::FP16:
        if (!bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::BF16:
        if (!clampModeExists && !ftzModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::BF8:
    case PPEFpConvertMode::HF8:
        if (!bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    case PPEFpConvertMode::NONE:
    case PPEFpConvertMode::I32:
        if (!clampModeExists && !ftzModeExists && !bf16RoundModeExists) {
            return ::mlir::success();
        }
        break;

    default:
        break;
    }

    return errorAt(getLoc(), "Operation {0} has unsupported combination of parameters", getOperationName());
}

mlir::LogicalResult PPEIntBiasAddOp::verify() {
    return verifyPPEBiasAdd(*this);
}

mlir::LogicalResult PPEIntScaleMultOp::verify() {
    return verifyPPEScaleMult(*this);
}

mlir::LogicalResult PPEIntScaleShiftOp::verify() {
    auto scaleTableExists = (getScaleTable() != nullptr);
    auto shiftStaticExists = getShiftStatic().has_value();

    // scale_table only
    if (scaleTableExists && !shiftStaticExists) {
        return ::mlir::success();
    }

    // scale_static only
    if (shiftStaticExists && !scaleTableExists) {
        return ::mlir::success();
    }

    return errorAt(getLoc(), "Operation {0} needs either scale_table or shift_static as parameter", getOperationName());
}

mlir::LogicalResult ODUSparsityOp::verify() {
    const bool sparsityMapExists = (getSparsityMap() != nullptr);
    const bool compressionEnabledExists = getCompressionEnabled().has_value();
    const bool compressionEnabled = getCompressionEnabled().value_or(false);
    const bool sparseValueExists = getSparseValue().has_value();

    if (compressionEnabled) {
        return ::mlir::success();
    }

    if (sparsityMapExists && (!compressionEnabledExists || !sparseValueExists)) {
        return ::mlir::success();
    }

    return errorAt(getLoc(), "Operation {0}: invalid params combination", getOperationName());
}

mlir::LogicalResult ODUOutActivationsOp::verify() {
    auto arch = VPU::getArch(*this);
    auto dataTypeExists = getDataType().has_value();
    auto dataWidthExists = getDataWidth().has_value();

    if (!dataTypeExists && !dataWidthExists) {
        return ::mlir::success();
    }

    if ((arch == VPU::ArchKind::NPU37XX) && !(dataTypeExists && !dataWidthExists)) {
        return errorAt(getLoc(), "Operation {0}: use data_type attr to specify data type", getOperationName());
    }
    if ((arch > VPU::ArchKind::NPU37XX) && !(!dataTypeExists && dataWidthExists)) {
        return errorAt(getLoc(), "Operation {0}: use data_width attr to specify data type", getOperationName());
    }

    return ::mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
