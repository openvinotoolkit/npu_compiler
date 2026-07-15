//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Template models for QuantizedLayerOpInterface, parameterized on arch-specific utilities.
//
// All ArchUtils types must provide:
//   static bool isMixPrecisionSupported(mlir::Operation*, bool);
//   static bool checkPostOp(IE::LayerWithPostOpInterface, bool, bool);
//   static int64_t getMaximumQuantizationLevels(mlir::Operation*);
//
// ConvLikeQuantModel and MatMulQuantModel additionally require:
//   static constexpr IE::InputTypeConstraints inputConstraints;
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/input_types_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace IE {

namespace detail {

// Shared implementations for QuantizedLayerOpInterface methods common to all model classes.
template <typename ArchUtils>
bool isMixPrecisionSupportedImpl(mlir::Operation* op, bool isPReLUSupported) {
    return ArchUtils::isMixPrecisionSupported(op, isPReLUSupported);
}

template <typename ArchUtils>
bool checkPostOpImpl(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) {
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
    if (layerWithPostOp == nullptr) {
        return true;
    }
    return ArchUtils::checkPostOp(layerWithPostOp, isPerAxisQuantizedOutput, isFloatInput);
}

template <typename ArchUtils>
int64_t getMaximumQuantizationLevelsImpl(mlir::Operation* op) {
    return ArchUtils::getMaximumQuantizationLevels(op);
}

}  // namespace detail

//
// ConvLikeQuantModel for Conv, GroupConv, TransposedConv, GroupTransposedConv
//
template <typename ArchUtils, typename OpType, bool CheckPerAxisOnInput = true, bool CheckInputTypes = false>
class ConvLikeQuantModel :
        public IE::QuantizedLayerOpInterface::ExternalModel<
                ConvLikeQuantModel<ArchUtils, OpType, CheckPerAxisOnInput, CheckInputTypes>, OpType> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        return detail::isMixPrecisionSupportedImpl<ArchUtils>(op, isPReLUSupported);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        return detail::checkPostOpImpl<ArchUtils>(op, isPerAxisQuantizedOutput, isFloatInput);
    }

    bool isInputQuantizationFusable(mlir::Operation* op) const {
        auto concreteOp = mlir::cast<OpType>(op);
        auto inputDequantizeOp = concreteOp.getInput().template getDefiningOp<IE::DequantizeOp>();
        auto filterDequantizeOp = concreteOp.getFilter().template getDefiningOp<IE::DequantizeOp>();
        if (inputDequantizeOp == nullptr || filterDequantizeOp == nullptr) {
            return false;
        }
        constexpr auto constraints = ArchUtils::inputConstraints;
        if constexpr (CheckPerAxisOnInput) {
            if (!constraints.allowPerAxisInput && IE::isPerAxisQuant(inputDequantizeOp.getInput())) {
                return false;
            }
        }
        if constexpr (CheckInputTypes) {
            return IE::areInputTypesSupported(inputDequantizeOp.getInput(), filterDequantizeOp.getInput(), constraints);
        }
        return true;
    }

    bool isOutputQuantizationFusable(mlir::Operation* op, bool isPerAxisQuantized, bool isFloatInput) const {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
            if (!ArchUtils::checkPostOp(layerWithPostOp, isPerAxisQuantized, isFloatInput)) {
                return false;
            }
        }
        return true;
    }

    bool isInOutQuantizationCompatible(mlir::Operation* op, mlir::Operation* quantizeOperation) const {
        if (!IE::areAllUsersQuantized(op)) {
            return false;
        }
        auto quantizeOp = mlir::cast<IE::QuantizeOp>(quantizeOperation);
        return IE::isQuantizationSupported(quantizeOp, op, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT);
    }

    int64_t getMaximumQuantizationLevels(mlir::Operation* op) const {
        return detail::getMaximumQuantizationLevelsImpl<ArchUtils>(op);
    }
};

//
// EltwiseQuantModel for Add, Multiply, Subtract
//
template <typename ArchUtils, typename OpType, VPU::EltwiseType EltwType>
class EltwiseQuantModel :
        public IE::QuantizedLayerOpInterface::ExternalModel<EltwiseQuantModel<ArchUtils, OpType, EltwType>, OpType> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        return detail::isMixPrecisionSupportedImpl<ArchUtils>(op, isPReLUSupported);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        return detail::checkPostOpImpl<ArchUtils>(op, isPerAxisQuantizedOutput, isFloatInput);
    }

    bool isInputQuantizationFusable(mlir::Operation* op) const {
        auto concreteOp = mlir::cast<OpType>(op);
        auto input1DequantizeOp = concreteOp.getInput1().template getDefiningOp<IE::DequantizeOp>();
        auto input2DequantizeOp = concreteOp.getInput2().template getDefiningOp<IE::DequantizeOp>();
        if (input1DequantizeOp == nullptr || input2DequantizeOp == nullptr ||
            IE::isPerAxisQuant(input1DequantizeOp.getInput()) || IE::isPerAxisQuant(input2DequantizeOp.getInput())) {
            return false;
        }
        // Verify eltwise quantization type compatibility.
        auto input1Type = mlir::cast<vpux::NDTypeInterface>(input1DequantizeOp.getInput().getType()).getElementType();
        auto input2Type = mlir::cast<vpux::NDTypeInterface>(input2DequantizeOp.getInput().getType()).getElementType();
        return vpux::isSupportedEltwiseQuantization(input1Type, input2Type, /*allowDifferentScales=*/true,
                                                    /*allowDifferentZp=*/true, EltwType);
    }

    bool isOutputQuantizationFusable(mlir::Operation* op, bool isPerAxisQuantized, bool isFloatInput) const {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
            if (!ArchUtils::checkPostOp(layerWithPostOp, isPerAxisQuantized, isFloatInput)) {
                return false;
            }
        }
        // Eltwise ops do not support per-axis output quantization
        if (isPerAxisQuantized) {
            return false;
        }
        return true;
    }

    bool isInOutQuantizationCompatible(mlir::Operation* op, mlir::Operation* quantizeOperation) const {
        if (!IE::areAllUsersQuantized(op)) {
            return false;
        }
        auto quantizeOp = mlir::cast<IE::QuantizeOp>(quantizeOperation);
        return IE::isQuantizationSupported(quantizeOp, op, IE::TypeComparisonMode::STRICT_EQUAL);
    }

    int64_t getMaximumQuantizationLevels(mlir::Operation* op) const {
        return detail::getMaximumQuantizationLevelsImpl<ArchUtils>(op);
    }
};

//
// MaxPoolQuantModel
//
template <typename ArchUtils, bool RejectPerAxisOutput = true>
class MaxPoolQuantModel :
        public IE::QuantizedLayerOpInterface::ExternalModel<MaxPoolQuantModel<ArchUtils, RejectPerAxisOutput>,
                                                            IE::MaxPoolOp> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        return detail::isMixPrecisionSupportedImpl<ArchUtils>(op, isPReLUSupported);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        return detail::checkPostOpImpl<ArchUtils>(op, isPerAxisQuantizedOutput, isFloatInput);
    }

    bool isInputQuantizationFusable(mlir::Operation* op) const {
        auto maxPoolOp = mlir::cast<IE::MaxPoolOp>(op);
        auto inputDequantizeOp = maxPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();
        return inputDequantizeOp != nullptr && !IE::isPerAxisQuant(inputDequantizeOp.getInput());
    }

    bool isOutputQuantizationFusable(mlir::Operation* op, bool isPerAxisQuantized, bool /*isFloatInput*/) const {
        if constexpr (RejectPerAxisOutput) {
            if (isPerAxisQuantized) {
                return false;
            }
        }
        // Pool IDU does not support zero-point subtraction, so it compensates by ignoring output
        // zero-point as well. Since the input zero-point is not subtracted, the non-linear post-op will operate
        // on improper data. Currently, quantized pool ops are disabled for all post-ops.
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        return layerWithPostOp == nullptr || !layerWithPostOp.hasPPE();
    }

    bool isInOutQuantizationCompatible(mlir::Operation* op, mlir::Operation* quantizeOperation) const {
        if (!IE::areAllUsersQuantized(op)) {
            return false;
        }
        auto quantizeOp = mlir::cast<IE::QuantizeOp>(quantizeOperation);
        return IE::isQuantizationSupported(quantizeOp, op, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT);
    }

    int64_t getMaximumQuantizationLevels(mlir::Operation* op) const {
        return detail::getMaximumQuantizationLevelsImpl<ArchUtils>(op);
    }
};

//
// AvgPoolQuantModel
//
template <typename ArchUtils>
class AvgPoolQuantModel :
        public IE::QuantizedLayerOpInterface::ExternalModel<AvgPoolQuantModel<ArchUtils>, IE::AvgPoolOp> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        return detail::isMixPrecisionSupportedImpl<ArchUtils>(op, isPReLUSupported);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        return detail::checkPostOpImpl<ArchUtils>(op, isPerAxisQuantizedOutput, isFloatInput);
    }

    bool isInputQuantizationFusable(mlir::Operation* op) const {
        auto avgPoolOp = mlir::cast<IE::AvgPoolOp>(op);
        auto inputDequantizeOp = avgPoolOp.getInput().getDefiningOp<IE::DequantizeOp>();
        return inputDequantizeOp != nullptr && !IE::isPerAxisQuant(inputDequantizeOp.getInput());
    }

    bool isOutputQuantizationFusable(mlir::Operation* op, bool isPerAxisQuantized, bool isFloatInput) const {
        // AveragePool IDU does not support zero-point subtraction, so it compensates by ignoring output
        // zero-point as well. Since the input zero-point is not subtracted, the non-linear post-op will
        // operate on improper data. In fully-quantized mode, quantized AvgPool is disabled for all post-ops;
        // for float-input mixed precision the arch-specific allowlist applies.
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
            if (!isFloatInput) {
                return false;
            }
            if (!ArchUtils::checkPostOp(layerWithPostOp, isPerAxisQuantized, isFloatInput)) {
                return false;
            }
        }
        if (isPerAxisQuantized) {
            return false;
        }
        return true;
    }

    bool isInOutQuantizationCompatible(mlir::Operation* op, mlir::Operation* quantizeOperation) const {
        if (!IE::areAllUsersQuantized(op)) {
            return false;
        }
        auto quantizeOp = mlir::cast<IE::QuantizeOp>(quantizeOperation);
        return IE::isQuantizationSupported(quantizeOp, op, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT);
    }

    int64_t getMaximumQuantizationLevels(mlir::Operation* op) const {
        return detail::getMaximumQuantizationLevelsImpl<ArchUtils>(op);
    }
};

//
// MatMulQuantModel
//
template <typename ArchUtils>
class MatMulQuantModel :
        public IE::QuantizedLayerOpInterface::ExternalModel<MatMulQuantModel<ArchUtils>, IE::MatMulOp> {
public:
    bool isMixPrecisionSupported(mlir::Operation* op, bool isPReLUSupported) const {
        return detail::isMixPrecisionSupportedImpl<ArchUtils>(op, isPReLUSupported);
    }

    bool checkPostOp(mlir::Operation* op, bool isPerAxisQuantizedOutput, bool isFloatInput) const {
        return detail::checkPostOpImpl<ArchUtils>(op, isPerAxisQuantizedOutput, isFloatInput);
    }

    bool isInputQuantizationFusable(mlir::Operation* op) const {
        auto matMulOp = mlir::cast<IE::MatMulOp>(op);
        auto input1DequantizeOp = matMulOp.getInput1().getDefiningOp<IE::DequantizeOp>();
        auto input2DequantizeOp = matMulOp.getInput2().getDefiningOp<IE::DequantizeOp>();

        if (input1DequantizeOp == nullptr || input2DequantizeOp == nullptr) {
            return false;
        }

        return IE::areInputTypesSupported(input1DequantizeOp.getInput(), input2DequantizeOp.getInput(),
                                          ArchUtils::inputConstraints);
    }

    bool isOutputQuantizationFusable(mlir::Operation* op, bool isPerAxisQuantized, bool isFloatInput) const {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(op);
        if (layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
            if (!ArchUtils::checkPostOp(layerWithPostOp, isPerAxisQuantized, isFloatInput)) {
                return false;
            }
        }
        return true;
    }

    bool isInOutQuantizationCompatible(mlir::Operation* op, mlir::Operation* quantizeOperation) const {
        if (!IE::areAllUsersQuantized(op)) {
            return false;
        }
        auto quantizeOp = mlir::cast<IE::QuantizeOp>(quantizeOperation);
        return IE::isQuantizationSupported(quantizeOp, op, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT);
    }

    int64_t getMaximumQuantizationLevels(mlir::Operation* op) const {
        return detail::getMaximumQuantizationLevelsImpl<ArchUtils>(op);
    }
};

}  // namespace IE
}  // namespace vpux
