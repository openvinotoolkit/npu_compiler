//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/error.hpp"

#include <numeric>

using namespace vpux;
using namespace VPU;

EltwiseQuantizationApproximation::EltwiseQuantizationApproximation(double input1Target, double input2Target,
                                                                   double outputTarget, VPU::EltwiseType eltwiseType)
        : _input1(input1Target), _input2(input2Target), _output(1 / outputTarget) {
    // We align shifts to the smaller one by dividing input MULT with 2^diff, inputs shift will be set to 0 in
    // nce_cluster_task.cpp and added to the output shift .
    //
    // what we actually do is input1 * MULT1 i32 --> + --> * MULT_OUT >> (SHIFT_OUT + SHIFT_IN) --> u8
    //                       input2 * MULT2 i32 ----^
    const auto minShift = std::min(_input1.shift(), _input2.shift());
    const auto maxShift = std::max(_input1.shift(), _input2.shift());
    // shift register is using 6 bits, so the maximum shift value is 2^6 - 1
    const int64_t maxRegisterShift = pow(2, 6) - 1;
    // the multiply register for each individual IDU unit is unsigned 16 bit;
    // so unlike the common PPE logic that uses signed 16 bit register and a maximum
    // multiply of pow(2, 15) - 1, here we can safely scale up to pow(2, 16) - 1
    const int64_t maxRegisterMult = pow(2, 16) - 1;

    const auto supportsShiftToMaximum = [&]() -> bool {
        if (maxShift + _output.shift() > maxRegisterShift) {
            return false;
        }
        if (_input1.mult() > maxRegisterMult >> (maxShift - _input1.shift())) {
            return false;
        }
        if (_input2.mult() > maxRegisterMult >> (maxShift - _input2.shift())) {
            return false;
        }

        return true;
    };

    // Currently handle just the case when both input scales are negative
    if (_input1.mult() < 0 || _input2.mult() < 0) {
        if (eltwiseType == VPU::EltwiseType::ADD || eltwiseType == VPU::EltwiseType::SUBTRACT) {
            // Can't handle cases when just one scale is negative
            if ((_input1.mult() < 0) ^ (_input2.mult() < 0)) {
                VPUX_THROW("Unsupported case for ADD/SUB eltwise, just one negative scale {0} {1}.", _input1.mult(),
                           _input2.mult());
            }
            _output.setMult(-1 * _output.mult());
        }
        if (eltwiseType == VPU::EltwiseType::MULTIPLY && ((_input1.mult() < 0) ^ (_input2.mult() < 0))) {
            _output.setMult(-1 * _output.mult());
        }
        _input1.setMult(static_cast<uint16_t>(std::abs(_input1.mult())));
        _input2.setMult(static_cast<uint16_t>(std::abs(_input2.mult())));
    }

    if (supportsShiftToMaximum()) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() << (maxShift - _input1.shift())));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() << (maxShift - _input2.shift())));
        _output.setShift(_output.shift() + maxShift);
    } else if (minShift + _output.shift() < maxRegisterShift) {
        _input1.setMult(static_cast<uint16_t>(_input1.mult() >> (_input1.shift() - minShift)));
        _input2.setMult(static_cast<uint16_t>(_input2.mult() >> (_input2.shift() - minShift)));
        _output.setShift(_output.shift() + minShift);
    } else {
        VPUX_THROW("Eltwise add input1_MULT/input2_MULT/output_SHIFT out of register range");
    }
}

const QuantizationApproximation& EltwiseQuantizationApproximation::input1() const {
    return _input1;
}

const QuantizationApproximation& EltwiseQuantizationApproximation::input2() const {
    return _input2;
}

const QuantizationApproximation& EltwiseQuantizationApproximation::output() const {
    return _output;
}

double VPU::computeQuantScale(mlir::Type inputType, mlir::Type outputType) {
    const auto inputScale = mlir::isa_and_nonnull<mlir::quant::QuantizedType>(inputType)
                                    ? extractScalesAndZeroPoints(inputType).first.front()
                                    : 1.0;
    const auto outputScale = mlir::isa_and_nonnull<mlir::quant::QuantizedType>(outputType)
                                     ? extractScalesAndZeroPoints(outputType).first.front()
                                     : 1.0;

    VPUX_THROW_WHEN(inputScale == 0, "Invalid input scale value '0'");
    VPUX_THROW_WHEN(outputScale == 0, "Invalid output scale value '0'");

    return inputScale / outputScale;
}

double VPU::computeQuantScaleWithWeightedOps(mlir::Type inputType, mlir::Type outputType, mlir::Type weightsType) {
    const auto weightsScale = mlir::isa_and_nonnull<mlir::quant::QuantizedType>(weightsType)
                                      ? extractScalesAndZeroPoints(weightsType).first.front()
                                      : 1.0;

    VPUX_THROW_WHEN(weightsScale == 0, "Invalid output scale value '0'");
    return computeQuantScale(inputType, outputType) * weightsScale;
}

double VPU::computeScale(mlir::Operation* operation) {
    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    const auto outputElemType = mlir::cast<vpux::NDTypeInterface>(operation->getResult(0).getType()).getElementType();
    if (mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::TransposedConvolutionOp, VPU::TransposedConvolutionOp,
                  IE::MatMulOp>(operation)) {
        const auto weightsElemType =
                mlir::cast<vpux::NDTypeInterface>(operation->getOperand(1).getType()).getElementType();
        // In case of per axis quantization it is needed to have the scales in scale table
        if (!mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputElemType) &&
            !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(weightsElemType) &&
            !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outputElemType)) {
            auto staticScale = 1.0;
            if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(operation)) {
                staticScale =
                        convOp.getStaticScaleAttr() != nullptr ? convOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
            }
            return computeQuantScaleWithWeightedOps(inputElemType, outputElemType, weightsElemType) * staticScale;
        }

    } else if (auto avgPoolOp = mlir::dyn_cast<IE::AvgPoolOp>(operation)) {
        const auto kernelSize = vpux::parseIntArrayAttr<int64_t>(avgPoolOp.getKernelSizeAttr());
        const auto staticScale =
                avgPoolOp.getStaticScaleAttr() != nullptr ? avgPoolOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
        return computeAvgPoolQuantScale(inputElemType, outputElemType, kernelSize) * staticScale;
    }
    if (auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(operation)) {
        VPUX_THROW_WHEN(
                inputElemType != outputElemType && (mlir::isa<mlir::quant::QuantizedType>(inputElemType) ||
                                                    mlir::isa<mlir::quant::QuantizedType>(outputElemType)),
                "Quantization-agnostic operation (MaxPool) has different quantized input ({0}) and output ({1}) types.",
                inputElemType, outputElemType);
        if (const auto staticScale = maxPoolOp.getStaticScale()) {
            return staticScale->convertToDouble();
        }
        return 1.0;
    }
    return computeQuantScale(inputElemType, outputElemType);
}

int64_t VPU::computeQuantZPForEltwise(mlir::Type type) {
    const auto qType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(type);
    if (qType == nullptr) {
        return 0;
    }

    const auto maybeZP = extractScalarOrUniformZP(qType);
    VPUX_THROW_WHEN(mlir::failed(maybeZP), "Per-axis quantized types with zero points != 0 aren't supported");
    return *maybeZP;
}

double VPU::computeAvgPoolQuantScale(mlir::Type inputType, mlir::Type outputType, mlir::ArrayRef<int64_t> kernelShape) {
    // avgFactor = 1 / (D1 *...* Dn), where kernel shape is <D1 x...x Dn>
    const auto avgFactor = 1.0 / static_cast<double>(std::accumulate(kernelShape.begin(), kernelShape.end(), 1.0,
                                                                     std::multiplies<int64_t>()));

    return computeQuantScale(inputType, outputType) * avgFactor;
}
