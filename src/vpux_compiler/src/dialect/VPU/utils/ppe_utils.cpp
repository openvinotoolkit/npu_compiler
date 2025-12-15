//
// Copyright (C) 2024-2025 Intel Corporation.
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
