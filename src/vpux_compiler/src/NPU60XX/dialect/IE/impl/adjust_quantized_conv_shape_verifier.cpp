//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/adjust_quantized_conv_shape_verifier.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

using namespace vpux::IE::arch60xx;

//
// AdjustQuantizedConvShapeVerifier
//

bool AdjustQuantizedConvShapeVerifier::isBeneficialConversion(IE::ConvolutionOp convOp, mlir::Value filter,
                                                              Logger log) const {
    auto convInput = convOp.getInput();
    auto convOutput = convOp.getOutput();
    auto isQuantizedType = [](mlir::Value val) {
        auto elementType = mlir::cast<vpux::NDTypeInterface>(val.getType()).getElementType();
        return mlir::isa<mlir::quant::QuantizedType>(elementType);
    };

    if (IE::isPerAxisQuant(convInput) || IE::isPerAxisQuant(convOutput)) {
        log.trace("Unsupported convolution with per-axis quantized input/output");
        return false;
    }

    if (isQuantizedType(convInput) != isQuantizedType(convOutput)) {
        log.trace("Unsupported convolution with mixed quantized and non-quantized I/O types");
        return false;
    }

    if (isQuantizedType(filter) != isQuantizedType(convInput)) {
        log.trace("Unsupported convolution with mixed-precision input and filter types");
        return false;
    }

    if (!mlir::cast<mlir::ShapedType>(convOp.getInput().getType()).hasStaticShape()) {
        log.trace("Unsupported convolution with dynamic input shape");
        return false;
    }

    if (!IE::isPerAxisQuant(filter)) {
        log.trace("Not beneficial for convolution without per-axis quantized filter");
        return false;
    }

    // in order to avoid performance regressions we need to filter out convolutions
    // with small kernels
    auto filterShape = vpux::getShape(filter);
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto KY = filterShape[Dims4D::Filter::KY];
    // a kernel smaller than 7x7 can't benefit from adjustment
    if (KX * KY <= 49) {
        log.trace("Not beneficial for convolutions with small kernel {0}x{1} ", KY, KX);
        return false;
    }

    return true;
}
