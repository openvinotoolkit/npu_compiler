//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/adjust_quantized_conv_shape_verifier.hpp"

using namespace vpux::IE::arch37xx;

//
// AdjustQuantizedConvShapeVerifier
//

bool AdjustQuantizedConvShapeVerifier::isBeneficialConversion(IE::ConvolutionOp, mlir::Value, Logger) const {
    return false;
}
