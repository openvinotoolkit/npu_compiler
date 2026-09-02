//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/adjust_quantized_conv_shape_verifier.hpp"

namespace vpux::IE::arch37xx {

/*
   Class for adjust quantized convolution shape conversion verifier for NPU37XX
*/
class AdjustQuantizedConvShapeVerifier : public vpux::IE::AdjustQuantizedConvShapeVerifierBase {
public:
    bool isBeneficialConversion(IE::ConvolutionOp convOp, mlir::Value filter, Logger log) const override;
};

}  // namespace vpux::IE::arch37xx
