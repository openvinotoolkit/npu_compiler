//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace IE {

/*
   Class for adjust quantized convolution shape conversion verifier
*/
class AdjustQuantizedConvShapeVerifierBase {
public:
    virtual ~AdjustQuantizedConvShapeVerifierBase() = default;

    virtual bool isBeneficialConversion(IE::ConvolutionOp convOp, mlir::Value filter, Logger log) const = 0;
};

}  // namespace IE
}  // namespace vpux
