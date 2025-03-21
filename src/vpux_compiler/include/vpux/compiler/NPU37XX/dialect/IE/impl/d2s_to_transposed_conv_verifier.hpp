//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/d2s_to_transposed_conv_verifier.hpp"

namespace vpux::IE::arch37xx {

/*
   Class for DepthSpace to TransposedConv conversion verifier for NPU37XX
*/
class D2SToTransposedConvVerifier : public D2SToTransposedConvVerifierBase {
public:
    mlir::LogicalResult isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                               IE::DepthToSpaceOp d2sOp) const override;
};

}  // namespace vpux::IE::arch37xx
