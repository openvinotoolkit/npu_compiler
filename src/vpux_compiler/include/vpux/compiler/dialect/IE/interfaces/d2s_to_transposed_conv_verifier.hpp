//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

/*
   Class for DepthSpace to TransposedConv conversion verifier
*/
class D2SToTransposedConvVerifierBase {
public:
    virtual ~D2SToTransposedConvVerifierBase() = default;

    virtual mlir::LogicalResult isBeneficialConversion(Logger log, mlir::PatternRewriter& rewriter,
                                                       IE::DepthToSpaceOp d2sOp) const;
};

/*
   Find right class to verify whether DepthSpace to TransposedConv conversion is beneficial for particular platform
*/
std::unique_ptr<D2SToTransposedConvVerifierBase> createD2SToTransposedConvVerifier(vpux::VPU::ArchKind arch);

}  // namespace IE
}  // namespace vpux
