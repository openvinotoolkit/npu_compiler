//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"
#include "vpux/compiler/NPU40XX/dialect/IE/impl/d2s_to_transposed_conv_verifier.hpp"

namespace vpux {
namespace IE {

//
// D2SToTransposedConvVerifierBase
//

mlir::LogicalResult D2SToTransposedConvVerifierBase::isBeneficialConversion(Logger, mlir::PatternRewriter&,
                                                                            IE::DepthToSpaceOp) const {
    return mlir::success();
}

std::unique_ptr<D2SToTransposedConvVerifierBase> createD2SToTransposedConvVerifier(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<IE::arch37xx::D2SToTransposedConvVerifier>();
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return std::make_unique<IE::arch40xx::D2SToTransposedConvVerifier>();
    default: {
        return std::make_unique<D2SToTransposedConvVerifierBase>();
    }
    }
}

}  // namespace IE
}  // namespace vpux
