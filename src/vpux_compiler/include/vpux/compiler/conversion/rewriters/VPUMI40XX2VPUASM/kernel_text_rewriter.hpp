//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class KernelTextRewriter : public VPUASMSymbolizationPattern<VPUMI40XX::DeclareKernelTextOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI40XX::DeclareKernelTextOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
