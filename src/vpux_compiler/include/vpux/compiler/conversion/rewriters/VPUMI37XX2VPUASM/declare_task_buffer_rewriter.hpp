//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI37XX2VPUASM/symbolization_pattern.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

class DeclareTaskBufferRewriter : public VPUASMSymbolizationPattern<VPUMI37XX::DeclareTaskBufferOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPUMI37XX::DeclareTaskBufferOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
    llvm::SmallVector<mlir::FlatSymbolRefAttr> getSymbolicNames(VPUMI37XX::DeclareTaskBufferOp op,
                                                                size_t counter) override;
};

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
