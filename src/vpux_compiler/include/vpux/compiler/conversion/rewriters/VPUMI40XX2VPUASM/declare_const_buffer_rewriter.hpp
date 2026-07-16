//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class DeclareConstBufferRewriter : public VPUASMSymbolizationPattern<Const::DeclareOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(Const::DeclareOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
