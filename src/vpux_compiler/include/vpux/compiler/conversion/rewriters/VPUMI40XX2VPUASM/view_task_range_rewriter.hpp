//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_pattern.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

class ViewTaskRangeRewriter : public VPUASMSymbolizationPattern<VPURegMapped::ViewTaskRangeOp> {
public:
    using Base::Base;
    mlir::FailureOr<SymbolizationResult> symbolize(VPURegMapped::ViewTaskRangeOp op, SymbolMapper& mapper,
                                                   mlir::ConversionPatternRewriter& rewriter) const override;
};

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
