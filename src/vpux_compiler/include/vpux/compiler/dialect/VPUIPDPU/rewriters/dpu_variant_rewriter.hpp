//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

namespace vpux {
namespace VPUIPDPU {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUASM::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap,
                       VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
    VPURegMapped::NPU5PPEBackwardsCompatibilityMode _npu5PPEBackwardsCompatibilityMode;
};

}  // namespace VPUIPDPU
}  // namespace vpux
