//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/attributes.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace vpuipdpu2npureg60xx {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::LogicalResult verifyDPUVariant(VPUIPDPU::DPUVariantOp op) const;

    void fillDPUConfigs(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillBarrierCfg(VPUIPDPU::DPUVariantOp op, vpux::NPUReg60XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp,
                          vpux::NPUReg60XX::Descriptors::DpuVariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg60xx
}  // namespace vpux
