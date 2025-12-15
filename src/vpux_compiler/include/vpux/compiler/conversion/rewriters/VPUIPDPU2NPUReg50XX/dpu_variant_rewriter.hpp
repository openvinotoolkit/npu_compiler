//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"

namespace vpux {
namespace vpuipdpu2npureg50xx {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log,
                       VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPURegMapped::NPU5PPEBackwardsCompatibilityMode _npu5PPEBackwardsCompatibilityMode;

    mlir::LogicalResult verifyDPUVariant(VPUIPDPU::DPUVariantOp op) const;

    void fillDPUConfigs(mlir::Region& DPURegion, vpux::NPUReg50XX::Descriptors::DpuVariantRegister& descriptor) const;

    void fillBarrierCfg(VPUIPDPU::DPUVariantOp op, vpux::NPUReg50XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp,
                          vpux::NPUReg50XX::Descriptors::DpuVariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg50xx
}  // namespace vpux
