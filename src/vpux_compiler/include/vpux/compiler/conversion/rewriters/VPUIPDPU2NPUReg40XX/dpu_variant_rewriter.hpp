//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"

using namespace NPUReg40XX;

namespace vpux {
namespace vpuipdpu2npureg40xx {

class DPUVariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp> {
public:
    DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUVariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPU::DPUDryRunMode _dryRunMode;

    mlir::LogicalResult verifyDPUVariant(VPUIPDPU::DPUVariantOp op) const;

    void fillDPUConfigs(mlir::Region& DPURegion, vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillBarrierCfg(VPUIPDPU::DPUVariantOp op, vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp,
                          vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const;
    void fillStubCfg(vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
