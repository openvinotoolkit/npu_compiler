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

class DPUInvariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp> {
public:
    DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log,
                         VPURegMapped::NPU5PPEBackwardsCompatibilityMode npu5PPEBackwardsCompatibilityMode);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    VPURegMapped::NPU5PPEBackwardsCompatibilityMode _npu5PPEBackwardsCompatibilityMode;

    void fillIDUCfg(mlir::Region& DPURegion, vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillMPECfg(mlir::Region& DPURegion, vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillPPECfg(mlir::Region& DPURegion, vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillODUCfg(mlir::Region& DPURegion, vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                        vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                          vpux::NPUReg50XX::Descriptors::DpuInvariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg50xx
}  // namespace vpux
