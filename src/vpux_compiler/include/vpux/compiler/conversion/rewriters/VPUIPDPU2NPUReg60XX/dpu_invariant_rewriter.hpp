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

class DPUInvariantRewriter final : public mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp> {
public:
    DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log);

public:
    mlir::LogicalResult matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    void fillIDUCfg(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillMPECfg(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillPPECfg(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillODUCfg(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillGeneralCfg(mlir::Region& DPURegion, vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                        vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
    void fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                          vpux::NPUReg60XX::Descriptors::DpuInvariantRegister& descriptor) const;
};

}  // namespace vpuipdpu2npureg60xx
}  // namespace vpux
