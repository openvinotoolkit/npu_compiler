//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/dpu_pvp_rewriter.hpp"

#include "vpux/compiler/dialect/VPUIPDPU/utils/utils.hpp"

using namespace vpux;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult DpuPVPRewriter::matchAndRewrite(VPUASM::DpuPVPOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    // only log PVP counts
    if (_log.isActive(LogLevel::Info)) {
        VPUIPDPU::computeDpuPvpCounts(_elfMain, _log);
    }
    // For NPU40XX, DpuPVP is not needed - erase the op
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
