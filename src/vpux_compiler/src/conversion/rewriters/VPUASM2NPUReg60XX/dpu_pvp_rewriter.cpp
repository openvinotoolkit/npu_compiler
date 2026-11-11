//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/dpu_pvp_rewriter.hpp"

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/utils/utils.hpp"

using namespace vpux;
using namespace NPUReg60XX;
using namespace NPUReg60XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg60xx {

mlir::LogicalResult DpuPVPRewriter::matchAndRewrite(VPUASM::DpuPVPOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto [countFPOps, totalOpCount] = VPUIPDPU::computeDpuPvpCounts(_elfMain, _log);

    VpuDpuPVP descriptor;
    descriptor.write<Fields::VpuDpuPVP_opFPCount>(countFPOps);
    descriptor.write<Fields::VpuDpuPVP_totalOpCount>(totalOpCount);

    NPUReg60XX::DpuPVPOp::create(rewriter, origOp->getLoc(),  //
                                 origOp.getSymNameAttr(),     //
                                 std::move(descriptor));

    rewriter.eraseOp(origOp);

    return mlir::success();
}
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
