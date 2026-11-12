//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/act_shave_rewriter.hpp"

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"

using namespace vpux;
using namespace NPUReg60XX;
using namespace NPUReg60XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg60xx {

mlir::LogicalResult ActShaveRtRewriter::matchAndRewrite(VPUASM::ActShaveRtOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    NPUReg60XX::ActShaveRtOp::create(rewriter, origOp->getLoc(), origOp.getSymNameAttr(), origOp.getKernelPathAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
