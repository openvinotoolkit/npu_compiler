//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/act_shave_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult ActShaveRtRewriter::matchAndRewrite(VPUASM::ActShaveRtOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.create<NPUReg50XX::ActShaveRtOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getKernelPathAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
