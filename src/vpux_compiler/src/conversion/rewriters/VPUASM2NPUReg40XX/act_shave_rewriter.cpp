//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/act_shave_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"

using namespace vpux;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

mlir::LogicalResult ActShaveRtRewriter::matchAndRewrite(VPUASM::ActShaveRtOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.create<NPUReg40XX::ActShaveRtOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getKernelPathAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg40xx
}  // namespace vpux
