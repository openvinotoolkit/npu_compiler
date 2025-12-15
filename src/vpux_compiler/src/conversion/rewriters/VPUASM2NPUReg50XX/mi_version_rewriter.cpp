//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg50XX/mi_version_rewriter.hpp"

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops.hpp"

using namespace NPUReg50XX;
using namespace NPUReg50XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg50xx {

mlir::LogicalResult MappedInferenceVersionRewriter::matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<NPUReg50XX::MappedInferenceVersionOp>(
            origOp, origOp.getSymNameAttr(), origOp.getMajorAttr(), origOp.getMinorAttr(), origOp.getPatchAttr());
    return mlir::success();
}
}  // namespace vpuasm2npureg50xx
}  // namespace vpux
