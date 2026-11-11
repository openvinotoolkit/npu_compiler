//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/mi_version_rewriter.hpp"

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"

using namespace vpux;
using namespace NPUReg60XX;
using namespace NPUReg60XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg60xx {

mlir::LogicalResult MappedInferenceVersionRewriter::matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<NPUReg60XX::MappedInferenceVersionOp>(
            origOp, origOp.getSymNameAttr(), origOp.getMajorAttr(), origOp.getMinorAttr(), origOp.getPatchAttr());

    return mlir::success();
}

}  // namespace vpuasm2npureg60xx
}  // namespace vpux
