//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/wrap_vf_rewriter.hpp"

namespace vpux::VPU::VF::v2 {

bool WrapVFRewriter::opNeedsTobeWrapped(VPU::VerticalFusionOpInterface op) const {
    if (mlir::isa<VPU::VerticalFusionOp>(op->getParentOp())) {
        _log.trace("operation '{0}' at '{1}' is already wrapped in VF op", op->getName(), op->getLoc());
        return false;
    }

    if (!op.isVFSupported()) {
        _log.trace("Operation '{0}' at '{1}' doesn't support VF", op->getName(), op->getLoc());
        return false;
    }
    return true;
}
}  // namespace vpux::VPU::VF::v2
