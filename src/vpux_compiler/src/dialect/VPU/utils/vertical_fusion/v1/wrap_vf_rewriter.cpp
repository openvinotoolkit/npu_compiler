//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/wrap_vf_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
namespace vpux::VPU::VF::v1 {

bool WrapVFRewriter::opNeedsTobeWrapped(VPU::VerticalFusionOpInterface op) const {
    if (mlir::isa<VPU::VerticalFusionOp>(op->getParentOp())) {
        _log.trace("operation '{0}' at '{1}' is already wrapped in VF op", op->getName(), op->getLoc());
        return false;
    }

    if (!op.isVFSupported()) {
        _log.trace("Operation '{0}' at '{1}' doesn't support VF", op->getName(), op->getLoc());
        return false;
    }

    if (op->hasAttr(tilingStrategy)) {
        const auto tilingSize = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(tilingStrategy)));
        const auto tilingDimCount = llvm::count_if(tilingSize, [](auto value) {
            return value > 1;
        });
        if (tilingDimCount > 1) {
            _log.trace("Operation '{0}' at '{1}' can not be wraped in VF since multi-dim tiling is not supported",
                       op->getName(), op->getLoc());
            return false;
        }
    }
    return true;
}
}  // namespace vpux::VPU::VF::v1
