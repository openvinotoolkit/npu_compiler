//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_convert_to_dpu_checker.hpp"

using namespace vpux::IE::arch37xx;

//
// FuseConvertToDPUChecker
//

bool FuseConvertToDPUChecker::isFusionToParentDPUOpSupported(mlir::Operation* dpuOp, Logger log) const {
    auto parentInputType = mlir::cast<NDTypeInterface>(dpuOp->getOperand(0).getType());
    if (!mlir::isa<mlir::FloatType>(parentInputType.getElementType())) {
        log.trace("Parent input is not a float type = {0}.", parentInputType.getElementType());
        return false;
    }

    // For MaxPool, DPU SCL outputs FP16, not F32, so just setting bypass to conversion to F16 in PPE will not be enough
    // to get a proper FP32 output
    if (mlir::isa<IE::MaxPoolOp>(dpuOp)) {
        log.trace("Parent op of type {0} at loc {1} does not support FP32 output.", dpuOp->getName(), dpuOp->getLoc());
        return false;
    }

    // There might be a way to set proper FP32 clamp values when bypassing conversion to FP16 in FpPPE
    // Ticket to investigate: E#150685
    if (auto postOpIf = mlir::dyn_cast<IE::LayerWithPostOpInterface>(dpuOp)) {
        if (postOpIf.getPostOp().has_value()) {
            auto postOp = postOpIf.getPostOp().value();
            if (postOp.getStringRef() == IE::ClampOp::getOperationName()) {
                log.trace("Parent op of type {0} at loc {1} has Clamp post op.", dpuOp->getName(), dpuOp->getLoc());
                return false;
            }
        }
    }

    return true;
}
