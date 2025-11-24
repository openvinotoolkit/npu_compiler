//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU::VF::v2 {

//
// MoveViewOpsRewriter
//

class MoveViewOpsRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    MoveViewOpsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace vpux::VPU::VF::v2
