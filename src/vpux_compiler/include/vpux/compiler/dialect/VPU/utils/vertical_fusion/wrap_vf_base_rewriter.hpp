//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

namespace vpux::VPU::VF {

//
// WrapVFRewriter
//

class WrapVFRewriterBase : public mlir::OpInterfaceRewritePattern<VPU::VerticalFusionOpInterface> {
public:
    WrapVFRewriterBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::VerticalFusionOpInterface>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

    virtual void wrapIntoVFRegion(VPU::VerticalFusionOpInterface op, mlir::PatternRewriter& rewriter) const;

    virtual bool opNeedsTobeWrapped(VPU::VerticalFusionOpInterface op) const = 0;

protected:
    Logger _log;
};
}  // namespace vpux::VPU::VF
