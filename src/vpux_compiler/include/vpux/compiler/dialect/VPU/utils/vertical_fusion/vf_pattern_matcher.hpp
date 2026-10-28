//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <memory>

namespace vpux {
namespace VPU {

class LayerVPUNNCost;

class VFPattern {
public:
    virtual ~VFPattern() = default;

    virtual mlir::FailureOr<VPU::VerticalFusionOp> tryMerge(VPU::VerticalFusionOp rootOp, mlir::RewriterBase& rewriter,
                                                            const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                            VFMergeCostMode costMode, Logger log) const = 0;
    virtual StringRef getPatternName() const = 0;

protected:
    mlir::Operation* getSingleInnerOp(VPU::VerticalFusionOp vfOp) const;
    VPU::VerticalFusionOp getUniqUser(VPU::VerticalFusionOp vfOp) const;
};

class VFPatternMatcher {
public:
    VFPatternMatcher(VFMergeCostMode costMode, Logger log);

    void addPattern(std::unique_ptr<VFPattern> pattern);
    bool applyPatterns(mlir::func::FuncOp func, mlir::RewriterBase& rewriter) const;

private:
    SmallVector<std::unique_ptr<VFPattern>> _patterns;
    VFMergeCostMode _costMode = VFMergeCostMode::PERFORMANCE;
    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
