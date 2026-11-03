//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_pattern_matcher.hpp"

namespace vpux {
namespace VPU {

class SDPAVFPattern : public VFPattern {
public:
    // Try to match and merge an SDPA-related VF chain rooted at rootOp.
    mlir::FailureOr<VPU::VerticalFusionOp> tryMerge(VPU::VerticalFusionOp rootOp, mlir::RewriterBase& rewriter,
                                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                    VFMergeCostMode costMode, Logger log) const override;

    // Return the stable pattern name used by VF pattern matcher diagnostics.
    StringRef getPatternName() const override {
        return "SDPAVFPattern";
    }

protected:
    // Extract single inner ops from each VF op; return nullopt if any VF op contains zero or multiple inner ops.
    std::optional<SmallVector<mlir::Operation*>> getInnerOps(ArrayRef<VPU::VerticalFusionOp> vfOps, Logger log,
                                                             StringRef validatorName) const;
    // Detect a supported SDPA chain starting from rootOp and return the ordered VF op chain.
    virtual std::optional<SmallVector<VPU::VerticalFusionOp>> getMatchedChain(VPU::VerticalFusionOp rootOp,
                                                                              Logger log) const;
    // Match SoftmaxDecomposedSDPA form: Conv-SoftMax-{Conv, Reduce->Conv}-Eltwise.
    std::optional<SmallVector<VPU::VerticalFusionOp>> tryMatchSoftmaxDecomposedSDPA(VPU::VerticalFusionOp rootOp,
                                                                                    VPU::VerticalFusionOp second,
                                                                                    Logger log) const;
    // Match SDPA form: Conv-Eltwise-SoftMax-Conv.
    std::optional<SmallVector<VPU::VerticalFusionOp>> tryMatchSDPA(VPU::VerticalFusionOp rootOp,
                                                                   VPU::VerticalFusionOp second, Logger log) const;
    // Verify that vfOps exactly follows the expected SoftmaxDecomposedSDPA op sequence.
    bool hasExpectedSoftmaxDecomposedChain(ArrayRef<VPU::VerticalFusionOp> vfOps, Logger log) const;
    // Verify that vfOps exactly follows the expected SDPA op sequence.
    bool hasExpectedSDPAChain(ArrayRef<VPU::VerticalFusionOp> vfOps, Logger log) const;
    // Align multi-cluster strategies for the matched pattern.
    virtual bool alignMultiClusterStrategy(VPU::VF::v2::VFConfig& vfConfig, Logger log) const;
    // Build and return the best VF scheduling case for mergedOp using the provided cost function.
    virtual std::optional<VPU::VF::v2::VFCase> buildVFCase(VPU::VerticalFusionOp mergedOp,
                                                           ArrayRef<VPU::VerticalFusionOp> matchedOps,
                                                           const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                           VFMergeCostMode costMode, Logger log) const;
};

}  // namespace VPU
}  // namespace vpux
