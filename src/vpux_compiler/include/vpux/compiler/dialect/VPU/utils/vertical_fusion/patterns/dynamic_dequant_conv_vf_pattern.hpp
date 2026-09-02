//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_pattern_matcher.hpp"

namespace vpux {
namespace VPU {

class DynamicDequantConvVFPattern final : public VFPattern {
public:
    mlir::FailureOr<VPU::VerticalFusionOp> tryMerge(VPU::VerticalFusionOp rootOp, mlir::RewriterBase& rewriter,
                                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                    Logger log) const override;

    StringRef getPatternName() const override {
        return "DynamicDequantConvVFPattern";
    }

private:
    std::optional<SmallVector<VPU::VerticalFusionOp>> getMatchedChain(VPU::VerticalFusionOp rootOp, Logger log) const;
    bool hasExpectedChain(VPU::VerticalFusionOp dequantVF, VPU::VerticalFusionOp convVF) const;
    bool hasExpectedMemPermuteProducer(VPU::VerticalFusionOp memPermuteVF, VPU::VerticalFusionOp dequantVF) const;
    VPU::VerticalFusionOp getExpectedMemPermuteProducer(VPU::VerticalFusionOp dequantVF) const;
    bool alignConvStrategyWithDequant(VPU::VerticalFusionOp dequantVF, VPU::VerticalFusionOp convVF, Logger log) const;
    std::optional<VPU::VF::v2::VFCase> buildVFCase(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp convVFOp,
                                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                   Logger log) const;
};

}  // namespace VPU
}  // namespace vpux
