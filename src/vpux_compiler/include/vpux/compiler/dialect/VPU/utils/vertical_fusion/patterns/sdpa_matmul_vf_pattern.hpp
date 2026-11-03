//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/sdpa_vf_pattern.hpp"

namespace vpux {
namespace VPU {

class SDPAMatMulVFPattern final : public SDPAVFPattern {
public:
    StringRef getPatternName() const override {
        return "SDPAMatMulVFPattern";
    }

protected:
    std::optional<SmallVector<VPU::VerticalFusionOp>> getMatchedChain(VPU::VerticalFusionOp rootOp,
                                                                      Logger log) const override;

    bool alignMultiClusterStrategy(VPU::VF::v2::VFConfig& vfConfig, Logger log) const override;

    std::optional<VPU::VF::v2::VFCase> buildVFCase(VPU::VerticalFusionOp mergedOp,
                                                   ArrayRef<VPU::VerticalFusionOp> matchedOps,
                                                   const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                   VFMergeCostMode costMode, Logger log) const override;
};

}  // namespace VPU
}  // namespace vpux
