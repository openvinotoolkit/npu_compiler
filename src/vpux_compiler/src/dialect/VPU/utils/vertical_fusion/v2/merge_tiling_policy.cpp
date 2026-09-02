//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/conv_post_eltwise_merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/general_merge_tiling_policy.hpp"

namespace vpux::VPU::VF::v2 {

std::unique_ptr<MergeTilingPolicy> createMergeTilingPolicy(MergeTilingPolicyType policyType,
                                                           bool enableVerticalFusionPipelining, Logger log,
                                                           VFCacheAnalysis& cache) {
    switch (policyType) {
    case MergeTilingPolicyType::ConvPostEltwise:
        return std::make_unique<ConvPostEltwiseMergeTilingPolicy>(enableVerticalFusionPipelining, log, cache);
    case MergeTilingPolicyType::General:
        return std::make_unique<GeneralMergeTilingPolicy>(enableVerticalFusionPipelining, log, cache);
    }

    VPUX_THROW("Unsupported merge tiling policy type");
}

}  // namespace vpux::VPU::VF::v2
