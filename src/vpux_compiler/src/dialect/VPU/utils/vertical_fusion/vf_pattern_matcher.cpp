//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_pattern_matcher.hpp"

#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/dynamic_dequant_conv_vf_pattern.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/sdpa_vf_pattern.hpp"

using namespace vpux;
using namespace VPU;

mlir::Operation* VFPattern::getSingleInnerOp(VPU::VerticalFusionOp vfOp) const {
    auto* body = vfOp.getBody();
    if (body == nullptr) {
        return nullptr;
    }

    mlir::Operation* singleOp = nullptr;
    for (auto& op : body->without_terminator()) {
        if (singleOp != nullptr) {
            return nullptr;
        }
        singleOp = &op;
    }
    return singleOp;
}

VPU::VerticalFusionOp VFPattern::getSingleVFUser(VPU::VerticalFusionOp vfOp) const {
    if (!vfOp->hasOneUse()) {
        return nullptr;
    }
    return mlir::dyn_cast<VPU::VerticalFusionOp>(*vfOp->user_begin());
}

VPU::VFPatternMatcher::VFPatternMatcher(Logger log): _log(log) {
    addPattern(std::make_unique<DynamicDequantConvVFPattern>());
    addPattern(std::make_unique<SDPAVFPattern>());
}

void VPU::VFPatternMatcher::addPattern(std::unique_ptr<VFPattern> pattern) {
    _patterns.push_back(std::move(pattern));
}

bool VPU::VFPatternMatcher::applyPatterns(mlir::func::FuncOp func, mlir::RewriterBase& rewriter) const {
    bool changed = false;
    bool localChange = true;
    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func, _log);

    while (localChange) {
        localChange = false;

        SmallVector<VPU::VerticalFusionOp> vfOps;
        func.walk([&](VPU::VerticalFusionOp op) {
            vfOps.push_back(op);
        });

        _log.trace("VFPatternMatcher::applyPatterns total root candidates={0}", vfOps.size());

        for (auto rootOp : vfOps) {
            if (rootOp == nullptr || rootOp->getParentOfType<VPU::VerticalFusionOp>() != nullptr ||
                rootOp->use_empty()) {
                continue;
            }

            const auto rootLoc = rootOp->getLoc();
            _log.trace("VFPatternMatcher::applyPatterns try root={0}", rootLoc);
            for (const auto& pattern : _patterns) {
                _log.trace("VFPatternMatcher::applyPatterns apply pattern={0} root={1}", pattern->getPatternName(),
                           rootLoc);
                auto mergedOp = pattern->tryMerge(rootOp, rewriter, costFunction, _log);
                if (mlir::failed(mergedOp)) {
                    continue;
                }

                _log.trace("VFPatternMatcher::applyPatterns applied pattern={0} root={1}", pattern->getPatternName(),
                           rootLoc);
                changed = true;
                localChange = true;
                break;
            }

            if (localChange) {
                break;
            }
        }
    }

    return changed;
}
