//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <cstddef>

namespace vpux {

class AliasesInfo;

}  // namespace vpux

namespace vpux::VPUIP::DmaOptimization {

class EditPlan;
struct EdgeCollection;
struct LoopRegionBoundaryInfo;

/// CopyReusePlanner is responsible for planning the reuse of CopyOp operations by extracting candidates,
/// obtaining approval for reuse, and generating an edit plan for the approved rewrites.
/// It has three main stages: candidate extraction, approval, and edit plan generation.
/// Stage 1: candidate extraction currently handles FanOut edges, where one source buffer may be used by multiple
/// CopyOp operations.
/// Stage 2: approval: The extracted candidates are evaluated and approved for reuse based on various
/// criteria. Stage 3: edit plan generation: An edit plan is created for the approved rewrites, detailing how the IR
/// should be transformed.
class CopyReusePlanner final {
public:
    explicit CopyReusePlanner(Logger log): _log(log) {
    }

    size_t planCopyReuse(const EdgeCollection& edges, const AliasesInfo& aliasesInfo, EditPlan& out) const;

private:
    Logger _log;
};

}  // namespace vpux::VPUIP::DmaOptimization
