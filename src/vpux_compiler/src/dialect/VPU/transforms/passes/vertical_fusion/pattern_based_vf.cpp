//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_pattern_matcher.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_PATTERNBASEDVF
#define GEN_PASS_DEF_PATTERNBASEDVF
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

class PatternBasedVFPass final : public VPU::impl::PatternBasedVFBase<PatternBasedVFPass> {
public:
    explicit PatternBasedVFPass(const WorkloadManagementMode workloadManagementMode, Logger log)
            : _workloadManagementMode(workloadManagementMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    WorkloadManagementMode _workloadManagementMode = WorkloadManagementMode::PWLM_V0_1_PAGES;
};

mlir::LogicalResult PatternBasedVFPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (workloadManagementModeOpt.hasValue()) {
        _workloadManagementMode = workloadManagementModeOpt.getValue();
    }
    return mlir::success();
}

void PatternBasedVFPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto funcName = func.getSymName();
    _log.trace("PatternBasedVFPass start, mode={0}, func={1}", _workloadManagementMode, funcName);

    if (_workloadManagementMode <= WorkloadManagementMode::PWLM_V0_1_PAGES) {
        _log.trace("PatternBasedVFPass skip in PWLM mode, func={0}", funcName);
        return;
    }

    auto& ctx = getContext();
    mlir::IRRewriter rewriter(&ctx);

    VFPatternMatcher matcher(_log);
    _log.trace("PatternBasedVFPass invoking matcher, func={0}", funcName);
    const auto changed = matcher.applyPatterns(func, rewriter);
    _log.trace("PatternBasedVFPass done, changed={0}, func={1}", changed, funcName);
}

}  // namespace

std::unique_ptr<mlir::Pass> VPU::createPatternBasedVFPass(const WorkloadManagementMode workloadManagementMode,
                                                          Logger log) {
    return std::make_unique<PatternBasedVFPass>(workloadManagementMode, log);
}
