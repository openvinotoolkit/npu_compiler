//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_cost_model.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_PRINTNNCACHESTATISTICS
#define GEN_PASS_DEF_PRINTNNCACHESTATISTICS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// PrintNNCacheStatisticsPass
//
class PrintNNCacheStatisticsPass final : public VPU::impl::PrintNNCacheStatisticsBase<PrintNNCacheStatisticsPass> {
public:
    PrintNNCacheStatisticsPass(Logger log, StringRef passName): passName(passName) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    StringRef passName;
};

void PrintNNCacheStatisticsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto costModelUtils = VPU::getICostModelUtilsInterface(moduleOp->getContext());
    if (costModelUtils->isNNCacheStatisticsSupported()) {
        auto maybeCostModelAnalysis = getCachedAnalysis<VPU::CostModelAnalysis>();
        const auto arch = config::getArch(moduleOp);
        auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, arch, _log);
        _log.info("[NN Cache statistics] for Pass {0}, costModel: {1}", passName,
                  costModel->getPreloadedCacheCounter().printString());
    }
}

}  // namespace

//
// createPrintNNCacheStatistics
//

std::unique_ptr<mlir::Pass> VPU::createPrintNNCacheStatisticsPass(Logger log, StringRef passName) {
    return std::make_unique<PrintNNCacheStatisticsPass>(log, passName);
}
