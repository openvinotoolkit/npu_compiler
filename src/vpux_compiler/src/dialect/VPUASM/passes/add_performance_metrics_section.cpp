//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUASM {
#define GEN_PASS_DECL_ADDPERFORMANCEMETRICSSECTION
#define GEN_PASS_DEF_ADDPERFORMANCEMETRICSSECTION
#include "vpux/compiler/dialect/VPUASM/passes.hpp.inc"
}  // namespace vpux::VPUASM

using namespace vpux;

namespace {
class AddPerformanceMetricsSectionPass :
        public VPUASM::impl::AddPerformanceMetricsSectionBase<AddPerformanceMetricsSectionPass> {
public:
    explicit AddPerformanceMetricsSectionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddPerformanceMetricsSectionPass::safeRunOnFunc() {
    auto funcOp = getOperation();
    VPUX_THROW_UNLESS(llvm::hasSingleElement(funcOp.getOps<ELF::MainOp>()), "Expected exactly one ELF mainOp. Got {0}",
                      llvm::range_size(funcOp.getOps<ELF::MainOp>()));
    auto elfMain = *funcOp.getOps<ELF::MainOp>().begin();

    auto perfMetricsOps = elfMain.getOps<ELF::PerformanceMetricsOp>();
    VPUX_THROW_UNLESS(llvm::range_size(perfMetricsOps) == 0,
                      "Expected no pre-existing PerformanceMetricsOp in AddPerformanceMetricsSectionPass. Got {0}",
                      llvm::range_size(perfMetricsOps));

    auto builder = mlir::OpBuilder::atBlockEnd(elfMain.getBody());
    auto perfMetricsOp = builder.create<ELF::PerformanceMetricsOp>(elfMain.getLoc());
    ELF::moveOpToSection(perfMetricsOp.getOperation(), builder);
}
}  // namespace

std::unique_ptr<mlir::Pass> VPUASM::createAddPerformanceMetricsSectionPass(Logger log) {
    return std::make_unique<AddPerformanceMetricsSectionPass>(log);
}
