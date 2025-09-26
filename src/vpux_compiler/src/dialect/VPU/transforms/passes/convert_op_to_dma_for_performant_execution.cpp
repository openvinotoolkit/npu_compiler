//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/convert_op_to_dma_for_performant_execution_getter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTOPTODMAFORPERFORMANTEXECUTION
#define GEN_PASS_DEF_CONVERTOPTODMAFORPERFORMANTEXECUTION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// MoveToDMAPass
//

class ConvertOpToDMAForPerformantExecutionPass final :
        public VPU::impl::ConvertOpToDMAForPerformantExecutionBase<ConvertOpToDMAForPerformantExecutionPass> {
public:
    explicit ConvertOpToDMAForPerformantExecutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertOpToDMAForPerformantExecutionPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto arch = config::getArch(func);
    auto conversionStrategy = VPU::createConvertOpToDMAForPerformantExecutionStrategy(arch);

    mlir::ConversionTarget target(ctx);
    conversionStrategy->markOpLegality(target, _log);

    mlir::RewritePatternSet patterns(&ctx);
    conversionStrategy->addPatterns(patterns, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertOpToDMAForPerformantExecutionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertOpToDMAForPerformantExecutionPass(Logger log) {
    return std::make_unique<ConvertOpToDMAForPerformantExecutionPass>(log);
}
