//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/OperationSupport.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOUTSTANDINGDEQUANT
#define GEN_PASS_DEF_FUSEOUTSTANDINGDEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

//
// FuseOutstandingDequantPass
//

class FuseOutstandingDequantPass final : public IE::impl::FuseOutstandingDequantBase<FuseOutstandingDequantPass> {
public:
    explicit FuseOutstandingDequantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseOutstandingDequantPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void FuseOutstandingDequantPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getFuseOutstandingDequantStrategy();
    strategy->addPatterns(patterns, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

//
// createFuseOutstandingDequant
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOutstandingDequant(Logger log) {
    return std::make_unique<FuseOutstandingDequantPass>(log);
}
