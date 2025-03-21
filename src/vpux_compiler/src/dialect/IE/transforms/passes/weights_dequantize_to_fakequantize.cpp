//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/weights_dequantize_to_fakequantize_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <vector>

namespace vpux::IE {
#define GEN_PASS_DECL_WEIGHTSDEQUANTIZETOFAKEQUANTIZE
#define GEN_PASS_DEF_WEIGHTSDEQUANTIZETOFAKEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

namespace vpux {

class WeightsDequantizeToFakeQuantizePass final :
        public IE::impl::WeightsDequantizeToFakeQuantizeBase<WeightsDequantizeToFakeQuantizePass> {
public:
    WeightsDequantizeToFakeQuantizePass() = default;
    explicit WeightsDequantizeToFakeQuantizePass(const IE::LowPrecisionTransformOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);

        initializeFromOptions();
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnFunc() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    bool _enableWDBlockArgumentInput = true;
};

mlir::LogicalResult WeightsDequantizeToFakeQuantizePass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void WeightsDequantizeToFakeQuantizePass::initializeFromOptions() {
    if (enableWDBlockArgumentInput.hasValue()) {
        _enableWDBlockArgumentInput = enableWDBlockArgumentInput.getValue();
    }
}

void WeightsDequantizeToFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    auto strategy = vpux::IE::createWeightsDequantizeToFakeQuantizeStrategy(func, _enableWDBlockArgumentInput);
    strategy->addPatterns(patterns, _log);

    auto config = getDefaultGreedyRewriteConfig();
    // Note: the implicit contract in the compiler mandates that all
    // quantization-like patterns are converted to FQ. thus, we have to run
    // forever until convergence. if this halts, there's a bug somewhere in the
    // pass.
    config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createWeightsDequantizeToFakeQuantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createWeightsDequantizeToFakeQuantizePass() {
    return std::make_unique<WeightsDequantizeToFakeQuantizePass>();
}

std::unique_ptr<mlir::Pass> vpux::IE::createWeightsDequantizeToFakeQuantizePass(
        const IE::LowPrecisionTransformOptions& options, Logger log) {
    return std::make_unique<WeightsDequantizeToFakeQuantizePass>(options, log);
}
