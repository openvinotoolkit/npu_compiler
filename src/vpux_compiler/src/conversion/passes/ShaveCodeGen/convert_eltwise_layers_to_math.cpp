//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Pass/Pass.h>

// Generated
namespace ConvertEltwiseLayersToMathPatterns {
#include <vpux/compiler/conversion/convert_eltwise_layers_to_math.hpp.inc>
}

namespace vpux {
#define GEN_PASS_DECL_CONVERTELTWISELAYERS2MATH
#define GEN_PASS_DEF_CONVERTELTWISELAYERS2MATH
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// ConvertSWLayers2LinalgPass
//

class ConvertEltwiseLayers2MathPass final : public impl::ConvertEltwiseLayers2MathBase<ConvertEltwiseLayers2MathPass> {
public:
    explicit ConvertEltwiseLayers2MathPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertEltwiseLayers2MathPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                           mlir::math::MathDialect, mlir::func::FuncDialect>();

    target.addLegalDialect<IE::IEDialect>();
    target.addIllegalOp<IE::CosOp>();

    mlir::RewritePatternSet patterns(&ctx);
    ConvertEltwiseLayersToMathPatterns::populateWithGenerated(patterns);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertEltwiseLayers2MathPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createConvertEltwiseLayers2MathPass(Logger log) {
    return std::make_unique<ConvertEltwiseLayers2MathPass>(log);
}
