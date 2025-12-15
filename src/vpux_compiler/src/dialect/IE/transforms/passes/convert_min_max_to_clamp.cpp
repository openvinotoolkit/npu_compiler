//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMINMAXTOCLAMP
#define GEN_PASS_DEF_CONVERTMINMAXTOCLAMP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// MinMaxConverter
//

template <class ConcreteOp>
class MinMaxConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MinMaxConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("MinMaxConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MinMaxConverter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto nonScalarOperand = origOp.getInput2();
    auto scalarOperand = origOp.getInput1();
    if (mlir::cast<NDTypeInterface>(scalarOperand.getType()).getNumElements() != 1) {
        std::swap(nonScalarOperand, scalarOperand);
    }
    if (mlir::cast<NDTypeInterface>(scalarOperand.getType()).getNumElements() != 1) {
        return mlir::failure();
    }

    const auto scalarOrFail = IE::getQuantizedSplatConstant(scalarOperand);
    if (mlir::failed(scalarOrFail)) {
        _log.nest().trace("Failed to retrieve scalar operand");
        return mlir::failure();
    }
    auto scalarValue = scalarOrFail.value();
    double fp16Max = checked_cast<double>(std::numeric_limits<vpux::type::float16>::max());
    double fp16Lowest = checked_cast<double>(std::numeric_limits<vpux::type::float16>::lowest());
    scalarValue = std::clamp(scalarValue, fp16Lowest, fp16Max);

    auto ctx = origOp->getContext();
    mlir::FloatAttr clampMax;
    mlir::FloatAttr clampMin;
    if constexpr (std::is_same_v<ConcreteOp, IE::MaximumOp>) {
        clampMax = getFPAttr(ctx, static_cast<double>(std::numeric_limits<type::float16>::max()));
        clampMin = getFPAttr(ctx, scalarValue);
    } else if constexpr (std::is_same_v<ConcreteOp, IE::MinimumOp>) {
        clampMax = getFPAttr(ctx, scalarValue);
        clampMin = getFPAttr(ctx, static_cast<double>(std::numeric_limits<type::float16>::lowest()));
    } else {
        static_assert(always_false<ConcreteOp>, "Unsupported operation");
    }

    auto newClampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), nonScalarOperand, clampMin, clampMax);
    rewriter.replaceOp(origOp, newClampOp.getOutput());

    return mlir::success();
}

//
// ConvertMinMaxToClampPass
//

class ConvertMinMaxToClampPass final : public IE::impl::ConvertMinMaxToClampBase<ConvertMinMaxToClampPass> {
public:
    explicit ConvertMinMaxToClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertMinMaxToClampPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MinMaxConverter<IE::MinimumOp>>(&ctx, _log);
    patterns.add<MinMaxConverter<IE::MaximumOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        _log.debug("Failed to replace Min or Max operation with Clamp");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertMinMaxToClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertMinMaxToClampPass(Logger log) {
    return std::make_unique<ConvertMinMaxToClampPass>(log);
}
