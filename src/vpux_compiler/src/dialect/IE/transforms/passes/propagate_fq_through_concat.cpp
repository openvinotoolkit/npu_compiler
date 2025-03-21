//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEFQTHROUGHCONCAT
#define GEN_PASS_DEF_PROPAGATEFQTHROUGHCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// PropagateFqThroughConcat
//

class PropagateFqThroughConcat final : public IE::impl::PropagateFqThroughConcatBase<PropagateFqThroughConcat> {
public:
    explicit PropagateFqThroughConcat(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

public:
    class ConcatOpConverter;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// ConcatOpConverter
//

class PropagateFqThroughConcat::ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ConcatOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isPerTensorFq(const mlir::Value in) {
    const auto maybeFqOp = in.getDefiningOp<IE::FakeQuantizeOp>();
    if (maybeFqOp == nullptr) {
        return false;
    }

    return IE::isPerTensorFQ({maybeFqOp});
}

mlir::LogicalResult PropagateFqThroughConcat::ConcatOpConverter::matchAndRewrite(
        IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: rewriting {1}", getDebugName(), origConcatOp->getLoc());
    const auto concatInputList = origConcatOp.getInputs();
    // Find FQ operation
    auto fqOutputIter = std::find_if(concatInputList.begin(), concatInputList.end(), isPerTensorFq);
    auto fqOp = (*fqOutputIter).getDefiningOp<IE::FakeQuantizeOp>();

    // try to use concat output quantization parameter if it has
    auto isFakeQuantizeOp = [](const mlir::Operation* operation) {
        return mlir::isa<IE::FakeQuantizeOp>(*operation);
    };
    bool outHasFq = false;
    auto outFq = std::find_if(origConcatOp.getOutput().getUsers().begin(), origConcatOp.getOutput().getUsers().end(),
                              isFakeQuantizeOp);
    if (outFq != origConcatOp.getOutput().getUsers().end()) {
        outHasFq = true;
        fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(*outFq);
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatInputList.size());
    for (const auto& concatInput : concatInputList) {
        if (isPerTensorFq(concatInput)) {
            newConcatInputs.push_back(concatInput);
        } else {
            auto newFqOp = rewriter.create<IE::FakeQuantizeOp>(
                    origConcatOp->getLoc(), concatInput, fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(),
                    fqOp.getOutputHigh(), fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());

            _log.nest().trace("Inserted new FQ: {0}", newFqOp);
            newConcatInputs.push_back(newFqOp.getOutput());
        }
    }

    auto newConcatOp =
            rewriter.create<IE::ConcatOp>(origConcatOp->getLoc(), newConcatInputs, origConcatOp.getPerAxisAttr(),
                                          origConcatOp.getStaticOffsetsAttr());

    if (outHasFq) {
        rewriter.replaceOp(origConcatOp, newConcatOp.getOutput());
    } else {
        rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                origConcatOp, newConcatOp.getOutput(), fqOp.getInputLow(), fqOp.getInputHigh(), fqOp.getOutputLow(),
                fqOp.getOutputHigh(), fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
    }
    _log.nest().trace("ConcatOp conversion done");

    return mlir::success();
}

void PropagateFqThroughConcat::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalConcat = [](IE::ConcatOp concatOp) -> bool {
        const auto concatInputList = concatOp.getInputs();
        // Either all concat inputs must be quantized, or none of them.
        if (std::all_of(concatInputList.begin(), concatInputList.end(), isPerTensorFq)) {
            return true;
        }

        if (std::none_of(concatInputList.begin(), concatInputList.end(), isPerTensorFq)) {
            return true;
        }

        if (std::count_if(concatInputList.begin(), concatInputList.end(), isPerTensorFq) == 1) {
            return false;
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::ConcatOp>(isLegalConcat);
    target.addLegalOp<IE::FakeQuantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateFqThroughConcat::ConcatOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateFqThroughConcatPass(Logger log) {
    return std::make_unique<PropagateFqThroughConcat>(log);
}
