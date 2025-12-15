//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <iterator>

namespace vpux::IE {
#define GEN_PASS_DECL_PROPAGATEDEQUANTTHROUGHCONCAT
#define GEN_PASS_DEF_PROPAGATEDEQUANTTHROUGHCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// PropagateDequantThroughConcat
//

class PropagateDequantThroughConcat final :
        public IE::impl::PropagateDequantThroughConcatBase<PropagateDequantThroughConcat> {
public:
    explicit PropagateDequantThroughConcat(Logger log): _log(log) {
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

class PropagateDequantThroughConcat::ConcatOpConverter final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ConcatOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ConcatOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PropagateDequantThroughConcat::ConcatOpConverter::matchAndRewrite(
        IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}]: rewriting {1}", getDebugName(), origConcatOp->getLoc());
    const auto concatInputList = origConcatOp.getInputs();

    // This check is needed in case if there is convolution with big kernel. Such convolution is
    // splitted into multiple ones and such case doesn't support further dequantize fusion.
    // So if we propagate dequantize through such concat, we may end up having more dequantize layers
    // and performance degradation as a result.
    if (std::distance(origConcatOp->getUsers().begin(), origConcatOp->getUsers().end()) > 1) {
        return mlir::failure();
    }

    auto isDequant = [](mlir::Value input) {
        return input.getDefiningOp<IE::DequantizeOp>() != nullptr;
    };

    if (std::count_if(concatInputList.begin(), concatInputList.end(), isDequant) != 1) {
        return mlir::failure();
    }

    // Find Dequant operation
    auto dequantInputIter = std::find_if(concatInputList.begin(), concatInputList.end(), isDequant);
    auto dequantOp = (*dequantInputIter).getDefiningOp<IE::DequantizeOp>();

    if (IE::isPerAxisQuant(dequantOp.getInput())) {
        return mlir::failure();
    }

    auto hasAnyConstInputs = [&]() {
        return llvm::any_of(origConcatOp.getInputs(), [&](mlir::Value buff) {
            return mlir::isa_and_nonnull<Const::DeclareOp>(buff.getDefiningOp());
        });
    };

    // If there are no constant inputs, the propagation is not efficient.
    if (!hasAnyConstInputs()) {
        _log.nest().trace("ConcatOp has no constant inputs, skipping dequant propagation");
        return mlir::failure();
    }

    auto dequantizeElemType = mlir::cast<vpux::NDTypeInterface>(dequantOp.getInput().getType()).getElementType();
    SmallVector<mlir::Value> newConcatInputs;
    for (const auto& concatInput : concatInputList) {
        if (isDequant(concatInput)) {
            newConcatInputs.push_back(dequantOp.getInput());
        } else {
            auto quantizeOp = rewriter.createOrFold<IE::QuantizeOp>(appendLoc(concatInput.getLoc(), "quantize"),
                                                                    concatInput, dequantizeElemType);
            _log.nest().trace("Inserted new Quantize: {0}", quantizeOp);
            newConcatInputs.push_back(quantizeOp);
        }
    }

    auto newConcatOp =
            rewriter.create<IE::ConcatOp>(origConcatOp->getLoc(), newConcatInputs, origConcatOp.getPerAxisAttr(),
                                          origConcatOp.getStaticOffsetsAttr());
    auto newDequantOp =
            rewriter.create<IE::DequantizeOp>(dequantOp->getLoc(), newConcatOp.getOutput(), dequantOp.getDstElemType());
    rewriter.replaceOp(origConcatOp, newDequantOp.getOutput());
    _log.nest().trace("ConcatOp conversion done");

    return mlir::success();
}

void PropagateDequantThroughConcat::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PropagateDequantThroughConcat::ConcatOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateDequantThroughConcatPass(Logger log) {
    return std::make_unique<PropagateDequantThroughConcat>(log);
}
