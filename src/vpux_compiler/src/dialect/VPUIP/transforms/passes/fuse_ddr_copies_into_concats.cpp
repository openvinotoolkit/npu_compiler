//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_FUSEDDRCOPIESINTOCONCATS
#define GEN_PASS_DEF_FUSEDDRCOPIESINTOCONCATS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool isInputEligibleForConversion(const mlir::Value input, const Logger& log) {
    log.trace("Processing input: {0}", input.getLoc());
    if (mlir::isa<mlir::BlockArgument>(input)) {
        log.trace("Input is a block argument.");
        return false;
    }
    auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
    if (copyOp == nullptr) {
        log.trace("Input producer is not a VPUIP.CopyOp.");
        return false;
    }
    if (!VPUIP::isCopyToDDR(copyOp) || !VPUIP::isCopyFromDDR(copyOp)) {
        log.trace("Input producer is not a DDR2DDR copy.");
        return false;
    }
    if (mlir::isa<mlir::BlockArgument>(copyOp.getInput())) {
        log.trace("Input copy producer is a block argument.");
        return false;
    }
    auto inputCopyProducer = copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
    if (inputCopyProducer == nullptr) {
        log.trace("Input copy producer is not a VPUIP.Copy.");
        return false;
    }
    if (vpux::VPUIP::hasDistributedOperand(inputCopyProducer)) {
        log.trace("Input copy producer is not a distributed VPUIP.Copy.");
        return false;
    }
    auto output = inputCopyProducer.getOutputBuff();
    if (mlir::isa<mlir::BlockArgument>(output)) {
        log.trace("CopyOp buffer is a block argument.");
        return false;
    }
    auto distributedOpAlloc = output.getDefiningOp<mlir::memref::AllocOp>();
    if (distributedOpAlloc == nullptr) {
        log.trace("CopyOp output buffer is not allocated via memref.alloc");
        return false;
    }

    log.trace("Input is eligible for conversion");
    return true;
}

//
// FuseCopies
//

class FuseCopies final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseCopies(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
        setDebugName("FuseCopies");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// The original subgraph looks like that
// NCE    memrefAlloc
//    \      /
//  distributedOp (CMX2DDR)   copyOpAlloc
//        |                /
//        +---------> copyOp (DDR2DDR)   anotherBranch
//                             \            /
//                              origConcatOp
//
// This transformation fuses distributedCopy and copyOp into newTilingCopy:
// distributedCopy   copyOpAlloc
//  |                /
// newDistributedCopy (CMX2DDR)  anotherBranch
//             \            /
//              origConcatOp
//
mlir::LogicalResult FuseCopies::matchAndRewrite(VPUIP::ConcatViewOp origConcatOp,
                                                mlir::PatternRewriter& rewriter) const {
    const auto concatInputs = origConcatOp.getInputs();
    const auto isEligible = [&](const mlir::Value in) -> bool {
        return in != nullptr && isInputEligibleForConversion(in, _log);
    };
    SmallVector<mlir::Value> eligibleInputs;
    std::copy_if(concatInputs.begin(), concatInputs.end(), std::back_inserter(eligibleInputs), isEligible);
    if (eligibleInputs.empty()) {
        return matchFailed(rewriter, origConcatOp, "No DDR2DDR copy to fuse. Skipping ConcatView.");
    }
    rewriter.setInsertionPoint(origConcatOp);
    for (const auto& input : eligibleInputs) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        auto distributedOp = copyOp.getInput().getDefiningOp<VPUIP::CopyOp>();
        if (distributedOp == nullptr) {
            _log.debug("Received a non-Copy operation");
            continue;
        }
        auto distributedOpBuffs = distributedOp.getOutputBuff();
        auto distributedOpAlloc = distributedOpBuffs.getDefiningOp<mlir::memref::AllocOp>();
        auto copyOpAlloc = copyOp.getOutputBuff();
        auto newCopy = rewriter.create<VPUIP::CopyOp>(copyOp->getLoc(), distributedOp.getInput(), copyOpAlloc);
        copyOp.replaceAllUsesWith(newCopy.getOperation());
        rewriter.eraseOp(copyOp);

        if (distributedOp.getOutput().use_empty()) {
            rewriter.eraseOp(distributedOp);
        }
        if (distributedOpAlloc.getMemref().use_empty()) {
            rewriter.eraseOp(distributedOpAlloc);
        }
    }

    return mlir::success();
}

//
// FuseDDRCopiesIntoConcats
//

class FuseDDRCopiesIntoConcats final : public VPUIP::impl::FuseDDRCopiesIntoConcatsBase<FuseDDRCopiesIntoConcats> {
public:
    explicit FuseDDRCopiesIntoConcats(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseDDRCopiesIntoConcats::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseCopies>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseDDRCopiesIntoConcats
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseDDRCopiesIntoConcats(Logger log) {
    return std::make_unique<FuseDDRCopiesIntoConcats>(log);
}
