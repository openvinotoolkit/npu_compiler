//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTMEMORYSPACE
#define GEN_PASS_DEF_ADJUSTMEMORYSPACE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

namespace {

mlir::Value copyIntoMemSpace(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                             vpux::IndexedSymbolAttr destinationMemSpace) {
    return builder.createOrFold<VPU::CopyOp>(loc, val, destinationMemSpace);
}

mlir::LogicalResult insertCmxCopies(mlir::Operation* origOp, mlir::PatternRewriter& rewriter) {
    const auto memSpaceCMX = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN), 0);

    DenseMap<mlir::Value, mlir::Value> copiedInputs;
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto origInputValue = inputOperand.get();
        // No need to copy the data if due to some reason it's in CMX already
        const auto inputMemSpace = mlir::cast<vpux::NDTypeInterface>(origInputValue.getType()).getMemSpace();
        if (inputMemSpace == memSpaceCMX) {
            continue;
        }

        /// Make sure that we copy each piece of data into CMX only once
        /// @example
        /// Bad:
        ///   Input --> Copy -> NCEEltwise(Abs)
        ///        \--> Copy --/
        /// OK:
        ///   Input -> Copy -> NCEEltwise(Abs)
        ///                \--/
        if (copiedInputs.count(origInputValue) == 0) {
            const auto inputCMX = copyIntoMemSpace(
                    rewriter, appendLoc(origOp->getLoc(), "input-{0}-CMX", inputOperand.getOperandNumber()),
                    origInputValue, memSpaceCMX);
            copiedInputs[origInputValue] = inputCMX;
        }
        inputOperand.set(copiedInputs[origInputValue]);
    }

    auto origOutput = origOp->getResult(0);
    const auto origOutType = mlir::cast<vpux::NDTypeInterface>(origOutput.getType());
    const auto origOutMemSpace = origOutType.getMemSpace();
    if (origOutMemSpace != memSpaceCMX) {
        // Leave the original operation but change it in-place and add a Copy after it
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(origOp);
        const auto newOutType = origOutType.changeMemSpace(memSpaceCMX);
        rewriter.modifyOpInPlace(origOp, [&]() {
            origOutput.setType(newOutType);
            const auto copiedOutput =
                    copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), "output-DDR"), origOutput, origOutMemSpace);
            origOutput.replaceAllUsesExcept(copiedOutput, copiedOutput.getDefiningOp());
        });
    }

    return mlir::success();
}

//
// CopiesForNCEOp
//

class CopiesForNCEOp final : public mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface> {
public:
    CopiesForNCEOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::NCEOpInterface>(ctx), _log(log) {
        setDebugName("CopiesForNCEOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopiesForNCEOp::matchAndRewrite(VPU::NCEOpInterface origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return insertCmxCopies(origOp, rewriter);
}

//
// CopiesForM2iOp
//

class CopiesForM2iOp final : public mlir::OpRewritePattern<VPU::M2ITaskOp> {
public:
    CopiesForM2iOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::M2ITaskOp>(ctx), _log(log) {
        setDebugName("CopiesForM2iOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2ITaskOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult CopiesForM2iOp::matchAndRewrite(VPU::M2ITaskOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return insertCmxCopies(origOp, rewriter);
}

//
// AdjustMemorySpacePass
//

class AdjustMemorySpacePass final : public VPU::impl::AdjustMemorySpaceBase<AdjustMemorySpacePass> {
public:
    explicit AdjustMemorySpacePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustMemorySpacePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // NCE/M2I operations are only legal if all their outputs and inputs (incl. weights) reside in CMX
    const auto isLegalOp = [](mlir::Operation* op) {
        if (mlir::isa<VPU::NCEOpInterface, VPU::M2ITaskOp>(op)) {
            const auto verifyLocationInCmx = [](mlir::Value operand) {
                return mlir::cast<vpux::NDTypeInterface>(operand.getType()).getMemoryKind() == MemoryKind::CMX_NN;
            };
            return llvm::all_of(op->getOperands(), verifyLocationInCmx) &&
                   llvm::all_of(op->getResults(), verifyLocationInCmx);
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegalOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<CopiesForNCEOp>(&ctx, _log);
    patterns.add<CopiesForM2iOp>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustMemorySpacePass
//

std::unique_ptr<mlir::Pass> createAdjustMemorySpacePass(Logger log) {
    return std::make_unique<AdjustMemorySpacePass>(log);
}

}  // namespace VPU
}  // namespace vpux
