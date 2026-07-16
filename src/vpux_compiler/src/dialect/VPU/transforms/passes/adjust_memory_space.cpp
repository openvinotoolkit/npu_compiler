//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
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
    // scf.forall indicates a multiclustered operation, therefore memory space should not indicate a particular cluster
    // as each iteration of the loop will move the data to a different cluster
    const auto isMulticlustered = origOp->getParentOfType<mlir::scf::ForallOp>() != nullptr;
    const auto memSpaceCMX =
            isMulticlustered ? IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN))
                             : IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN), 0);

    DenseMap<mlir::Value, mlir::Value> copiedInputs;
    for (auto& inputOperand : origOp->getOpOperands()) {
        auto origInputValue = inputOperand.get();
        const auto ndInputType = mlir::dyn_cast<vpux::NDTypeInterface>(origInputValue.getType());
        // Non-tensor operands (e.g. index types) have no memory space and do not need a CMX copy
        if (!ndInputType) {
            continue;
        }
        // No need to copy the data if due to some reason it's in CMX already
        const auto inputMemSpace = ndInputType.getMemSpace();
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

    // Check whether any NDType result is outside CMX; ops with multiple results (e.g. ConvolutionOp with a
    // reduceMax output) require all tensor results to be moved to CMX
    const auto anyOutputNeedsMove = llvm::any_of(origOp->getResults(), [&](mlir::Value result) {
        const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(result.getType());
        return ndType && ndType.getMemSpace() != memSpaceCMX;
    });
    if (anyOutputNeedsMove) {
        // Leave the original operation but change it in-place and add a Copy after it for each result
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(origOp);
        rewriter.modifyOpInPlace(origOp, [&]() {
            for (auto origOutput : origOp->getResults()) {
                const auto ndOutType = mlir::dyn_cast<vpux::NDTypeInterface>(origOutput.getType());
                // Non-tensor results (e.g. index types) have no memory space
                if (!ndOutType || ndOutType.getMemSpace() == memSpaceCMX) {
                    continue;
                }
                const auto origOutMemSpace = ndOutType.getMemSpace();
                origOutput.setType(ndOutType.changeMemSpace(memSpaceCMX));
                const auto copiedOutput = copyIntoMemSpace(rewriter, appendLoc(origOp->getLoc(), "output-DDR"),
                                                           origOutput, origOutMemSpace);
                origOutput.replaceAllUsesExcept(copiedOutput, copiedOutput.getDefiningOp());
            }
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

    // NCE operations are only legal if all their outputs and inputs (incl. weights) reside in CMX
    const auto isLegalOp = [](mlir::Operation* op) {
        if (mlir::isa<VPU::NCEOpInterface>(op)) {
            const auto verifyLocationInCmx = [](mlir::Value operand) {
                if (operand == nullptr) {
                    return true;
                }
                const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType());
                // Non-tensor types (e.g. index results from reduceMinMax) do not reside in any memory space
                if (!ndType) {
                    return true;
                }
                return ndType.getMemoryKind() == MemoryKind::CMX_NN;
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
