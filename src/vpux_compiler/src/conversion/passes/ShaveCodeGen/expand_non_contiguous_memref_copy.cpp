//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
#define GEN_PASS_DECL_EXPANDNONCONTIGUOUSMEMREFCOPY
#define GEN_PASS_DEF_EXPANDNONCONTIGUOUSMEMREFCOPY
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// lowersToMemCpyIntrinsic
//

bool lowersToMemCpyIntrinsic(mlir::BaseMemRefType type) {
    // This mirrors the predicate from MemRefCopyOpLowering::matchAndRewrite.
    const auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    if (memrefType == nullptr) {
        return false;
    }
    return memrefType.getLayout().isIdentity() || (memrefType.hasStaticShape() && memrefType.getNumElements() > 0 &&
                                                   mlir::memref::isStaticShapeAndContiguousRowMajor(memrefType));
}

//
// ExpandCopyRewrite
//

class ExpandCopyRewrite final : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
public:
    ExpandCopyRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::memref::CopyOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::memref::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ExpandCopyRewrite::matchAndRewrite(mlir::memref::CopyOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto sourceType = origOp.getSource().getType();
    const auto targetType = origOp.getTarget().getType();

    // Both operands must be ranked in order to construct the linalg.generic op.
    if (!mlir::isa<mlir::MemRefType>(sourceType) || !mlir::isa<mlir::MemRefType>(targetType)) {
        return matchFailed(_log, rewriter, origOp, "Unranked memref operands are not supported");
    }

    _log.trace("Expanding non-contiguous '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto copyOp = mlir::linalg::makeMemRefCopyOp(rewriter, origOp.getLoc(), origOp.getSource(), origOp.getTarget());
    rewriter.replaceOp(origOp, copyOp);

    return mlir::success();
}

//
// ExpandNonContiguousMemRefCopyPass
//

class ExpandNonContiguousMemRefCopyPass final :
        public impl::ExpandNonContiguousMemRefCopyBase<ExpandNonContiguousMemRefCopyPass> {
public:
    explicit ExpandNonContiguousMemRefCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ExpandNonContiguousMemRefCopyPass::safeRunOnModule() {
    auto& ctx = getContext();
    // Note that this can create an empty SW module, however we temporarily accept this.
    // This will be reworked in E#191882.
    auto swModule = VPUIP::getVPUSWModule(getOperation(), _log);

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });
    target.addDynamicallyLegalOp<mlir::memref::CopyOp>([](mlir::memref::CopyOp op) {
        // A memref.copy is legal only if the llvm conversion will lower it into an llvm.intr.memcpy.
        return lowersToMemCpyIntrinsic(op.getSource().getType()) && lowersToMemCpyIntrinsic(op.getTarget().getType());
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandCopyRewrite>(&ctx, _log);
    const mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (mlir::failed(mlir::applyPartialConversion(swModule, target, frozenPatterns))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandNonContiguousMemRefCopyPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createExpandNonContiguousMemRefCopyPass(Logger log) {
    return std::make_unique<ExpandNonContiguousMemRefCopyPass>(log);
}
