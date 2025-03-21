//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MOVEVIEWOPSTOVF
#define GEN_PASS_DEF_MOVEVIEWOPSTOVF
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// ViewOpsRewriter
//

class ViewOpsRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    ViewOpsRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ViewOpsRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                     mlir::PatternRewriter& rewriter) const {
    auto isOpWeightsFromVFOperandIndex = [](mlir::Operation* op, size_t operandIdx) -> bool {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr) {
            return false;
        }
        if (auto opWeights = llvm::cast_if_present<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            return opWeights.getArgNumber() == operandIdx;
        }
        return false;
    };
    auto tilingStrategy = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());

    for (auto vfOperand : vfOp->getOperands() | indexed) {
        auto parentOp = vfOperand.value().getDefiningOp<VPU::TilingViewLikeOpInterface>();

        if (parentOp == nullptr || !VPU::isPureViewOp(parentOp)) {
            continue;
        }

        // Exclude weights moving for non-SOC tiling
        // As only under SOC case, the producer op for weights (if exists) need to be tiled and merged into VF
        if (llvm::any_of(vfOp.getBody()->getArgument(vfOperand.index()).getUsers(), [&](auto user) {
                return isOpWeightsFromVFOperandIndex(user, vfOperand.index()) &&
                       (tilingStrategy[Dims4D::Act::C.ind()] == 1);
            })) {
            continue;
        }

        if (llvm::all_of(parentOp->getOperands(), [](auto value) {
                return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                       mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
            })) {
            continue;
        }

        auto newVFOp = fuseOpsInBlock(rewriter, vfOp, parentOp);
        rewriter.replaceOp(vfOp, newVFOp.getResult(0));
        return mlir::success();
    }
    return mlir::failure();
}

//
// MoveViewOpsToVFPass
//

class MoveViewOpsToVFPass final : public VPU::impl::MoveViewOpsToVFBase<MoveViewOpsToVFPass> {
public:
    explicit MoveViewOpsToVFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MoveViewOpsToVFPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ViewOpsRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveViewOpsToVerticalFusionPass
//

std::unique_ptr<mlir::Pass> VPU::createMoveViewOpsToVerticalFusionPass(Logger log) {
    return std::make_unique<MoveViewOpsToVFPass>(log);
}
