//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_UNROLLUNUSEDVERTICALFUSIONREGION
#define GEN_PASS_DEF_UNROLLUNUSEDVERTICALFUSIONREGION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

bool canUnrollUnusedVF(VPU::VerticalFusionOp op) {
    auto vfOps = op.getOps().front().getOps<VPU::VerticalFusionOpInterface>();
    const bool isSingleOp = (std::distance(vfOps.begin(), vfOps.end()) == 1);

    const auto tilingInfo = parseIntArrayAttr<int64_t>(op.getTilingStrategy());
    const auto hasTiling = llvm::any_of(tilingInfo, [](int64_t i) {
        return i != 1;
    });

    return isSingleOp || !hasTiling;
}

//
// VerticalFusionUnrollRewriter
//

class VerticalFusionUnrollRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    VerticalFusionUnrollRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult VerticalFusionUnrollRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    if (!canUnrollUnusedVF(vfOp)) {
        _log.debug("Failed to unroll VerticalFusionOp '{0}' at '{1}'.", vfOp->getName(), vfOp->getLoc());
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    for (auto arg : vfOp.getBody()->getArguments()) {
        mapper.map(arg, vfOp.getOperand(arg.getArgNumber()));
    }

    const auto tilingInfo = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy());
    const auto hasTiling = llvm::any_of(tilingInfo, [](int64_t i) {
        return i != 1;
    });

    auto vfOps = vfOp.getOps().front().getOps<VPU::VerticalFusionOpInterface>();
    auto numOperations = std::distance(vfOps.begin(), vfOps.end());

    mlir::Operation* clonedOp = nullptr;
    for (auto& op : vfOp.getBody()->without_terminator()) {
        clonedOp = rewriter.clone(op, mapper);
        if (hasTiling && mlir::isa<VPU::VerticalFusionOpInterface>(clonedOp)) {
            clonedOp->setAttr(vfOp.getTilingStrategyAttrName(), vfOp.getTilingStrategyAttr());
        }

        if (numOperations != 1 && clonedOp->hasTrait<VPU::EltwiseOp>()) {
            SmallVector<mlir::Operation*> parents;
            for (auto operand : clonedOp->getOperands()) {
                if (auto parentOp = operand.getDefiningOp()) {
                    parents.push_back(parentOp);
                }
            }
            if (!parents.empty()) {
                llvm::sort(parents, [](auto* lhs, auto* rhs) {
                    return lhs->isBeforeInBlock(rhs);
                });
                clonedOp->moveAfter(parents.back());
            }
        }

        for (auto resultIdx = 0U; resultIdx < op.getNumResults(); ++resultIdx) {
            mapper.map(op.getResult(resultIdx), clonedOp->getResult(resultIdx));
        }
    }

    rewriter.replaceOp(vfOp, clonedOp->getResults());

    return mlir::success();
}

//
// UnrollUnusedVerticalFusionRegionPass
//

class UnrollUnusedVerticalFusionRegionPass final :
        public VPU::impl::UnrollUnusedVerticalFusionRegionBase<UnrollUnusedVerticalFusionRegionPass> {
public:
    explicit UnrollUnusedVerticalFusionRegionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void UnrollUnusedVerticalFusionRegionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionUnrollRewriter>(&ctx, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUnrollUnusedVerticalFusionRegionPass
//

std::unique_ptr<mlir::Pass> VPU::createUnrollUnusedVerticalFusionRegionPass(Logger log) {
    return std::make_unique<UnrollUnusedVerticalFusionRegionPass>(log);
}
