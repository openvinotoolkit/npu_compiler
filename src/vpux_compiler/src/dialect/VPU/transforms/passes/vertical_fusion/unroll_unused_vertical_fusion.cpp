//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_UNROLLUNUSEDVERTICALFUSIONREGION
#define GEN_PASS_DEF_UNROLLUNUSEDVERTICALFUSIONREGION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

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

        mapper.map(op.getResult(0), clonedOp->getResult(0));
    }

    rewriter.replaceOp(vfOp, clonedOp->getResult(0));

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

    const auto isLegalVF = [&](VPU::VerticalFusionOp op) {
        auto vfOps = op.getOps().front().getOps<VPU::VerticalFusionOpInterface>();
        if (std::distance(vfOps.begin(), vfOps.end()) == 1) {
            return false;
        }

        const auto tilingInfo = parseIntArrayAttr<int64_t>(op.getTilingStrategy());
        const auto hasTiling = llvm::any_of(tilingInfo, [](int64_t i) {
            return i != 1;
        });
        if (!hasTiling) {
            return false;
        }
        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::VerticalFusionOp>(isLegalVF);
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPU::VPUDialect>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VerticalFusionUnrollRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollUnusedVerticalFusionRegionPass
//

std::unique_ptr<mlir::Pass> VPU::createUnrollUnusedVerticalFusionRegionPass(Logger log) {
    return std::make_unique<UnrollUnusedVerticalFusionRegionPass>(log);
}
