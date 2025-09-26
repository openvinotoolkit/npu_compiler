//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_FLATTENELTWISEKERNEL
#define GEN_PASS_DEF_FLATTENELTWISEKERNEL
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// FlattenEltwiseKernelPass
//

class FlattenEltwiseKernelPass final :
        public vpux::ShaveCodeGen::impl::FlattenEltwiseKernelBase<FlattenEltwiseKernelPass> {
public:
    explicit FlattenEltwiseKernelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void flattenKernel(mlir::func::FuncOp);
    mlir::linalg::LinalgOp getCandidateOp(mlir::func::FuncOp);
};

mlir::linalg::LinalgOp FlattenEltwiseKernelPass::getCandidateOp(mlir::func::FuncOp func) {
    auto tilingOps = func.getOps<mlir::TilingInterface>();

    // Flatten when we only have one linalg op for now to avoid
    // possible interactions with kernel tiling.
    // collapseOpIterationDims also only supports linalg.GenericOp and linalg.CopyOp.
    if (std::distance(tilingOps.begin(), tilingOps.end()) != 1 ||
        !mlir::isa<mlir::linalg::GenericOp, mlir::linalg::CopyOp>(*tilingOps.begin())) {
        return nullptr;
    }

    auto op = mlir::cast<mlir::linalg::LinalgOp>(**tilingOps.begin());
    if (op.getNumLoops() <= 1 || !mlir::linalg::isElementwise(op)) {
        return nullptr;
    }

    return op;
}

void FlattenEltwiseKernelPass::flattenKernel(mlir::func::FuncOp func) {
    if (getCandidateOp(func) == nullptr) {
        // We couldn't find a candidate op so we're not eligible for this transformation.
        return;
    }

    // Fold unit dimensions to enable more flattening.
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    mlir::linalg::ControlDropUnitDims options;
    mlir::linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
    if (failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        return signalPassFailure();
    }

    // The unit dimension folding could have replaced our previous op
    // so we need to find it again.
    auto op = getCandidateOp(func);
    if (op == nullptr) {
        return;
    }

    if (llvm::any_of(op.getIndexingMapsArray(), [](mlir::AffineMap map) {
            // The implementation of collapseOpIterationDims has a limitation
            // and only supports the case where all indexing maps are projected
            // permutations.
            return !map.isProjectedPermutation();
        })) {
        return;
    }

    // Note getCandidateOp checks that we have at least two loops.
    for (int i = op.getNumLoops() - 2; i >= 0; --i) {
        // Try to collapse dimensions for adjacent loops. The linalg op is emitted
        // with at least one operand iterated over in memory order, so collapsing
        // non-adjacent iterations is not expected to be profitable.
        mlir::IRRewriter rewriter(op);
        mlir::ReassociationIndices rai = {i, i + 1};
        if (!mlir::linalg::areDimSequencesPreserved(op.getIndexingMapsArray(), rai)) {
            continue;
        }
        auto flattened = mlir::linalg::collapseOpIterationDims(op, rai, rewriter);
        if (failed(flattened)) {
            continue;
        }

        rewriter.replaceOp(op, flattened->results);
        op = flattened->collapsedOp;
    }

    return;
}

void FlattenEltwiseKernelPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    auto funcOps = swModule.getOps<mlir::func::FuncOp>();
    for (auto func : funcOps) {
        if (func.isExternal()) {
            continue;
        }
        flattenKernel(func);
    }
    return;
}

}  // namespace

//
// createFlattenEltwiseKernelPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createFlattenEltwiseKernelPass(Logger log) {
    return std::make_unique<FlattenEltwiseKernelPass>(log);
}
