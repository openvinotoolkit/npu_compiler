//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SHAVEKERNELSIMPLIFY
#define GEN_PASS_DEF_SHAVEKERNELSIMPLIFY
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// ShaveKernelSimplifyPass
//

class ShaveKernelSimplifyPass final :
        public vpux::ShaveCodeGen::impl::ShaveKernelSimplifyBase<ShaveKernelSimplifyPass> {
public:
    explicit ShaveKernelSimplifyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ShaveKernelSimplifyPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    mlir::linalg::populateBubbleUpExtractSliceOpPatterns(patterns);
    mlir::linalg::populateSwapExtractSliceWithFillPatterns(patterns);
    mlir::tensor::populateFoldTensorEmptyPatterns(patterns);

    mlir::tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::tensor::EmptyOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::linalg::FillOp::getCanonicalizationPatterns(patterns, &ctx);
    mlir::linalg::GenericOp::getCanonicalizationPatterns(patterns, &ctx);

    const mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    getOperation()->walk<mlir::WalkOrder::PreOrder>([&](IE::CodeGenCapsuleOp capsuleOp) -> mlir::WalkResult {
        if (failed(mlir::applyPatternsGreedily(capsuleOp, frozenPatterns, getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
            return mlir::WalkResult::interrupt();
        }
        // Avoid walking the capsule contents.
        return mlir::WalkResult::skip();
    });
}

}  // namespace

//
// createShaveKernelSimplifyPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createShaveKernelSimplifyPass(Logger log) {
    return std::make_unique<ShaveKernelSimplifyPass>(log);
}
