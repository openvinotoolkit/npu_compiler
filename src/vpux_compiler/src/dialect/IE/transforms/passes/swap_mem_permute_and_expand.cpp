//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/expand_with_layer_rewriter.hpp"
#include "vpux/compiler/dialect/IE/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPMEMPERMUTEANDEXPAND
#define GEN_PASS_DEF_SWAPMEMPERMUTEANDEXPAND
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
bool isConvertedFromReroder(IE::MemPermuteOp memPermuteOp) {
    if (memPermuteOp == nullptr) {
        return false;
    }
    auto inShape = getShape(memPermuteOp.getInput());
    auto outShape = getShape(memPermuteOp.getOutput());
    if (inShape != outShape) {
        return false;
    }

    auto inOrder = DimsOrder::fromValue(memPermuteOp.getInput());
    auto outOrder = DimsOrder::fromValue(memPermuteOp.getOutput());
    auto expectMemPerm = getPermutationFromOrders(inOrder, outOrder, memPermuteOp->getContext());
    return memPermuteOp.getMemPerm() == expectMemPerm;
}

//
//  The beneficial pattern:
//
//     input               input
//       |                   |
//     Mempermute          Expand
//       |                   |
//     Expand   ==>        MemPermute
//       |                   |
//     Slice(s)            Slice(s)
//       |                   |
//     MemPermute(s)       MemPermute(s)
//       |                   |
//     output              output
//
//  It's worth to swap parent Reorder-like MemPermute and Expand,  the swapped MemPermute will be handled by follow-up
//  optimizations.

bool isBeneficialToSwapExpandMemPermute(IE::ExpandOp origExpandOp, mlir::Operation* layerOp) {
    auto memPermuteOp = mlir::dyn_cast<IE::MemPermuteOp>(layerOp);
    if (memPermuteOp == nullptr) {
        return false;
    }
    if (mlir::isa<mlir::BlockArgument>(origExpandOp.getInput())) {
        return false;
    }
    if (!isConvertedFromReroder(memPermuteOp)) {
        return false;
    }

    const auto users = SmallVector<mlir::Operation*>(origExpandOp->getUsers());
    if (IE::anyIDUAutopadCandidate(users)) {
        return false;
    }

    const auto permuteInput = memPermuteOp.getInput();
    const auto inMemShape = getMemShape(permuteInput);
    const auto memPerm = memPermuteOp.getMemPerm();
    if (!isTrivialPermute(inMemShape, memPerm)) {
        return true;
    }

    const auto expandOutput = origExpandOp.getOutput();
    SmallVector<IE::SliceOp> slices;

    for (auto userOp : expandOutput.getUsers()) {
        auto maybeSlice = mlir::dyn_cast_or_null<IE::SliceOp>(*userOp);
        if (maybeSlice == nullptr) {
            return false;
        }
        slices.push_back(maybeSlice);
    }

    if (slices.empty()) {
        return false;
    }
    SmallVector<mlir::Value> memPermuteOps;
    for (auto& userOp : slices) {
        auto sliceOutput = userOp.getResult();
        if (!sliceOutput.hasOneUse()) {
            return false;
        }
        auto maybeMemPermuteOp = mlir::dyn_cast_or_null<IE::MemPermuteOp>(*sliceOutput.getUsers().begin());
        if (maybeMemPermuteOp == nullptr) {
            return false;
        }
        memPermuteOps.push_back(maybeMemPermuteOp);
    }

    return !memPermuteOps.empty();
}

//
// SwapMemPermuteAndExpandPass
//

class SwapMemPermuteAndExpandPass final : public IE::impl::SwapMemPermuteAndExpandBase<SwapMemPermuteAndExpandPass> {
public:
    explicit SwapMemPermuteAndExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapMemPermuteAndExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<IE::ExpandWithLayer>(&ctx, isBeneficialToSwapExpandMemPermute, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapMemPermuteAndExpandPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMemPermuteAndExpandPass(Logger log) {
    return std::make_unique<SwapMemPermuteAndExpandPass>(log);
}
