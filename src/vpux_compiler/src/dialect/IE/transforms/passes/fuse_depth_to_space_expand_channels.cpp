//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cstdint>
#include <memory>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSED2SEXPANDCHANNELS
#define GEN_PASS_DEF_FUSED2SEXPANDCHANNELS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseDepthToSpaceWithExpand
//

//                                         Input(1x12x540x960)
//                                                  |
//                                       DepthToSpace(1x3x1080x1920)
//                                                  |
//                                        Expand(1x16x1080x1920)
//
//                  To
//
//                                          Input(1x12x540x960)
//                                                  |
//     DepthToSpace(1x3x1080x1920) {padded_channels = #IE.ChannelPadding<input = 0: i64, output = 13: i64>}
//

mlir::FailureOr<IE::ExpandOp> getExpandOpToFuse(IE::DepthToSpaceOp origOp) {
    auto users = origOp->getUsers();
    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }
    auto expandOp = mlir::dyn_cast<IE::ExpandOp>(*users.begin());
    if (expandOp == nullptr) {
        return mlir::failure();
    }
    return expandOp;
}

mlir::LogicalResult fuseDepthToSpaceWithExpand(mlir::MLIRContext* ctx, IE::DepthToSpaceOp origOp,
                                               mlir::OpBuilder& builder, Logger& log) {
    log.trace("[{0}] Got depth-to-space layer at '{1}'", origOp->getName(), origOp->getLoc());
    auto nestedLogger = log.nest();

    if (origOp.getPaddedChannelsAttr() != nullptr) {
        return mlir::failure();
    }
    auto expandOpOrFailure = getExpandOpToFuse(origOp);
    if (mlir::failed(expandOpOrFailure)) {
        return mlir::failure();
    }
    auto expandOp = expandOpOrFailure.value();

    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    if (llvm::any_of(padsBegin, [](int64_t value) {
            return value != 0;
        })) {
        return mlir::failure();
    }
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    // Only allow padding on the channel dimension, all other padsEnd values must be zero
    for (size_t i = 0; i < padsEnd.size(); ++i) {
        if (i != checked_cast<size_t>(Dims4D::Act::C.ind()) && padsEnd[i] != 0) {
            return mlir::failure();
        }
    }
    // Track: E#182999: Check consumers chain to detect ops, which accuracy might be affected due to padding

    auto paddedChannelsAttr = IE::ChannelPaddingAttr::get(ctx, builder.getI64IntegerAttr(0),
                                                          builder.getI64IntegerAttr(padsEnd[Dims4D::Act::C.ind()]));

    // Create new DepthToSpaceOp with padded output channels
    builder.setInsertionPointAfter(origOp);
    auto newDepthToSpaceOp = builder.create<IE::DepthToSpaceOp>(
            origOp->getLoc(), origOp.getInput(), origOp.getBlockSizeAttr(), origOp.getModeAttr(), paddedChannelsAttr);

    // replace DepthToSpaceOp->Expand chain with new DepthToSpaceOp
    expandOp->replaceAllUsesWith(newDepthToSpaceOp->getResults());

    nestedLogger.trace("fuse DepthToSpace with Expand success");
    return mlir::success();
}

//
// FuseDepthToSpaceWithExpandPass
//

class FuseDepthToSpaceWithExpandPass final :
        public IE::impl::FuseD2SExpandChannelsBase<FuseDepthToSpaceWithExpandPass> {
public:
    explicit FuseDepthToSpaceWithExpandPass(Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseDepthToSpaceWithExpandPass::safeRunOnFunc() {
    auto func = getOperation();

    auto& ctx = getContext();
    mlir::OpBuilder builder(&ctx);
    SmallVector<IE::DepthToSpaceOp> depthToSpaceOpsToRemove;

    func.walk([&](IE::DepthToSpaceOp d2s) {
        if (fuseDepthToSpaceWithExpand(&ctx, d2s, builder, _log).succeeded()) {
            _log.trace("fuse DepthToSpace with Expand success");
            depthToSpaceOpsToRemove.emplace_back(d2s);
        } else {
            _log.trace("fuse DepthToSpace with Expand failed. Pattern not matched.");
        }
    });
    // cleanup dangling DepthToSpace and Expand ops
    for (auto& d2s : make_early_inc_range(depthToSpaceOpsToRemove)) {
        getExpandOpToFuse(d2s).value().erase();
        d2s->erase();
    }
}

}  // namespace

//
// createFuseD2SExpandChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseD2SExpandChannelsPass(Logger log) {
    return std::make_unique<FuseDepthToSpaceWithExpandPass>(log);
}
