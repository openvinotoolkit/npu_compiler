//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/fuse_conv_like_ops_with_slice_rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSECONVWITHSLICE
#define GEN_PASS_DEF_FUSECONVWITHSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseConvWithSlicePass
//

class FuseConvWithSlicePass final : public IE::impl::FuseConvWithSliceBase<FuseConvWithSlicePass> {
public:
    explicit FuseConvWithSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseConvWithSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::FuseConvWithSlice>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseConvWithSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseConvWithSlicePass(Logger log) {
    return std::make_unique<FuseConvWithSlicePass>(log);
}
