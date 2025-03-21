//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_vpu_nce_ops_interface.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux {
#define GEN_PASS_DECL_ONESHOTBUFFERIZEVPU2VPUIP
#define GEN_PASS_DEF_ONESHOTBUFFERIZEVPU2VPUIP
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

void removeBufferizationAttributes(mlir::Operation* op) {
    // removed "__inplace_operands_attr__" on each in-place analyzed operations
    op->walk([&](mlir::Operation* op) {
        if (op->hasAttr("__inplace_operands_attr__")) {
            op->removeAttr("__inplace_operands_attr__");
        }
    });
}

//
// OneshotBufferizeVPU2VPUIPPass
//

class OneShotBufferizeVPU2VPUIPPass final : public impl::OneShotBufferizeVPU2VPUIPBase<OneShotBufferizeVPU2VPUIPPass> {
private:
    void safeRunOnModule() final;
};

void OneShotBufferizeVPU2VPUIPPass::safeRunOnModule() {
    mlir::bufferization::OneShotBufferizationOptions options = vpux::getOneShotBufferizationOptions();
    mlir::ModuleOp moduleOp = getOperation();
    auto& ctx = getContext();

    // E#112397: special case: "bufferize" multi-tile vpu nce permute operation
    // before anything else. can be (safely) deleted once NCEClusterTiling is
    // removed and necessary functionality is supported directly in NCEPermuteOp
    auto log = Logger::global().nest("one-shot-bufferize-MultiTileNcePermute", 0);
    for (auto funcOp : moduleOp.getRegion().getOps<mlir::func::FuncOp>()) {
        if (mlir::failed(vpux::lowerMultiTileVpuNcePermuteOneShot(&ctx, funcOp, log))) {
            signalPassFailure();
            return;
        }
    }

    if (mlir::failed(mlir::bufferization::bufferizeOp(moduleOp, options, /*statistics=*/nullptr))) {
        signalPassFailure();
        return;
    }

    removeBufferizationAttributes(moduleOp);
}

}  // namespace

//
// createOneShotBufferizeVPU2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createOneShotBufferizeVPU2VPUIPPass() {
    return std::make_unique<OneShotBufferizeVPU2VPUIPPass>();
}
