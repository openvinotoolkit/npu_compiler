//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SHAVESTACKALLOCATION
#define GEN_PASS_DEF_SHAVESTACKALLOCATION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

static constexpr unsigned MAX_ALLOC_BYTES = 64;

class ShaveStackAllocationPass final : public ShaveCodeGen::impl::ShaveStackAllocationBase<ShaveStackAllocationPass> {
public:
    explicit ShaveStackAllocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final {
        auto moduleOp = getOperation();
        auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

        mlir::PassManager pm(&getContext());
        pm.addPass(mlir::bufferization::createPromoteBuffersToStackPass(/*maxAllocSizeInBytes=*/MAX_ALLOC_BYTES,
                                                                        /*maxRankOfAllocatedMemRef=*/1));
        auto funcOps = swModule.getOps<mlir::func::FuncOp>();

        for (auto func : funcOps) {
            if (mlir::failed(pm.run(func))) {
                _log.trace("Failed to promote buffers to stack");
                return signalPassFailure();
            }

            // Abort on any surviving memref.alloc ops.
            bool hasAllocOps = false;
            for (auto allocOp : func.getOps<mlir::memref::AllocOp>()) {
                mlir::emitError(allocOp.getLoc()) << "Unexpected allocation in shave kernel.";
                hasAllocOps = true;
            }
            if (hasAllocOps) {
                return signalPassFailure();
            }
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createShaveStackAllocationPass(Logger log) {
    return std::make_unique<ShaveStackAllocationPass>(log);
}
