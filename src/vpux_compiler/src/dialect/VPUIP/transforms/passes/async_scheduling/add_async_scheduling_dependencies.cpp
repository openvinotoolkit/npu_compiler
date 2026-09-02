//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"

#include "vpux/utils/core/range.hpp"
namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADDASYNCSCHEDULINGDEPENDENCIES
#define GEN_PASS_DEF_ADDASYNCSCHEDULINGDEPENDENCIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool containsDynamicDequantize(mlir::async::ExecuteOp execOp) {
    const auto walkResult = execOp.getBody()->walk([](VPUIP::SwKernelOp swKernelOp) {
        if (swKernelOp.getKernelFunction().getLeafReference() == "builtin_DynamicDequantize") {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return walkResult.wasInterrupted();
}

bool containsConvolution(mlir::async::ExecuteOp execOp) {
    const auto walkResult = execOp.getBody()->walk([](VPUIP::NCEClusterTaskOp nceTaskOp) {
        if (nceTaskOp.getTaskType() == VPUIP::NCETaskType::CONV) {
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return walkResult.wasInterrupted();
}

bool canMoveAfter(mlir::Operation* op, mlir::Operation* insertionPoint) {
    return llvm::all_of(op->getResults(), [&](mlir::Value result) {
        return llvm::all_of(result.getUsers(), [&](mlir::Operation* user) {
            return user->getBlock() == insertionPoint->getBlock() && insertionPoint->isBeforeInBlock(user);
        });
    });
}

class AddAsyncSchedulingDependenciesPass final :
        public VPUIP::impl::AddAsyncSchedulingDependenciesBase<AddAsyncSchedulingDependenciesPass> {
public:
    explicit AddAsyncSchedulingDependenciesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddAsyncSchedulingDependenciesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.buildConsMap();

    mlir::async::ExecuteOp lastConvolutionConsumer;
    const auto execOps = to_small_vector(func.getOps<mlir::async::ExecuteOp>());
    for (auto dynamicDequantizeExecOp : execOps) {
        if (!containsDynamicDequantize(dynamicDequantizeExecOp)) {
            continue;
        }

        if (lastConvolutionConsumer != nullptr) {
            if (!lastConvolutionConsumer->isBeforeInBlock(dynamicDequantizeExecOp)) {
                if (!canMoveAfter(dynamicDequantizeExecOp, lastConvolutionConsumer)) {
                    _log.trace("Cannot safely add a dependency from '{0}' to '{1}' without breaking dominance",
                               lastConvolutionConsumer.getLoc(), dynamicDequantizeExecOp.getLoc());
                } else {
                    dynamicDequantizeExecOp->moveAfter(lastConvolutionConsumer);
                    depsInfo.addDependency(lastConvolutionConsumer, dynamicDequantizeExecOp);
                }
            } else {
                depsInfo.addDependency(lastConvolutionConsumer, dynamicDequantizeExecOp);
            }
        }

        lastConvolutionConsumer = nullptr;
        for (const auto consumerIdx : depsInfo.getConsumerOps(depsInfo.getIndex(dynamicDequantizeExecOp))) {
            auto consumerExecOp = depsInfo.getExecuteOpAtIndex(consumerIdx);
            if (containsConvolution(consumerExecOp)) {
                lastConvolutionConsumer = consumerExecOp;
            }
        }
    }

    depsInfo.updateTokenDependencies();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAddAsyncSchedulingDependenciesPass(Logger log) {
    return std::make_unique<AddAsyncSchedulingDependenciesPass>(log);
}
