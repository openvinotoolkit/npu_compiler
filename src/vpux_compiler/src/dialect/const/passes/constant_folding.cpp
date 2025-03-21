//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <llvm/Support/ThreadPool.h>

using namespace vpux;

namespace vpux::Const {
#define GEN_PASS_DECL_CONSTANTFOLDING
#define GEN_PASS_DEF_CONSTANTFOLDING
#include "vpux/compiler/dialect/const/passes.hpp.inc"
}  // namespace vpux::Const

namespace {

//
// ConstantFoldingPass
//

int64_t getMaxIntermediateSize(Const::DeclareOp& origOp) {
    const auto& contentAttr = origOp.getContentAttr();
    auto inputType = contentAttr.getBaseContent().getType().cast<NDTypeInterface>();
    auto maximumSize = inputType.getTotalAllocSize().count();
    auto transformations = contentAttr.getTransformations();
    for (auto transformation : transformations) {
        inputType = transformation.inferOutputType(inputType);
        auto size = inputType.getTotalAllocSize().count();
        maximumSize = (size > maximumSize) ? size : maximumSize;
    }
    return maximumSize;
}

class ConstantFoldingPass final : public Const::impl::ConstantFoldingBase<ConstantFoldingPass> {
public:
    explicit ConstantFoldingPass(Logger log, const int64_t threshold = 300 * 1024 * 1024): _threshold(threshold) {
        Base::initLogger(log, Base::getArgumentName());
    }  // set threshold to 300MB

private:
    void safeRunOnFunc() final;

    const int64_t _threshold;
    const int64_t _tiny_const_threshold = 1024;  // set threshold to 1KB
};

void ConstantFoldingPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    auto ops = func.getOps<Const::DeclareOp>();

    _log.info("Constant folding size threshold is set to {0}", _threshold);

    // If multi-threading is not enabled, fold constants sequentially
    if (!ctx.isMultithreadingEnabled()) {
        for (auto origOp : ops) {
            vpux::Const::foldSingleConstant(origOp);
        }
        return;
    }
    // Fold all const that size is with in 1KB by using single thread
    SmallVector<Const::DeclareOp> bigConstOps;
    for (auto origOp : ops) {
        int64_t constSize = getMaxIntermediateSize(origOp);
        if (constSize > _tiny_const_threshold) {
            bigConstOps.push_back(origOp);
            continue;
        }
        foldSingleConstant(origOp);
    }

    auto& threadPool = ctx.getThreadPool();
    auto op = bigConstOps.begin();
    auto opsEnd = bigConstOps.end();
    while (op != opsEnd) {
        auto origOp = *op;
        int64_t constSize = getMaxIntermediateSize(origOp);
        if (constSize > _threshold) {
            // If the const exceeds size limit, fold it directly
            foldSingleConstant(origOp);
            ++op;
            continue;
        }

        // If the const is within the limit, decide the number of consts that
        // can be folded in parallel
        int64_t constSizeInParallel = 0;
        unsigned int constNumInParallel = 0;
        auto opReserve = op;
        while (constSizeInParallel <= _threshold && op != opsEnd) {
            auto origOpInParallel = *op;
            int64_t cstSize = getMaxIntermediateSize(origOpInParallel);
            constSizeInParallel += cstSize;
            constNumInParallel++;
            ++op;
        }

        mlir::ParallelDiagnosticHandler handler(&ctx);
        std::atomic<unsigned int> curIndex(0);
        auto processConstants = [&] {
            while (true) {
                const unsigned int index = curIndex++;
                if (index >= constNumInParallel) {
                    break;
                }
                handler.setOrderIDForThread(index);
                Const::DeclareOp curOp = *(std::next(opReserve, index));
                foldSingleConstant(curOp);
                handler.eraseOrderIDForThread();
            }
        };

        llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
        unsigned int numActions = std::min(constNumInParallel, threadPool.getThreadCount());
        for (unsigned int i = 0; i < numActions; ++i) {
            tasksGroup.async(processConstants);
        }
        tasksGroup.wait();
    }
}

}  // namespace

//
// createConstantFoldingPass
//

std::unique_ptr<mlir::Pass> vpux::Const::createConstantFoldingPass(Logger log, const int64_t threshold) {
    return std::make_unique<ConstantFoldingPass>(log, threshold);
}
