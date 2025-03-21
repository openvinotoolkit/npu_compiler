//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/format.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DEDEBATCHER
#define GEN_PASS_DEF_DEDEBATCHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> getDeDebatchNum(mlir::func::CallOp callOp, int64_t& castOpCnt,
                                         bool injectDebatchingReorderingAttr) {
    const auto privateFuncOperands = callOp.getOperands();
    int64_t dedebatchNum = 0;
    for (auto operand : privateFuncOperands) {
        if (auto convertCast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            const auto batchedInput = getShape(convertCast.getOperand(0))[Dims4D::Act::N];
            const auto debatchedInput = getShape(convertCast.getResult(0))[Dims4D::Act::N];
            const auto ratio = batchedInput / debatchedInput;
            VPUX_THROW_UNLESS(batchedInput % debatchedInput == 0, "Batch dim is not divisible by de-batched dim");
            if (dedebatchNum == 0) {
                dedebatchNum = ratio;
            }
            VPUX_THROW_UNLESS(dedebatchNum == ratio, "De-de-batch number is not matched for various inputs");
            DebatchedCallOpAttributeView::inject(callOp, castOpCnt, dedebatchNum);
            if (injectDebatchingReorderingAttr) {
                DebatchedCallOpAttributeView::setReorderingAttr(callOp);
            }
            castOpCnt++;
        }
    }
    return dedebatchNum;
}

mlir::SmallVector<mlir::Operation*> sliceCallsOp(mlir::OpBuilder& builder, mlir::func::CallOp callOp,
                                                 const int64_t dedebatchNum, bool injectDebatchingReorderingAttr) {
    const auto callLoc = callOp.getLoc();
    const auto privateFuncOperands = callOp.getOperands();
    auto newCallOps = SmallVector<mlir::Operation*>();
    for (int i = 0; i < dedebatchNum; i++) {
        // Create sliced private function operands
        mlir::SmallVector<mlir::Value> newOperands;
        size_t sliceIdx = 0;
        for (auto operand : privateFuncOperands) {
            if (auto convertCast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
                auto batchedInput = convertCast.getOperand(0);
                auto debatchedInput = convertCast.getResult(0);
                builder.setInsertionPoint(callOp);
                // prepare  slice offset: we must create slice offset with shape rank
                // equal to the batched operand rank
                Shape sliceOffset{
                        SmallVector<int64_t>(batchedInput.getType().cast<vpux::NDTypeInterface>().getRank(), 0)};
                sliceOffset[Dims4D::Act::N] = getShape(debatchedInput)[Dims4D::Act::N] * i;
                const auto sliceLoc = appendLoc(callLoc, "slice_{0}_op_{1}", sliceIdx + 1, i + 1);
                auto slicedOperand =
                        builder.create<IE::SliceOp>(sliceLoc, batchedInput, sliceOffset, getShape(debatchedInput));
                newOperands.push_back(slicedOperand.getResult());
                sliceIdx++;
            } else {
                newOperands.push_back(operand);
            }
        }

        // Create multi-batched private function calls
        auto newCall =
                builder.create<mlir::func::CallOp>(callLoc, callOp.getCallee(), callOp->getResultTypes(), newOperands);
        DebatchedCallOpAttributeView::inject(newCall, i, dedebatchNum);
        if (injectDebatchingReorderingAttr) {
            DebatchedCallOpAttributeView::setReorderingAttr(newCall);
        }
        newCallOps.push_back(newCall);
    }
    return newCallOps;
}

void concatenateCallOps(mlir::OpBuilder& builder, mlir::func::CallOp callOp, SmallVector<mlir::Operation*> newCallOps) {
    const auto callLoc = callOp.getLoc();
    const auto privateFuncResNum = callOp.getResults().size();
    for (size_t i = 0; i < privateFuncResNum; i++) {
        mlir::SmallVector<mlir::Value> newCallResults;
        for (auto newCall : newCallOps) {
            auto res = newCall->getResult(i);
            newCallResults.push_back(res);
        }

        auto newConcatResult = builder.create<IE::ConcatOp>(callLoc, newCallResults, 0);
        auto origCallResUsers = callOp.getResult(i).getUsers();
        for (auto usr : origCallResUsers) {
            usr->getResult(0).replaceAllUsesWith(newConcatResult->getResult(0));
        }
    }
    return;
}

//
// DeDebatcherPass
//

class DeDebatcherPass final : public IE::impl::DeDebatcherBase<DeDebatcherPass> {
public:
    explicit DeDebatcherPass(const DebatcherOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
        log.debug("Create {0}", getName());
    }

    mlir::LogicalResult initializeOptions(StringRef options) override;

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult parseFromOptions();
};

mlir::LogicalResult DeDebatcherPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    _log.trace("{0}: {1}", debatcherMethod.getArgStr(), debatcherMethod.getValue());
    _log.trace("initializing of {0} succeeded", getName());
    return mlir::success();
}

//
// safeRunOnModule
//

void DeDebatcherPass::safeRunOnFunc() {
    _log.debug("{0}::safeRunOnModule", getName());

    bool injectDebatchingReorderingMethodAttr = false;
    auto parsedDebatcherMethod = this->debatcherMethod.getValue();
    if (parsedDebatcherMethod == "reordering") {
        _log.debug("{0} applying method: {1}", getName(), parsedDebatcherMethod);
        injectDebatchingReorderingMethodAttr = true;
    }
    auto main = getOperation();
    mlir::OpBuilder builder(main);
    if (main.isPrivate()) {
        return;
    }
    // Check all private function calls in main function
    auto callOps = main.getFunctionBody().getOps<mlir::func::CallOp>();
    for (auto callOp : callOps) {
        //  Acquire and validate de-debatch number
        int64_t castOpCnt = 0;
        const auto dedebatchNum = getDeDebatchNum(callOp, castOpCnt, injectDebatchingReorderingMethodAttr);

        // Not batched case
        if (castOpCnt == 0) {
            continue;
        }

        // Get multi-batch sliced private function calls
        auto newCallOps = sliceCallsOp(builder, callOp, dedebatchNum.value(), injectDebatchingReorderingMethodAttr);

        // Create concat for multi-batched private function results
        concatenateCallOps(builder, callOp, newCallOps);
    }
}
}  // namespace

//
// createDeDebatcherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDeDebatcherPass(const DebatcherOptions& options, Logger log) {
    return std::make_unique<DeDebatcherPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createDeDebatcherPass(Logger log) {
    return createDeDebatcherPass(DebatcherOptions{}, log);
}
