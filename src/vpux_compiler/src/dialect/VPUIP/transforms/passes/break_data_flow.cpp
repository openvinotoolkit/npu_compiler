//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_BREAKDATAFLOW
#define GEN_PASS_DEF_BREAKDATAFLOW
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// BreakDataFlowPass
//

class BreakDataFlowPass final : public VPUIP::impl::BreakDataFlowBase<BreakDataFlowPass> {
public:
    explicit BreakDataFlowPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void BreakDataFlowPass::safeRunOnFunc() {
    auto funcOp = getOperation();

    funcOp.walk([](VPUIP::LayerOpInterface op) {
        for (auto res : op->getOpResults()) {
            const auto ind = res.getResultNumber();
            const auto resBuf = op.getOutputs()[ind];
            res.replaceAllUsesWith(resBuf);
        }
    });

    // ConcatViewOp is a unique op since this is pure view-like op(not VPUIP::LayerOpInterface),
    // but it has output_buff whose type is equal to the output type.
    // So when we replace ConcatViewOp with output_buff, we simultaneously break data flow and convert viewop to a
    // declaration. Therefore it does not really matter when to get rid of this within of the Hardware adaptation
    // pipeline. On the other hand there is an advantage in erasing op here instead of ConvertViewOpsToDeclarations,
    // since later in ConvertFuncArgsToDeclarations we can replace safely function argument with result of
    // DeclareBufferOp when Return op is consumer of ConcatView:

    // where %arg2 is output argument of function
    // %n = VPUIP.ConcatView inputs(%i, %j : !type1, !type2) outputs(%arg2 : !out_type) -> !out_type
    // return %n;
    //
    // after ConvertFuncArgsToDeclarations is applied %n is no longer alias of function argument:
    // %out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> out_type
    // %n = VPUIP.ConcatView inputs(%i, %j : !type1, !type2) outputs(%out : !out_type) -> !out_type
    // // expected-error@+1 {{function output at index=0 should be an alias of the output buffer, but it's not}}
    // return %n;

    funcOp.walk([](VPUIP::ConcatViewOp op) {
        for (auto input : op.getInputs()) {
            if (auto waitOp = input.getDefiningOp<mlir::async::AwaitOp>()) {
                if (waitOp->hasOneUse()) {
                    waitOp->dropAllUses();
                    waitOp->erase();
                }
            }
        }
        auto res = op.getResult();
        res.replaceAllUsesWith(op.getOutputBuff());
        op->erase();
    });
}

}  // namespace

//
// createBreakDataFlowPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createBreakDataFlowPass(Logger log) {
    return std::make_unique<BreakDataFlowPass>(log);
}
