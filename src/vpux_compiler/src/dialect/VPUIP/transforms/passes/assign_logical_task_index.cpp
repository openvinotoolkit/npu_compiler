//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ASSIGNLOGICALTASKINDEX
#define GEN_PASS_DEF_ASSIGNLOGICALTASKINDEX
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class AssignLogicalTaskIndexPass final : public VPUIP::impl::AssignLogicalTaskIndexBase<AssignLogicalTaskIndexPass> {
public:
    explicit AssignLogicalTaskIndexPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AssignLogicalTaskIndexPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto* ctx = moduleOp->getContext();

    int64_t logicalTaskIndex = 0;
    moduleOp.walk([&](VPUIP::SwKernelOp swOp) {
        if (VPUIP::isIoDmaSwKernel(swOp)) {
            swOp->setAttr(VPUIP::LOGICAL_TASK_INDEX_ATTR_NAME,
                          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), logicalTaskIndex++));
        }
    });
}

}  // namespace

//
// createAssignLogicalTaskIndexPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAssignLogicalTaskIndexPass(Logger log) {
    return std::make_unique<AssignLogicalTaskIndexPass>(log);
}
