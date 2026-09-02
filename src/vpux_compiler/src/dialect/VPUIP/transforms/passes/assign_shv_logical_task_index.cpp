//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ASSIGNSHVLOGICALTASKINDEX
#define GEN_PASS_DEF_ASSIGNSHVLOGICALTASKINDEX
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class AssignShvLogicalTaskIndexPass final :
        public VPUIP::impl::AssignShvLogicalTaskIndexBase<AssignShvLogicalTaskIndexPass> {
public:
    explicit AssignShvLogicalTaskIndexPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AssignShvLogicalTaskIndexPass::safeRunOnModule() {
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
// createAssignShvLogicalTaskIndexPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAssignShvLogicalTaskIndexPass(Logger log) {
    return std::make_unique<AssignShvLogicalTaskIndexPass>(log);
}
