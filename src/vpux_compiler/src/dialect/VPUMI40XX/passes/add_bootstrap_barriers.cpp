//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/passes.hpp"

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_ADDBOOTSTRAPBARRIERS
#define GEN_PASS_DEF_ADDBOOTSTRAPBARRIERS
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

class AddBootstrapBarriersPass : public VPUMI40XX::impl::AddBootstrapBarriersBase<AddBootstrapBarriersPass> {
public:
    explicit AddBootstrapBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddBootstrapBarriersPass::safeRunOnFunc() {
    auto ctx = &(getContext());
    auto netFunc = getOperation();

    int64_t bootstrapID = 0;
    mlir::Value first;
    int64_t numberOfAvailablePhysicalBarriers = VPUIP::getNumAvailableBarriers(netFunc);
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    std::vector<bool> initialized(numberOfAvailablePhysicalBarriers, false);
    for (auto op : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(bootstrapID));
        auto pid = op.getId();
        if (initialized[pid]) {
            continue;
        }

        auto bootsTrapTask = builder.create<VPUMI40XX::BootstrapOp>(op.getLoc(), trivialIndexType, op->getResult(0));
        if (bootstrapID == 0) {
            first = bootsTrapTask;
        }
        if (bootstrapID == numberOfAvailablePhysicalBarriers) {
            break;
        }
        ++bootstrapID;
        initialized[pid] = true;
    }
    if (first) {
        mpi.getBootstrapBarriersMutable().assign(first);
        mpi.setBootstrapBarriersCountAttr(
                builder.getI64IntegerAttr(std::min(numberOfAvailablePhysicalBarriers, bootstrapID)));
    }
}

}  // namespace

//
// createAddBootstrapBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddBootstrapBarriersPass(Logger log) {
    return std::make_unique<AddBootstrapBarriersPass>(log);
}
