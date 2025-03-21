//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/format.hpp"

namespace vpux::VPUMI37XX {
#define GEN_PASS_DECL_ASSIGNFULLKERNELPATH
#define GEN_PASS_DEF_ASSIGNFULLKERNELPATH
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp.inc"
}  // namespace vpux::VPUMI37XX

using namespace vpux;

namespace {

//
// AssignFullKernelPathPass
//

class AssignFullKernelPathPass final : public VPUMI37XX::impl::AssignFullKernelPathBase<AssignFullKernelPathPass> {
public:
    explicit AssignFullKernelPathPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignFullKernelPathPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUMI37XX::ActKernelInvocationOp origOp) {
        auto kernelRangeOp = origOp.getRangeIndex().getDefiningOp<VPUMI37XX::ActKernelRangeOp>();
        auto kernelRangeOpTaskType = kernelRangeOp.getKernelTaskType();
        auto isCacheOp = VPUIP::isCacheOpTaskType(kernelRangeOpTaskType);
        if (isCacheOp) {
            return;
        }

        auto kernelParamsOp = origOp.getParamsIndex().getDefiningOp<VPUMI37XX::KernelParamsOp>();
        if (kernelParamsOp == nullptr) {
            return;
        }

        auto kernelParamsOpInputs = kernelParamsOp.getInputs();
        auto kernelPath = kernelParamsOp.getKernelType();
        auto cpu = ELFNPU37XX::getSwKernelArchString(VPU::getArch(origOp));

        std::string newKernelType;
        bool hasDDRInputBuffers = !VPUIP::getDDRBuffers(kernelParamsOpInputs).empty();
        if (hasDDRInputBuffers) {
            newKernelType = printToString("{0}_{1}_{2}", kernelPath, cpu, "lsu0_wo");
        } else {
            newKernelType = printToString("{0}_{1}", kernelPath, cpu);
        }

        auto kernelTextOp = kernelRangeOp.getKernelTextIndex().getDefiningOp<VPUMI37XX::DeclareKernelTextOp>();
        auto kernelArgsOp = kernelRangeOp.getKernelArgsIndex().getDefiningOp<VPUMI37XX::DeclareKernelArgsOp>();
        auto kernelEntryOp = kernelRangeOp.getKernelEntryIndex().getDefiningOp<VPUMI37XX::DeclareKernelEntryOp>();

        kernelTextOp.setKernelPath(newKernelType);
        kernelArgsOp.setKernelPath(newKernelType);
        kernelEntryOp.setKernelPath(newKernelType);
    });
}
}  // namespace

//
// createAssignFullKernelPathPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI37XX::createAssignFullKernelPathPass(Logger log) {
    return std::make_unique<AssignFullKernelPathPass>(log);
}
