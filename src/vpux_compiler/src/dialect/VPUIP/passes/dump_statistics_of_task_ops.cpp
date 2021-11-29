//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

//
// DumpStatisticsOfTaskOpsPass
//

class DumpStatisticsOfTaskOpsPass final : public VPUIP::DumpStatisticsOfTaskOpsBase<DumpStatisticsOfTaskOpsPass> {
public:
    explicit DumpStatisticsOfTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DumpStatisticsOfTaskOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    _log.info("VPUIP tasks statistics:");

    llvm::DenseSet<mlir::OperationName> dpuOperations{
            mlir::OperationName(VPUIP::ConvolutionUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::PoolingUPAOp::getOperationName(), &ctx),
            mlir::OperationName(VPUIP::EltwiseUPAOp::getOperationName(), &ctx)};

    llvm::DenseMap<mlir::OperationName, size_t> taskMap;
    func->walk([&](VPUIP::TaskOpInterface op) {
        taskMap[op->getName()]++;
    });

    for (auto& taskOp : taskMap) {
        _log.nest().info("{0} - {1} ops", taskOp.first, taskOp.second);

        if (VPU::getCompilationMode(func) == VPU::CompilationMode::ReferenceSW) {
            continue;
        }

        if (dpuOperations.contains(taskOp.first)) {
            _log.nest().warning("'{0}' was not converted to 'VPUIP.NCETask'", taskOp.first);
        }
    }
}

}  // namespace

//
// createDumpStatisticsOfTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDumpStatisticsOfTaskOpsPass(Logger log) {
    return std::make_unique<DumpStatisticsOfTaskOpsPass>(log);
}
