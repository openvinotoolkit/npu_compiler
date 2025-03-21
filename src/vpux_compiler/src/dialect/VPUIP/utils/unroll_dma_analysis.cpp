//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/unroll_dma_analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

using namespace vpux;
namespace {
void ProcessOp(VPURT::TaskOp vpurtTask, VPUIP::UnrollDMAAnalysis::StorageType& lookupArray) {
    if (vpurtTask.getInnerTaskOpOfType<VPUIP::ExpandDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollExpandDMAPass)] = 1;
    } else if (vpurtTask.getInnerTaskOpOfType<VPUIP::PermuteDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollPermuteToNNDMAPass)] = 1;
    } else if (vpurtTask.getInnerTaskOpOfType<VPUIP::DepthToSpaceDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollDepthToSpaceDMAPass)] = 1;
    } else if (vpurtTask.getInnerTaskOpOfType<VPUIP::SpaceToDepthDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollSpaceToDepthDMAPass)] = 1;
    } else if (vpurtTask.getInnerTaskOpOfType<VPUIP::UpsamplingDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollUpsamplingDMAPass)] = 1;
    } else if (vpurtTask.getInnerTaskOpOfType<VPUIP::PerAxisTileDMAOp>() != nullptr) {
        lookupArray[static_cast<size_t>(VPUIP::UnrollDMAAnalysisNeeded::UnrollPerAxisTileDMAPass)] = 1;
    }
}
}  // namespace

namespace vpux::VPUIP {

UnrollDMAAnalysis::UnrollDMAAnalysis(mlir::Operation* operation): _operation(operation) {
    auto func = mlir::dyn_cast<mlir::func::FuncOp>(_operation);
    auto vpurtTasks = func.getOps<VPURT::TaskOp>();
    std::for_each(vpurtTasks.begin(), vpurtTasks.end(), [&](VPURT::TaskOp vpurtTask) {
        ProcessOp(vpurtTask, _lookupArray);
    });
}

bool UnrollDMAAnalysis::passNeeded(UnrollDMAAnalysisNeeded passTag) {
    return static_cast<bool>(_lookupArray[static_cast<size_t>(passTag)]);
}

}  // namespace vpux::VPUIP
