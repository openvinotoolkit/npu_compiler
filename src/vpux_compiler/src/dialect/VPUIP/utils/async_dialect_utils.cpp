//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Async/IR/Async.h>

using namespace vpux;
namespace {
bool isDmaDataOp(mlir::async::ExecuteOp execOp, VPU::MemoryKind expectedSrc, VPU::MemoryKind expectedDst) {
    if (VPUIP::getExecutorType(execOp) != config::ExecutorKind::DMA_NN) {
        return false;
    }

    if (auto dmaTask = VPUIP::getDmaTypeOp(execOp)) {
        auto srcMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getInput().getType()).getMemoryKind();
        auto dstMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaTask.getOutput().getType()).getMemoryKind();
        return (expectedSrc == srcMemSpace && expectedDst == dstMemSpace);
    }

    return false;
}
}  // namespace

namespace vpux::VPUIP {

config::ExecutorKind getExecutorType(mlir::async::ExecuteOp execOp) {
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        return VPUIP::VPUIPDialect::getExecutorKind(execOp);
    }
    // treat all other executors as DPU
    return config::ExecutorKind::DPU;
}

config::ExecutorKind getExecutorType(size_t opIdx, AsyncDepsInfo& depsInfo) {
    auto execOp = depsInfo.getExecuteOpAtIndex(opIdx);
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        return VPUIP::VPUIPDialect::getExecutorKind(execOp);
    }
    VPUX_THROW("Can not get executor for 'async.exec' op with id {0}", opIdx);
}

VPUIP::DMATypeOpInterface getDmaTypeOp(mlir::async::ExecuteOp execOp) {
    auto* bodyBlock = execOp.getBody();

    for (auto& op : bodyBlock->getOperations()) {
        if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
            return dmaOp;
        }
    }

    return nullptr;
}

bool isDmaDDR2CMX(mlir::async::ExecuteOp execOp) {
    return isDmaDataOp(execOp, VPU::MemoryKind::DDR, VPU::MemoryKind::CMX_NN);
}

bool isDmaCMX2DDR(mlir::async::ExecuteOp execOp) {
    return isDmaDataOp(execOp, VPU::MemoryKind::CMX_NN, VPU::MemoryKind::DDR);
}

bool isDmaDDR2DDR(mlir::async::ExecuteOp execOp) {
    return isDmaDataOp(execOp, VPU::MemoryKind::DDR, VPU::MemoryKind::DDR);
}

bool isDmaCMX2CMX(mlir::async::ExecuteOp execOp) {
    return isDmaDataOp(execOp, VPU::MemoryKind::CMX_NN, VPU::MemoryKind::CMX_NN);
}
}  // namespace vpux::VPUIP
