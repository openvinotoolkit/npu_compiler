//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_expand_dma_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_expand_dma_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_expand_dma_strategy.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

using namespace vpux;

std::unique_ptr<IGreedilyPassStrategy> VPUIP::createUnrollExpandDMAStrategy(mlir::func::FuncOp funcOp) {
    const auto arch = config::getArch(funcOp);

    auto module = funcOp->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.getCount();

    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<VPUIP::arch37xx::UnrollExpandDMAStrategy>(dmaPortCount);
        break;
    default:
        return std::make_unique<VPUIP::arch40xx::UnrollExpandDMAStrategy>(dmaPortCount);
        break;
    }
}
