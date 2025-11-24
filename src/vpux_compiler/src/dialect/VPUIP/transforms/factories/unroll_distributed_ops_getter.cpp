//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/unroll_distributed_ops_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/unroll_distributed_ops_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_distributed_ops_strategy.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPUIP {

std::unique_ptr<IUnrollDistributedOpsStrategy> createUnrollDistributedOpsStrategy(
        mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion) {
    const auto arch = config::getArch(funcOp);
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<arch37xx::UnrollDistributedOpsStrategy>(funcOp, enableSegmentedDmaFusion);
    default:
        return std::make_unique<arch40xx::UnrollDistributedOpsStrategy>(funcOp, enableSegmentedDmaFusion);
    }
}

}  // namespace vpux::VPUIP
