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

// Before NPU4, input data could be read from other clusters. SE pointers could also address data
// that is not placed in the local cluster of the DPU task, which means that a mechanism was necessary
// to select which cluster to read from. The base pointer part of the SE pointers would do this.
// Starting with NPU4, the base pointers no longer need to be configured differently for each cluster
// as the DPU is only able to read from the local cluster, so they can be reset.
bool resetBasePtrs(const vpux::config::ArchKind arch) {
    return arch >= config::ArchKind::NPU40XX;
}

}  // namespace vpux::VPUIP
