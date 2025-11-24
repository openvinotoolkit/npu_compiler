//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/unroll_distributed_ops_strategy.hpp"

namespace vpux::VPUIP::arch37xx {
class UnrollDistributedOpsStrategy final : public vpux::VPUIP::IUnrollDistributedOpsStrategy {
public:
    UnrollDistributedOpsStrategy(mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion)
            : IUnrollDistributedOpsStrategy(funcOp, enableSegmentedDmaFusion) {
    }

    void prepareOps(mlir::MLIRContext& ctx, Logger& log) override;
};

}  // namespace vpux::VPUIP::arch37xx
