//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::VPUIP {

class IUnrollDistributedOpsStrategy {
public:
    IUnrollDistributedOpsStrategy(mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion)
            : _funcOp(funcOp), _enableSegmentedDmaFusion(enableSegmentedDmaFusion) {
    }

    virtual void prepareOps(mlir::MLIRContext& ctx, Logger& log) = 0;

    virtual ~IUnrollDistributedOpsStrategy() = default;

protected:
    mlir::func::FuncOp _funcOp;
    std::optional<bool> _enableSegmentedDmaFusion;
};

}  // namespace vpux::VPUIP
