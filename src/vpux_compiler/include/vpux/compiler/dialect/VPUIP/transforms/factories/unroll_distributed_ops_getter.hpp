//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/unroll_distributed_ops_strategy.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <optional>

namespace vpux::VPUIP {

std::unique_ptr<IUnrollDistributedOpsStrategy> createUnrollDistributedOpsStrategy(
        mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion);

}  // namespace vpux::VPUIP
