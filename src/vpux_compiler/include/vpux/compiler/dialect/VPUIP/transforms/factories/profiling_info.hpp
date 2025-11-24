//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP {

using TimestampTypeCb = mlir::Type (*)(mlir::MLIRContext* ctx);
using SetWorkloadIdsCb = void (*)(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

TimestampTypeCb getTimestampTypeCb(config::ArchKind arch);
SetWorkloadIdsCb setWorkloadsIdsCb(config::ArchKind arch);

}  // namespace vpux::VPUIP
