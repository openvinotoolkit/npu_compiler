//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

#include <mlir/IR/Dialect.h>

namespace vpux::VPUIPDPU::arch40xx {

mlir::LogicalResult verifyODUCfgOp(VPUIPDPU::ODUCfgOp op);

void registerDPUExpandOpInterfaces(mlir::DialectRegistry& registry);
void registerVerifiersOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::VPUIPDPU::arch40xx
