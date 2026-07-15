//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPU::arch40xx {

void registerClusterBroadcastingOpInterfaces(mlir::DialectRegistry& registry);
void registerSCFTilingOpsInterfaces(mlir::DialectRegistry& registry);
void registerLayerWithDmaInterface(mlir::DialectRegistry& registry);

}  // namespace vpux::VPU::arch40xx
