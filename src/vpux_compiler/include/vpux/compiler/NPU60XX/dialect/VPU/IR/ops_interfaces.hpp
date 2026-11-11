//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPU::arch60xx {

void registerLayerWithPermuteInterfaceForIE(mlir::DialectRegistry& registry);
void registerClusterBroadcastingOpInterfaces(mlir::DialectRegistry& registry);
void registerLayerWithDmaInterface(mlir::DialectRegistry& registry);
void registerNCEOpInterface(mlir::DialectRegistry& registry);

}  // namespace vpux::VPU::arch60xx
