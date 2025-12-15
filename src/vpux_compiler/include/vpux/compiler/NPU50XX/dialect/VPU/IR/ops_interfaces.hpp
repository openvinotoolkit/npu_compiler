//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPU::arch50xx {

void registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry);
void registerClusterBroadcastingOpInterfaces(mlir::DialectRegistry& registry);
void registerUnrollBatchOpInterfaces(mlir::DialectRegistry& registry);
void registerICostModelUtilsInterface(mlir::DialectRegistry& registry);
void registerSWTilingInfoOpInterface(mlir::DialectRegistry& registry);

}  // namespace vpux::VPU::arch50xx
