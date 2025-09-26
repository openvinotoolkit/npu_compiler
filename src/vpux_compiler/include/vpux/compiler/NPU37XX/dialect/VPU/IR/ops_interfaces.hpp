//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Dialect.h>

namespace vpux::VPU::arch37xx {

void registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry);
void registerLayerWithPermuteInterfaceForIE(mlir::DialectRegistry& registry);
void registerLayoutInfoOpInterfaces(mlir::DialectRegistry& registry);
void registerDDRAccessOpModelInterface(mlir::DialectRegistry& registry);
void registerNCEOpInterface(mlir::DialectRegistry& registry);
void registerClusterBroadcastingOpInterfaces(mlir::DialectRegistry& registry);
void registerUnrollBatchOpInterfaces(mlir::DialectRegistry& registry);
void registerICostModelUtilsInterface(mlir::DialectRegistry& registry);
void registerSWTilingInfoOpInterface(mlir::DialectRegistry& registry);

}  // namespace vpux::VPU::arch37xx
