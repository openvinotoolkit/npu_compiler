//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "intel_npu/network_metadata.hpp"

namespace vpux::VPUMI37XX {

// E#-140887: replace mlir::ArrayRef<uint8_t> with BlobView
intel_npu::NetworkMetadata getNetworkMetadata(mlir::ArrayRef<uint8_t> blob);
intel_npu::NetworkMetadata getNetworkMetadata(mlir::ModuleOp module);

// Returns network metadata by deserializing serialized metadata
intel_npu::NetworkMetadata getNetworkMetadata(uint8_t* serializedMetadata, size_t serializedMetadataSize);

}  // namespace vpux::VPUMI37XX
