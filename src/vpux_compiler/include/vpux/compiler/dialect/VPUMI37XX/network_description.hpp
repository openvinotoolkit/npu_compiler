//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "vpux/compiler/network_metadata.hpp"

namespace vpux::VPUMI37XX {

// E#-140887: replace mlir::ArrayRef<uint8_t> with BlobView
NetworkMetadata getNetworkMetadata(mlir::ArrayRef<uint8_t> blob);

// Returns network metadata by deserializing serialized metadata
NetworkMetadata getNetworkMetadata(uint8_t* serializedMetadata, size_t serializedMetadataSize);

// Returns network metadata by extracting it from the module's NetworkInfoOp or global variable
// of a serialized metadata for host compilation
NetworkMetadata getHostCompileNetworkMetadata(mlir::ModuleOp module);

}  // namespace vpux::VPUMI37XX
