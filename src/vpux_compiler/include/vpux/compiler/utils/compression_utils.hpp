//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/Types.h>

namespace vpux {

//
// Activation compression
//

constexpr uint32_t ACT_COMPRESSION_RESERVED_MEM_SIZE = 64;
constexpr uint32_t ACT_COMPRESSION_SIZE_ENTRY_SIZE = 32;
constexpr uint32_t ACT_COMPRESSION_BUF_SIZE_ALIGNMENT = 32;
constexpr uint32_t ACT_COMPRESSION_MIN_BUF_SIZE = 256;

// For compression reserved size of buffer needs to be updated for worst case compression
int64_t updateSizeForCompression(int64_t origTensorSize, llvm::ArrayRef<int64_t> origShape = llvm::ArrayRef<int64_t>(),
                                 int64_t sparsityMapSize = 0);

bool isSupportedBufferSizeForCompression(vpux::NDTypeInterface ndType);

mlir::Type setCompressionState(mlir::Type type, VPUIP::CompressionState compression);

VPUIP::CompressionState getCompressionState(mlir::Type type);

VPUIP::CompressionStateAttr getCompressionStateAttr(mlir::Type type);

}  // namespace vpux
