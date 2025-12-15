//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux {

//
// Swizzling constants
//

// Swizzling key 5 requires swizzled buffer to have start address aligned to 16k/32K for different arch
constexpr uint64_t SWIZZLING_KEY_5 = 5;

// Swizzled buffers need to have space in CMX of size aligned to 512/1024 based on arch, because this
// it the smallest unit of CMX RAM cut
constexpr int64_t SWIZZLING_SIZE_ALIGNMENT_VPUX50XX = 1024;
constexpr int64_t SWIZZLING_SIZE_ALIGNMENT_VPUX40XX = 1024;
constexpr int64_t SWIZZLING_SIZE_ALIGNMENT_VPUX37XX = 512;

int64_t getSizeAlignmentForSwizzling(config::ArchKind arch);

/// @brief Required alignment of buffers in CMX memory required swizzling operations
/// @param swizzlingKey
/// @param archKind
/// @return alignment [bytes]
int64_t getAddressAlignmentForSwizzling(int64_t swizzlingKey, config::ArchKind archKind);

VPUIP::SwizzlingSchemeAttr createSwizzlingSchemeAttr(mlir::MLIRContext* ctx, config::ArchKind archKind,
                                                     int64_t swizzlingKey);

// For swizzling buffer size needs to be aligned to 512/1024 as dictated by arch
int64_t alignSizeForSwizzling(int64_t size, int64_t sizeAlignment);

/// @brief calculate size of buffers with requested initial memory allocation offset and fixed minimal allocation
/// increments
/// @param bufferSizes - reference to vector containing sizes [bytes] of buffers to be allocated
/// @param offsetAlignment - alignment of buffer start [bytes]. The default value corresponds to offset required for
/// vpux::SWIZZLING_KEY_5. Must be > 0. The start address ADDR of any allocated buffer yields ADDR % offsetAlignment ==
/// 0.
/// @param sizeAlignment - memory allocation increment. The memory is allocated  in units of sizeAlignment so that
/// bufferSize[i] % sizeAlignment == 0. The default value corresponds to SWIZZLING_SIZE_ALIGNMENT_VPUX37XX or
/// SWIZZLING_SIZE_ALIGNMENT_VPUX40XX.
/// @return required memory taking into account the allocation requirements for swizzled buffers [bytes].
Byte calculateAlignedBuffersMemoryRequirement(mlir::SmallVector<Byte>& bufferSizes,
                                              const Byte offsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT),
                                              const Byte sizeAlignment = Byte(0));

VPUIP::SwizzlingSchemeAttr getSwizzlingSchemeAttr(mlir::Type type);

// Retrieve swizzling key setting embedded in layout with buffer types
int64_t getSwizzlingKey(mlir::Type type);

mlir::Type setSwizzlingKey(mlir::Type type, mlir::IntegerAttr swizzlingKeyAttr, config::ArchKind archKind);

mlir::Type setSwizzlingKey(mlir::Type type, int64_t swizzlingKey, config::ArchKind archKind);

SmallVector<int64_t> getPerClusterBytesAddedForSwizzling(VPUIP::DistributedBufferType distributedBuffer);

vpux::NDTypeInterface updateSwizzlingSchemeBasedOnDistributedType(VPUIP::DistributedBufferType inputType,
                                                                  vpux::NDTypeInterface newType);

}  // namespace vpux
