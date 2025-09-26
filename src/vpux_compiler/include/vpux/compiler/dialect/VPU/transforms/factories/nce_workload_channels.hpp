//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux {
namespace VPU {

/**
 * @brief Returns a vector of supported channel sizes for Depthwise operations based on the architecture
 *
 * @param arch VPU architecture kind (e.g. NPU37XX, NPU40XX)
 * @return SmallVector<int64_t> Vector of supported channel sizes in descending order
 */
SmallVector<int64_t> getSupportedChannelsDW(config::ArchKind arch);

/**
 * @brief Checks if at least one channel is supported by kernel optimization. If true, proceeds with
 * getChannelsSupportedByKernelOptimization function
 *
 * @param op The MLIR operation
 * @param supportedChannels Array of channel sizes that are supported by the architecture
 * @param kx Kernel width parameter
 * @param sx Stride parameter in X dimension
 * @return bool True if kernel optimization is supported, false otherwise
 */
bool hasAnyChannelSupportedByKernelOptimization(mlir::Operation* op, ArrayRef<int64_t> supportedChannels, int64_t KX,
                                                int64_t SX);

/**
 * @brief Returns a vector of channel sizes that can be optimized
 *
 * @param op The MLIR operation
 * @param workloadsChannels Array of available channel configurations
 * @param maxSlotsSum Maximum slots sum constraint for optimization
 * @return SmallVector<int64_t> Vector of channel sizes that can be used for optimized workload split
 */
SmallVector<int64_t> getChannelsSupportedByKernelOptimization(mlir::Operation* op, ArrayRef<int64_t> workloadsChannels,
                                                              int64_t maxSlotsSum);

/**
 * @brief Determines if NCE permute offsets correction is needed
 *
 * @param nceOp NCE operation interface
 * @return bool True if NCE permute offsets correction is required, false otherwise
 */
bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp);

}  // namespace VPU
}  // namespace vpux
