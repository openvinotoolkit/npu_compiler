//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

// Encode tile index and index of SHAVE unit inside the tile into a single integer for a convenient use in scheduling
// passes. The queue indexing is done in listIndex major order. If list index attribute of a shave task is not provided
// then it is assumed 0 and the encoding does not distinguish between different SHAVE FIFOs and the encoding follows
// the tileIndex.
int64_t getShaveQueueIdEncoding(int64_t numTiles, int64_t tileIndex, int64_t listIndex);
int64_t getShaveTileIndexFromEncodedId(int64_t shaveQueueIdEncoding, int64_t numTiles);
int64_t getShaveListIndexFromEncodedId(int64_t shaveQueueIdEncoding, int64_t numTiles);
namespace VPU {
constexpr StringRef USE_DEDICATED_FIFO_PER_SHAVE_ENGINE = "VPU.UseDedicatedFifoPerShaveEngine";
bool isFifoPerShaveEngineEnabled(mlir::Operation* op);
bool hasSupportForFifoPerShaveEngine(config::ArchKind arch, bool enableWorkloadManagement);
}  // namespace VPU
}  // namespace vpux
