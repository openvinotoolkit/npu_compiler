//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

namespace NCEInvariant {

//
// Constants
//
// TODO: E113153, config to be moved to init compiler pass
constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

constexpr int64_t NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 1;

constexpr int64_t SUPPORTED_BATCH_SIZE = 1;

constexpr int64_t MAX_STRIDE = 8;

constexpr int64_t VPU_CHANNEL_ALIGNMENT = 16;

constexpr int64_t VPU_COMPRESSED_INPUT_CHANNEL_NUM = 4;

constexpr int64_t VPU_WEIGHT_SET_BYTE_ALIGNMENT = 16;

constexpr int64_t VPU_DIMENSION_LIMIT = 8192;

constexpr int64_t VPU_SEGMENT_SIZE_DENSE = 4;
constexpr int64_t VPU_SEGMENT_SIZE_SPARSE = 8;

constexpr int64_t VPU_CHANNEL_SIZE_FOR_L1OPT16 = 16;
constexpr int64_t VPU_CHANNEL_SIZE_FOR_L1OPT32 = 32;
constexpr int64_t VPU_CHANNEL_SIZE_FOR_L1OPT64 = 64;

constexpr int64_t VPU_SPATIAL_ALIGNMENT = 4;

constexpr std::array<int64_t, 3> DEPTHWISE_WORKLOAD_SIZES{16, 32, 64};

//
// Attributes checks
//

bool isAttrsSupported(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop,
                      int64_t padBottom, int64_t padLeft, int64_t padRight, LogCb logCb = globalLogCb);

//
// Activation type checks
//

bool isAligned(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb);

int64_t getAlignment(mlir::Type elemType);

bool isInputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, bool supportsInputActCompression = false,
                             LogCb logCb = globalLogCb);
bool isOutputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb = globalLogCb);

//
// WeightsTable information
//

Byte getWeightsTableSize(int64_t OC);

mlir::FailureOr<SmallVector<Byte>> getWeightsTableSize(int64_t OC, mlir::Value weightsTable,
                                                       mlir::Value weightTableScale, mlir::Value weightTableBias);

mlir::LogicalResult getWeightTableBuffers(mlir::Operation* op, SmallVector<Byte>& buffers, int64_t OC);

//
// Fuse PadOp check
//

bool verifyPads(mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBeginAttr, mlir::ArrayAttr padEndAttr,
                LogCb logCb = globalLogCb);
bool verifyPads(int64_t KY, int64_t KX, int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                LogCb logCb = globalLogCb);

//
// Common utility for AvgPool, MaxPool, Eltwise and DWConv
//

bool checkLayouts(mlir::TypeRange operandTypes, mlir::TypeRange resultTypes, const config::ArchKind& arch,
                  const unsigned numInputOperands, LogCb logCb);

mlir::LogicalResult isSupported(mlir::Operation* op, Logger log = Logger::global());

//
// Check if small kernel optimization is supported
//
bool doesWorkloadSupportSmallKernelOpt([[maybe_unused]] config::ArchKind arch, int64_t KX, int64_t SX,
                                       ArrayRef<int64_t> workloadOutSz, bool isFp16Input, [[maybe_unused]] int64_t KY,
                                       [[maybe_unused]] int64_t padLeft);
bool isSmallKernelOptimizationSupported(const config::ArchKind arch, mlir::Operation* op);

//
// Verify kernel utils
//
mlir::LogicalResult verifyKernel(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop,
                                 int64_t padBottom, int64_t padLeft, int64_t padRight, Logger log = Logger::global());

mlir::LogicalResult verifyKernel(mlir::Operation* origOp, Logger log = Logger::global());

mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType,
                                  vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelSize,
                                  mlir::ArrayAttr kernelStrides, Logger log = Logger::global());

//
// Verify weights table utils
//
mlir::LogicalResult verifyWeightTables(mlir::Operation* origOp, Logger log = Logger::global());

//
// Check if given architecture supports Elementwise multiply operation
//

bool isEltwiseMultiplySubtractSupported(const config::ArchKind arch);

//
// Check whether alignment is beneficial for the operation
//

bool isParentOptimalForAlignment(mlir::Operation* parentOp);
bool isAlignmentBeneficial(mlir::Operation* op);

bool hasDimensionExceedingVPULimit(ShapeRef shape);
}  // namespace NCEInvariant

}  // namespace VPU
}  // namespace vpux
