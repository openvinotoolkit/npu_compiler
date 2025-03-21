//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

namespace NCEInvariant {

//
// Constants
//
// TODO: E113153, config to be moved to init compiler pass
constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

constexpr int64_t SUPPORTED_BATCH_SIZE = 1;

constexpr int64_t MAX_STRIDE = 8;

constexpr int64_t VPU_CHANNEL_ALIGNMENT = 16;

constexpr int64_t VPU_COMPRESSED_INPUT_CHANNEL_NUM = 4;

constexpr int64_t VPU_WEIGHT_SET_BYTE_ALIGNMENT = 16;

constexpr int64_t VPU_DIMENSION_LIMIT = 8192;

constexpr int64_t VPU_SEGMENT_SIZE_DENSE = 4;
constexpr int64_t VPU_SEGMENT_SIZE_SPARSE = 8;

constexpr int64_t VPU_CHANNEL_SIZE_FOR_L1OPT = 32;

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

bool checkLayouts(mlir::TypeRange operandTypes, mlir::TypeRange resultTypes, const VPU::ArchKind& arch,
                  const unsigned numInputOperands, LogCb logCb);

mlir::LogicalResult isSupported(mlir::Operation* op, Logger log = Logger::global());

//
// Check if small kernel optimization is supported
//
bool isSmallKernelOptimizationSupported(const VPU::ArchKind arch, mlir::Operation* op);

mlir::LogicalResult verifyKernel(mlir::Operation* op, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop,
                                 int64_t padBottom, int64_t padLeft, int64_t padRight, Logger log = Logger::global());

mlir::LogicalResult verifyKernel(mlir::Operation* origOp, Logger log = Logger::global());

mlir::LogicalResult verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module, vpux::NDTypeInterface inputType,
                                  vpux::NDTypeInterface outputType, mlir::ArrayAttr kernelSize,
                                  mlir::ArrayAttr kernelStrides, Logger log = Logger::global());
//
// Check if given architecture supports Elementwise multiply operation
//

bool isEltwiseMultiplySubtractSupported(const VPU::ArchKind arch);

}  // namespace NCEInvariant

}  // namespace VPU
}  // namespace vpux
