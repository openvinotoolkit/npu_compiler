//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"

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

constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

constexpr int64_t KMB_CMCONV_WIDTH_ALIGNMENT = 16;
constexpr int64_t KMB_CMCONV_CHANNELS_LIMIT = 16;

constexpr int64_t SUPPORTED_BATCH_SIZE = 1;
constexpr int64_t MAX_KERNEL_SIZE = 11;
constexpr int64_t MAX_STRIDE = 8;

constexpr int64_t VPU_CHANNEL_ALIGNMENT = 16;

constexpr int64_t VPU_COMPRESSED_INPUT_CHANNEL_NUM = 4;

constexpr int64_t VPU_WEIGHT_SET_BYTE_ALIGNMENT = 16;

constexpr int64_t VPU_DIMENSION_LIMIT = 8192;

//
// Precision checks
//

bool isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb = globalLogCb);

//
// Attributes checks
//

bool isAttrsSupported(ArchKind arch, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop, int64_t padBottom,
                      int64_t padLeft, int64_t padRight, LogCb logCb = globalLogCb);

//
// Activation type checks
//

bool isAligned(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb);

int64_t getAlignment(mlir::Type elemType);

bool isInputActTypeSupported(ArchKind arch, vpux::NDTypeInterface type, int64_t alignment,
                             bool supportsInputActCompression = false, LogCb logCb = globalLogCb);
bool isOutputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb = globalLogCb);

//
// WeightsTable information
//

Byte getWeightsTableSize(int64_t OC);

//
// Channel major Convolution
//

bool isChannelMajorCompatible(ArchKind arch, vpux::NDTypeInterface inputType);

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

bool checkLayouts(mlir::ValueRange operands, mlir::Value result, const VPU::ArchKind& arch,
                  const unsigned numInputOperands, LogCb logCb);

//
// Compress Convolution
//

bool isCompressConvolution(ArchKind arch, mlir::Operation* op);

}  // namespace NCEInvariant

}  // namespace VPU
}  // namespace vpux
