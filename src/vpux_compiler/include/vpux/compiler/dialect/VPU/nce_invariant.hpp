//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <llvm/Support/FormatVariadic.h>

namespace vpux {
namespace VPU {

namespace NCEInvariant {

//
// Constants
//

constexpr int64_t WEIGHT_TABLE_NUM_ELEMENTS_PER_OC = 4;

//
// Logging callback
//

using LogCb = FuncRef<void(const llvm::formatv_object_base&)>;
void emptyLogCb(const llvm::formatv_object_base&);

//
// Precision checks
//

bool isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb = emptyLogCb);

//
// Attributes checks
//

bool isAttrsSupported(ArchKind arch, int64_t KY, int64_t KX, int64_t SY, int64_t SX, int64_t padTop, int64_t padBottom,
                      int64_t padLeft, int64_t padRight, LogCb logCb = emptyLogCb);

//
// Activation type checks
//

int64_t getAlignment(mlir::Type elemType);

bool isActTypeSupported(mlir::ShapedType type, LogCb logCb = emptyLogCb);

//
// PostOp checks
//

bool isPostOpSupported(mlir::Operation* postOp);

//
// WeightsTable information
//

Byte getWeightsTableSize(int64_t OC);

}  // namespace NCEInvariant

}  // namespace VPU
}  // namespace vpux
