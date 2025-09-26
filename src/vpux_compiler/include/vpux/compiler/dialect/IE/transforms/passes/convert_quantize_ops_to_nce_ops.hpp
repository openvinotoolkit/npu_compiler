//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux::IE {

mlir::Value buildDwWeights(const mlir::Location& loc, const int64_t OC, const mlir::Type& elementType,
                           mlir::PatternRewriter& rewriter);
bool isQuantizedPerAxis(mlir::Value val);
// cst -> dequant case.
// When a dequantize operation has a constant producer, execute dequantization on activation shaves.
// ConvertQuantizeOpsToNCEOps must skip such dequantize operations.
// Mark them legal.
inline bool hasConstProducer(IE::DequantizeOp dequantOp) {
    return mlir::isa_and_nonnull<Const::DeclareOp>(dequantOp.getInput().getDefiningOp());
}
bool isLegalQuantizeOp(IE::QuantizeOp quantizeOp, bool canUseCMajor);
bool isPerChannelQuantizedType(mlir::Value val);
bool is16BitStorageType(vpux::NDTypeInterface type);
bool is8BitStorageType(vpux::NDTypeInterface type);

}  // namespace vpux::IE
