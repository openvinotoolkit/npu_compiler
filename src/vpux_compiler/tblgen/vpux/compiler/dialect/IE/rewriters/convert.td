//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_REWRITERS_CONVERT
#define VPUX_COMPILER_DIALECT_IE_REWRITERS_CONVERT

include "vpux/compiler/dialect/IE/ops/data_type.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

//
// FuseConverts is no longer expressed as a TableGen pattern: the fusion is only valid when the intermediate
// conversion is redundant (the value is unchanged). Lossy intermediates (float -> int, f32 -> f16, ...) must be
// preserved, otherwise the truncation/rounding performed by the inner conversion is lost. The guarded pattern
// lives in ConvertOp::getCanonicalizationPatterns (see IE/IR/ops/convert.cpp).
//

#endif
