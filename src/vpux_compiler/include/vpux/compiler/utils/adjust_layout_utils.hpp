//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

namespace vpux {

struct AdjustConvShapeParams {
    Shape filterShape;     // New constructed filter shape after reshape conv's shape
    Shape inputShape;      // New conv's input shape after adjust
    Shape outputShape;     // New conv's output shape after adjust
    int64_t borrowFactor;  // The borrowed fator from W for C
    int64_t filterPading;  // The padding num for filter after construct new filter
    int64_t padNum;        // The padding num for aligned shape
};

void insertReorderForInput(mlir::Operation* op, mlir::OpOperand& input, const DimsOrder& dstOrder,
                           mlir::PatternRewriter& rewriter, Logger log);
IE::ReorderOp insertReorderForOutput(mlir::Operation* op, mlir::Value output, const DimsOrder& dstOrder,
                                     mlir::PatternRewriter& rewriter, Logger log);

void changeDimsOrder(mlir::Value value, const DimsOrder& newOrder, Logger log);

mlir::FailureOr<AdjustConvShapeParams> getAdjustConvShapeParameters(IE::ConvolutionOp convOp, mlir::Value filter,
                                                                    ShapeRef outputShape, Logger _log);

int64_t calculateAlignmentFactor(const vpux::NDTypeInterface sliceInType, const vpux::NDTypeInterface sliceOutType);

}  // namespace vpux
