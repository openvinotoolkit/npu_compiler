//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reify_shape.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <utility>

using namespace vpux;

// Concatenate dynamic dimensions with static dimensions.
// tensor<1x2x?> produces:
// %dim_1 = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
// %dim_2 = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
// %tensor_dim = tensor.dim %0, %c3
// %tensor_dim_i64 = arith.index_cast %tensor_dim : index to i64
// %from_elements = tensor.from_elements %tensor_dim_i64 : tensor<1xi64>
// %dim_x = tensor.bitcast %from_elements : tensor<1xi64> to tensor<1xsi64>
// IE.Concat(%dim_1, %dim_2, %dim_x)
//
// arith.index_cast -> tensor.from_elements -> tensor.bitcast is required because:
// 1. tensor.dim produces index data type
// 2. arith.index_cast converts index to i64 scalar (arith prefers to work with signless types)
// 3. tensor.from_elements promotes scalars to tensors
// 4. tensor.bitcast converts i64 tensors to si64 tensors
//
// IE dialect prefers to work with signed (or unsigned) tensors, not with signless scalars.
IE::ConcatOp vpux::buildConcat(const mlir::Location loc, mlir::OpBuilder& builder, ShapeRef producerShape,
                               mlir::ValueRange dynamicOperands) {
    SmallVector<mlir::Value> concatInputs{};
    // Offset inside dynamicOperands container.
    // Increases when an element of dynamicOperands goes to concatInputs.
    size_t dynamicOpIdx = 0;
    for (const auto& dim : producerShape) {
        if (dim == mlir::ShapedType::kDynamic) {
            auto toI64 = builder.create<mlir::arith::IndexCastOp>(appendLoc(loc, "to_i64_{0}", dim),
                                                                  getInt64Type(builder.getContext()),
                                                                  dynamicOperands[dynamicOpIdx]);
            auto tensorI64Type = mlir::RankedTensorType::get({1}, getInt64Type(builder.getContext()));
            auto toTensor = builder.create<mlir::tensor::FromElementsOp>(appendLoc(loc, "to_tensor_{0}", dim),
                                                                         tensorI64Type, toI64->getResult(0));
            auto tensorSI64Type = mlir::RankedTensorType::get({1}, getSInt64Type(builder.getContext()));
            auto toSI64 = builder.create<mlir::tensor::BitcastOp>(appendLoc(loc, "to_si64_{0}", dim), tensorSI64Type,
                                                                  toTensor->getResult(0));
            concatInputs.push_back(toSI64->getResult(0));
            dynamicOpIdx++;
        } else {
            const SmallVector<int64_t> dimValues{dim};
            auto tensorType = mlir::RankedTensorType::get({1}, getSInt64Type(builder.getContext()));
            concatInputs.push_back(
                    Const::createConst(builder, appendLoc(loc, "_dim_{0}", dim), tensorType, ArrayRef(dimValues)));
        }
    }

    const auto axisAttr = getIntAttr(builder.getContext(), 0);
    return builder.create<IE::ConcatOp>(appendLoc(loc, "concat"), concatInputs, axisAttr);
}

mlir::Value vpux::repackDynamicTensor(mlir::OpBuilder& builder, mlir::Operation* producer, NDTypeInterface operandType,
                                      IE::ConcatOp newShapeValue) {
    auto ctx = builder.getContext();

    const auto tensorRank = checked_cast<int64_t>(operandType.getRank());
    const auto operandShape = operandType.getShape();
    mlir::Value dynReshapeOperand = producer->getResult(0);
    if (!IE::isDynamicDataContiguous(operandShape, operandType.getDimsOrder())) {
        const SmallVector<int64_t> begins(tensorRank, 0);
        const SmallVector<int64_t> strides(tensorRank, 1);

        auto sliceOp =
                builder.create<IE::StridedSliceOp>(appendLoc(producer->getLoc(), "slice"),
                                                   /*data=*/producer->getResult(0),
                                                   /*begins=*/nullptr,
                                                   /*ends=*/newShapeValue.getOutput(),
                                                   /*strides=*/nullptr,
                                                   /*beginsAttr=*/getIntArrayAttr(ctx, begins),
                                                   /*endsAttr=*/nullptr,
                                                   /*stridesAttr=*/getIntArrayAttr(ctx, strides),
                                                   /*beginMask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*endMask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*newAxisMask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*shrinkAxisMask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*ellipsisMask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}));

        dynReshapeOperand = sliceOp.getOutput();
    }

    // Reshape is required because strided slice infers all dimensions as dynamic
    // [Track number: S#154699]
    auto reshape = builder.create<IE::DynamicReshapeOp>(appendLoc(producer->getLoc(), "reshape"),
                                                        /*data=*/dynReshapeOperand,
                                                        /*shape=*/newShapeValue.getOutput(),
                                                        /*output_shape=*/getIntArrayAttr(ctx, operandShape.raw()),
                                                        /*output_bounds=*/getBounds(producer->getResult(0)));
    return reshape.getOutput();
}
