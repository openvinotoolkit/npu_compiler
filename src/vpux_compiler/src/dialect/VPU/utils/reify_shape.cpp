//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/reify_shape.hpp"
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

// VPU::DynamicReshapeOp/VPU::StridedSliceOp require a si32 `shape`/`ends` operand, so the concat
// built here packs each dimension as a 1-element si32 tensor instead of the si64 tensor that
// vpux::buildConcat (see dialect/IE/utils/reify_shape.cpp) uses.
VPU::ConcatOp vpux::VPU::buildConcat(const mlir::Location loc, mlir::OpBuilder& builder, ShapeRef producerShape,
                                     mlir::ValueRange dynamicOperands) {
    SmallVector<mlir::Value> concatInputs{};
    size_t dynamicOpIdx = 0;
    for (const auto& dim : producerShape) {
        if (dim == mlir::ShapedType::kDynamic) {
            auto dynOpLoc = dynamicOperands[dynamicOpIdx].getLoc();
            auto toI32 =
                    mlir::arith::IndexCastOp::create(builder, appendLoc(dynOpLoc, "to_i32_{0}", dynamicOpIdx),
                                                     getInt32Type(builder.getContext()), dynamicOperands[dynamicOpIdx]);
            auto tensorI32Type = mlir::RankedTensorType::get({1}, getInt32Type(builder.getContext()));
            auto toTensor = mlir::tensor::FromElementsOp::create(
                    builder, appendLoc(dynOpLoc, "to_tensor_{0}", dynamicOpIdx), tensorI32Type, toI32->getResult(0));
            auto tensorSI32Type = mlir::RankedTensorType::get({1}, getSInt32Type(builder.getContext()));
            auto toSI32 = mlir::tensor::BitcastOp::create(builder, appendLoc(dynOpLoc, "to_si32_{0}", dynamicOpIdx),
                                                          tensorSI32Type, toTensor->getResult(0));
            concatInputs.push_back(toSI32->getResult(0));
            dynamicOpIdx++;
        } else {
            const SmallVector<int32_t> dimValues{checked_cast<int32_t>(dim)};
            auto tensorType = mlir::RankedTensorType::get({1}, getSInt32Type(builder.getContext()));
            concatInputs.push_back(
                    Const::createConst(builder, appendLoc(loc, "dim_{0}", dim), tensorType, ArrayRef(dimValues)));
        }
    }

    const auto axisAttr = getIntAttr(builder.getContext(), 0);
    return VPU::ConcatOp::create(builder, appendLoc(loc, "concat"), concatInputs, axisAttr);
}

mlir::Value vpux::VPU::repackDynamicTensor(mlir::OpBuilder& builder, mlir::Operation* producer,
                                           NDTypeInterface operandType, VPU::ConcatOp newShapeValue) {
    auto ctx = builder.getContext();

    const auto tensorRank = checked_cast<int64_t>(operandType.getRank());
    const auto operandShape = operandType.getShape();
    mlir::Value dynReshapeOperand = producer->getResult(0);
    if (!IE::isDynamicDataContiguous(operandShape, operandType.getDimsOrder())) {
        const SmallVector<int64_t> begins(tensorRank, 0);
        const SmallVector<int64_t> strides(tensorRank, 1);

        auto sliceOp = VPU::StridedSliceOp::create(builder, appendLoc(producer->getLoc(), "slice"),
                                                   /*input=*/producer->getResult(0),
                                                   /*begins=*/nullptr,
                                                   /*ends=*/newShapeValue.getOutput(),
                                                   /*strides=*/nullptr,
                                                   /*begins_attr=*/getIntArrayAttr(ctx, begins),
                                                   /*ends_attr=*/nullptr,
                                                   /*strides_attr=*/getIntArrayAttr(ctx, strides),
                                                   /*begin_mask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*end_mask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*new_axis_mask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*shrink_axis_mask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*ellipsis_mask=*/getIntArrayAttr(ctx, SmallVector<int64_t>{}),
                                                   /*bounds_representation=*/VPU::BoundsRepresentation::BOUNDS);

        dynReshapeOperand = sliceOp.getOutput();
    }

    auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(producer->getResult(0).getType());
    VPUX_THROW_WHEN(boundedType == nullptr, "Expected to get BoundedTensorType at {0}",
                    producer->getResult(0).getLoc());
    auto bounds = boundedType.getBounds();

    // Reshape is required because strided slice infers all dimensions as dynamic
    // [Track number: S#154699]
    auto reshape = VPU::DynamicReshapeOp::create(builder, appendLoc(producer->getLoc(), "reshape"),
                                                 /*data=*/dynReshapeOperand,
                                                 /*shape=*/newShapeValue.getOutput(),
                                                 /*output_shape=*/getIntArrayAttr(ctx, operandShape.raw()),
                                                 /*output_bounds=*/getIntArrayAttr(ctx, bounds),
                                                 /*only_set_shape=*/false,
                                                 /*bounds_representation=*/VPU::BoundsRepresentation::BOUNDS);
    return reshape.getOutput();
}
