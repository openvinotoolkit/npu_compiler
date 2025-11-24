//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace vpux {
namespace IE {

void DynamicDimOpBuilder::notifyOperationInserted(mlir::Operation* op, mlir::OpBuilder::InsertPoint) {
    if (auto dimOp = mlir::dyn_cast_or_null<mlir::tensor::DimOp>(op)) {
        auto index = dimOp.getConstantIndex().value_or(0);
        extendOpLoc(op, StringLiteral("dim_{0}"), index);
    } else if (mlir::isa<mlir::arith::ConstantIndexOp>(op)) {
        extendOpLoc(op, StringLiteral("const_index"));
    }
}

bool hasDynamicShapeAttr(mlir::Value value) {
    auto type = value.getType();
    return mlir::isa<Core::BoundedTensorType>(type) || mlir::isa<Core::DynamicDimsMaskTensorType>(type);
}

bool hasDynamicShape(mlir::Value value) {
    if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType())) {
        return !rankedTensorType.hasStaticShape();
    }
    return false;
}

bool hasDynamicTensors(mlir::Operation* op) {
    const auto hasDynamicInputs = llvm::any_of(op->getOperands(), [&](mlir::Value val) {
        return hasDynamicShapeAttr(val) || hasDynamicShape(val);
    });
    const auto hasDynamicOutputs = llvm::any_of(op->getResults(), [&](mlir::Value val) {
        return hasDynamicShapeAttr(val) || hasDynamicShape(val);
    });

    return hasDynamicInputs || hasDynamicOutputs;
}

// The list of operations may seem completely arbitrary, but they share one common trait.
// Convolution, MaxPool, Add and ReLU all run on static shapes only.
// In most cases these operations are mapped to DPU.
// DPU cannot adjust the workload size once it is set during the parsing stage.
// Right now these operations use upper bounds to set the size of workloads.
bool needsStaticShape(mlir::Operation* op) {
    auto requireStaticShape = [&]() {
        if (auto staticShapeOpInterface = mlir::dyn_cast_or_null<StaticShapeOpInterface>(op)) {
            return staticShapeOpInterface.requiresStaticShape();
        }
        return false;
    };

    return op ? requireStaticShape() : false;
}

// Given a dynamic shape and DimsOrder, decide if the dynamic data is contiguous or strided
// e.g. ?x16x32x4 NCHW -> contiguous
//      1x32x?x10 NHWC -> contiguous
//      1x14x10x? NCHW -> strided
bool isDynamicDataContiguous(vpux::ShapeRef shape, vpux::DimsOrder order) {
    bool foundDynamicDim = false;
    for (size_t idx = 0; idx < shape.size(); idx++) {
        const auto dimPos = order.dimAt(idx);

        if (shape[dimPos] == mlir::ShapedType::kDynamic) {
            // more than one dynamic dim
            if (foundDynamicDim) {
                return false;
            }

            // outer-most dynamic dim found
            foundDynamicDim = true;
        }

        // for dynamic data to be contiguous, the dynamic dim must be the outer-most dim that is not 1
        if (!foundDynamicDim && shape[dimPos] != 1 && shape[dimPos] != mlir::ShapedType::kDynamic) {
            return false;
        }
    }

    return foundDynamicDim;
}

}  // namespace IE

Shape extractShape(const Shape& shape) {
    return shape;
}
Shape extractShape(const BoundedShape& shape) {
    return shape.toShape();
}
Shape extractShape(const DimsMaskedShape& shape) {
    return shape.toReifiedShape();
}

Shape reifyShape(ShapeRef shape) {
    VPUX_THROW_WHEN(shape.isDynamic(), "Tried to reify a dynamic shape without known bounds: {0}", shape);
    return Shape(shape);
}

Shape reifyShape(BoundedShapeRef shape) {
    return shape.toReifiedShape();
}

Shape reifyShape(DimsMaskedShapeRef shape) {
    return shape.toReifiedShape();
}

Shape reifyShape(const Shape& shape) {
    VPUX_THROW_WHEN(shape.isDynamic(), "Tried to reify a dynamic shape without known bounds: {0}", shape);
    return shape;
}

Shape reifyShape(const BoundedShape& shape) {
    return shape.toReifiedShape();
}

Shape reifyShape(const DimsMaskedShape& shape) {
    return shape.toReifiedShape();
}

std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const Shape& shape) {
    return std::make_tuple(shape, Bounds(), DynamicDimsMask());
}
std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const BoundedShape& shape) {
    return std::make_tuple(shape.toShape(), shape.toRepresentation(), DynamicDimsMask());
}
std::tuple<Shape, Bounds, DynamicDimsMask> splitShapeAndRepresentation(const DimsMaskedShape& shape) {
    return std::make_tuple(shape.toReifiedShape(), Bounds(), shape.toRepresentation());
}

}  // namespace vpux
