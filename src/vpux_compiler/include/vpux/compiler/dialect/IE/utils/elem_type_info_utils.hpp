//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux {
namespace IE {

void propagateElementTypeDown(LayerDataInfo<mlir::Type>& info);
void propagateElementTypeUp(LayerDataInfo<mlir::Type>& info);
bool isSupportedNearestNCEInterpolate(InterpolateOp interpolateOp, LogCb logCb = globalLogCb);
bool isSupportedElemTypeInfoCase(mlir::Operation* op, bool seOpsEnabled, LogCb logCb = globalLogCb);

void propagateElemTypeDownForAffineReshapeOp(AffineReshapeOp affineReshape, LayerDataInfo<mlir::Type>& info);
void propagateElemTypeDownForConcatOp(ConcatOp concat, LayerDataInfo<mlir::Type>& info);
void propagateElemTypeDownForExpandDilatedOp(ExpandDilatedOp expandDilated, LayerDataInfo<mlir::Type>& info);
void propagateElemTypeDownForReorderOp(ReorderOp reorder, LayerDataInfo<mlir::Type>& info);
void propagateElemTypeDownForTransposeOp(TransposeOp transpose, LayerDataInfo<mlir::Type>& info);
void propagateElemTypeUpForExpandDilatedOp(ExpandDilatedOp expandDilated, LayerDataInfo<mlir::Type>& info);

std::optional<mlir::Type> inferElemTypeAffineReshape(AffineReshapeOpAdaptor affineReshapeOp, mlir::Type inputElemType);
mlir::FailureOr<mlir::Type> inferOutElemTypeWithAxis(ArrayRef<mlir::Type> elemTypes, IE::ConcatOpAdaptor concat,
                                                     LogCb logCb = emptyLogCb);
mlir::FailureOr<mlir::Type> inferOutElemTypeWithOffsets(ArrayRef<mlir::Type> elemTypes, IE::ConcatOpAdaptor concat,
                                                        ShapeRef outShape, LogCb logCb = emptyLogCb);
mlir::FailureOr<Shape> inferOutShapeWithOffsets(IE::ConcatOpAdaptor concat, mlir::Location loc);
std::unordered_set<Dim> getConcatAxesFromOffsets(IE::ConcatOpAdaptor concat, ShapeRef outShape);
Dim normalizeAxis(IE::ConcatOpAdaptor concat);
mlir::Type inferElemTypeReorder(IE::ReorderOpAdaptor reorder, mlir::Type inputElemType, mlir::MLIRContext* ctx);
mlir::Type inferElemTypeTranspose(mlir::AffineMap map, mlir::Type inputElemType);

class PerTensorElemTypeInfoOpModel final :
        public IE::ElemTypeInfoOpInterface::FallbackModel<PerTensorElemTypeInfoOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeDown(info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeUp(info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoAffineReshapeOpModel final :
        public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoAffineReshapeOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::AffineReshapeOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected AffineReshapeOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeDownForAffineReshapeOp(origOp, info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeUp(info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoConcatOpModel final : public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoConcatOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::ConcatOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected ConcatOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeDownForConcatOp(origOp, info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeUp(info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoExpandDilatedOpModel final :
        public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoExpandDilatedOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::ExpandDilatedOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected ExpandDilatedOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeDownForExpandDilatedOp(origOp, info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::ExpandDilatedOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected ExpandDilatedOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeUpForExpandDilatedOp(origOp, info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoMaxPoolOpModel final : public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoMaxPoolOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected MaxPoolOp, got {0} at loc {1}", op->getName(), op->getLoc());

        if (VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, /*checkLayout=*/false,
                                           /*checkChannelAlignment=*/false)) {
            propagateElementTypeDown(info);
        }
    }
    static void inferElemTypeInfoUp(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected MaxPoolOp, got {0} at loc {1}", op->getName(), op->getLoc());

        if (VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, /*checkLayout=*/false,
                                           /*checkChannelAlignment=*/false)) {
            propagateElementTypeUp(info);
        }
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoReorderOpModel final : public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoReorderOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::ReorderOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected ReorderOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeDownForReorderOp(origOp, info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeUp(info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

class ElemTypeInfoTransposeOpModel final :
        public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoTransposeOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::TransposeOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected TransposeOp, got {0} at loc {1}", op->getName(), op->getLoc());

        propagateElemTypeDownForTransposeOp(origOp, info);
    }
    static void inferElemTypeInfoUp(mlir::Operation* /*op*/, LayerDataInfo<mlir::Type>& info) {
        propagateElementTypeUp(info);
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

}  // namespace IE
}  // namespace vpux
