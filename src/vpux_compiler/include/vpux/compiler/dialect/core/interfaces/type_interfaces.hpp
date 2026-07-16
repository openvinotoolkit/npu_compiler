//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPU {
enum class MemoryKind : uint64_t;
}

namespace vpux {

//
// TypeComponents
//

struct TypeComponents {
    std::optional<Shape> shape = std::nullopt;
    std::optional<mlir::Type> elementType = std::nullopt;
    std::optional<DimsOrder> dimsOrder = std::nullopt;
    std::optional<IndexedSymbolAttr> memSpace = std::nullopt;
    std::optional<Strides> strides = std::nullopt;
    std::optional<Bounds> bounds = std::nullopt;
    std::optional<DynamicDimsMask> dynamicDimsMask = std::nullopt;

    TypeComponents& setShape(ShapeRef newShape);
    TypeComponents& setShapeWithRepresentation(Shape&& newShape);
    TypeComponents& setShapeWithRepresentation(BoundedShape&& newShape);
    TypeComponents& setShapeWithRepresentation(DimsMaskedShape&& newShape);
    TypeComponents& setElementType(mlir::Type newElementType);
    TypeComponents& setDimsOrder(const DimsOrder& newDimsOrder);
    TypeComponents& setMemSpace(IndexedSymbolAttr newMemSpace);
    TypeComponents& setStrides(StridesRef newStrides);
    TypeComponents& setBounds(Bounds&& newBounds);
    TypeComponents& setDynamicDimsMask(DynamicDimsMask&& newDynamicDimsMask);
};

}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/core/type_interfaces.hpp.inc>

namespace vpux {

class TensorNDTypeInterface : public NDTypeInterface::FallbackModel<TensorNDTypeInterface> {
public:
    vpux::ShapeRef getShape(mlir::Type type) const;
    vpux::MemShape getMemShape(mlir::Type type) const;
    bool hasRank(mlir::Type type) const;
    int64_t getRank(mlir::Type type) const;
    int64_t getNumElements(mlir::Type type) const;
    mlir::Type getElementType(mlir::Type type) const;
    vpux::DimsOrder getDimsOrder(mlir::Type type) const;
    vpux::IndexedSymbolAttr getMemSpace(mlir::Type type) const;
    vpux::VPU::MemoryKind getMemoryKind(mlir::Type type) const;
    vpux::Strides getStrides(mlir::Type type) const;
    vpux::MemStrides getMemStrides(mlir::Type type) const;
    vpux::Bit getElemTypeSize(mlir::Type type) const;
    vpux::Byte getTotalAllocSize(mlir::Type type) const;
    vpux::Byte getCompactAllocSize(mlir::Type type) const;
    vpux::NDTypeInterface changeShape(mlir::Type type, vpux::ShapeRef shape) const;
    vpux::NDTypeInterface changeElemType(mlir::Type type, mlir::Type elemType) const;
    vpux::NDTypeInterface changeShapeElemType(mlir::Type type, vpux::ShapeRef shape, mlir::Type elemType) const;
    vpux::NDTypeInterface changeDimsOrder(mlir::Type type, const vpux::DimsOrder& order) const;
    vpux::NDTypeInterface changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const;
    vpux::NDTypeInterface changeStrides(mlir::Type type, vpux::StridesRef strides) const;
    vpux::NDTypeInterface changeTypeComponents(mlir::Type type, const vpux::TypeComponents& typeComponents) const;
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
    vpux::NDTypeInterface extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                          vpux::ShapeRef tileElemStrides) const;
    vpux::NDTypeInterface eraseTiledInfo(mlir::Type type) const;
    vpux::NDTypeInterface pad(mlir::Type type, vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
};

class MemRefNDTypeInterface : public vpux::NDTypeInterface::FallbackModel<MemRefNDTypeInterface> {
public:
    vpux::ShapeRef getShape(mlir::Type type) const;
    vpux::MemShape getMemShape(mlir::Type type) const;
    bool hasRank(mlir::Type type) const;
    int64_t getRank(mlir::Type type) const;
    int64_t getNumElements(mlir::Type type) const;
    mlir::Type getElementType(mlir::Type type) const;
    vpux::DimsOrder getDimsOrder(mlir::Type type) const;
    vpux::IndexedSymbolAttr getMemSpace(mlir::Type type) const;
    vpux::VPU::MemoryKind getMemoryKind(mlir::Type type) const;
    vpux::Strides getStrides(mlir::Type type) const;
    vpux::MemStrides getMemStrides(mlir::Type type) const;
    vpux::Bit getElemTypeSize(mlir::Type type) const;
    vpux::Byte getTotalAllocSize(mlir::Type type) const;
    vpux::Byte getCompactAllocSize(mlir::Type type) const;
    vpux::NDTypeInterface changeShape(mlir::Type type, vpux::ShapeRef shape) const;
    vpux::NDTypeInterface changeElemType(mlir::Type type, mlir::Type elemType) const;
    vpux::NDTypeInterface changeShapeElemType(mlir::Type type, vpux::ShapeRef shape, mlir::Type elemType) const;
    vpux::NDTypeInterface changeDimsOrder(mlir::Type type, const vpux::DimsOrder& order) const;
    vpux::NDTypeInterface changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const;
    vpux::NDTypeInterface changeStrides(mlir::Type type, vpux::StridesRef strides) const;
    vpux::NDTypeInterface changeTypeComponents(mlir::Type type, const vpux::TypeComponents& typeComponents) const;
    vpux::NDTypeInterface extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape) const;
    vpux::NDTypeInterface extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                          vpux::ShapeRef tileElemStrides) const;
    vpux::NDTypeInterface eraseTiledInfo(mlir::Type type) const;
    vpux::NDTypeInterface pad(mlir::Type type, vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) const;
};

}  // namespace vpux
