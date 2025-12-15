//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/utils/core/error.hpp"

//
// BoundedBuffer
//

namespace vpux {

ShapeRef VPUIP::BoundedBufferType::getShape() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getShape();
}

MemShape VPUIP::BoundedBufferType::getMemShape() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemShape();
}

bool VPUIP::BoundedBufferType::hasRank() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.hasRank();
}

int64_t VPUIP::BoundedBufferType::getRank() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getRank();
}

int64_t VPUIP::BoundedBufferType::getNumElements() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getNumElements();
}

mlir::Type VPUIP::BoundedBufferType::getElementType() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getElementType();
}

DimsOrder VPUIP::BoundedBufferType::getDimsOrder() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getDimsOrder();
}

IndexedSymbolAttr VPUIP::BoundedBufferType::getMemSpace() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemSpace();
}

VPU::MemoryKind VPUIP::BoundedBufferType::getMemoryKind() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemoryKind();
}

Strides VPUIP::BoundedBufferType::getStrides() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getStrides();
}

MemStrides VPUIP::BoundedBufferType::getMemStrides() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemStrides();
}

Bit VPUIP::BoundedBufferType::getElemTypeSize() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getElemTypeSize();
}

Byte VPUIP::BoundedBufferType::getTotalAllocSize() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto shape = mlir::cast<vpux::NDTypeInterface>(getDynamicShape());
    return data.getTotalAllocSize() + shape.getTotalAllocSize();
}

Byte VPUIP::BoundedBufferType::getCompactAllocSize() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto shape = mlir::cast<vpux::NDTypeInterface>(getDynamicShape());
    return data.getCompactAllocSize() + shape.getCompactAllocSize();
}

NDTypeInterface VPUIP::BoundedBufferType::changeShape(ShapeRef shape) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeShape(shape);

    const auto dynamicShape = mlir::cast<vpux::NDTypeInterface>(getDynamicShape());
    const auto newShape = Shape({checked_cast<Shape::ValueType>(shape.size())});
    const auto newDynamicShape = dynamicShape.changeShape(newShape);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeElemType(mlir::Type elemType) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeElemType(elemType);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeShapeElemType(shape, elemType);

    const auto dynamicShape = mlir::cast<vpux::NDTypeInterface>(getDynamicShape());
    const auto newShape = Shape({checked_cast<Shape::ValueType>(shape.size())});
    const auto newDynamicShape = dynamicShape.changeShape(newShape);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeDimsOrder(DimsOrder order) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeDimsOrder(order);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeMemSpace(memSpace);

    const auto dynamicShape = mlir::cast<vpux::NDTypeInterface>(getDynamicShape());
    const auto newDynamicShape = dynamicShape.changeMemSpace(memSpace);

    return VPUIP::BoundedBufferType::get(newData, newDynamicShape);
}

NDTypeInterface VPUIP::BoundedBufferType::changeStrides(StridesRef strides) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.changeStrides(strides);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::changeTypeComponents(const TypeComponents& typeComponents) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto data = ndData.changeTypeComponents(typeComponents);
    return VPUIP::BoundedBufferType::get(data, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.extractDenseTile(tileOffsets, tileShape);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::extractViewTile(ShapeRef tileOffsets, ShapeRef tileShape,
                                                          ShapeRef tileElemStrides) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.extractViewTile(tileOffsets, tileShape, tileElemStrides);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::eraseTiledInfo() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.eraseTiledInfo();
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

NDTypeInterface VPUIP::BoundedBufferType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto newData = data.pad(padBefore, padAfter);
    return VPUIP::BoundedBufferType::get(newData, getDynamicShape());
}

void VPUIP::BoundedBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<data=" << getData() << ", dynamic_shape=" << getDynamicShape() << ">";
}

mlir::Type VPUIP::BoundedBufferType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    mlir::Type data;
    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(data)) {
        return Type();
    }

    if (parser.parseComma()) {
        return Type();
    }

    mlir::Type dynamicShape;
    if (parser.parseKeyword("dynamic_shape")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(dynamicShape)) {
        return Type();
    }

    if (parser.parseGreater()) {
        return Type();
    }

    return get(data, dynamicShape);
}

mlir::LogicalResult VPUIP::BoundedBufferType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                     mlir::Type data, mlir::Type dynamicShape) {
    if (!mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(data)) {
        return printTo(emitError(), "Data type is not a MemRef or DistributedBuffer. Got {0}", data);
    }

    if (!mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(dynamicShape)) {
        return printTo(emitError(), "Dynamic shape type is not a MemRef or DistributedBuffer. Got {0}", dynamicShape);
    }

    return mlir::success();
}

mlir::ShapedType vpux::VPUIP::BoundedBufferType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                           mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPUIP::BoundedBufferType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPUIP::BoundedBufferType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

//
// DistributedTypeInterface
//
bool vpux::VPUIP::BoundedBufferType::containsDistributedTypes() const {
    // If the data is a distributed type, the dynamicShape will be as well
    return mlir::isa<vpux::VPUIP::DistributedBufferType>(getData());
}

SmallVector<mlir::Type> vpux::VPUIP::BoundedBufferType::getDistributedTypes() const {
    SmallVector<mlir::Type> distributedTypes;
    if (mlir::isa<vpux::VPUIP::DistributedBufferType>(getData())) {
        distributedTypes.push_back(getData());
    }
    if (mlir::isa<vpux::VPUIP::DistributedBufferType>(getDynamicShape())) {
        distributedTypes.push_back(getDynamicShape());
    }
    return distributedTypes;
}

NDTypeInterface vpux::VPUIP::BoundedBufferType::changeShapeForExplicitDistribution(
        ShapeRef shape, VPU::DistributionInfoAttr distributedAttr) const {
    return changeShapeElemTypeForExplicitDistribution(shape, getElementType(), distributedAttr);
}

NDTypeInterface vpux::VPUIP::BoundedBufferType::changeShapeElemTypeForExplicitDistribution(
        ShapeRef /*shape*/, mlir::Type /*elemType*/, VPU::DistributionInfoAttr /*distributedAttr*/) const {
    VPUX_THROW("Not implemented");
    return nullptr;
}

NDTypeInterface vpux::VPUIP::BoundedBufferType::changeTypeComponentsForExplicitDistribution(
        const TypeComponents& /*typeComponents*/, VPU::DistributionInfoAttr /*distributedAttr*/) const {
    VPUX_THROW("Not implemented");
    return nullptr;
}

NDTypeInterface vpux::VPUIP::BoundedBufferType::extractDenseTileForExplicitDistribution(
        vpux::ShapeRef /*tileOffsets*/, vpux::ShapeRef /*tileShape*/,
        VPU::DistributionInfoAttr /*distributedAttr*/) const {
    VPUX_THROW("Not implemented");
    return nullptr;
}

NDTypeInterface vpux::VPUIP::BoundedBufferType::extractViewTileForExplicitDistribution(
        vpux::ShapeRef, vpux::ShapeRef, vpux::ShapeRef, VPU::DistributionInfoAttr) const {
    VPUX_THROW("Not implemented");
}

}  // namespace vpux
