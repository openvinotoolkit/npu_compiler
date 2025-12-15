//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUIPDialect::registerTypes
//

void vpux::VPUIP::VPUIPDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUIP/types.cpp.inc>
            >();
}

//
// SparseBufferType::Accessors
//

// Note: clonewith are defined for compliance with ShapedTypeInterface.

mlir::ShapedType vpux::VPUIP::SparseBufferType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                          mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPUIP::SparseBufferType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPUIP::SparseBufferType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

//
// DistributedBufferType::Accessors
//

// Note: clonewith are defined for compliance with ShapedTypeInterface.

mlir::ShapedType vpux::VPUIP::DistributedBufferType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                               mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPUIP::DistributedBufferType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPUIP::DistributedBufferType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

vpux::ShapeRef vpux::VPUIP::DistributedBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::DistributedBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::DistributedBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::DistributedBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributionInfoAttr vpux::VPUIP::DistributedBufferType::getDistribution() const {
    return getImpl()->distribution;
}

VPUIP::SparsityCompressionAttr vpux::VPUIP::DistributedBufferType::getSparsityCompression() const {
    return getImpl()->sparsityCompression;
}

//
// ITIBufferType::Accessors
//

vpux::ShapeRef vpux::VPUIP::ITIBufferType::getShape() const {
    return vpux::ShapeRef(getImpl()->shape);
}

mlir::Type vpux::VPUIP::ITIBufferType::getElementType() const {
    return getImpl()->elementType;
}

mlir::MemRefLayoutAttrInterface vpux::VPUIP::ITIBufferType::getLayout() const {
    return getImpl()->layout;
}

vpux::IndexedSymbolAttr vpux::VPUIP::ITIBufferType::getMemSpace() const {
    return getImpl()->memSpace;
}

mlir::UnitAttr vpux::VPUIP::ITIBufferType::getIduSegmentation() const {
    return getImpl()->iduSegmentation;
}

ArrayRef<vpux::VPUIP::HaloRegionAttr> vpux::VPUIP::ITIBufferType::getInwardHaloRegions() const {
    return getImpl()->inwardHaloRegions;
}

ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> vpux::VPUIP::ITIBufferType::getOutwardHaloRegions() const {
    return getImpl()->outwardHaloRegions;
}
