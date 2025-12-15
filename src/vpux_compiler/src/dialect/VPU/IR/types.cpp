//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/rewriter.hpp"  // for vpux::getBufferType()

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// VPUDialect::registerTypes
//

void VPU::VPUDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPU/types.cpp.inc>
            >();
}

//
// VPU::DistributedTensorType accessors
//

ShapeRef VPU::DistributedTensorType::getShape() const {
    return ShapeRef(getImpl()->shape);
}

mlir::Type VPU::DistributedTensorType::getElementType() const {
    return getImpl()->elementType;
}

mlir::AffineMapAttr VPU::DistributedTensorType::getOrder() const {
    return getImpl()->order;
}

IndexedSymbolAttr VPU::DistributedTensorType::getMemSpace() const {
    return getImpl()->memSpace;
}

VPU::DistributionInfoAttr VPU::DistributedTensorType::getDistribution() const {
    return getImpl()->distribution;
}

Const::OpaqueI64ElementsAttr VPU::DistributedTensorType::getDynamicDimsMask() const {
    return getImpl()->dynamicDimsMask;
}

mlir::ShapedType VPU::DistributedTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                       mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPU::DistributedTensorType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPU::DistributedTensorType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

mlir::FailureOr<::mlir::bufferization::BufferLikeType> VPU::DistributedTensorType::getBufferType(
        const mlir::bufferization::BufferizationOptions&, llvm::function_ref<mlir::InFlightDiagnostic()>) const {
    return mlir::cast<mlir::bufferization::BufferLikeType>(vpux::getBufferType(*this));
}

mlir::LogicalResult VPU::DistributedTensorType::verifyCompatibleBufferType(
        mlir::bufferization::BufferLikeType bufferType, llvm::function_ref<mlir::InFlightDiagnostic()>) const {
    // Note: in theory, this can be done differently, without needing to convert
    // buffer to tensor
    return mlir::success(*this == reconstructTensorType(bufferType));
}

//
// VPU::SparseTensorType accessors
//

mlir::ShapedType VPU::SparseTensorType::cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                                  mlir::Type elementType) const {
    if (!shape.has_value()) {
        return mlir::cast<vpux::VPU::SparseTensorType>(changeElemType(elementType));
    }
    return mlir::cast<vpux::VPU::SparseTensorType>(changeShapeElemType(ShapeRef(shape.value()), elementType));
}

mlir::FailureOr<::mlir::bufferization::BufferLikeType> VPU::SparseTensorType::getBufferType(
        const mlir::bufferization::BufferizationOptions&, llvm::function_ref<mlir::InFlightDiagnostic()>) const {
    return mlir::cast<mlir::bufferization::BufferLikeType>(vpux::getBufferType(*this));
}

mlir::LogicalResult VPU::SparseTensorType::verifyCompatibleBufferType(
        mlir::bufferization::BufferLikeType bufferType, llvm::function_ref<mlir::InFlightDiagnostic()>) const {
    // Note: in theory, this can be done differently, without needing to convert
    // buffer to tensor
    return mlir::success(*this == reconstructTensorType(bufferType));
}
