//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <cstdint>
#include <functional>
#include <numeric>

using namespace vpux;

//
// TypeComponents
//

TypeComponents& TypeComponents::setShape(ShapeRef newShape) {
    shape = Shape(newShape.toValues());
    return *this;
}
TypeComponents& TypeComponents::setElementType(mlir::Type newElementType) {
    elementType = newElementType;
    return *this;
}
TypeComponents& TypeComponents::setDimsOrder(DimsOrder newDimsOrder) {
    dimsOrder = newDimsOrder;
    return *this;
}
TypeComponents& TypeComponents::setMemSpace(IndexedSymbolAttr newMemSpace) {
    memSpace = newMemSpace;
    return *this;
}
TypeComponents& TypeComponents::setBounds(mlir::ArrayAttr newBounds) {
    bounds = newBounds;
    return *this;
}
TypeComponents& TypeComponents::setStrides(StridesRef newStrides) {
    strides = Strides(newStrides.toValues());
    return *this;
}

//
// Generated
//

#include <vpux/compiler/dialect/core/type_interfaces.cpp.inc>

//
// TensorNDTypeInterface
//

vpux::ShapeRef TensorNDTypeInterface::getShape(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::ShapeRef>(type)
            .Case<mlir::RankedTensorType, mlir::UnrankedTensorType>([](auto tensor) {
                return vpux::ShapeRef(tensor.getShape());
            })
            .Default([](mlir::Type type) -> vpux::ShapeRef {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::MemShape TensorNDTypeInterface::getMemShape(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemShape'. Got '{0}'", type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool TensorNDTypeInterface::hasRank(mlir::Type type) const {
    return type.isa<mlir::RankedTensorType>();
}

int64_t TensorNDTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return tensor.getRank();
}

int64_t TensorNDTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getNumElements'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    if (tensor.hasStaticShape()) {
        return tensor.getNumElements();
    }
    auto boundedTensorType = type.cast<vpux::BoundedTypeInterface>();
    auto parsedBounds = parseIntArrayAttr<int64_t>(boundedTensorType.getBounds());
    return std::accumulate(parsedBounds.begin(), parsedBounds.end(), 1, std::multiplies<int64_t>());
}

mlir::Type TensorNDTypeInterface::getElementType(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::RankedTensorType, mlir::UnrankedTensorType>([](auto tensor) {
                return tensor.getElementType();
            })
            .Default([](mlir::Type type) -> mlir::Type {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::DimsOrder TensorNDTypeInterface::getDimsOrder(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getDimsOrder'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return DimsOrder::fromAffineMap(vpux::getOrder(tensor));
}

vpux::IndexedSymbolAttr TensorNDTypeInterface::getMemSpace(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemSpace'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return vpux::getMemorySpace(tensor);
}

vpux::VPU::MemoryKind TensorNDTypeInterface::getMemoryKind(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemoryKind'. Got '{0}'", type);
    const auto memSpace = getMemSpace(type);

    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

vpux::Strides TensorNDTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getStrides'. Got '{0}'", type);
    const auto memStrides = getMemStrides(type);
    const auto order = getDimsOrder(type);
    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides TensorNDTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getMemStrides'. Got '{0}'", type);
    auto tensor = mlir::cast<mlir::RankedTensorType>(type);
    const auto order = getDimsOrder(type);
    // Tensors are always compact
    if (!tensor.hasStaticShape()) {
        auto shape = tensor.getShape();
        auto boundedTensorType = mlir::cast<vpux::BoundedTypeInterface>(tensor);
        auto parsedBounds = parseIntArrayAttr<int64_t>(boundedTensorType.getBounds());
        VPUX_THROW_UNLESS(parsedBounds.size() == shape.size(), "Bounds and shape mismatch : {0} vs {1}", parsedBounds,
                          shape);
        auto newTensor =
                vpux::getTensorType(Shape(parsedBounds), tensor.getElementType(), order, getMemSpace(type), nullptr);
        return StrideReqs::compact(order.numDims()).calcStrides(order, newTensor);
    }
    return StrideReqs::compact(order.numDims()).calcStrides(order, tensor);
}

vpux::Bit TensorNDTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte TensorNDTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getTotalAllocSize'. Got '{0}'", type);
    if (getRank(type) == 0) {
        return alignMemSize(getElemTypeSize(type), Byte(1));
    }

    const auto ndType = type.dyn_cast<vpux::NDTypeInterface>();
    if (ndType != nullptr && ndType.getShape().isDynamic()) {
        // Bounded ranked tensors must always be compact.
        return getCompactAllocSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);
    const auto totalSizeBits = alignMemSize(memStrides.front() * memShape.front(), Byte(1));
    return Byte(totalSizeBits);
}

vpux::Byte TensorNDTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getCompactAllocSize'. Got '{0}'", type);
    const Bit typeSize = getElemTypeSize(type);
    if (getRank(type) == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    const auto tensorType = type.cast<mlir::RankedTensorType>();
    if (auto boundsAttr = vpux::getBounds(tensorType)) {
        // TODO: #113258 consider removing this code since getShape will return bounded buffer
        const auto bounds = parseIntArrayAttr<int64_t>(boundsAttr);
        auto totalSize = std::accumulate(bounds.begin(), bounds.end(), 1, std::multiplies<int64_t>());
        VPUX_THROW_WHEN(totalSize <= 0, "Only shapes > 0 are supported for 'getCompactAllocSize'.");
        return totalSize * typeSize;
    }

    const auto shape = getShape(type);
    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeShape'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto elemType = getElementType(type);
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = vpux::getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(type), shape);
        if (axis.has_value()) {
            elemType = changeAxis(perAxisType, axis.value());
        }
    }
    auto boundedType = type.cast<vpux::BoundedTypeInterface>();
    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type), boundedType.getBounds());

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType =
            llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                    .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType) {
                        return vpux::getTensorType(getShape(type), elemType, getDimsOrder(type), getMemSpace(type),
                                                   getBounds(type.cast<mlir::RankedTensorType>()));
                    })
                    .Case<mlir::UnrankedTensorType>([&](mlir::UnrankedTensorType) {
                        return mlir::UnrankedTensorType::get(elemType);
                    })
                    .Default([](mlir::Type type) -> mlir::ShapedType {
                        VPUX_THROW("Unsupported type '{0}'", type);
                    });

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeShapeElemType(mlir::Type type, vpux::ShapeRef shape,
                                                                 mlir::Type elemType) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeShapeElemType'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);
    auto boundedType = type.cast<vpux::BoundedTypeInterface>();
    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type), boundedType.getBounds());

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeDimsOrder'. Got '{0}'", type);

    return vpux::getTensorType(getShape(type), getElementType(type), order, getMemSpace(type),
                               vpux::getBounds(type.cast<mlir::RankedTensorType>()));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeMemSpace'. Got '{0}'", type);
    return vpux::getTensorType(getShape(type), getElementType(type), getDimsOrder(type), memSpace,
                               vpux::getBounds(type.cast<mlir::RankedTensorType>()));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeStrides(mlir::Type /*type*/, vpux::StridesRef /*strides*/) const {
    VPUX_THROW("Tensors only support compact strides");
}

vpux::NDTypeInterface TensorNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  const vpux::TypeComponents& typeComponents) const {
    const auto shape = typeComponents.shape.value_or(Shape(getShape(type).toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType(type));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder(type));
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace(type));

    const auto boundedType = type.dyn_cast<vpux::BoundedTypeInterface>();
    const auto bounds = typeComponents.bounds.value_or((boundedType != nullptr) ? boundedType.getBounds() : nullptr);

    return vpux::getTensorType(shape, elementType, dimsOrder, memSpace, bounds);
}

vpux::NDTypeInterface TensorNDTypeInterface::extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                              vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'extractDenseTile'. Got '{0}'", type);
    auto elemType = getElementType(type);
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto boundedType = type.cast<vpux::BoundedTypeInterface>();
    const auto newType =
            vpux::getTensorType(tileShape, elemType, getDimsOrder(type), getMemSpace(type), boundedType.getBounds());

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::extractViewTile(mlir::Type /*type*/, vpux::ShapeRef /*tileOffsets*/,
                                                             vpux::ShapeRef /*tileShape*/,
                                                             vpux::ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("Tensors only support compact strides");
}

vpux::NDTypeInterface TensorNDTypeInterface::eraseTiledInfo(mlir::Type type) const {
    return type;
}

vpux::NDTypeInterface TensorNDTypeInterface::pad(mlir::Type type, vpux::ShapeRef padBefore,
                                                 vpux::ShapeRef padAfter) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(), "Only RankedTensorType is supported for 'pad'. Got '{0}'",
                      type);
    const auto origShape = getShape(type);

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(), "Got non consistent 'padBefore' and 'padAfter' values");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(), "Paddings and input shape are not consistent");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto elemType = getElementType(type);
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    auto boundedType = type.cast<vpux::BoundedTypeInterface>();
    const auto newType =
            vpux::getTensorType(newShape, elemType, getDimsOrder(type), getMemSpace(type), boundedType.getBounds());

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

//
// MemRefNDTypeInterface
//

vpux::ShapeRef MemRefNDTypeInterface::getShape(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::ShapeRef>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                return vpux::ShapeRef(memref.getShape());
            })
            .Default([](mlir::Type type) -> vpux::ShapeRef {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::MemShape MemRefNDTypeInterface::getMemShape(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getMemShape'. Got '{0}'", type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool MemRefNDTypeInterface::hasRank(mlir::Type type) const {
    return type.isa<mlir::MemRefType>();
}

int64_t MemRefNDTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto memref = type.cast<mlir::MemRefType>();
    return memref.getRank();
}

int64_t MemRefNDTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getNumElements'. Got '{0}'",
                      type);

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    if (sparsityCompression != nullptr) {
        return sparsityCompression.getTotalNumElems();
    }

    const auto memref = type.cast<mlir::MemRefType>();
    return memref.getNumElements();
}

mlir::Type MemRefNDTypeInterface::getElementType(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                return memref.getElementType();
            })
            .Default([](mlir::Type type) -> mlir::Type {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::DimsOrder MemRefNDTypeInterface::getDimsOrder(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getDimsOrder'. Got '{0}'", type);
    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();
    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }
    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }
    VPUX_THROW("Missing layout information");
}

vpux::IndexedSymbolAttr MemRefNDTypeInterface::getMemSpace(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::IndexedSymbolAttr>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                const auto memSpaceAttr = memref.getMemorySpace();
                if (memSpaceAttr == nullptr) {
                    return vpux::IndexedSymbolAttr();
                }

                auto memSpace = memSpaceAttr.template dyn_cast<vpux::IndexedSymbolAttr>();
                VPUX_THROW_UNLESS(memSpace != nullptr, "Unsupported memory space attribute'{0}'", memSpaceAttr);

                return memSpace;
            })
            .Default([](mlir::Type type) -> vpux::IndexedSymbolAttr {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::VPU::MemoryKind MemRefNDTypeInterface::getMemoryKind(mlir::Type type) const {
    const auto memSpace = getMemSpace(type);

    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

vpux::Strides MemRefNDTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getStrides'. Got '{0}'", type);

    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();

    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>()) {
        if (auto stridesAttr = descAttr.strides()) {
            const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
            const Bit elemSize = getElemTypeSize(type);

            return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                               return stride * elemSize;
                                           })));
        }
    }

    // Missing strides specification means compact strides.
    const auto order = getDimsOrder(type);
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(order, memref);

    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides MemRefNDTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getMemStrides'. Got '{0}'",
                      type);
    const auto order = getDimsOrder(type);
    const auto strides = getStrides(type);
    return order.toMemoryOrder(strides);
}

vpux::Bit MemRefNDTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte MemRefNDTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getTotalAllocSize'. Got '{0}'",
                      type);

    const auto layout = type.cast<mlir::MemRefType>().getLayout();
    const auto memRefAttr = layout.dyn_cast<vpux::MemRefAttr>();
    if (memRefAttr) {
        if (auto allocSizeAttr = memRefAttr.allocSize()) {
            return Byte(allocSizeAttr.getInt());
        }
    }

    if (getRank(type) == 0) {
        return alignMemSize(getElemTypeSize(type), Byte(1));
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    vpux::Byte allocSizeByte{0};
    // memStrides.front() being equal to 0 implies a dimension with 0 size which can arise
    // in some transformer models and are essentially an empty tensor. Below algorithm will
    // find largest alloc size which won't work for tensors with one of the dimensions set to 0
    // Possibly can be removed after E#-156188.
    if (memStrides.front().count() != 0) {
        // With DMA fusion we can have front stride smaller than actual buffer size. It can lead to wrong allocations,
        // so we're looking for the largest combination of stride*dim, like in DistributedTensor. CMX is opposite.
        // Because of large inter-cluster stride we can get size larger than actual, so skip first dim
        bool hasInterClusterStride =
                getMemoryKind(type) == vpux::VPU::MemoryKind::CMX_NN && memStrides.front() >= Bit(2_MB);
        const size_t startDim = hasInterClusterStride ? 1 : 0;
        for (size_t i = startDim; i < memShape.size(); ++i) {
            auto newAllocSizeByte = alignMemSize(memStrides[MemDim(i)] * memShape[MemDim(i)], Byte(1)).to<Byte>();
            if (newAllocSizeByte > allocSizeByte) {
                allocSizeByte = newAllocSizeByte;
            }
        }
    }

    if (memRefAttr) {
        const auto sparsityCompression = memRefAttr.hwSpecificField<VPUIP::SparsityCompressionAttr>();
        if (sparsityCompression != nullptr) {
            const auto order = getDimsOrder(type);
            const auto compactMemStrides = StrideReqs::compact(order.numDims()).calcStrides(order, type);
            VPUX_THROW_UNLESS(memStrides == compactMemStrides, "Non-compact type is not supported with compression");
            allocSizeByte = sparsityCompression.getAllocSize(getElementType(type));
        }

        auto swizzlingScheme = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
            // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
            allocSizeByte =
                    Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
        }

        auto compressionTypeAttr = memRefAttr.hwSpecificField<vpux::VPUIP::CompressionStateAttr>();
        if (compressionTypeAttr &&
            ((compressionTypeAttr.getValue() == VPUIP::CompressionState::RuntimeCompressed) ||
             (compressionTypeAttr.getValue() == VPUIP::CompressionState::CompressionCandidate))) {
            allocSizeByte = Byte(updateSizeForCompression(allocSizeByte.count()));
        }
    }

    return allocSizeByte;
}

vpux::Byte MemRefNDTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getCompactAllocSize'. Got '{0}'",
                      type);
    const Bit typeSize = getElemTypeSize(type);
    if (getRank(type) == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    if (sparsityCompression != nullptr) {
        return sparsityCompression.getAllocSize(getElementType(type));
    }

    const auto shape = getShape(type);
    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeShape'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();

    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    VPUIP::SparsityCompressionAttr sparsityCompressionAttr = nullptr;
    mlir::IntegerAttr allocSizeAttr = nullptr;
    VPUIP::CompressionStateAttr compressionStateAttr = nullptr;
    const auto descAttr = layout.dyn_cast<vpux::MemRefAttr>();
    if (descAttr != nullptr) {
        swizzlingSchemeAttr = descAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        sparsityCompressionAttr = descAttr.hwSpecificField<VPUIP::SparsityCompressionAttr>();
        allocSizeAttr = descAttr.allocSize();
        compressionStateAttr = descAttr.hwSpecificField<VPUIP::CompressionStateAttr>();
    }
    auto newType =
            vpux::getMemRefType(shape, getElementType(type), newOrder, getMemSpace(type), StridesRef(),
                                swizzlingSchemeAttr, sparsityCompressionAttr, allocSizeAttr, compressionStateAttr);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType = llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                           .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                               return vpux::getMemRefType(getShape(type), elemType, getDimsOrder(type),
                                                          getMemSpace(type), StridesRef(), getSwizzlingSchemeAttr(type),
                                                          VPUIP::getSparsityCompressionAttr(type),
                                                          getAllocSizeAttr(type), getCompressionStateAttr(type));
                           })
                           .Case<mlir::UnrankedMemRefType>([&](mlir::UnrankedMemRefType) {
                               return mlir::UnrankedMemRefType::get(elemType, getMemSpace(type));
                           })
                           .Default([](mlir::Type type) -> mlir::ShapedType {
                               VPUX_THROW("Unsupported type '{0}'", type);
                           });

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeShapeElemType(mlir::Type type, vpux::ShapeRef shape,
                                                                 mlir::Type elemType) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeShapeElemType'. Got '{0}'",
                      type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto newType = vpux::getMemRefType(shape, elemType, newOrder, getMemSpace(type), StridesRef(),
                                             getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                             getAllocSizeAttr(type), getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeDimsOrder'. Got '{0}'",
                      type);
    return vpux::getMemRefType(getShape(type), getElementType(type), order, getMemSpace(type), StridesRef(),
                               getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                               getAllocSizeAttr(type), getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    return llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), memSpace,
                                           getStrides(type), getSwizzlingSchemeAttr(type),
                                           VPUIP::getSparsityCompressionAttr(type), getAllocSizeAttr(type),
                                           getCompressionStateAttr(type));
            })
            .Case<mlir::UnrankedMemRefType>([&](mlir::UnrankedMemRefType) {
                return mlir::UnrankedMemRefType::get(getElementType(type), memSpace);
            })
            .Default([](mlir::Type type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeStrides(mlir::Type type, vpux::StridesRef strides) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeStrides'. Got '{0}'",
                      type);
    return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), getMemSpace(type), strides,
                               getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                               getAllocSizeAttr(type), getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  const vpux::TypeComponents& typeComponents) const {
    const auto shape = typeComponents.shape.value_or(Shape(getShape(type).toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType(type));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder(type));
    const auto strides = typeComponents.strides.value_or(getStrides(type));
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace(type));
    return vpux::getMemRefType(shape, elementType, dimsOrder, memSpace, strides, getSwizzlingSchemeAttr(type),
                               VPUIP::getSparsityCompressionAttr(type), getAllocSizeAttr(type),
                               getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                              vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'extractDenseTile'. Got '{0}'",
                      type);
    return eraseTiledInfo(extractViewTile(type, tileOffsets, tileShape, {}));
}

vpux::NDTypeInterface MemRefNDTypeInterface::extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                             vpux::ShapeRef tileShape,
                                                             vpux::ShapeRef tileElemStrides) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'extractViewTile'. Got '{0}'",
                      type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);

    auto tileElemType = getElementType(type);
    if (const auto perAxisQType = tileElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides(type);
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    sparsityCompression = VPUIP::tileSparsityCompression(sparsityCompression, tileOffsets, tileShape);

    const auto tileType =
            vpux::getMemRefType(tileShape, tileElemType, order, memSpace, tileStrides, getSwizzlingSchemeAttr(type),
                                sparsityCompression, getAllocSizeAttr(type), getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, tileType).succeeded(), "Got invalid tile type '{0}'", tileType);

    return tileType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::eraseTiledInfo(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'eraseTiledInfo'. Got '{0}'",
                      type);
    const auto shape = getShape(type);
    const auto elemType = getElementType(type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);
    return vpux::getMemRefType(shape, elemType, order, memSpace, StridesRef(), getSwizzlingSchemeAttr(type),
                               VPUIP::getSparsityCompressionAttr(type), getAllocSizeAttr(type),
                               getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::pad(mlir::Type type, vpux::ShapeRef padBefore,
                                                 vpux::ShapeRef padAfter) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'pad'. Got '{0}'", type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);

    const auto origShape = getShape(type);
    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(), "Got non consistent 'padBefore' and 'padAfter' values");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(), "Paddings and input shape are not consistent");

    Shape newShape(origShape.size());
    for (auto ind : irange(newShape.size())) {
        const auto d = Dim(ind);
        newShape[d] = origShape[d] + padBefore[d] + padAfter[d];
    }

    auto newElemType = getElementType(type);
    if (const auto perAxisQType = newElemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        newElemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    const auto newType = vpux::getMemRefType(newShape, newElemType, order, memSpace, /*strides=*/StridesRef(),
                                             getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                             getAllocSizeAttr(type), getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

//
// TensorBoundedTypeInterface
//

vpux::BoundedTypeInterface TensorBoundedTypeInterface::changeBounds(mlir::Type type, mlir::ArrayAttr bounds) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeBounds'. Got '{0}'", type);
    auto ndType = type.cast<vpux::NDTypeInterface>();
    return vpux::getTensorType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(), ndType.getMemSpace(),
                               bounds)
            .dyn_cast<vpux::BoundedTypeInterface>();
}

mlir::ArrayAttr TensorBoundedTypeInterface::getBounds(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getBounds'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return vpux::getBounds(tensor);
}
