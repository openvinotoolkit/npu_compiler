//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/compression_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/memref_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vpux/utils/core/mem_size.hpp>

using namespace vpux;

//
// TypeComponents
//

TypeComponents& TypeComponents::setShape(ShapeRef newShape) {
    shape = Shape(newShape.toValues());
    return *this;
}
TypeComponents& TypeComponents::setShapeWithRepresentation(Shape&& newShape) {
    shape = std::move(newShape);
    return *this;
}
TypeComponents& TypeComponents::setShapeWithRepresentation(BoundedShape&& newShape) {
    shape = newShape.toShape();
    setBounds(newShape.toRepresentation());
    return *this;
}
TypeComponents& TypeComponents::setShapeWithRepresentation(DimsMaskedShape&& newShape) {
    shape = newShape.toReifiedShape();
    setDynamicDimsMask(newShape.toRepresentation());
    return *this;
}
TypeComponents& TypeComponents::setElementType(mlir::Type newElementType) {
    elementType = newElementType;
    return *this;
}
TypeComponents& TypeComponents::setDimsOrder(const DimsOrder& newDimsOrder) {
    dimsOrder = newDimsOrder;
    return *this;
}
TypeComponents& TypeComponents::setMemSpace(IndexedSymbolAttr newMemSpace) {
    memSpace = newMemSpace;
    return *this;
}
TypeComponents& TypeComponents::setBounds(Bounds&& newBounds) {
    bounds = std::move(newBounds);
    if (!bounds->empty()) {
        dynamicDimsMask = DynamicDimsMask{};
    }
    return *this;
}
TypeComponents& TypeComponents::setDynamicDimsMask(DynamicDimsMask&& newDynamicDimsMask) {
    dynamicDimsMask = std::move(newDynamicDimsMask);
    if (!dynamicDimsMask->empty()) {
        bounds = Bounds{};
    }
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getMemShape'. Got '{0}'", type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool TensorNDTypeInterface::hasRank(mlir::Type type) const {
    return mlir::isa<mlir::RankedTensorType>(type);
}

int64_t TensorNDTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto tensor = mlir::cast<mlir::RankedTensorType>(type);
    return tensor.getRank();
}

int64_t TensorNDTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getNumElements'. Got '{0}'", type);

    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        auto bounds = boundedType.getBounds();
        return std::accumulate(bounds.begin(), bounds.end(), 1, std::multiplies<int64_t>());
    }

    const auto rankedType = mlir::cast<mlir::RankedTensorType>(type);
    return rankedType.getNumElements();
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getDimsOrder'. Got '{0}'", type);
    const auto tensor = mlir::cast<mlir::RankedTensorType>(type);
    return DimsOrder::fromAffineMap(vpux::getOrder(tensor));
}

vpux::IndexedSymbolAttr TensorNDTypeInterface::getMemSpace(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getMemSpace'. Got '{0}'", type);
    const auto tensor = mlir::cast<mlir::RankedTensorType>(type);
    return vpux::getMemorySpace(tensor);
}

vpux::VPU::MemoryKind TensorNDTypeInterface::getMemoryKind(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
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
    const auto order = getDimsOrder(type);
    const auto memStrides = getMemStrides(type);
    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides TensorNDTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getMemStrides'. Got '{0}'", type);
    auto tensor = mlir::cast<mlir::RankedTensorType>(type);
    const auto order = getDimsOrder(type);

    // Tensors are always compact
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        auto shape = tensor.getShape();
        auto bounds = boundedType.getBounds();
        VPUX_THROW_UNLESS(bounds.size() == shape.size(), "Bounds and shape mismatch : {0} vs {1}", bounds, shape);
        auto newTensor = vpux::getTensorType(ShapeRef(bounds.raw()), tensor.getElementType(), order, getMemSpace(type),
                                             getBounds(type), getDynamicDimsMask(type));
        return StrideReqs::compact(order.numDims()).calcStrides(order, newTensor);
    }

    return StrideReqs::compact(order.numDims()).calcStrides(order, tensor);
}

vpux::Bit TensorNDTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte TensorNDTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getTotalAllocSize'. Got '{0}'", type);
    if (getRank(type) == 0) {
        return alignMemSize(getElemTypeSize(type), Byte(1));
    }

    const auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'getCompactAllocSize'. Got '{0}'", type);
    const Bit typeSize = getElemTypeSize(type);
    if (getRank(type) == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    const auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(tensorType)) {
        // TODO: #113258 consider removing this code since getShape will return bounded buffer
        auto bounds = boundedType.getBounds();
        auto totalSize = std::accumulate(bounds.begin(), bounds.end(), 1, std::multiplies<int64_t>());
        VPUX_THROW_WHEN(totalSize <= 0, "Only shapes > 0 are supported for 'getCompactAllocSize'.");
        return totalSize * typeSize;
    }

    const auto shape = getShape(type);
    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'changeShape'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto elemType = getElementType(type);
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto axis = vpux::getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(type), shape);
        if (axis.has_value()) {
            elemType = changeAxis(perAxisType, axis.value());
        }
    }

    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type), getBounds(type),
                                             getDynamicDimsMask(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType = llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                           .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType) {
                               return vpux::getTensorType(getShape(type), elemType, getDimsOrder(type),
                                                          getMemSpace(type), getBounds(type), getDynamicDimsMask(type));
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'changeShapeElemType'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type), getBounds(type),
                                             getDynamicDimsMask(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeDimsOrder(mlir::Type type, const vpux::DimsOrder& order) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'changeDimsOrder'. Got '{0}'", type);

    return vpux::getTensorType(getShape(type), getElementType(type), order, getMemSpace(type), getBounds(type),
                               getDynamicDimsMask(type));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'changeMemSpace'. Got '{0}'", type);

    return vpux::getTensorType(getShape(type), getElementType(type), getDimsOrder(type), memSpace, getBounds(type),
                               getDynamicDimsMask(type));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeStrides(mlir::Type /*type*/, vpux::StridesRef /*strides*/) const {
    VPUX_THROW("Tensors only support compact strides");
}

vpux::NDTypeInterface TensorNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  const vpux::TypeComponents& typeComponents) const {
    // Note: *Ref is OK since both branches return non-Refs with longer lifetime.
    const ShapeRef shape = typeComponents.shape.has_value() ? ShapeRef(*typeComponents.shape) : getShape(type);
    const auto elementType =
            typeComponents.elementType.has_value() ? *typeComponents.elementType : getElementType(type);
    const auto dimsOrder = typeComponents.dimsOrder.has_value() ? *typeComponents.dimsOrder : getDimsOrder(type);
    const auto memSpace = typeComponents.memSpace.has_value() ? *typeComponents.memSpace : getMemSpace(type);
    const BoundsRef bounds =
            typeComponents.bounds.has_value() ? BoundsRef(typeComponents.bounds.value()) : getBounds(type);
    const DynamicDimsMaskRef dynamicDimsMask = typeComponents.dynamicDimsMask.has_value()
                                                       ? DynamicDimsMaskRef(typeComponents.dynamicDimsMask.value())
                                                       : getDynamicDimsMask(type);

    return vpux::getTensorType(shape, elementType, dimsOrder, memSpace, bounds, dynamicDimsMask);
}

vpux::NDTypeInterface TensorNDTypeInterface::extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                              vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'extractDenseTile'. Got '{0}'", type);

    return callOnShapeOf(type, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);

        for (auto ind : irange(inShape.size())) {
            const auto d = Dim(ind);
            if (tileShape[d] != mlir::ShapedType::kDynamic) {
                outShape[d] = tileShape[d];
            }
        }

        auto [outStaticShape, outBounds, outDimMask] = splitShapeAndRepresentation(outShape);

        auto elemType = getElementType(type);
        if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
        }

        const auto newType = vpux::getTensorType(outStaticShape, elemType, getDimsOrder(type), getMemSpace(type),
                                                 outBounds, outDimMask);

        const auto loc = mlir::UnknownLoc::get(type.getContext());
        VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'",
                          newType);

        return newType;
    });
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::RankedTensorType>(type),
                      "Only RankedTensorType is supported for 'pad'. Got '{0}'", type);
    const auto origShape = getShape(type);

    VPUX_THROW_UNLESS(padBefore.size() == padAfter.size(), "Got non consistent 'padBefore' and 'padAfter' values");
    VPUX_THROW_UNLESS(origShape.size() == padBefore.size(), "Paddings and input shape are not consistent");

    return callOnShapeOf(type, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        for (auto ind : irange(inShape.size())) {
            const auto d = Dim(ind);
            outShape[d] = inShape[d] + padBefore[d] + padAfter[d];
        }
        auto [outStaticShape, outBounds, outDimMask] = splitShapeAndRepresentation(outShape);

        auto elemType = getElementType(type);
        if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
            elemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
        }

        const auto newType = vpux::getTensorType(outStaticShape, elemType, getDimsOrder(type), getMemSpace(type),
                                                 outBounds, outDimMask);

        const auto loc = mlir::UnknownLoc::get(type.getContext());
        VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'",
                          newType);

        return newType;
    });
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'getMemShape'. Got '{0}'",
                      type);
    const auto dimsOrder = getDimsOrder(type);
    const auto shape = getShape(type);
    return dimsOrder.toMemoryOrder(shape);
}

bool MemRefNDTypeInterface::hasRank(mlir::Type type) const {
    return mlir::isa<mlir::MemRefType>(type);
}

int64_t MemRefNDTypeInterface::getRank(mlir::Type type) const {
    VPUX_THROW_UNLESS(hasRank(type), "Type '{0}' has no rank", type);
    const auto memref = mlir::cast<mlir::MemRefType>(type);
    return memref.getRank();
}

int64_t MemRefNDTypeInterface::getNumElements(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'getNumElements'. Got '{0}'",
                      type);

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    if (sparsityCompression != nullptr) {
        return sparsityCompression.getTotalNumElems();
    }

    const auto memref = mlir::cast<mlir::MemRefType>(type);
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'getDimsOrder'. Got '{0}'",
                      type);
    const auto memref = mlir::cast<mlir::MemRefType>(type);
    const auto layout = memref.getLayout();
    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }
    if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    // return default order if no layout is specified
    return DimsOrder::fromAffineMap(mlir::AffineMap::getMultiDimIdentityMap(getRank(type), type.getContext()));
}

vpux::IndexedSymbolAttr MemRefNDTypeInterface::getMemSpace(mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, vpux::IndexedSymbolAttr>(type)
            .Case<mlir::MemRefType, mlir::UnrankedMemRefType>([](auto memref) {
                const auto memSpaceAttr = memref.getMemorySpace();
                if (memSpaceAttr == nullptr) {
                    return vpux::IndexedSymbolAttr();
                }

                auto memSpace = mlir::dyn_cast<vpux::IndexedSymbolAttr>(memSpaceAttr);
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'getStrides'. Got '{0}'",
                      type);

    const auto memref = mlir::cast<mlir::MemRefType>(type);
    const auto layout = memref.getLayout();

    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'getMemStrides'. Got '{0}'",
                      type);
    const auto order = getDimsOrder(type);
    const auto strides = getStrides(type);
    return order.toMemoryOrder(strides);
}

vpux::Bit MemRefNDTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte MemRefNDTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'getTotalAllocSize'. Got '{0}'", type);

    const auto shape = getShape(type);
    const auto layout = mlir::cast<mlir::MemRefType>(type).getLayout();
    const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout);
    if (memRefAttr) {
        if (auto allocSizeAttr = memRefAttr.allocSize()) {
            return Byte(allocSizeAttr.getInt());
        }
    }

    if (getRank(type) == 0) {
        return alignMemSize(getElemTypeSize(type), Byte(1));
    }

    auto memShape = getMemShape(type);
    if (shape.isDynamic()) {
        BoundsRef bounds = {};
        if (memRefAttr != nullptr) {
            bounds = memRefAttr.bounds();
        }

        VPUX_THROW_WHEN(bounds.empty(), "Can't compute total allocation size for dynamic shape '{0}' without bounds",
                        shape);
        VPUX_THROW_UNLESS(bounds.size() == shape.size(), "Bounds '{0}' don't match shape '{1}'", bounds, shape);
        memShape = getDimsOrder(type).toMemoryOrder(ShapeRef(bounds.raw()));
    }

    MemStrides memStrides;
    if (shape.isDynamic() && memRefAttr != nullptr && memRefAttr.strides() == nullptr) {
        const auto order = getDimsOrder(type);
        memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(type), memShape);
    } else {
        memStrides = getMemStrides(type);
    }

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    vpux::Byte allocSizeByte{0};
    // memStrides.front() being equal to 0 implies a dimension with 0 size which can arise
    // in some transformer models and are essentially an empty tensor. Below algorithm will
    // find largest alloc size which won't work for tensors with one of the dimensions set to 0
    // Possibly can be removed after E#-156188.
    if (memStrides.front().count() != 0) {
        // With DMA fusion we can have front stride smaller than actual buffer size. It can lead to wrong
        // allocations, so we're looking for the largest combination of stride*dim, like in DistributedTensor. CMX
        // is opposite. Because of large inter-cluster stride we can get size larger than actual, so skip first dim
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
            const auto compactMemStrides =
                    shape.isDynamic()
                            ? StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(type), memShape)
                            : StrideReqs::compact(order.numDims()).calcStrides(order, type);
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'getCompactAllocSize'. Got '{0}'", type);
    const Bit typeSize = getElemTypeSize(type);
    const auto shape = getShape(type);

    if (getRank(type) == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    VPUX_THROW_WHEN(sparsityCompression != nullptr && shape.isDynamic(),
                    "Sparsity and dynamic shapes type: {0} cannot be combined at the moment", type);
    if (sparsityCompression != nullptr) {
        return sparsityCompression.getAllocSize(getElementType(type));
    }

    BoundsRef bounds = {};
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(mlir::cast<mlir::MemRefType>(type).getLayout())) {
        bounds = memRefAttr.bounds();
    }

    if (shape.isDynamic()) {
        VPUX_THROW_WHEN(bounds.empty(), "Can't compute compact allocation size for dynamic shape '{0}' without bounds",
                        shape);
        VPUX_THROW_UNLESS(bounds.size() == shape.size(), "Bounds '{0}' don't match shape '{1}'", bounds, shape);
    }

    if (!bounds.empty()) {
        const int64_t totalSize = details::calcTotalShapeSize(bounds.raw());
        VPUX_THROW_WHEN(totalSize <= 0, "Only shapes > 0 are supported for 'getCompactAllocSize'.");
        return alignMemSize(totalSize * typeSize, Byte(1));
    }

    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'changeShape'. Got '{0}'",
                      type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto memref = mlir::cast<mlir::MemRefType>(type);
    const auto layout = memref.getLayout();

    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    VPUIP::SparsityCompressionAttr sparsityCompressionAttr = nullptr;
    mlir::IntegerAttr allocSizeAttr = nullptr;
    VPUIP::CompressionStateAttr compressionStateAttr = nullptr;
    const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout);
    if (descAttr != nullptr) {
        swizzlingSchemeAttr = descAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        sparsityCompressionAttr = descAttr.hwSpecificField<VPUIP::SparsityCompressionAttr>();
        allocSizeAttr = descAttr.allocSize();
        compressionStateAttr = descAttr.hwSpecificField<VPUIP::CompressionStateAttr>();
    }
    // E#224862: it is not clear whether bounds should be preserved / changed
    // anyhow when shape is changed; as of now, follow strides semantics, that
    // is, bounds are dropped (since shape changes).
    auto newType =
            vpux::getMemRefType(shape, getElementType(type), newOrder, getMemSpace(type), StridesRef(), BoundsRef(),
                                swizzlingSchemeAttr, sparsityCompressionAttr, allocSizeAttr, compressionStateAttr);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType =
            llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                    .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                        return vpux::getMemRefType(getShape(type), elemType, getDimsOrder(type), getMemSpace(type),
                                                   StridesRef(), getBounds(type), VPUIP::getSwizzlingSchemeAttr(type),
                                                   VPUIP::getSparsityCompressionAttr(type),
                                                   VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));
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
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'changeShapeElemType'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    auto newOrder = inferNewDimsOrder(origOrder, shape.size());
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    // E#224862: similarly to changeShape(), bounds are dropped since shape
    // changes; there is no better semantics at the moment.
    const auto newType =
            vpux::getMemRefType(shape, elemType, newOrder, getMemSpace(type), StridesRef(), BoundsRef(),
                                VPUIP::getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeDimsOrder(mlir::Type type, const vpux::DimsOrder& order) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'changeDimsOrder'. Got '{0}'", type);
    return vpux::getMemRefType(getShape(type), getElementType(type), order, getMemSpace(type), StridesRef(),
                               getBounds(type), VPUIP::getSwizzlingSchemeAttr(type),
                               VPUIP::getSparsityCompressionAttr(type), VPUIP::getAllocSizeAttr(type),
                               VPUIP::getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    return llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType memrefType) {
                // TODO (E#211610): implement for strided layout support for other methods where it makes sense
                if (hasStridedLayout(memrefType)) {
                    return mlir::MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                                                 memrefType.getLayout(), memSpace);
                }
                const auto strides = getStrides(type);
                return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), memSpace, strides,
                                           getBounds(type), VPUIP::getSwizzlingSchemeAttr(type),
                                           VPUIP::getSparsityCompressionAttr(type), VPUIP::getAllocSizeAttr(type),
                                           VPUIP::getCompressionStateAttr(type));
            })
            .Case<mlir::UnrankedMemRefType>([&](mlir::UnrankedMemRefType) {
                return mlir::UnrankedMemRefType::get(getElementType(type), memSpace);
            })
            .Default([](mlir::Type type) -> mlir::ShapedType {
                VPUX_THROW("Unsupported type '{0}'", type);
            });
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeStrides(mlir::Type type, vpux::StridesRef strides) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'changeStrides'. Got '{0}'",
                      type);
    return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), getMemSpace(type), strides,
                               getBounds(type), VPUIP::getSwizzlingSchemeAttr(type),
                               VPUIP::getSparsityCompressionAttr(type), VPUIP::getAllocSizeAttr(type),
                               VPUIP::getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  const vpux::TypeComponents& typeComponents) const {
    // Note: *Ref is OK since both branches return non-Refs with longer lifetime.
    const ShapeRef shape = typeComponents.shape.has_value() ? ShapeRef(*typeComponents.shape) : getShape(type);
    const auto elementType =
            typeComponents.elementType.has_value() ? *typeComponents.elementType : getElementType(type);
    const auto dimsOrder = typeComponents.dimsOrder.has_value() ? *typeComponents.dimsOrder : getDimsOrder(type);
    const auto memSpace = typeComponents.memSpace.has_value() ? *typeComponents.memSpace : getMemSpace(type);
    const auto strides = typeComponents.strides.has_value() ? *typeComponents.strides : getStrides(type);
    const BoundsRef bounds = typeComponents.bounds.has_value() ? *typeComponents.bounds : getBounds(type);
    return vpux::getMemRefType(shape, elementType, dimsOrder, memSpace, strides, bounds,
                               VPUIP::getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                               VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                              vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'extractDenseTile'. Got '{0}'", type);
    return eraseTiledInfo(extractViewTile(type, tileOffsets, tileShape, {}));
}

vpux::NDTypeInterface MemRefNDTypeInterface::extractViewTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                             vpux::ShapeRef tileShape,
                                                             vpux::ShapeRef tileElemStrides) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type),
                      "Only MemRefType is supported for 'extractViewTile'. Got '{0}'", type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);

    auto tileElemType = getElementType(type);
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
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

    auto originalShape = getShape(type);
    auto tileBounds = Bounds(getBounds(type));
    if (!tileBounds.empty()) {
        VPUX_THROW_UNLESS(tileBounds.size() == tileShape.size(), "Tile bounds '{0}' is not aligned with rank '{1}'",
                          tileBounds, tileShape.size());
        VPUX_THROW_UNLESS(tileBounds.size() == originalShape.size(),
                          "Tile bounds '{0}' is not aligned with original rank '{1}'", tileBounds,
                          originalShape.size());

        for (auto ind : irange(tileBounds.size())) {
            const auto d = Dim(ind);
            if (bool isTiledAtThisDim = (tileShape[d] != originalShape[d]); isTiledAtThisDim) {
                assert(tileShape[d] != mlir::ShapedType::kDynamic && "Tiled dimension should not be dynamic");
                tileBounds[d] = tileShape[d];
            }
        }
    }

    auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    sparsityCompression = VPUIP::tileSparsityCompression(sparsityCompression, tileOffsets, tileShape);

    const auto tileType = vpux::getMemRefType(tileShape, tileElemType, order, memSpace, tileStrides, tileBounds,
                                              VPUIP::getSwizzlingSchemeAttr(type), sparsityCompression,
                                              VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, tileType).succeeded(), "Got invalid tile type '{0}'", tileType);

    return tileType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::eraseTiledInfo(mlir::Type type) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'eraseTiledInfo'. Got '{0}'",
                      type);
    const auto shape = getShape(type);
    const auto elemType = getElementType(type);
    const auto order = getDimsOrder(type);
    const auto memSpace = getMemSpace(type);
    // E#224862: it is not yet clear how tile info erasure affects bounds, so
    // follow strides semantics and drop any present bounds.
    return vpux::getMemRefType(shape, elemType, order, memSpace, StridesRef(), BoundsRef(),
                               VPUIP::getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                               VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::pad(mlir::Type type, vpux::ShapeRef padBefore,
                                                 vpux::ShapeRef padAfter) const {
    VPUX_THROW_UNLESS(mlir::isa<mlir::MemRefType>(type), "Only MemRefType is supported for 'pad'. Got '{0}'", type);
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
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(newElemType)) {
        newElemType = expandScalesAndZP(perAxisQType, padBefore, padAfter);
    }

    // E#224862: padding likely affects bounds, however, there's no proper
    // design yet, so follow strides - drop any present bounds.
    const auto newType =
            vpux::getMemRefType(newShape, newElemType, order, memSpace, /*strides=*/StridesRef(), BoundsRef(),
                                VPUIP::getSwizzlingSchemeAttr(type), VPUIP::getSparsityCompressionAttr(type),
                                VPUIP::getAllocSizeAttr(type), VPUIP::getCompressionStateAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}
