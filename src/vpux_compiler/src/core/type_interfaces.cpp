//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/type_interfaces.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/QuantTypes.h>

using namespace vpux;

//
// TypeComponents
//

TypeComponents& TypeComponents::setShape(ShapeRef newShape) {
    shape = newShape;
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
TypeComponents& TypeComponents::setStrides(StridesRef newStrides) {
    strides = newStrides;
    return *this;
}

//
// Generated
//

#include <vpux/compiler/core/generated/type_interfaces.cpp.inc>

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
    return tensor.getNumElements();
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
    return DimsOrder::fromAffineMap(IE::getOrder(tensor));
}

vpux::IndexedSymbolAttr TensorNDTypeInterface::getMemSpace(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemSpace'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    return IE::getMemorySpace(tensor);
}

vpux::VPU::MemoryKind TensorNDTypeInterface::getMemoryKind(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemoryKind'. Got '{0}'", type);
    const auto memSpace = getMemSpace(type);

    if (memSpace == nullptr) {
        return vpux::VPU::MemoryKind::DDR;
    }

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides TensorNDTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getStrides'. Got '{0}'", type);
    const auto memStrides = getMemStrides(type);
    const auto order = getDimsOrder(type);
    return order.toLogicalOrder(memStrides);
}

vpux::MemStrides TensorNDTypeInterface::getMemStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getMemStrides'. Got '{0}'", type);
    const auto tensor = type.cast<mlir::RankedTensorType>();
    const auto order = getDimsOrder(type);
    // Tensors are always compact
    return StrideReqs::compact(order.numDims()).calcStrides(order, tensor);
}

vpux::Bit TensorNDTypeInterface::getElemTypeSize(mlir::Type type) const {
    return vpux::getElemTypeSize(type);
}

vpux::Byte TensorNDTypeInterface::getTotalAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getTotalAllocSize'. Got '{0}'", type);
    if (getRank(type) == 0) {
        return getElemTypeSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    return Byte(memStrides.front() * memShape.front());
}

vpux::Byte TensorNDTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'getCompactAllocSize'. Got '{0}'", type);
    const auto typeSize = static_cast<Bit>(getElemTypeSize(type));
    if (getRank(type) == 0) {
        return typeSize;
    }

    const auto shape = getShape(type);
    return shape.totalSize() * typeSize;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeShape'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto elemType = getElementType(type);
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = vpux::getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(type), shape);
        if (axis.hasValue()) {
            elemType = changeAxis(perAxisType, axis.getValue());
        }
    }
    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType =
            llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                    .Case<mlir::RankedTensorType>([&](mlir::RankedTensorType) {
                        return vpux::getTensorType(getShape(type), elemType, getDimsOrder(type), getMemSpace(type));
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
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto newType = vpux::getTensorType(shape, elemType, newOrder, getMemSpace(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface TensorNDTypeInterface::changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeDimsOrder'. Got '{0}'", type);
    return vpux::getTensorType(getShape(type), getElementType(type), order, getMemSpace(type));
}

vpux::NDTypeInterface TensorNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'changeMemSpace'. Got '{0}'", type);
    return vpux::getTensorType(getShape(type), getElementType(type), getDimsOrder(type), memSpace);
}

vpux::NDTypeInterface TensorNDTypeInterface::changeStrides(mlir::Type /*type*/, vpux::StridesRef /*strides*/) const {
    VPUX_THROW("Tensors only support compact strides");
}

vpux::NDTypeInterface TensorNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  vpux::TypeComponents typeComponents) const {
    const auto shape = typeComponents.shape.getValueOr(getShape(type));
    const auto elementType = typeComponents.elementType.getValueOr(getElementType(type));
    const auto dimsOrder = typeComponents.dimsOrder.getValueOr(getDimsOrder(type));
    const auto memSpace = typeComponents.memSpace.getValueOr(getMemSpace(type));
    return vpux::getTensorType(shape, elementType, dimsOrder, memSpace);
}

vpux::NDTypeInterface TensorNDTypeInterface::extractDenseTile(mlir::Type type, vpux::ShapeRef tileOffsets,
                                                              vpux::ShapeRef tileShape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::RankedTensorType>(),
                      "Only RankedTensorType is supported for 'extractDenseTile'. Got '{0}'", type);
    auto elemType = getElementType(type);
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto newType = vpux::getTensorType(tileShape, elemType, getDimsOrder(type), getMemSpace(type));

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

    const auto newType = vpux::getTensorType(newShape, elemType, getDimsOrder(type), getMemSpace(type));

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

    auto compressionScheme = VPUIP::getCompressionSchemeAttr(type);
    if (compressionScheme != nullptr) {
        return compressionScheme.getTotalNumElems();
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
    if (const auto descAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
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

    return vpux::VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

vpux::Strides MemRefNDTypeInterface::getStrides(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getStrides'. Got '{0}'", type);

    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();

    if (const auto mapAttr = layout.dyn_cast<mlir::AffineMapAttr>()) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
        if (auto stridesAttr = descAttr.strides()) {
            const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
            const auto elemSize = getElemTypeSize(type);

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
    if (getRank(type) == 0) {
        return getElemTypeSize(type);
    }

    const auto memShape = getMemShape(type);
    const auto memStrides = getMemStrides(type);

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    auto allocSizeByte = Byte(memStrides.front() * memShape.front());

    const auto layout = type.cast<mlir::MemRefType>().getLayout();

    if (const auto memRefAttr = layout.dyn_cast<VPUIP::MemRefAttr>()) {
        const auto compressionScheme = memRefAttr.compressionScheme();
        if (compressionScheme != nullptr) {
            const auto order = getDimsOrder(type);
            const auto compactMemStrides = StrideReqs::compact(order.numDims()).calcStrides(order, type);
            VPUX_THROW_UNLESS(memStrides == compactMemStrides, "Non-compact type is not supported with compression");
            allocSizeByte = compressionScheme.getAllocSize(getElementType(type));
        }

        auto swizzlingScheme = memRefAttr.swizzlingScheme();
        if (!swizzlingScheme || swizzlingScheme.getKey().getInt() == 0) {
            return allocSizeByte;
        }

        // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
        allocSizeByte = Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
    }

    return allocSizeByte;
}

vpux::Byte MemRefNDTypeInterface::getCompactAllocSize(mlir::Type type) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'getCompactAllocSize'. Got '{0}'",
                      type);
    const auto typeSize = static_cast<Bit>(getElemTypeSize(type));
    if (getRank(type) == 0) {
        return typeSize;
    }

    auto compressionScheme = VPUIP::getCompressionSchemeAttr(type);
    if (compressionScheme != nullptr) {
        return compressionScheme.getAllocSize(getElementType(type));
    }

    const auto shape = getShape(type);
    return shape.totalSize() * typeSize;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeShape(mlir::Type type, vpux::ShapeRef shape) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeShape'. Got '{0}'", type);

    const auto origOrder = getDimsOrder(type);
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto memref = type.cast<mlir::MemRefType>();
    const auto layout = memref.getLayout();

    VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr = nullptr;
    VPUIP::CompressionSchemeAttr compressionSchemeAttr = nullptr;
    const auto descAttr = layout.dyn_cast<VPUIP::MemRefAttr>();
    if (descAttr != nullptr) {
        swizzlingSchemeAttr = descAttr.swizzlingScheme();
        compressionSchemeAttr = descAttr.compressionScheme();
    }
    auto newType = vpux::getMemRefType(shape, getElementType(type), newOrder, getMemSpace(type), StridesRef(),
                                       swizzlingSchemeAttr, compressionSchemeAttr);

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeElemType(mlir::Type type, mlir::Type elemType) const {
    auto newType = llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
                           .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                               return vpux::getMemRefType(getShape(type), elemType, getDimsOrder(type),
                                                          getMemSpace(type), StridesRef(), getSwizzlingSchemeAttr(type),
                                                          VPUIP::getCompressionSchemeAttr(type));
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
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    const auto newType = vpux::getMemRefType(shape, elemType, newOrder, getMemSpace(type), StridesRef(),
                                             getSwizzlingSchemeAttr(type), VPUIP::getCompressionSchemeAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeDimsOrder(mlir::Type type, vpux::DimsOrder order) const {
    VPUX_THROW_UNLESS(type.isa<mlir::MemRefType>(), "Only MemRefType is supported for 'changeDimsOrder'. Got '{0}'",
                      type);
    return vpux::getMemRefType(getShape(type), getElementType(type), order, getMemSpace(type), StridesRef(),
                               getSwizzlingSchemeAttr(type), VPUIP::getCompressionSchemeAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeMemSpace(mlir::Type type, vpux::IndexedSymbolAttr memSpace) const {
    return llvm::TypeSwitch<mlir::Type, mlir::ShapedType>(type)
            .Case<mlir::MemRefType>([&](mlir::MemRefType) {
                return vpux::getMemRefType(getShape(type), getElementType(type), getDimsOrder(type), memSpace,
                                           getStrides(type), getSwizzlingSchemeAttr(type),
                                           VPUIP::getCompressionSchemeAttr(type));
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
                               getSwizzlingSchemeAttr(type), VPUIP::getCompressionSchemeAttr(type));
}

vpux::NDTypeInterface MemRefNDTypeInterface::changeTypeComponents(mlir::Type type,
                                                                  vpux::TypeComponents typeComponents) const {
    const auto shape = typeComponents.shape.getValueOr(getShape(type));
    const auto elementType = typeComponents.elementType.getValueOr(getElementType(type));
    const auto dimsOrder = typeComponents.dimsOrder.getValueOr(getDimsOrder(type));
    const auto strides = typeComponents.strides.getValueOr(getStrides(type));
    const auto memSpace = typeComponents.memSpace.getValueOr(getMemSpace(type));
    return vpux::getMemRefType(shape, elementType, dimsOrder, memSpace, strides, getSwizzlingSchemeAttr(type),
                               VPUIP::getCompressionSchemeAttr(type));
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

    auto compressionScheme = VPUIP::getCompressionSchemeAttr(type);
    compressionScheme = VPUIP::tileCompressionScheme(compressionScheme, tileOffsets, tileShape);

    const auto tileType = vpux::getMemRefType(tileShape, tileElemType, order, memSpace, tileStrides,
                                              getSwizzlingSchemeAttr(type), compressionScheme);

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
                               VPUIP::getCompressionSchemeAttr(type));
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
                                             getSwizzlingSchemeAttr(type), VPUIP::getCompressionSchemeAttr(type));

    const auto loc = mlir::UnknownLoc::get(type.getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(), "Got invalid ShapedType '{0}'", newType);

    return newType;
}
