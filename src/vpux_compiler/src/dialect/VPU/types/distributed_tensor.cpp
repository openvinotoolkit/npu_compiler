//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// SubElementTypeInterface
//

void VPU::DistributedTensorType::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                                                          llvm::function_ref<void(Type)> walkTypesFn) const {
    walkTypesFn(getElementType());
    if (!getOrder().isIdentity()) {
        walkAttrsFn(getOrder());
    }
    walkAttrsFn(getMemSpace());
    walkAttrsFn(getDistribution());
}

//
// print/parse
//

void VPU::DistributedTensorType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();
    printer << ", " << getOrder();
    printer << ", " << getMemSpace();
    printer << ", {";

    auto distribution = getDistribution();
    printer << "mode = \"" << VPU::stringifyDistributionMode(distribution.mode().getValue()) << "\"";
    if (distribution.num_tiles() != nullptr) {
        printer << ", num_tiles = " << distribution.num_tiles();
    }
    if (distribution.kernel() != nullptr) {
        printer << ", kernel = " << distribution.kernel();
    }
    if (distribution.pads() != nullptr) {
        printer << ", pads = " << distribution.pads();
    }
    if (distribution.strides() != nullptr) {
        printer << ", strides = " << distribution.strides();
    }
    if (distribution.num_clusters() != nullptr) {
        printer << ", num_clusters = " << distribution.num_clusters();
    }
    if (distribution.alignment() != nullptr) {
        printer << ", alignment = " << distribution.alignment();
    }
    printer << "}";

    printer << ">";
}

mlir::Type VPU::DistributedTensorType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).hasValue() && parser.parseXInDimensionList().succeeded()) {
        shape.push_back(dim);
    }

    mlir::Type elemType;
    if (parser.parseType(elemType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    mlir::AffineMapAttr order;
    if (parser.parseAttribute(order)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    IndexedSymbolAttr memSpace;
    if (parser.parseAttribute(memSpace)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }

    // DistributedTensorAttr

    if (parser.parseLBrace()) {
        return Type();
    }

    // DistributionModeAttr

    if (parser.parseKeyword("mode")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    std::string distributionModeStr;
    if (parser.parseKeywordOrString(&distributionModeStr)) {
        return Type();
    }
    const auto distributionMode = VPU::symbolizeDistributionMode(distributionModeStr);
    if (!distributionMode.hasValue()) {
        return Type();
    }
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(parser.getContext(), distributionMode.getValue());

    mlir::ArrayAttr numTiles;
    mlir::ArrayAttr kernel;
    VPU::PaddingAttr pads;
    mlir::ArrayAttr strides;
    mlir::IntegerAttr numClusters;
    mlir::ArrayAttr alignment;

    while (parser.parseOptionalRBrace()) {
        if (parser.parseComma()) {
            return Type();
        }
        std::string attrName;
        if (parser.parseKeywordOrString(&attrName)) {
            return Type();
        }
        if (parser.parseEqual()) {
            return Type();
        }
        if (attrName == "num_tiles") {
            if (parser.parseAttribute(numTiles)) {
                return Type();
            }
        } else if (attrName == "kernel") {
            if (parser.parseAttribute(kernel)) {
                return Type();
            }
        } else if (attrName == "pads") {
            if (parser.parseAttribute(pads)) {
                return Type();
            }
        } else if (attrName == "strides") {
            if (parser.parseAttribute(strides)) {
                return Type();
            }
        } else if (attrName == "num_clusters") {
            if (parser.parseAttribute(numClusters)) {
                return Type();
            }
        } else if (attrName == "alignment") {
            if (parser.parseAttribute(alignment)) {
                return Type();
            }
        } else {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }
    auto distributedAttr = VPU::DistributedTensorAttr::get(distributionModeAttr, numTiles, kernel, pads, strides,
                                                           numClusters, alignment, parser.getContext());
    return static_cast<mlir::Type>(
            get(parser.getContext(), makeArrayRef(shape), elemType, order, memSpace, distributedAttr));
}

//
// verify
//

mlir::LogicalResult VPU::DistributedTensorType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       ::llvm::ArrayRef<int64_t> shape, mlir::Type /*elementType*/,
                                                       mlir::AffineMapAttr /*order*/, IndexedSymbolAttr /*memSpace*/,
                                                       DistributedTensorAttr distribution) {
    return VPU::verify(emitError, distribution, shape);
}

//
// getCompactType
//

mlir::RankedTensorType VPU::DistributedTensorType::getCompactType() const {
    return mlir::RankedTensorType::get(getShape().raw(), getElementType(),
                                       IE::TensorAttr::get(getOrder(), getMemSpace(), getContext()));
}

//
// Shape utils
//

// @brief Retrieve the array of compute shapes.
// @warning An important thing to consider with regards to compute shapes,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
// In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
// tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
// [1, 16, 4, 4], which is needed when tiling and generating workloads,
// while the allocated shape is [1, 64, 4, 4] (because of duplicated)
// information which is needed for scheduler and strategy manager,
// in order to estimate memory
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapes() const {
    return VPU::getPerClusterComputeShapes(getShape(), getDistribution());
}

// @brief Retrieve the offsets for each compute shape with regards to full tensor shape.
// @warning An important thing to consider with regards to compute offsets,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapeOffsets() const {
    return VPU::getPerClusterComputeShapeOffsets(getShape(), getDistribution());
}

// @brief Get largest compact compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPU::DistributedTensorType::getLargestCompactShape() const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    return *std::max_element(tiledComputeShapes.begin(), tiledComputeShapes.end(), [](ShapeRef a, ShapeRef b) {
        return vpux::details::calcTotalShapeSize(a.raw()) < vpux::details::calcTotalShapeSize(b.raw());
    });
}

// @brief Get the compact compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPU::DistributedTensorType::getCompactShape(int64_t tileInd) const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(tiledComputeShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return tiledComputeShapes[tileInd];
}

// @brief Retrieve the array of padding for each cluster
// @warning This function is needed for getting padding in OVERLAPPED mode.
SmallVector<PadInfo> VPU::DistributedTensorType::getPerClusterPadding() const {
    return VPU::getPerClusterPadding(getDistribution());
}

// @brief Retrieve the array of strided compute shapes
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
SmallVector<StridedShape> VPU::DistributedTensorType::getPerClusterStridedShapes() const {
    const auto strideInReqs = StrideReqs::compact(getShape().size());
    VPUX_THROW_UNLESS(strideInReqs.checkStrides(*this), "Only compact strides are supported");
    return VPU::getPerClusterStridedShapes(getShape(), getStrides(), getDimsOrder(), getDistribution());
}

// @brief Get largest strided compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getLargestStridedShape() const {
    const auto stridedShapeSize = [](const StridedShape& stridedShape) {
        return stridedShape.shape.front() * stridedShape.strides.front();
    };

    const auto stridedShapes = getPerClusterStridedShapes();
    VPUX_THROW_UNLESS(!stridedShapes.empty(), "Missing per-cluster strided shapes");
    return *std::max_element(stridedShapes.begin(), stridedShapes.end(),
                             [&](const StridedShape& a, const StridedShape& b) {
                                 return stridedShapeSize(a) < stridedShapeSize(b);
                             });
}

// @brief Get the strided compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getStridedShape(int64_t tileInd) const {
    const auto stridedShapes = getPerClusterStridedShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(stridedShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return stridedShapes[tileInd];
}

//
// NDTypeInterface
//

MemShape VPU::DistributedTensorType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool VPU::DistributedTensorType::hasRank() const {
    return true;
}

int64_t VPU::DistributedTensorType::getRank() const {
    return static_cast<int64_t>(getShape().size());
}

DimsOrder VPU::DistributedTensorType::getDimsOrder() const {
    return DimsOrder::fromAffineMap(getOrder().getValue());
}

int64_t VPU::DistributedTensorType::getNumElements() const {
    return vpux::details::calcTotalShapeSize(getShape().raw());
}

VPU::MemoryKind VPU::DistributedTensorType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }
    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).getValue();
}

Strides VPU::DistributedTensorType::getStrides() const {
    const auto mapAttr = getOrder();
    VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", mapAttr);

    const auto memStrides = getMemStrides();
    const auto order = getDimsOrder();
    return order.toLogicalOrder(memStrides);
}

MemStrides VPU::DistributedTensorType::getMemStrides() const {
    const auto order = getDimsOrder();
    // Tensors are always compact
    const auto elemSize = getElemTypeSize();
    const auto shape = getShape();
    const auto memShape = order.toMemoryOrder(shape);
    return StrideReqs::compact(order.numDims()).calcStrides(elemSize, memShape);
}

Bit VPU::DistributedTensorType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPU::DistributedTensorType::getTotalAllocSize() const {
    // Tensors are always compact
    return getCompactAllocSize();
}

Byte VPU::DistributedTensorType::getCompactAllocSize() const {
    auto shape = getShape();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.mode();

    // DUPLICATED|MULTICASTED takes priority since it means that each cluster will have the entire
    // tensor, regardless whether it's tiled or not.
    Shape tiledShape;
    if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::MULTICASTED)) {
        tiledShape = Shape(shape.raw());
    } else if (VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContains(distributionMode.getValue(), VPU::DistributionMode::OVERLAPPED)) {
        tiledShape = getLargestCompactShape();
    } else {
        // No distribution mode.
        tiledShape = Shape(shape.raw());
    }

    if (distribution.alignment() != nullptr) {
        const auto alignment = parseIntArrayAttr<int64_t>(distribution.alignment());
        const auto optionalAlignment = Optional<ArrayRef<int64_t>>(alignment);
        tiledShape = Shape(alignShape(tiledShape.raw(), optionalAlignment));
    }

    const auto totalSize = vpux::details::calcTotalShapeSize(tiledShape.raw());
    const auto elemSize = getElemTypeSize();
    const auto byteSize = static_cast<int64_t>(CHAR_BIT);
    if (elemSize.count() < byteSize) {
        return Byte(vpux::divUp(totalSize, byteSize));
    }

    return Byte(elemSize) * totalSize;
}

NDTypeInterface VPU::DistributedTensorType::changeShape(ShapeRef shape) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      getDimsOrder(), shape);
    auto elemType = getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
        if (axis.hasValue()) {
            elemType = changeAxis(perAxisType, axis.getValue());
        }
    }
    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeElemType(mlir::Type elemType) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      getDimsOrder(), shape);
    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeDimsOrder(DimsOrder order) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), getElementType(),
                                           mlir::AffineMapAttr::get(order.toAffineMap(getContext())), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), getElementType(), getOrder(), memSpace,
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("DistributedTensorType only supports compact strides");
}

NDTypeInterface VPU::DistributedTensorType::changeTypeComponents(TypeComponents typeComponents) const {
    const auto shape = typeComponents.shape.getValueOr(getShape());
    const auto dimsOrder = typeComponents.dimsOrder.getValueOr(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.getValueOr(getMemSpace());

    auto elementType = getElementType();
    if (typeComponents.elementType.hasValue()) {
        elementType = typeComponents.elementType.getValue();
    } else {
        if (auto perAxisType = elementType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
            if (axis.hasValue()) {
                elementType = changeAxis(perAxisType, axis.getValue());
            }
        }
    }

    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elementType,
                                           mlir::AffineMapAttr::get(dimsOrder.toAffineMap(getContext())), memSpace,
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    auto elemType = getElementType();
    if (const auto perAxisQType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    return VPU::DistributedTensorType::get(getContext(), tileShape.raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::extractViewTile(vpux::ShapeRef /*tileOffsets*/,
                                                            vpux::ShapeRef /*tileShape*/,
                                                            vpux::ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("DistributedTensorType only supports compact strides");
}

NDTypeInterface VPU::DistributedTensorType::eraseTiledInfo() const {
    return *this;
}

NDTypeInterface VPU::DistributedTensorType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for DistributedTensorType");
}
