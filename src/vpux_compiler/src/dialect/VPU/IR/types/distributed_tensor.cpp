//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

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
    printer << "mode = \"" << VPU::stringifyDistributionMode(distribution.getMode().getValue()) << "\"";
    if (distribution.getNumTiles() != nullptr) {
        printer << ", num_tiles = " << distribution.getNumTiles();
    }
    if (distribution.getKernel() != nullptr) {
        printer << ", kernel = " << distribution.getKernel();
    }
    if (distribution.getPads() != nullptr) {
        printer << ", pads = " << distribution.getPads();
    }
    if (distribution.getStrides() != nullptr) {
        printer << ", strides = " << distribution.getStrides();
    }
    if (distribution.getNumClusters() != nullptr) {
        printer << ", num_clusters = " << distribution.getNumClusters();
    }
    if (distribution.getAlignment() != nullptr) {
        printer << ", alignment = " << distribution.getAlignment();
    }
    if (distribution.getUniformDistributedSegments() != nullptr) {
        printer << ", uniform_distributed_segments";
    }
    if (distribution.getComputeShapes() != nullptr) {
        printer << ", compute_shapes = " << distribution.getComputeShapes();
    }
    if (distribution.getComputeOffsets() != nullptr) {
        printer << ", compute_offsets = " << distribution.getComputeOffsets();
    }
    if (distribution.getMemoryShapes() != nullptr) {
        printer << ", memory_shapes = " << distribution.getMemoryShapes();
    }
    if (distribution.getMemoryOffsets() != nullptr) {
        printer << ", memory_offsets = " << distribution.getMemoryOffsets();
    }
    if (distribution.getEqualMemoryAndComputeView() != nullptr) {
        printer << ", equal_memory_and_compute_view";
    }
    printer << "}";

    if (getDynamicDimsMask() != nullptr) {
        printer << ", dynamic_dims_mask = " << getDynamicDimsMask();
    }

    printer << ">";
}

mlir::Type VPU::DistributedTensorType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }

    SmallVector<int64_t> shape;
    int64_t dim = 0;
    while (parser.parseOptionalInteger(dim).has_value() && parser.parseXInDimensionList().succeeded()) {
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

    // DistributionInfoAttr

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
    if (!distributionMode.has_value()) {
        return Type();
    }
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(parser.getContext(), distributionMode.value());

    mlir::ArrayAttr numTiles;
    mlir::ArrayAttr kernel;
    VPU::PaddingAttr pads;
    mlir::ArrayAttr strides;
    mlir::IntegerAttr numClusters;
    mlir::ArrayAttr alignment;
    mlir::UnitAttr uniformDistributedSegments;
    mlir::ArrayAttr computeShapes;
    mlir::ArrayAttr computeOffsets;
    mlir::ArrayAttr memoryShapes;
    mlir::ArrayAttr memoryOffsets;
    mlir::UnitAttr equalComputeAndMemoryView;

    while (parser.parseOptionalRBrace()) {
        if (parser.parseComma()) {
            return Type();
        }
        std::string attrName;
        if (parser.parseKeywordOrString(&attrName)) {
            return Type();
        }

        // Handle UnitAttr first since they don't have value assigned
        if (attrName == "uniform_distributed_segments") {
            uniformDistributedSegments = mlir::UnitAttr::get(parser.getContext());
            continue;
        }

        if (attrName == "equal_memory_and_compute_view") {
            equalComputeAndMemoryView = mlir::UnitAttr::get(parser.getContext());
            continue;
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
        } else if (attrName == "compute_shapes") {
            if (parser.parseAttribute(computeShapes)) {
                return Type();
            }
        } else if (attrName == "compute_offsets") {
            if (parser.parseAttribute(computeOffsets)) {
                return Type();
            }
        } else if (attrName == "memory_shapes") {
            if (parser.parseAttribute(memoryShapes)) {
                return Type();
            }
        } else if (attrName == "memory_offsets") {
            if (parser.parseAttribute(memoryOffsets)) {
                return Type();
            }
        } else {
            return Type();
        }
    }
    auto distributedAttr =
            VPU::DistributionInfoAttr::get(parser.getContext(), distributionModeAttr, numTiles, kernel, pads, strides,
                                           numClusters, alignment, uniformDistributedSegments, computeShapes,
                                           computeOffsets, memoryShapes, memoryOffsets, equalComputeAndMemoryView);

    if (mlir::succeeded(parser.parseOptionalGreater())) {
        return static_cast<mlir::Type>(
                get(parser.getContext(), ArrayRef(shape), elemType, order, memSpace, distributedAttr));
    }
    //
    // DynamicDimsMask (optional)
    //

    if (parser.parseComma()) {
        return Type();
    }

    if (parser.parseKeyword("dynamic_dims_mask")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }

    Const::OpaqueI64ElementsAttr dynamicDimsMask;
    if (parser.parseAttribute(dynamicDimsMask)) {
        return Type();
    }

    if (parser.parseGreater()) {
        return Type();
    }

    return static_cast<mlir::Type>(
            get(parser.getContext(), ArrayRef(shape), elemType, order, memSpace, distributedAttr, dynamicDimsMask));
}

//
// verify
//

mlir::LogicalResult VPU::DistributedTensorType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                       ::llvm::ArrayRef<int64_t> shape, mlir::Type /*elementType*/,
                                                       mlir::AffineMapAttr /*order*/, IndexedSymbolAttr /*memSpace*/,
                                                       DistributionInfoAttr distribution,
                                                       Const::OpaqueI64ElementsAttr /*dynamicDimsMask*/) {
    return VPU::verify(emitError, distribution, shape);
}

//
// getCompactType
//

mlir::RankedTensorType VPU::DistributedTensorType::getCompactType() const {
    return mlir::RankedTensorType::get(
            getShape().raw(), getElementType(),
            vpux::TensorAttr::get(getContext(), getOrder(), getMemSpace(), /*bounds=*/{}, /*dynamicDimsMask=*/{}));
}

//
// Shape utils
//

// @brief Retrieve the array of compute shapes.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like SEGMENTED and OVERLAPPED take precedence over
//  DUPLICATED and MULTICASTED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
//  [1, 16, 4, 4], which is needed when tiling and generating workloads,
//  while the allocated shape is [1, 64, 4, 4] (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
//  In an example of OVERLAPPED over H with  k3x3s1x1 pad (1, 1, 1, 1) and
//  uniform segmentation for 4 clusters, a tensor of shape [1, 64, 22, 16]
//  will have the following compute distribution across clusters:
//  [1 64 6 16] [1 64 6 16] [1 64 5 16] [1 64 5 16]
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapes() const {
    auto distribution = getDistribution();
    if (distribution.getComputeShapes() == nullptr) {
        return VPU::getPerClusterComputeShapes(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeShapes());
}

// @brief Retrieve the offsets for each compute shape with regards to full tensor shape.
// @warning An important thing to consider with regards to compute offsets,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterComputeShapeOffsets() const {
    auto distribution = getDistribution();
    if (distribution.getComputeOffsets() == nullptr) {
        return VPU::getPerClusterComputeShapeOffsets(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeOffsets());
}

// @brief Retrieve the array of memory shapes.
// @warning An important thing to consider with regards to memory shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the memory shape in each cluster is
//  [1, 64, 4, 4], which is the allocated shape (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
//  In an example of OVERLAPPED over H with k3x3s1x1 pad (1, 1, 1, 1) and
//  uniform segmentation across 4 clusters, a tensor of shape [1, 64, 22, 16]
//  will have the following memory distribution across clusters:
//  [1 64 7 16] [1 64 8 16] [1 64 7 16] [1 64 6 16]
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterMemoryShapes() const {
    auto distribution = getDistribution();
    if (distribution.getMemoryShapes() == nullptr) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Shape {0}, Unsupported distribution: {1}", getShape(),
                          distribution);
        return optionalPerClusterMemoryShapes.value();
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getMemoryShapes());
}

// @brief Retrieve the array of memory buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to memory shape offsets,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
SmallVector<Shape> VPU::DistributedTensorType::getPerClusterMemoryShapeOffsets() const {
    auto distribution = getDistribution();
    if (distribution.getMemoryOffsets() == nullptr) {
        return VPU::getPerClusterMemoryShapeOffsets(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getMemoryOffsets());
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
SmallVector<PadInfo> VPU::DistributedTensorType::getPerClusterPadding(PadInfo kernelPadding) const {
    return VPU::getPerClusterPadding(getDistribution(), kernelPadding);
}

// @brief Retrieve the array of strided memory shapes
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
SmallVector<StridedShape> VPU::DistributedTensorType::getPerClusterMemoryStridedShapes() const {
    const auto strideInReqs = StrideReqs::compact(getShape().size());
    VPUX_THROW_UNLESS(strideInReqs.checkStrides(*this), "Only compact strides are supported");

    const auto memoryShapes = getPerClusterMemoryShapes();
    const auto strides = getStrides();
    return VPU::getPerClusterMemoryStridedShapes(getShape(), strides, getDimsOrder(), getDistribution().getMode(),
                                                 memoryShapes);
}

// @brief Get largest strided memory shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getLargestStridedShape() const {
    const auto stridedShapeSize = [](const StridedShape& stridedShape) {
        return stridedShape.shape.front() * stridedShape.strides.front();
    };

    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(!stridedShapes.empty(), "Missing per-cluster strided shapes");
    return *std::max_element(stridedShapes.begin(), stridedShapes.end(),
                             [&](const StridedShape& a, const StridedShape& b) {
                                 return stridedShapeSize(a) < stridedShapeSize(b);
                             });
}

// @brief Get the strided memory shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPU::DistributedTensorType::getStridedShape(int64_t tileInd) const {
    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(stridedShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return stridedShapes[tileInd];
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offsets adjusted for the new shape.
NDTypeInterface VPU::DistributedTensorType::changeShapeForExplicitDistribution(
        ShapeRef shape, VPU::DistributionInfoAttr distributedAttr) const {
    const auto typeComponents = TypeComponents().setShape(shape);
    return changeTypeComponentsForExplicitDistribution(typeComponents, distributedAttr);
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and element type and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offsets adjusted for the new shape.
NDTypeInterface VPU::DistributedTensorType::changeShapeElemTypeForExplicitDistribution(
        ShapeRef shape, mlir::Type elemType, VPU::DistributionInfoAttr distributedAttr) const {
    const auto typeComponents = TypeComponents().setShape(shape).setElementType(elemType);
    return changeTypeComponentsForExplicitDistribution(typeComponents, distributedAttr);
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested type components. If shape is one of the changed
// components, it will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offsets
// adjusted for the new shape. Otherwise, it leaves the DistributedAttr untouched.
NDTypeInterface VPU::DistributedTensorType::changeTypeComponentsForExplicitDistribution(
        const TypeComponents& typeComponents, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return changeTypeComponents(typeComponents);
    }

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    VPUX_THROW_UNLESS(dimsOrder.numDims() == shape.size(), "Order '{0}' is incompatible with shape '{1}' ", dimsOrder,
                      shape);

    auto elementType = getElementType();
    if (typeComponents.elementType.has_value()) {
        elementType = typeComponents.elementType.value();
    } else {
        if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
            if (axis.has_value()) {
                elementType = changeAxis(perAxisType, axis.value());
            }
        }
    }

    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elementType,
                                           mlir::AffineMapAttr::get(dimsOrder.toAffineMap(getContext())), memSpace,
                                           distributedAttr);
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType obtained by extracting a dense tile from the original DistributedType.
// It will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offsets
// adjusted for the resulting dense tile.
NDTypeInterface VPU::DistributedTensorType::extractDenseTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractDenseTile(tileOffsets, tileShape);
    }

    auto elemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    return VPU::DistributedTensorType::get(getContext(), tileShape.raw(), elemType, getOrder(), getMemSpace(),
                                           distributedAttr);
}

NDTypeInterface VPU::DistributedTensorType::extractViewTileForExplicitDistribution(
        vpux::ShapeRef /*tileOffsets*/, vpux::ShapeRef /*tileShape*/, vpux::ShapeRef /*tileElemStrides*/,
        VPU::DistributionInfoAttr /*distributedAttr*/) const {
    VPUX_THROW("DistributedTensorType only supports compact strides");
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
    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
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
    const Bit elemSize = getElemTypeSize();
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
    const auto perClusterShapes = getPerClusterMemoryShapes();
    const Shape tiledShape =
            *std::max_element(perClusterShapes.begin(), perClusterShapes.end(), [](ShapeRef a, ShapeRef b) {
                return vpux::details::calcTotalShapeSize(a.raw()) < vpux::details::calcTotalShapeSize(b.raw());
            });

    const auto totalSize = vpux::details::calcTotalShapeSize(tiledShape.raw());
    const Bit elemSize = getElemTypeSize();

    return alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>();
}

NDTypeInterface VPU::DistributedTensorType::changeShape(ShapeRef shape) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}' ",
                      getDimsOrder(), shape);

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot change shape when having explicit per cluster shapes/offsets");

    auto elemType = getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
        if (axis.has_value()) {
            elemType = changeAxis(perAxisType, axis.value());
        }
    }
    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
                                           distribution);
}

NDTypeInterface VPU::DistributedTensorType::changeElemType(mlir::Type elemType) const {
    return VPU::DistributedTensorType::get(getContext(), getShape().raw(), elemType, getOrder(), getMemSpace(),
                                           getDistribution());
}

NDTypeInterface VPU::DistributedTensorType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    VPUX_THROW_UNLESS(getDimsOrder().numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}' ",
                      getDimsOrder(), shape);

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot change shape when having explicit per cluster shapes/offsets");

    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elemType, getOrder(), getMemSpace(),
                                           distribution);
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

NDTypeInterface VPU::DistributedTensorType::changeTypeComponents(const vpux::TypeComponents& typeComponents) const {
    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());
    auto distribution = getDistribution();

    // If there is a shape change requested
    if (shape != Shape(getShape().toValues())) {
        VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                        "Cannot change shape when having explicit per cluster shapes/offsets");
    }

    auto elementType = getElementType();
    if (typeComponents.elementType.has_value()) {
        elementType = typeComponents.elementType.value();
    } else {
        if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elementType)) {
            const auto axis = getQuantizedAxis(perAxisType.getQuantizedDimension(), getShape(), shape);
            if (axis.has_value()) {
                elementType = changeAxis(perAxisType, axis.value());
            }
        }
    }

    return VPU::DistributedTensorType::get(getContext(), shape.raw(), elementType,
                                           mlir::AffineMapAttr::get(dimsOrder.toAffineMap(getContext())), memSpace,
                                           distribution);
}

NDTypeInterface VPU::DistributedTensorType::extractDenseTile(vpux::ShapeRef tileOffsets,
                                                             vpux::ShapeRef tileShape) const {
    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot get DistributedTensorType with new shape from old one when having explicit per cluster "
                    "shapes/offsets");

    auto elemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        elemType = tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    return VPU::DistributedTensorType::get(getContext(), tileShape.raw(), elemType, getOrder(), getMemSpace(),
                                           distribution);
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
