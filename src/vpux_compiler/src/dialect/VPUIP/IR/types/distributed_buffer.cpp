//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {
vpux::MemRefAttr::HwFields getHwSpecificFields(mlir::MemRefLayoutAttrInterface layout) {
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        return memRefAttr.hwSpecificFields();
    }
    return {};
}

StridedShape alignStridedShape(const StridedShape& stridedTiledShape, VPU::DistributionInfoAttr distribution,
                               const DimsOrder order) {
    if (distribution.getAlignment() == nullptr) {
        return stridedTiledShape;
    }
    const auto alignment = parseIntArrayAttr<int64_t>(distribution.getAlignment());
    const auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    const auto alignedTiledShape =
            Shape(alignShape(stridedTiledShape.shape.raw(), optionalAlignment, alignValUp<int64_t>));
    const auto alignedTiledStrides =
            adaptStrides(stridedTiledShape.shape, stridedTiledShape.strides, {alignedTiledShape}, order);
    return StridedShape(alignedTiledShape, alignedTiledStrides.front());
}

Byte getStridedAllocSize(const StridedShape& stridedTiledShape, ShapeRef stridedTiledOffsets,
                         VPUIP::SparsityCompressionAttr sparsityCompression, const DimsOrder order,
                         const Bit elemBitSize) {
    if (sparsityCompression == nullptr) {
        // the size should be calcuted base on shape and stride based on memory d0
        // if use stridedTiledShape.shape.front() and stridedTiledShape.strides.front()
        // when the order is [d0,d1,d2,d3] -> [d1,d2,d3,d0], the size is not correct.
        return alignMemSize(stridedTiledShape.shape[order.dimAt(0)] * stridedTiledShape.strides[order.dimAt(0)],
                            Byte(1));
    }

    const auto axis = sparsityCompression.getAxis().getInt();
    const auto numElems = sparsityCompression.getNumElems().getValues<int64_t>();
    const int64_t alignment =
            (sparsityCompression.getAlignment() != nullptr) ? sparsityCompression.getAlignment().getInt() : 1;

    const auto startTileIt = numElems.begin() + stridedTiledOffsets[Dim(axis)];
    const auto endTileIt = startTileIt + stridedTiledShape.shape[Dim(axis)];
    int64_t totalBytes = 0;

    for (auto it = startTileIt; it != endTileIt; ++it) {
        auto tileByteSize = alignMemSize(elemBitSize * (*it), Byte(1)).to<Byte>().count();
        totalBytes += alignValUp<int64_t>(tileByteSize, alignment);
    }
    return Byte(totalBytes);
}

// Returns strides attribute only when the provided strides are not compact.
// This is aligned to what getMemRefType() does for mlir::MemRefType.
mlir::ArrayAttr getStridesAttr(mlir::MLIRContext* ctx, StridesRef strides, const DimsOrder& dimsOrder, Bit elemSize,
                               ShapeRef shape) {
    if (strides.empty()) {
        return nullptr;
    }
    const auto memStrides = dimsOrder.toMemoryOrder(strides);
    const auto memShape = dimsOrder.toMemoryOrder(shape);

    const bool isCompact = StrideReqs::compact(shape.size()).checkStrides(memStrides, elemSize, memShape);
    if (isCompact) {
        return nullptr;
    }

    // Have strides only if they are not compact
    const auto elemStrides = to_small_vector(strides | transformed([&](Bit stride) {
                                                 return stride.count() / elemSize.count();
                                             }));
    return getIntArrayAttr(ctx, elemStrides);
}

// Returns either a vpux::MemRefAttr or an mlir::AffineMapAttr (order) depending
// on the specified input: when nothing except order is specified, only order is
// returned. This aligns the behaviour to getMemRefType() that creates a
// mlir::MemRefType.
mlir::MemRefLayoutAttrInterface getMemrefLayout(mlir::AffineMapAttr order, mlir::ArrayAttr optionalStrides,
                                                mlir::IntegerAttr optionalAllocSize,
                                                mlir::ArrayRef<vpux::HwSpecificMemRefField> fields) {
    const bool hwSpecificFieldsEmpty =
            std::all_of(fields.begin(), fields.end(), [](const vpux::HwSpecificMemRefField& field) {
                return field == nullptr;
            });
    if (optionalStrides == nullptr && optionalAllocSize == nullptr && hwSpecificFieldsEmpty) {
        return order;
    }
    return vpux::MemRefAttr::get(order, optionalStrides, optionalAllocSize, fields, order.getContext());
}
}  // namespace

//
// print/parse
//

void VPUIP::DistributedBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    for (auto& dim : getShape()) {
        printer << dim << "x";
    }
    printer << getElementType();

    const auto layout = getLayout();
    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        printer << ", " << mapAttr;
    } else if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        printer << ", " << descAttr;
    } else {
        VPUX_THROW("Unsupported MemRefType layout '{0}'", layout);
    }

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
    if (getSparsityCompression() != nullptr) {
        printer << ", " << getSparsityCompression();
    }

    printer << ">";
}

mlir::Type VPUIP::DistributedBufferType::parse(mlir::AsmParser& parser) {
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

    mlir::MemRefLayoutAttrInterface layout;

    mlir::AffineMapAttr mapAttr;
    vpux::MemRefAttr memRefAttr;
    if (parser.parseOptionalAttribute(mapAttr).has_value()) {
        layout = mapAttr;
    } else if (parser.parseOptionalAttribute(memRefAttr).has_value()) {
        layout = memRefAttr;
    } else {
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
        } else if (attrName == "equal_memory_and_compute_view") {
            equalComputeAndMemoryView = mlir::UnitAttr::get(parser.getContext());
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
        } else if (attrName == "compute_offsets") {
            if (parser.parseAttribute(computeOffsets)) {
                return Type();
            }
        } else if (attrName == "compute_shapes") {
            if (parser.parseAttribute(computeShapes)) {
                return Type();
            }
        } else if (attrName == "memory_offsets") {
            if (parser.parseAttribute(memoryOffsets)) {
                return Type();
            }
        } else if (attrName == "memory_shapes") {
            if (parser.parseAttribute(memoryShapes)) {
                return Type();
            }
        } else {
            return Type();
        }
    }

    VPUIP::SparsityCompressionAttr sparsityCompression;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (parser.parseAttribute(sparsityCompression)) {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }
    auto distributedAttr =
            VPU::DistributionInfoAttr::get(parser.getContext(), distributionModeAttr, numTiles, kernel, pads, strides,
                                           numClusters, alignment, uniformDistributedSegments, computeShapes,
                                           computeOffsets, memoryShapes, memoryOffsets, equalComputeAndMemoryView);
    return static_cast<mlir::Type>(get(parser.getContext(), ArrayRef(shape), elemType, layout, memSpace,
                                       distributedAttr, sparsityCompression));
}

//
// verify
//

mlir::LogicalResult VPUIP::DistributedBufferType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                         ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                                                         mlir::MemRefLayoutAttrInterface layout,
                                                         IndexedSymbolAttr /*memSpace*/,
                                                         VPU::DistributionInfoAttr distribution,
                                                         VPUIP::SparsityCompressionAttr sparsityCompression) {
    if (mlir::failed(VPU::verify(emitError, distribution, shape))) {
        return mlir::failure();
    }

    if (auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        auto compressionStateAttr = descAttr.hwSpecificField<vpux::VPUIP::CompressionStateAttr>();
        if (compressionStateAttr != nullptr) {
            return printTo(emitError(), "Distributed buffer can't be compressed");
        }
    }

    if (sparsityCompression != nullptr) {
        if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
            const Bit elemTypeSize = vpux::getElemTypeSize(elementType);
            if (auto stridesAttr = descAttr.strides()) {
                const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
                const auto strides = Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                                                 return stride * elemTypeSize;
                                                             })));
                const auto order = DimsOrder::fromAffineMap(descAttr.order().getValue());
                const auto memShape = order.toMemoryOrder(Shape(shape));
                const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(elemTypeSize, memShape);
                const auto compactStrides = order.toLogicalOrder(memStrides);
                if (strides != compactStrides) {
                    return printTo(emitError(), "Cannot compress strided buffer");
                }
            }
        }

        const auto distributionMode = distribution.getMode().getValue();
        if (distributionMode != VPU::DistributionMode::SEGMENTED &&
            distributionMode != VPU::DistributionMode::OVERLAPPED) {
            return mlir::success();
        }
        if (sparsityCompression.getAxis() == nullptr) {
            return printTo(emitError(), "Cannot compressed the entire buffer for SEGMENTED/OVERLAPPED modes");
        }
        const auto axis = sparsityCompression.getAxis().getInt();
        if (axis != Dims4D::Filter::OC.ind()) {
            return printTo(emitError(),
                           "Only constants can be compressed and the compression can only be done over OC");
        }
        auto tilesOnAxis = parseIntArrayAttr<int64_t>(distribution.getNumTiles())[axis];
        if (tilesOnAxis == 1) {
            return printTo(emitError(), "Cannot segment and compress buffer on different dimensions");
        }
    }

    return mlir::success();
}

//
// getCompactType
//

mlir::MemRefType VPUIP::DistributedBufferType::getCompactType() const {
    auto distributionMode = getDistribution().getMode().getValue();
    auto swizzlingSchemeAttr = vpux::getSwizzlingSchemeAttr(*this);

    // Compact shapes from SEGMENTED/OVERLAPPED buffers may require more memory after their individual sizeAlignment and
    // thus the producer segmented buffer may require extra adjustment in sizeAlignment
    if (swizzlingSchemeAttr != nullptr && (distributionMode == VPU::DistributionMode::SEGMENTED ||
                                           distributionMode == VPU::DistributionMode::OVERLAPPED)) {
        const auto distributionAttr = getDistribution();
        VPUX_THROW_UNLESS(distributionAttr.getNumClusters() != distributionAttr.getNumTiles(),
                          "Unsupported case to re-align compact buffer with swizzling");

        const auto ctx = getContext();
        // In case of spilling swizzled buffer each per cluster buffer needs to be copied as is together with
        // additional alignment to DDR.
        // For example: 1x48x8x8xf16, SOH overlaped, each tile need 3840 bytes(1x48x5x8xf16) on CMX, after swizzled
        // buffer, will aligned to 4096 bytes.
        // When spilling to DDR, need static allocation on DDR is 8192 bytes. However, static allocation thinks DDR
        // buffer just needs 6144 bytes(1x48x8x8xf16). If we use sizeAlignment * numClusters as new alignment (for
        // example on VPUX40XX is 2048), the original size 6144 bytes is 2048 aligned. It's will casuse buffer overflow.
        // Here we increase the alignment until the memory meet the real requirements. For this case, the new alignment
        // will increase to 4096.
        size_t expectedAlignedByteSize = 0;
        for (auto perClusterShape : getPerClusterMemoryShapes()) {
            auto perClusterByteSize =
                    alignMemSize(getElemTypeSize() * perClusterShape.totalSize(), Byte(1)).to<Byte>().count();
            expectedAlignedByteSize +=
                    alignValUp<int64_t>(perClusterByteSize, swizzlingSchemeAttr.getSizeAlignment().getInt());
        }

        auto alignment = swizzlingSchemeAttr.getSizeAlignment().getInt();
        auto origSize = getShape().totalSize();
        auto currentNonAlignedByteSize = alignMemSize(getElemTypeSize() * origSize, Byte(1)).to<Byte>().count();

        while (alignValUp<size_t>(currentNonAlignedByteSize, alignment) != expectedAlignedByteSize) {
            alignment += swizzlingSchemeAttr.getSizeAlignment().getInt();
        }

        swizzlingSchemeAttr =
                VPUIP::SwizzlingSchemeAttr::get(ctx, swizzlingSchemeAttr.getKey(), getIntAttr(ctx, alignment));
    }

    const auto strides = getStrides();
    return vpux::getMemRefType(getShape(), getElementType(), getDimsOrder(), getMemSpace(), strides,
                               swizzlingSchemeAttr, VPUIP::getSparsityCompressionAttr(*this));
}

namespace {

//
// Shape utils
//

Shape* getLargestShapeIt(SmallVector<Shape>& shapes) {
    return std::max_element(shapes.begin(), shapes.end(), [](ShapeRef a, ShapeRef b) {
        return details::calcTotalShapeSize(a.raw()) < details::calcTotalShapeSize(b.raw());
    });
}

StridedShape* getLargestStridedShapeIt(SmallVector<StridedShape>& stridedShapes) {
    const auto stridedShapeSize = [](const StridedShape& stridedShape) {
        return stridedShape.shape.front() * stridedShape.strides.front();
    };
    return std::max_element(stridedShapes.begin(), stridedShapes.end(),
                            [&](const StridedShape& a, const StridedShape& b) {
                                return stridedShapeSize(a) < stridedShapeSize(b);
                            });
}

//
// Helper to extract subview quantized type when input is already distributed over quant-axis
//
// e.g. full shape     = 16x?x?x?
//      full qElemType = !quant.uniform<u8:f16:0, {s00:0, s01:0, s02:0, ... s15:0}
//
//         | Tile_0 | Tile_1 | Tile_2 |
// ========|========|========|========|
// Shave_0 |  s00   |  s06   |  s11   |
// 9x?x?x? |  s01   |  s07   |  s12   | qElemType0 : {s00, s01, s02, s06, s07, s08, s11, s12, s13}
//         |  s02   |  s08   |  s13   |
// ========|========|========|========|
// Shave_1 |  s03   |  s09   |  s14   |
// 7x?x?x? |  s04   |  s10   |  s15   | qElemType1 : {s03, s04, s05, s09, s10, s14, s15}
//         |  s05   |========|========|
//         |========|

mlir::Type getQuantTypeForExplicitDistribution(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                               VPU::DistributionInfoAttr inDistribution,
                                               VPU::DistributionInfoAttr subDistribution, vpux::ShapeRef tileOffsets) {
    auto axis = perAxisQType.getQuantizedDimension();
    auto numTiles = vpux::parseIntArrayAttr<int64_t>(inDistribution.getNumTiles());
    auto inputOffsets = vpux::parseIntArrayOfArrayAttr<int64_t>(inDistribution.getComputeOffsets());
    auto subviewShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(subDistribution.getMemoryShapes());
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    VPUX_THROW_UNLESS(tileOffsets[Dim(axis)] % numTiles[axis] == 0,
                      "Previous subview shapes are not identical in all tiles.");

    int64_t bias = tileOffsets[Dim(axis)] / numTiles[axis];  // bias within tile
    for (auto ind : irange(inputOffsets.size())) {
        offsets.push_back(inputOffsets[ind][axis] + bias);
        sizes.push_back(subviewShapes[ind][axis]);
    }

    return vpux::tileScalesAndZP(perAxisQType, offsets, sizes);
}

}  // namespace

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
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterComputeShapes() const {
    auto distribution = getDistribution();
    if (distribution.getComputeShapes() == nullptr) {
        return VPU::getPerClusterComputeShapes(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeShapes());
}

// @brief Retrieve the array of compute buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to compute shapes,
// is that modes like SEGMENTED and OVERLAPPED take precedence over
// DUPLICATED and MULTICASTED.
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterComputeShapeOffsets() const {
    auto distribution = getDistribution();
    if (distribution.getComputeOffsets() == nullptr) {
        return VPU::getPerClusterComputeShapeOffsets(getShape(), distribution);
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getComputeOffsets());
}

// @brief Retrieve the array of memory shapes.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
//  In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
//  tensor with shape [1, 64, 4, 4], the memory shape in each cluster is
//  [1, 64, 4, 4], which is the allocated shape (because of duplicated)
//  information which is needed for scheduler and strategy manager,
//  in order to estimate memory
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterMemoryShapes() const {
    auto distribution = getDistribution();
    if (distribution.getMemoryShapes() == nullptr) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Unsupported distribution: {0}", distribution);
        return optionalPerClusterMemoryShapes.value();
    }

    return VPU::arrayAttrToVecOfShapes(distribution.getMemoryShapes());
}

// @brief Retrieve the array of memory buffer offsets with regards to the full buffer.
// @warning An important thing to consider with regards to compute shapes,
//  is that modes like DUPLICATED and MULTICASTED take precedence over
//  SEGMENTED and OVERLAPPED.
SmallVector<Shape> VPUIP::DistributedBufferType::getPerClusterMemoryShapeOffsets() const {
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
Shape VPUIP::DistributedBufferType::getLargestCompactShape() const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    return *getLargestShapeIt(tiledComputeShapes);
}

// @brief Get the compact compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
Shape VPUIP::DistributedBufferType::getCompactShape(int64_t tileInd) const {
    auto tiledComputeShapes = getPerClusterComputeShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(tiledComputeShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return tiledComputeShapes[tileInd];
}

// @brief Retrieve the array of padding for each cluster
// @warning This function is needed for getting padding in OVERLAPPED mode.
SmallVector<PadInfo> VPUIP::DistributedBufferType::getPerClusterPadding(PadInfo kernelPadding) const {
    return VPU::getPerClusterPadding(getDistribution(), kernelPadding);
}

// @brief Retrieve the array of strided memory shapes
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
SmallVector<StridedShape> VPUIP::DistributedBufferType::getPerClusterMemoryStridedShapes() const {
    const auto memoryShapes = getPerClusterMemoryShapes();
    const auto strides = getStrides();
    return VPU::getPerClusterMemoryStridedShapes(getShape(), strides, getDimsOrder(), getDistribution().getMode(),
                                                 memoryShapes);
}

// @brief Get largest strided compute shape
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPUIP::DistributedBufferType::getLargestStridedShape() const {
    auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(!stridedShapes.empty(), "Missing per-cluster strided shapes");
    return *getLargestStridedShapeIt(stridedShapes);
}

// @brief Get the strided compute shape for a specific cluster
// @warning This function should not be used for memory size calculation,
// because it does not retrieve the true allocate shape in cases
// of broadcasting.
StridedShape VPUIP::DistributedBufferType::getStridedShape(int64_t tileInd) const {
    const auto stridedShapes = getPerClusterMemoryStridedShapes();
    VPUX_THROW_UNLESS(tileInd < static_cast<int64_t>(stridedShapes.size()),
                      "Requesting tiled shape outside of cluster pool");
    return stridedShapes[tileInd];
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offets adjusted for the new shape.
NDTypeInterface VPUIP::DistributedBufferType::changeShapeForExplicitDistribution(
        ShapeRef shape, VPU::DistributionInfoAttr distributedAttr) const {
    return changeShapeElemTypeForExplicitDistribution(shape, getElementType(), distributedAttr);
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested shape and element type and DistributedAttr with
// memory_shapes/memory_offsets/computes_shapes/compute_offets adjusted for the new shape.
NDTypeInterface VPUIP::DistributedBufferType::changeShapeElemTypeForExplicitDistribution(
        ShapeRef shape, mlir::Type elemType, VPU::DistributionInfoAttr distributedAttr) const {
    const auto ctx = getContext();
    const auto origOrder = getDimsOrder();
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;

    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto layoutAttr = getLayout();
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        const auto orderAttr = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
        // If swizzlingKey is set get rid of strides settings
        if (memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>()) {
            layoutAttr = getMemrefLayout(orderAttr, nullptr,
                                         /*allocSize=*/nullptr, memRefAttr.hwSpecificFields());
        } else {
            layoutAttr = orderAttr;
        }
    }

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, getMemSpace(),
                                                     distributedAttr, getSparsityCompression());

    const auto loc = mlir::UnknownLoc::get(ctx);
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "ChangeShape caused mismatch with quantization settings'{0}'", newType);

    return newType;
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType with requested type components. If shape is one of the changed
// components, it will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offets
// adjusted for the new shape. Otherwise, it leaves the DistributedAttr untouched.
NDTypeInterface VPUIP::DistributedBufferType::changeTypeComponentsForExplicitDistribution(
        const TypeComponents& typeComponents, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return changeTypeComponents(typeComponents);
    }

    const auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType());
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto strides = typeComponents.strides.value_or(getStrides());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    VPUX_THROW_UNLESS(dimsOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the shape '{1}'",
                      dimsOrder, shape);

    const auto elemSize = vpux::getElemTypeSize(elementType);
    const auto order = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(ctx));
    const auto newStridesAttr = getStridesAttr(ctx, strides, dimsOrder, elemSize, shape);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = getMemrefLayout(order, newStridesAttr, /*allocSize=*/nullptr, hwSpecificFields);

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elementType, newDescAttr, memSpace,
                                                     distributedAttr, getSparsityCompression());

    const auto loc = mlir::UnknownLoc::get(getContext());
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "changeTypeComponentsForExplicitDistribution caused mismatch with quantization settings'{0}'",
                      newType);

    return newType;
}

// @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
// them. This method creates DistributedType obtained by extracting a dense tile from the original DistributedType.
// It will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offets
// adjusted for the resulting dense tile.
NDTypeInterface VPUIP::DistributedBufferType::extractDenseTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractDenseTile(tileOffsets, tileShape);
    }

    const auto ctx = getContext();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    auto tileElemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, order, getMemSpace(), distributedAttr,
                                             sparsityCompression);
}

NDTypeInterface VPUIP::DistributedBufferType::extractViewTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, vpux::ShapeRef tileElemStrides,
        VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractViewTile(tileOffsets, tileShape, tileElemStrides);
    }
    const auto ctx = getContext();

    const auto elemSize = getElemTypeSize();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto memSpace = getMemSpace();

    auto tileElemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
        auto inMode = getDistribution().getMode().getValue();
        auto isMultiClusterAndMultiShaveOnAxis = false;
        if (inMode == VPU::DistributionMode::SEGMENTED) {
            auto axis = perAxisQType.getQuantizedDimension();
            auto inTiles = vpux::parseIntArrayAttr<int64_t>(getDistribution().getNumTiles());
            auto subTiles = vpux::parseIntArrayAttr<int64_t>(distributedAttr.getNumTiles());
            isMultiClusterAndMultiShaveOnAxis = (inTiles[axis] > 1) && (subTiles[axis] > 1);
        }

        if (isMultiClusterAndMultiShaveOnAxis) {
            tileElemType =
                    getQuantTypeForExplicitDistribution(perAxisQType, getDistribution(), distributedAttr, tileOffsets);
        } else {
            tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
        }
    }

    auto tileStrides = getStrides();
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto newStridesAttr = getStridesAttr(ctx, tileStrides, getDimsOrder(), elemSize, tileShape);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = getMemrefLayout(order, newStridesAttr, /*allocSize=*/nullptr, hwSpecificFields);

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, newDescAttr, memSpace, distributedAttr,
                                             sparsityCompression);
}

//
// NDTypeInterface
//

MemShape VPUIP::DistributedBufferType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getShape();
    return dimsOrder.toMemoryOrder(shape);
}

bool VPUIP::DistributedBufferType::hasRank() const {
    return true;
}

int64_t VPUIP::DistributedBufferType::getRank() const {
    return checked_cast<int64_t>(getShape().size());
}

int64_t VPUIP::DistributedBufferType::getNumElements() const {
    if (getSparsityCompression() != nullptr) {
        return getSparsityCompression().getTotalNumElems();
    }
    auto shape = getShape().raw();
    VPUX_THROW_UNLESS(!details::isDynamicDimValues(shape), "Cannot get element count of dynamic shaped type");
    return details::calcTotalShapeSize(shape);
}

DimsOrder VPUIP::DistributedBufferType::getDimsOrder() const {
    const auto layout = getLayout();
    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

VPU::MemoryKind VPUIP::DistributedBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }

    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

Strides VPUIP::DistributedBufferType::getStrides() const {
    const auto layout = getLayout();

    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        VPUX_THROW_UNLESS(mapAttr.getValue().isPermutation(), "Got non permutation layout attribute '{0}'", layout);
    }

    if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        if (auto stridesAttr = descAttr.strides()) {
            const auto elemStrides = parseIntArrayAttr<int64_t>(stridesAttr);
            const Bit elemSize = getElemTypeSize();

            return Strides(to_small_vector(elemStrides | transformed([&](int64_t stride) {
                                               return stride * elemSize;
                                           })));
        }
    }

    // Missing strides specification means compact strides.
    const auto order = getDimsOrder();
    const auto memShape = getMemShape();
    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

    return order.toLogicalOrder(memStrides);
}

MemStrides VPUIP::DistributedBufferType::getMemStrides() const {
    const auto order = getDimsOrder();
    const auto strides = getStrides();
    return order.toMemoryOrder(strides);
}

Bit VPUIP::DistributedBufferType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPUIP::DistributedBufferType::getTotalAllocSize() const {
    auto shape = getShape();
    auto strides = getStrides();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    auto sparsityCompression = getSparsityCompression();
    const auto order = getDimsOrder();
    const auto elemBitSize = getElemTypeSize();

    Byte allocSizeByte;

    if (distributionMode != VPU::DistributionMode::NONE) {
        const auto perClusterStridedShapes = getPerClusterMemoryStridedShapes();
        const auto perClusterOffsets = getPerClusterMemoryShapeOffsets();
        for (auto p : zip(perClusterStridedShapes, perClusterOffsets)) {
            const auto tileShape = std::get<0>(p);
            const auto tileOffsets = std::get<1>(p);
            const auto stridedTiledShape = alignStridedShape(tileShape, distribution, order);
            allocSizeByte = std::max(allocSizeByte, getStridedAllocSize(stridedTiledShape, tileOffsets,
                                                                        sparsityCompression, order, elemBitSize));
        }
    } else {
        // No distribution mode.
        Shape stridedTiledOffsets(SmallVector<int64_t>(shape.size(), 0));
        const auto stridedTiledShape = alignStridedShape(StridedShape(shape, strides), distribution, order);
        allocSizeByte =
                getStridedAllocSize(stridedTiledShape, stridedTiledOffsets, sparsityCompression, order, elemBitSize);
    }

    if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        auto swizzlingScheme = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        if (!swizzlingScheme || swizzlingScheme.getKey().getInt() == 0) {
            return allocSizeByte;
        }

        // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
        allocSizeByte = Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
    }

    return allocSizeByte;
}

Byte VPUIP::DistributedBufferType::getAllocSizeOfCluster(size_t clusterId) const {
    auto shape = getShape();
    auto strides = getStrides();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    auto sparsityCompression = getSparsityCompression();
    const auto order = getDimsOrder();
    const auto elemBitSize = getElemTypeSize();

    Byte allocSizeByte;

    if (distributionMode != VPU::DistributionMode::NONE) {
        const auto clusterStridedShapes = getPerClusterMemoryStridedShapes()[clusterId];
        const auto clusterOffsets = getPerClusterMemoryShapeOffsets()[clusterId];
        const auto stridedTiledShape = alignStridedShape(clusterStridedShapes, distribution, order);
        allocSizeByte = getStridedAllocSize(stridedTiledShape, clusterOffsets, sparsityCompression, order, elemBitSize);
    } else {
        // No distribution mode.
        Shape stridedTiledOffsets(SmallVector<int64_t>(shape.size(), 0));
        const auto stridedTiledShape = alignStridedShape(StridedShape(shape, strides), distribution, order);
        allocSizeByte =
                getStridedAllocSize(stridedTiledShape, stridedTiledOffsets, sparsityCompression, order, elemBitSize);
    }

    if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        auto swizzlingScheme = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        if (!swizzlingScheme || swizzlingScheme.getKey().getInt() == 0) {
            return allocSizeByte;
        }

        // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
        allocSizeByte = Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
    }

    return allocSizeByte;
}

Byte VPUIP::DistributedBufferType::getCompactAllocSize() const {
    auto shape = getShape();
    const Bit elemSize = getElemTypeSize();
    const auto distribution = getDistribution();
    const auto distributionMode = distribution.getMode().getValue();
    auto sparsityCompression = getSparsityCompression();

    const auto alignTiledShape = [&](ShapeRef tiledShape) -> Shape {
        if (distribution.getAlignment() == nullptr) {
            return tiledShape.raw();
        }
        const auto alignment = parseIntArrayAttr<int64_t>(distribution.getAlignment());
        const auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        return Shape(alignShape(tiledShape.raw(), optionalAlignment, alignValUp<int64_t>));
    };

    const auto getAllocSize = [&](ShapeRef tiledShape, ShapeRef tiledOffsets) -> Byte {
        if (sparsityCompression == nullptr) {
            return Byte(alignMemSize(elemSize * details::calcTotalShapeSize(tiledShape.raw()), Byte(1)));
        }

        const auto axis = sparsityCompression.getAxis().getInt();
        const auto numElems = sparsityCompression.getNumElems().getValues<int64_t>();
        const int64_t alignment =
                (sparsityCompression.getAlignment() != nullptr) ? sparsityCompression.getAlignment().getInt() : 1;

        const auto startTileIt = numElems.begin() + tiledOffsets[Dim(axis)];
        const auto endTileIt = startTileIt + tiledShape[Dim(axis)];
        int64_t tileElemsBytes = 0;
        for (auto it = startTileIt; it != endTileIt; ++it) {
            tileElemsBytes += alignValUp<int64_t>(*it * Byte(elemSize).count(), alignment);
        }
        return Byte(tileElemsBytes);
    };

    Byte allocSizeByte(0);

    // DUPLICATED|MULTICASTED takes priority since it means that each cluster will have the entire
    // tensor, regardless whether it's tiled or not.
    Shape tiledOffsets(SmallVector<int64_t>(shape.size(), 0));
    if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::DUPLICATED) ||
        VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::MULTICASTED)) {
        const auto tiledShape = alignTiledShape(shape);
        allocSizeByte = getAllocSize(tiledShape, tiledOffsets);
    } else if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED) ||
               VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::OVERLAPPED)) {
        const auto perClusterShapes = getPerClusterMemoryShapes();
        const auto perClusterOffsets = getPerClusterMemoryShapeOffsets();
        for (auto p : zip(perClusterShapes, perClusterOffsets)) {
            const auto tileShape = std::get<0>(p);
            const auto tileOffsets = std::get<1>(p);
            const auto alignedTiledShape = alignTiledShape(tileShape);
            allocSizeByte = std::max(allocSizeByte, getAllocSize(alignedTiledShape, tileOffsets));
        }
    } else {
        // No distribution mode.
        const auto tiledShape = alignTiledShape(shape);
        allocSizeByte = getAllocSize(tiledShape, tiledOffsets);
    }

    return allocSizeByte;
}

NDTypeInterface VPUIP::DistributedBufferType::changeShape(ShapeRef shape) const {
    return changeShapeElemType(shape, getElementType());
}

NDTypeInterface VPUIP::DistributedBufferType::changeElemType(mlir::Type elemType) const {
    const auto ctx = getContext();

    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), elemType, getLayout(), getMemSpace(),
                                             getDistribution(), getSparsityCompression());
}

NDTypeInterface VPUIP::DistributedBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ctx = getContext();

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot change shape when having explicit per cluster shapes/offsets");

    const auto origOrder = getDimsOrder();
    const auto newOrder = origOrder.isIdentity() ? DimsOrder::fromNumDims(shape.size()) : origOrder;
    VPUX_THROW_UNLESS(newOrder.numDims() == shape.size(), "Order '{0}' is incompatible with the new shape '{1}'",
                      newOrder, shape);

    auto layoutAttr = getLayout();
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        const auto orderAttr = mlir::AffineMapAttr::get(newOrder.toAffineMap(ctx));
        // If swizzlingKey is set get rid of strides settings
        if (memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>()) {
            layoutAttr = getMemrefLayout(orderAttr, nullptr,
                                         /*allocSize=*/nullptr, memRefAttr.hwSpecificFields());
        } else {
            layoutAttr = orderAttr;
        }
    }

    auto newType = VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, layoutAttr, getMemSpace(),
                                                     distribution, getSparsityCompression());

    const auto loc = mlir::UnknownLoc::get(ctx);
    VPUX_THROW_UNLESS(vpux::validateQuantElemType(loc, newType).succeeded(),
                      "ChangeShape caused mismatch with quantization settings'{0}'", newType);

    return newType;
}

NDTypeInterface VPUIP::DistributedBufferType::changeDimsOrder(DimsOrder order) const {
    const auto ctx = getContext();

    auto layoutAttr = getLayout();
    auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
    if (auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        // Assume compact strides
        layoutAttr = getMemrefLayout(orderAttr, nullptr,
                                     /*allocSize=*/nullptr, memRefAttr.hwSpecificFields());
    } else {
        layoutAttr = orderAttr;
    }

    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), getElementType(), layoutAttr, getMemSpace(),
                                             getDistribution(), getSparsityCompression());
}

NDTypeInterface VPUIP::DistributedBufferType::changeMemSpace(IndexedSymbolAttr /*memSpace*/) const {
    VPUX_THROW("changeMemSpace method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::changeStrides(StridesRef strides) const {
    const auto ctx = getContext();
    const auto elemSize = getElemTypeSize();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto newStridesAttr = getStridesAttr(ctx, strides, getDimsOrder(), elemSize, getShape());
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = getMemrefLayout(order, newStridesAttr,
                                             /*allocSize=*/nullptr, hwSpecificFields);
    return VPUIP::DistributedBufferType::get(ctx, getShape().raw(), getElementType(), newDescAttr, getMemSpace(),
                                             getDistribution(), getSparsityCompression());
}

NDTypeInterface VPUIP::DistributedBufferType::changeTypeComponents(const vpux::TypeComponents& typeComponents) const {
    const auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto elementType = typeComponents.elementType.value_or(getElementType());
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto strides = typeComponents.strides.value_or(getStrides());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());
    auto distribution = getDistribution();

    // If there is a shape change requested
    if (shape != Shape(getShape().toValues())) {
        VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                        "Cannot change shape when having explicit per cluster shapes/offsets");
    }

    const auto elemSize = vpux::getElemTypeSize(elementType);
    const auto order = mlir::AffineMapAttr::get(dimsOrder.toAffineMap(ctx));
    const auto newStridesAttr = getStridesAttr(ctx, strides, dimsOrder, elemSize, shape);

    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = getMemrefLayout(order, newStridesAttr,
                                             /*allocSize=*/nullptr, hwSpecificFields);

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elementType, newDescAttr, memSpace, distribution,
                                             getSparsityCompression());
}

NDTypeInterface VPUIP::DistributedBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ctx = getContext();

    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot get DistributedBufferType with new shape from old one when having explicit per cluster "
                    "shapes/offsets");

    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));

    auto tileElemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, order, getMemSpace(), distribution,
                                             sparsityCompression);
}

NDTypeInterface VPUIP::DistributedBufferType::extractViewTile(vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape,
                                                              vpux::ShapeRef tileElemStrides) const {
    const auto ctx = getContext();
    auto distribution = getDistribution();
    VPUX_THROW_WHEN(isDistributedAttrWithExplicitShapesAndOffsets(distribution),
                    "Cannot get DistributedBufferType with new shape from old one when having explicit per cluster "
                    "shapes/offsets");

    const auto elemSize = getElemTypeSize();
    const auto order = mlir::AffineMapAttr::get(getDimsOrder().toAffineMap(ctx));
    const auto memSpace = getMemSpace();

    auto tileElemType = getElementType();
    if (const auto perAxisQType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(tileElemType)) {
        tileElemType = vpux::tileScalesAndZP(perAxisQType, tileShape, tileOffsets);
    }

    auto tileStrides = getStrides();
    if (!tileElemStrides.empty()) {
        VPUX_THROW_UNLESS(tileElemStrides.size() == tileStrides.size(),
                          "Tile elem strides '{0}' is not aligned with rank '{1}'", tileElemStrides,
                          tileStrides.size());

        for (auto ind : irange(tileElemStrides.size())) {
            tileStrides[Dim(ind)] *= tileElemStrides[Dim(ind)];
        }
    }

    const auto newStridesAttr = getStridesAttr(ctx, tileStrides, getDimsOrder(), elemSize, tileShape);
    auto hwSpecificFields = getHwSpecificFields(getLayout());
    const auto newDescAttr = getMemrefLayout(order, newStridesAttr,
                                             /*allocSize=*/nullptr, hwSpecificFields);

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::DistributedBufferType::get(ctx, tileShape.raw(), tileElemType, newDescAttr, memSpace, distribution,
                                             sparsityCompression);
}

NDTypeInterface VPUIP::DistributedBufferType::eraseTiledInfo() const {
    VPUX_THROW("eraseTiledInfo method is not implemented for DistributedBufferType");
}

NDTypeInterface VPUIP::DistributedBufferType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not implemented for DistributedBufferType");
}
