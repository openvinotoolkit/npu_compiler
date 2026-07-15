//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

mlir::LogicalResult checkHaloCorrectness(FuncRef<mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> haloShape,
                                         ArrayRef<int64_t> haloOffset, ArrayRef<int64_t> bufferShape,
                                         const bool isIduSegmentation) {
    if (haloShape.size() != haloOffset.size() || haloShape.size() != bufferShape.size()) {
        return printTo(emitError(),
                       "Inward halo region shape/offset ranks do not match: ITI buffer rank '{0}', halo region rank "
                       "'{1}', halo offsets rank '{2}'",
                       bufferShape.size(), haloShape.size(), haloOffset.size());
    }

    for (size_t dim = 0; dim < bufferShape.size(); ++dim) {
        if (haloOffset[dim] + haloShape[dim] > bufferShape[dim]) {
            return printTo(emitError(),
                           "Halo region does not fit in ITI buffer: full ITI buffer '{0}', halo region "
                           "'{1}', halo region offsets '{2}'",
                           bufferShape, haloShape, haloOffset);
        }

        // For IDU segmentation, only allow "halo-ing" over H or W
        if (isIduSegmentation) {
            if (haloShape[dim] != bufferShape[dim] && Dim(dim) != Dims4D::Act::H && Dim(dim) != Dims4D::Act::W) {
                return printTo(emitError(),
                               "Halo region for IDU slices over dim '{0}', only H & W supported, halo region = "
                               "'{1}', full buffer '{2}'",
                               Dim(dim), haloShape, bufferShape);
            }
        }
    }

    return mlir::success();
}

// Returns the actual tensor volume found in the current cluster, whether it is produced
// in this cluster or not.
vpux::Shape getVolumeInCurrentCluster(ShapeRef itiShape, bool isIduSegmented,
                                      ArrayRef<vpux::VPUIP::HaloRegionAttr> inwardHaloRegions) {
    if (!isIduSegmented) {
        return Shape(itiShape.toValues());
    }

    auto shapeWithoutHalos = SmallVector<int64_t>(itiShape.begin(), itiShape.end());
    for (const auto& inHalo : inwardHaloRegions) {
        const auto inHaloShape = parseIntArrayAttr<int64_t>(inHalo.getShape());
        for (size_t dim = 0; dim < inHaloShape.size(); dim++) {
            if (inHaloShape[dim] != itiShape[Dim(dim)]) {
                shapeWithoutHalos[dim] = itiShape[Dim(dim)] - inHaloShape[dim];
            }
        }
    }

    return Shape(shapeWithoutHalos);
}

}  // namespace

//
// print/parse
//

void VPUIP::ITIBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<";
    printer.increaseIndent();
    printer.increaseIndent();
    printer.printNewline();
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

    if (getIduSegmentation() != nullptr) {
        printer << ", idu_segmentation";
    }

    if (!getInwardHaloRegions().empty()) {
        printer << ",";
        printer.printNewline();
        printer << "inwardHaloRegions = [";
        printer.increaseIndent();
        printer.increaseIndent();
        printer.printNewline();

        const auto inwardHaloRegions = getInwardHaloRegions();
        for (const auto haloRegion : inwardHaloRegions) {
            printer << haloRegion;
            if (haloRegion != inwardHaloRegions.back()) {
                printer << ",";
                printer.printNewline();
            }
        }
        printer.decreaseIndent();
        printer.decreaseIndent();
        printer.printNewline();
        printer << "]";
    }

    if (!getOutwardHaloRegions().empty()) {
        printer << ",";
        printer.printNewline();
        printer << "outwardHaloRegions = [";
        printer.increaseIndent();
        printer.increaseIndent();
        printer.printNewline();

        const auto outwardHaloRegions = getOutwardHaloRegions();
        for (const auto haloRegion : outwardHaloRegions) {
            printer << haloRegion;
            if (haloRegion != outwardHaloRegions.back()) {
                printer << ",";
                printer.printNewline();
            }
        }

        for (size_t i = 0; i < 4; ++i) {
            printer.decreaseIndent();
        }

        printer.printNewline();
        printer << "]";
    }

    printer << ">";
}
mlir::Type VPUIP::ITIBufferType::parse(mlir::AsmParser& parser) {
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

    const auto isIduSegmentation = parser.parseOptionalKeyword("idu_segmentation");

    SmallVector<VPUIP::HaloRegionAttr> inwardHaloVec;
    SmallVector<VPUIP::OutwardHaloRegionAttr> outwardHaloVec;
    bool noInwardHaloRegions = false;
    if (mlir::succeeded(parser.parseOptionalComma())) {
        const auto hasInwardHaloRegions = parser.parseOptionalKeyword("inwardHaloRegions");
        noInwardHaloRegions = hasInwardHaloRegions.failed();
        if (hasInwardHaloRegions.succeeded()) {
            if (parser.parseEqual()) {
                return Type();
            }

            if (parser.parseLSquare()) {
                return Type();
            }

            VPUIP::HaloRegionAttr inwardHalo;
            while (parser.parseOptionalAttribute(inwardHalo).has_value()) {
                inwardHaloVec.push_back(inwardHalo);
                (void)parser.parseOptionalComma();
            }

            if (parser.parseRSquare()) {
                return Type();
            }
        }
    }

    if (noInwardHaloRegions || mlir::succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("outwardHaloRegions")) {
            return Type();
        }

        if (parser.parseEqual()) {
            return Type();
        }

        if (parser.parseLSquare()) {
            return Type();
        }

        VPUIP::OutwardHaloRegionAttr outwardHalo;
        while (parser.parseOptionalAttribute(outwardHalo).has_value()) {
            outwardHaloVec.push_back(outwardHalo);
            (void)parser.parseOptionalComma();
        }

        if (parser.parseRSquare()) {
            return Type();
        }
    }

    if (parser.parseGreater()) {
        return Type();
    }

    if (isIduSegmentation.succeeded()) {
        return static_cast<mlir::Type>(get(parser.getContext(), ArrayRef(shape), elemType, layout, memSpace,
                                           mlir::UnitAttr::get(parser.getContext()), inwardHaloVec, outwardHaloVec));
    }

    return static_cast<mlir::Type>(get(parser.getContext(), ArrayRef(shape), elemType, layout, memSpace, nullptr,
                                       inwardHaloVec, outwardHaloVec));
}

//
// verify
//

mlir::LogicalResult VPUIP::ITIBufferType::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                 ::llvm::ArrayRef<int64_t> shape, mlir::Type /*elementType*/,
                                                 mlir::MemRefLayoutAttrInterface /*layout*/,
                                                 IndexedSymbolAttr /*memSpace*/, mlir::UnitAttr iduSegmentation,
                                                 ArrayRef<HaloRegionAttr> inwardHaloRegions,
                                                 ArrayRef<OutwardHaloRegionAttr> outwardHaloRegions) {
    const bool isIduSegmented = iduSegmentation != nullptr;
    int64_t heightInCurrentCluster = shape[Dims4D::Act::H.ind()];
    int64_t widthInCurrentCluster = shape[Dims4D::Act::W.ind()];
    for (const auto& inHalo : inwardHaloRegions) {
        const auto inHaloShape = parseIntArrayAttr<int64_t>(inHalo.getShape());
        const auto inHaloOffset = parseIntArrayAttr<int64_t>(inHalo.getOffset());

        const auto result = checkHaloCorrectness(emitError, inHaloShape, inHaloOffset, shape, isIduSegmented);
        if (result.failed()) {
            return result;
        }

        if (inHaloShape[Dims4D::Act::H.ind()] != shape[Dims4D::Act::H.ind()]) {
            heightInCurrentCluster -= inHaloShape[Dims4D::Act::H.ind()];
        }

        if (inHaloShape[Dims4D::Act::W.ind()] != shape[Dims4D::Act::W.ind()]) {
            widthInCurrentCluster -= inHaloShape[Dims4D::Act::W.ind()];
        }
    }

    // For idu segmented cases, allocated memory in current cluster is computed by
    // subtracting the h/w halo from full shape. Ensure it is a valid size.
    if (isIduSegmented) {
        if (heightInCurrentCluster <= 0 || widthInCurrentCluster <= 0) {
            return printTo(emitError(),
                           "ITI Buffer with IDU segmentation has height or width in current cluster <= 0, "
                           "h = '{0}', w = '{1}'.",
                           heightInCurrentCluster, widthInCurrentCluster);
        }
    }

    for (const auto& outHalo : outwardHaloRegions) {
        const auto outHaloShape = parseIntArrayAttr<int64_t>(outHalo.getShape());
        const auto outHaloOffset = parseIntArrayAttr<int64_t>(outHalo.getOffset());

        const auto result = checkHaloCorrectness(emitError, outHaloShape, outHaloOffset, shape, isIduSegmented);
        if (result.failed()) {
            return result;
        }

        for (const auto& inHalo : outHalo.getInwardHaloRegions().getValue()) {
            const auto inHaloAttr = mlir::cast<vpux::VPUIP::HaloRegionAttr>(inHalo);
            const auto inHaloShape = parseIntArrayAttr<int64_t>(inHaloAttr.getShape());

            for (size_t dim = 0; dim < shape.size(); ++dim) {
                if (inHaloShape[dim] != outHaloShape[dim]) {
                    return printTo(emitError(),
                                   "Inward halo region shape does not match the outward halo shape: inward halo '{0}', "
                                   "outward halo shape '{1}'",
                                   inHaloShape, outHaloShape);
                }
            }
        }
    }

    return mlir::success();
}

//
// NDTypeInterface
//

MemShape VPUIP::ITIBufferType::getMemShape() const {
    const auto dimsOrder = getDimsOrder();
    const auto shape = getVolumeInCurrentCluster(getShape(), getIduSegmentation() != nullptr, getInwardHaloRegions());
    return dimsOrder.toMemoryOrder(shape);
}

bool VPUIP::ITIBufferType::hasRank() const {
    return true;
}

int64_t VPUIP::ITIBufferType::getRank() const {
    return checked_cast<int64_t>(getShape().size());
}

int64_t VPUIP::ITIBufferType::getNumElements() const {
    auto memShape = getMemShape().raw();
    VPUX_THROW_UNLESS(!details::isDynamicDimValues(memShape), "Cannot get element count of dynamic shaped type");
    return details::calcTotalShapeSize(memShape);
}

DimsOrder VPUIP::ITIBufferType::getDimsOrder() const {
    const auto layout = getLayout();
    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout)) {
        return DimsOrder::fromAffineMap(mapAttr.getValue());
    }

    if (const auto descAttr = mlir::dyn_cast<vpux::MemRefAttr>(layout)) {
        return DimsOrder::fromAffineMap(descAttr.order().getValue());
    }

    VPUX_THROW("Missing layout information");
}

VPU::MemoryKind VPUIP::ITIBufferType::getMemoryKind() const {
    const auto memSpace = getMemSpace();
    if (memSpace == nullptr) {
        return VPU::MemoryKind::DDR;
    }

    return VPU::symbolizeEnum<VPU::MemoryKind>(memSpace.getLeafName()).value();
}

Strides VPUIP::ITIBufferType::getStrides() const {
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
    auto memShape = getMemShape();

    const auto memStrides = StrideReqs::compact(order.numDims()).calcStrides(getElemTypeSize(), memShape);

    return order.toLogicalOrder(memStrides);
}

MemStrides VPUIP::ITIBufferType::getMemStrides() const {
    const auto order = getDimsOrder();
    const auto strides = getStrides();
    return order.toMemoryOrder(strides);
}

Bit VPUIP::ITIBufferType::getElemTypeSize() const {
    return vpux::getElemTypeSize(getElementType());
}

Byte VPUIP::ITIBufferType::getTotalAllocSize() const {
    if (getRank() == 0) {
        return alignMemSize(getElemTypeSize(), Byte(1));
    }

    const auto memShape = getMemShape();
    const auto memStrides = getMemStrides();

    VPUX_THROW_UNLESS(memShape.size() == memStrides.size(), "Shape and strides mismatch : {0} vs {1}", memShape,
                      memStrides);

    auto allocSizeByte = alignMemSize(memStrides.front() * memShape.front(), Byte(1)).to<Byte>();

    if (const auto memRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(getLayout())) {
        auto swizzlingScheme = memRefAttr.hwSpecificField<vpux::VPUIP::SwizzlingSchemeAttr>();
        if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
            // If swizzling is enabled total buffer size needs to be aligned to 512 or 1024 as required by HW
            allocSizeByte =
                    Byte(alignSizeForSwizzling(allocSizeByte.count(), swizzlingScheme.getSizeAlignment().getInt()));
        }

        auto compressionStateAttr = memRefAttr.hwSpecificField<vpux::VPUIP::CompressionStateAttr>();
        if (compressionStateAttr &&
            ((compressionStateAttr.getValue() == VPUIP::CompressionState::RuntimeCompressed) ||
             (compressionStateAttr.getValue() == VPUIP::CompressionState::CompressionCandidate))) {
            allocSizeByte = Byte(updateSizeForCompression(allocSizeByte.count()));
        }
    }

    return allocSizeByte;
}

Byte VPUIP::ITIBufferType::getCompactAllocSize() const {
    const Bit typeSize = getElemTypeSize();
    if (getRank() == 0) {
        return alignMemSize(typeSize, Byte(1));
    }

    const auto shape = getVolumeInCurrentCluster(getShape(), getIduSegmentation() != nullptr, getInwardHaloRegions());
    return alignMemSize(typeSize * shape.totalSize(), Byte(1));
}

NDTypeInterface VPUIP::ITIBufferType::changeShape(ShapeRef /*shape*/) const {
    VPUX_THROW("changeShape method is not implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::changeElemType(mlir::Type elemType) const {
    return VPUIP::ITIBufferType::get(getContext(), getShape().raw(), elemType, getLayout(), getMemSpace(),
                                     getIduSegmentation(), getInwardHaloRegions(), getOutwardHaloRegions());
}

NDTypeInterface VPUIP::ITIBufferType::changeShapeElemType(ShapeRef /*shape*/, mlir::Type /*elemType*/) const {
    VPUX_THROW("changeShapeElemType method is not implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::changeDimsOrder(const DimsOrder& /*order*/) const {
    VPUX_THROW("changeDimsOrder method is not implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    return VPUIP::ITIBufferType::get(getContext(), getShape().raw(), getElementType(), getLayout(), memSpace,
                                     getIduSegmentation(), getInwardHaloRegions(), getOutwardHaloRegions());
}

NDTypeInterface VPUIP::ITIBufferType::changeStrides(StridesRef /*strides*/) const {
    VPUX_THROW("changeStrides method is not yet implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::changeTypeComponents(const vpux::TypeComponents& /*memSpace*/) const {
    VPUX_THROW("changeTypeComponents method is not yet implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::extractDenseTile(ShapeRef /*tileOffsets*/, ShapeRef /*tileShape*/) const {
    VPUX_THROW("extractDenseTile method is not yet implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::extractViewTile(vpux::ShapeRef /*tileOffsets*/, vpux::ShapeRef /*tileShape*/,
                                                      vpux::ShapeRef /*tileElemStrides*/) const {
    VPUX_THROW("extractViewTile method is not yet implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::eraseTiledInfo() const {
    VPUX_THROW("eraseTiledInfo method is not yet implemented for ITIBufferType");
}

NDTypeInterface VPUIP::ITIBufferType::pad(ShapeRef /*padBefore*/, ShapeRef /*padAfter*/) const {
    VPUX_THROW("pad method is not yet implemented for ITIBufferType");
}
