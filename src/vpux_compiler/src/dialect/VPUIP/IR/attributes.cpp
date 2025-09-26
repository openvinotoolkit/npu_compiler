//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <numeric>

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUIP/attributes.cpp.inc>

#include <vpux/compiler/dialect/VPUIP/enums.cpp.inc>

using namespace vpux;

//
// Dialect hooks
//

void VPUIP::VPUIPDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPUIP/attributes.cpp.inc>
            >();
}

//
// SparsityCompressionAttr
//

int64_t VPUIP::SparsityCompressionAttr::getTotalNumElems() const {
    if (getNumElems().empty()) {
        return 0;
    }
    auto numElems = getNumElems().getValues<int64_t>();
    return std::accumulate(numElems.begin(), numElems.end(), static_cast<int64_t>(0));
}

int64_t VPUIP::SparsityCompressionAttr::getNumElemsInRange(int64_t startIdx, int64_t size) const {
    const auto numElems = getNumElems().getValues<int64_t>();
    const auto startIt = numElems.begin() + startIdx;
    const auto endIt = startIt + size;
    return std::accumulate(startIt, endIt, static_cast<int64_t>(0));
}

Byte VPUIP::SparsityCompressionAttr::getAllocSize(mlir::Type elemType) const {
    const auto elemByteSize = getElemTypeSize(elemType).to<Byte>().count();
    const int64_t alignment = (getAlignment() != nullptr) ? getAlignment().getInt() : 1;
    const auto numElems = getNumElems().getValues<int64_t>();
    int64_t totalAllocSize = 0;
    for (auto num : numElems) {
        totalAllocSize += alignValUp<int64_t>(num * elemByteSize, alignment);
    }
    return Byte(totalAllocSize);
}

VPUIP::SparsityCompressionAttr VPUIP::getSparsityCompressionAttr(mlir::Type type) {
    if (type == nullptr) {
        return nullptr;
    }

    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
        if (const auto memRefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(memref.getLayout())) {
            return memRefAttr.hwSpecificField<VPUIP::SparsityCompressionAttr>();
        }
    } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        return distributedBuffer.getSparsityCompression();
    } else if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(type)) {
        return sparseType.getSparsityCompression();
    }

    return nullptr;
}

mlir::Type VPUIP::setSparsityCompressionAttr(mlir::Type type, VPUIP::SparsityCompressionAttr sparsityCompressionAttr) {
    if (type == nullptr) {
        return nullptr;
    }

    if (mlir::isa<mlir::MemRefType>(type)) {
        auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
        const auto strides = ndType.getStrides();
        return getMemRefType(ndType.getShape(), ndType.getElementType(), ndType.getDimsOrder(), ndType.getMemSpace(),
                             strides, getSwizzlingSchemeAttr(type), sparsityCompressionAttr);
    } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
        return VPUIP::DistributedBufferType::get(type.getContext(), distributedBuffer.getShape().raw(),
                                                 distributedBuffer.getElementType(), distributedBuffer.getLayout(),
                                                 distributedBuffer.getMemSpace(), distributedBuffer.getDistribution(),
                                                 sparsityCompressionAttr);
    } else if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(type)) {
        return VPUIP::SparseBufferType::get(sparseType.getData(), sparseType.getSparsityMap(),
                                            sparseType.getStorageElementTable(), sparseType.getIsWeights(),
                                            sparsityCompressionAttr);
    }

    return type;
}

VPUIP::SparsityCompressionAttr VPUIP::tileSparsityCompression(VPUIP::SparsityCompressionAttr sparsityCompression,
                                                              ShapeRef tileOffsets, ShapeRef tileShape) {
    if (sparsityCompression == nullptr) {
        return nullptr;
    }
    VPUX_THROW_UNLESS(sparsityCompression.getAxis() != nullptr,
                      "Cannot tile compression scheme that is not over an axis");
    const size_t axis = sparsityCompression.getAxis().getInt();
    VPUX_THROW_UNLESS(axis < tileOffsets.size() && axis < tileShape.size(),
                      "Axis {0} outside the range of tile dimensions: offsets size {1}, shape size {2}", axis,
                      tileOffsets.size(), tileShape.size());

    const auto numElems = sparsityCompression.getNumElems().getValues<int64_t>();
    const auto dimOffset = tileOffsets[Dim(axis)];
    const auto dimShape = tileShape[Dim(axis)];

    const auto startIt = numElems.begin() + dimOffset;
    const auto endIt = startIt + dimShape;
    const auto tileNumElems = SmallVector<int64_t>(startIt, endIt);

    auto ctx = sparsityCompression.getContext();
    const auto tileNumElemsType =
            mlir::RankedTensorType::get({static_cast<int64_t>(tileNumElems.size())}, getInt64Type(ctx));
    const auto tileNumElemsAttr = mlir::DenseElementsAttr::get(tileNumElemsType, ArrayRef(tileNumElems));
    return VPUIP::SparsityCompressionAttr::get(ctx, sparsityCompression.getAxis(), tileNumElemsAttr,
                                               sparsityCompression.getAlignment());
}

mlir::Type VPUIP::tileTypeSparsityCompression(mlir::Type type, ShapeRef tileOffsets, ShapeRef tileShape) {
    if (type == nullptr) {
        return nullptr;
    }

    const auto sparsityCompression = VPUIP::getSparsityCompressionAttr(type);
    if (sparsityCompression == nullptr) {
        return type;
    }

    const auto tiledSparsityCompression = VPUIP::tileSparsityCompression(sparsityCompression, tileOffsets, tileShape);
    return VPUIP::setSparsityCompressionAttr(type, tiledSparsityCompression);
}

//
// CustomPrinterOutwardHaloRegionAttr
//

::mlir::Attribute VPUIP::OutwardHaloRegionAttr::parse(::mlir::AsmParser& odsParser,
                                                      [[maybe_unused]] ::mlir::Type odsType) {
    ::mlir::Builder odsBuilder(odsParser.getContext());
    ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
    (void)odsLoc;
    ::mlir::FailureOr<mlir::ArrayAttr> resultShape;
    ::mlir::FailureOr<mlir::ArrayAttr> resultOffset;
    ::mlir::FailureOr<mlir::IntegerAttr> resultClusterId;
    ::mlir::FailureOr<mlir::ArrayAttr> resultInwardHaloRegions;
    // Parse literal '<'
    if (odsParser.parseLess()) {
        return {};
    }
    // Parse parameter struct
    bool seenShape = false;
    bool seenOffset = false;
    bool seenClusterId = false;
    bool seenInwardHaloRegions = false;
    {
        const auto loopBody = [&](::llvm::StringRef paramKey) -> bool {
            // Parse literal '='
            if (odsParser.parseEqual()) {
                return {};
            }
            if (!seenShape && paramKey == "shape") {
                seenShape = true;

                // Parse variable 'shape'
                resultShape = ::mlir::FieldParser<mlir::ArrayAttr>::parse(odsParser);
                if (::mlir::failed(resultShape)) {
                    odsParser.emitError(odsParser.getCurrentLocation(),
                                        "failed to parse VPUIP_OutwardHaloRegionAttr parameter 'shape' which is to be "
                                        "a `mlir::ArrayAttr`");
                    return {};
                }
            } else if (!seenOffset && paramKey == "offset") {
                seenOffset = true;

                // Parse variable 'offset'
                resultOffset = ::mlir::FieldParser<mlir::ArrayAttr>::parse(odsParser);
                if (::mlir::failed(resultOffset)) {
                    odsParser.emitError(odsParser.getCurrentLocation(),
                                        "failed to parse VPUIP_OutwardHaloRegionAttr parameter 'offset' which is to be "
                                        "a `mlir::ArrayAttr`");
                    return {};
                }
            } else if (!seenClusterId && paramKey == "cluster_id") {
                seenClusterId = true;

                // Parse variable 'cluster_id'
                resultClusterId = ::mlir::FieldParser<mlir::IntegerAttr>::parse(odsParser);
                if (::mlir::failed(resultClusterId)) {
                    odsParser.emitError(odsParser.getCurrentLocation(),
                                        "failed to parse VPUIP_OutwardHaloRegionAttr parameter 'cluster_id' which is "
                                        "to be a `mlir::IntegerAttr`");
                    return {};
                }
            } else if (!seenInwardHaloRegions && paramKey == "inwardHaloRegions") {
                seenInwardHaloRegions = true;

                // Parse variable 'inwardHaloRegions'
                resultInwardHaloRegions = ::mlir::FieldParser<mlir::ArrayAttr>::parse(odsParser);
                if (::mlir::failed(resultInwardHaloRegions)) {
                    odsParser.emitError(odsParser.getCurrentLocation(),
                                        "failed to parse VPUIP_OutwardHaloRegionAttr parameter 'inwardHaloRegions' "
                                        "which is to be a `mlir::ArrayAttr`");
                    return {};
                }
            } else {
                odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ")
                        << paramKey;
                return {};
            }
            return true;
        };
        for (unsigned odsStructIndex = 0; odsStructIndex < 4; ++odsStructIndex) {
            ::llvm::StringRef paramKey;
            if (odsParser.parseKeyword(&paramKey)) {
                odsParser.emitError(odsParser.getCurrentLocation(), "expected a parameter name in struct");
                return {};
            }
            if (!loopBody(paramKey)) {
                return {};
            }
            if ((odsStructIndex != 4 - 1) && odsParser.parseComma()) {
                return {};
            }
        }
    }
    // Parse literal '>'
    if (odsParser.parseGreater()) {
        return {};
    }
    assert(::mlir::succeeded(resultShape));
    assert(::mlir::succeeded(resultOffset));
    assert(::mlir::succeeded(resultClusterId));
    assert(::mlir::succeeded(resultInwardHaloRegions));
    return OutwardHaloRegionAttr::get(odsParser.getContext(), mlir::ArrayAttr((*resultShape)),
                                      mlir::ArrayAttr((*resultOffset)), mlir::IntegerAttr((*resultClusterId)),
                                      mlir::ArrayAttr((*resultInwardHaloRegions)));
}

void VPUIP::OutwardHaloRegionAttr::print(::mlir::AsmPrinter& odsPrinter) const {
    odsPrinter << "<";

    odsPrinter << "shape = ";
    odsPrinter.printStrippedAttrOrType(getShape());

    odsPrinter << ", ";
    odsPrinter << "offset = ";
    odsPrinter.printStrippedAttrOrType(getOffset());

    odsPrinter << ", ";
    odsPrinter << "cluster_id = ";
    odsPrinter.printStrippedAttrOrType(getClusterId());

    odsPrinter << ", inwardHaloRegions = [";

    odsPrinter.increaseIndent();
    odsPrinter.increaseIndent();
    odsPrinter.printNewline();

    const auto inwardHaloRegions = getInwardHaloRegions();
    for (const auto haloRegion : inwardHaloRegions) {
        odsPrinter.printStrippedAttrOrType(haloRegion);
        if (haloRegion != *(inwardHaloRegions.end() - 1)) {
            odsPrinter << ",";
            odsPrinter.printNewline();
        }
    }
    odsPrinter.decreaseIndent();
    odsPrinter.decreaseIndent();
    odsPrinter.printNewline();
    odsPrinter << "]";

    odsPrinter << ">";
}
