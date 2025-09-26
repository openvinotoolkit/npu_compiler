//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

using namespace vpux;

//
// print/parse
//

void VPUIP::SparseBufferType::print(mlir::AsmPrinter& printer) const {
    printer << "<data=" << getData();
    if (const auto& sparsityMap = getSparsityMap()) {
        printer << ", sparsity_map=" << sparsityMap;
    }
    if (const auto& storageElementTable = getStorageElementTable()) {
        printer << ", storage_element_table=" << storageElementTable;
    }
    if (getIsWeights() != nullptr) {
        printer << ", is_weights";
    }
    if (getSparsityCompression() != nullptr) {
        printer << ", " << getSparsityCompression();
    }
    if (getSeAttr() != nullptr) {
        printer << ", " << getSeAttr();
    }
    printer << ">";
}

mlir::Type VPUIP::SparseBufferType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return Type();
    }
    mlir::Type data;
    mlir::Type sparsityMap;
    mlir::Type storageElementTable;
    mlir::UnitAttr isWeights;
    VPUIP::SparsityCompressionAttr sparsityCompression;
    VPU::SEAttr seAttr;

    if (parser.parseKeyword("data")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(data)) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalGreater())) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("is_weights"))) {
        if (mlir::succeeded(parser.parseOptionalComma()) && parser.parseAttribute(sparsityCompression)) {
            return Type();
        }
        if (parser.parseGreater()) {
            return Type();
        }
        isWeights = mlir::UnitAttr::get(parser.getContext());
        return get(data, sparsityMap, storageElementTable, isWeights, sparsityCompression);
    }
    if (parser.parseKeyword("sparsity_map")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(sparsityMap)) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalGreater())) {
        return get(data, sparsityMap, storageElementTable);
    }

    if (parser.parseComma()) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalKeyword("is_weights"))) {
        if (mlir::succeeded(parser.parseOptionalComma())) {
            if (parser.parseAttribute(sparsityCompression)) {
                return Type();
            }
        }
        if (parser.parseGreater()) {
            return Type();
        }
        isWeights = mlir::UnitAttr::get(parser.getContext());
        return get(data, sparsityMap, storageElementTable, isWeights, sparsityCompression);
    }
    if (parser.parseKeyword("storage_element_table")) {
        return Type();
    }
    if (parser.parseEqual()) {
        return Type();
    }
    if (parser.parseType<mlir::Type>(storageElementTable)) {
        return Type();
    }
    if (mlir::succeeded(parser.parseOptionalComma())) {
        if (mlir::succeeded(parser.parseOptionalKeyword("is_weights"))) {
            if (mlir::succeeded(parser.parseOptionalComma()) && parser.parseAttribute(sparsityCompression)) {
                return Type();
            }
            if (parser.parseGreater()) {
                return Type();
            }
            isWeights = mlir::UnitAttr::get(parser.getContext());
            return get(data, sparsityMap, storageElementTable, isWeights, sparsityCompression);
        }

        if (parser.parseAttribute(seAttr)) {
            return Type();
        }
        if (parser.parseGreater()) {
            return Type();
        }
        return get(data, sparsityMap, storageElementTable, isWeights, sparsityCompression, seAttr);
    }
    if (parser.parseGreater()) {
        return Type();
    }

    return get(data, sparsityMap, storageElementTable);
}

//
// verify
//

mlir::LogicalResult VPUIP::SparseBufferType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                    mlir::Type data, mlir::Type sparsityMap, mlir::Type seTable,
                                                    mlir::UnitAttr isWeights,
                                                    VPUIP::SparsityCompressionAttr sparsityCompression,
                                                    VPU::SEAttr seAttribute) {
    if (!mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(data)) {
        return printTo(emitError(), "Data type is not a memref or distributed buffer. Got {0}", data);
    }
    if (sparsityMap != nullptr && !mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(sparsityMap)) {
        return printTo(emitError(), "Sparsity map type is not a memref or distributed buffer. Got {0}", sparsityMap);
    }
    if (seTable != nullptr && !mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(seTable)) {
        return printTo(emitError(), "Storage element table type is not a memref or distributed buffer. Got {0}",
                       seTable);
    }
    if ((seAttribute != nullptr) && ((isWeights != nullptr) || (sparsityCompression != nullptr))) {
        return printTo(emitError(),
                       "SEAttr and (isWeights or SparsityCompressionAttr) cannot be present at the same time.");
    }
    if (mlir::isa<vpux::VPUIP::DistributedBufferType>(data)) {
        if (sparsityMap != nullptr && !mlir::isa<vpux::VPUIP::DistributedBufferType>(sparsityMap)) {
            return printTo(emitError(), "Sparsity map of type {0} is not a distributed buffer while data is",
                           sparsityMap);
        }
        if (seTable != nullptr && !mlir::isa<vpux::VPUIP::DistributedBufferType>(seTable)) {
            return printTo(emitError(), "Storage element table of type {0} is not a distributed buffer while data is",
                           seTable);
        }
    }

    return mlir::success();
}

//
// NDTypeInterface
//

ShapeRef VPUIP::SparseBufferType::getShape() const {
    const auto data = VPU::getEffectiveSparseOutputType(*this);
    return mlir::cast<NDTypeInterface>(data).getShape();
}

MemShape VPUIP::SparseBufferType::getMemShape() const {
    const auto data = VPU::getEffectiveSparseOutputType(*this);
    return mlir::cast<NDTypeInterface>(data).getMemShape();
}

bool VPUIP::SparseBufferType::hasRank() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.hasRank();
}

int64_t VPUIP::SparseBufferType::getRank() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getRank();
}

int64_t VPUIP::SparseBufferType::getNumElements() const {
    if (getSparsityCompression() != nullptr) {
        return getSparsityCompression().getTotalNumElems();
    }
    const auto data = VPU::getEffectiveSparseOutputType(*this);
    return mlir::cast<NDTypeInterface>(data).getNumElements();
}

mlir::Type VPUIP::SparseBufferType::getElementType() const {
    const auto data = VPU::getEffectiveSparseOutputType(*this);
    return mlir::cast<NDTypeInterface>(data).getElementType();
}

DimsOrder VPUIP::SparseBufferType::getDimsOrder() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getDimsOrder();
}

IndexedSymbolAttr VPUIP::SparseBufferType::getMemSpace() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemSpace();
}

VPU::MemoryKind VPUIP::SparseBufferType::getMemoryKind() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getMemoryKind();
}

Strides VPUIP::SparseBufferType::getStrides() const {
    auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    if (getSeAttr() != nullptr) {
        // If SEAttr is set then return effective shape, srides are compact
        data = mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(*this));
    }
    return data.getStrides();
}

MemStrides VPUIP::SparseBufferType::getMemStrides() const {
    auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    if (getSeAttr() != nullptr) {
        // If SEAttr is set then return effective shape, srides are compact
        data = mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(*this));
    }
    return data.getMemStrides();
}

Bit VPUIP::SparseBufferType::getElemTypeSize() const {
    const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
    return data.getElemTypeSize();
}

Byte VPUIP::SparseBufferType::getTotalAllocSize() const {
    Byte totalSize(0);
    if (getSparsityCompression() != nullptr) {
        totalSize = getSparsityCompression().getAllocSize(getElementType());
    } else {
        const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
        totalSize = data.getTotalAllocSize();
    }
    if (getSparsityMap() != nullptr) {
        const auto sparsityMap = mlir::cast<vpux::NDTypeInterface>(getSparsityMap());
        totalSize += sparsityMap.getTotalAllocSize();
    }
    if (getStorageElementTable() != nullptr) {
        const auto storageElementTable = mlir::cast<vpux::NDTypeInterface>(getStorageElementTable());
        totalSize += storageElementTable.getTotalAllocSize();
    }
    return totalSize;
}

Byte VPUIP::SparseBufferType::getCompactAllocSize() const {
    Byte compactSize(0);
    if (getSparsityCompression() != nullptr) {
        compactSize = getSparsityCompression().getAllocSize(getElementType());
    } else {
        const auto data = mlir::cast<vpux::NDTypeInterface>(getData());
        compactSize = data.getTotalAllocSize();
    }
    if (getSparsityMap() != nullptr) {
        const auto sparsityMap = mlir::cast<vpux::NDTypeInterface>(getSparsityMap());
        compactSize += sparsityMap.getCompactAllocSize();
    }
    if (getStorageElementTable() != nullptr) {
        const auto storageElementTable = mlir::cast<vpux::NDTypeInterface>(getStorageElementTable());
        compactSize += storageElementTable.getCompactAllocSize();
    }
    return compactSize;
}

NDTypeInterface VPUIP::SparseBufferType::changeShape(ShapeRef shape) const {
    return changeShapeElemType(shape, getElementType());
}

NDTypeInterface VPUIP::SparseBufferType::changeElemType(mlir::Type elemType) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto data = ndData.changeElemType(elemType);
    return VPUIP::SparseBufferType::get(data, getSparsityMap(), getStorageElementTable(), getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::changeShapeElemType(ShapeRef shape, mlir::Type elemType) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    Shape inputDataShape(shape.toValues());
    if (auto seAttr = getSeAttr()) {
        inputDataShape = seAttr.backInferInputShape(shape);
    }
    const auto data = ndData.changeShapeElemType(inputDataShape, elemType);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(shape);
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.changeShape(shape);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape);
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::changeDimsOrder(DimsOrder order) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto data = ndData.changeDimsOrder(order);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        if (getIsWeights() == nullptr) {
            sparsityMap = ndSparsityMap.changeDimsOrder(order);
        }
    }

    // The order of the storage element table should not be changed since it is always 1xDxHxW
    const auto storageElementTable = getStorageElementTable();

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::changeMemSpace(IndexedSymbolAttr memSpace) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto data = ndData.changeMemSpace(memSpace);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        sparsityMap = ndSparsityMap.changeMemSpace(memSpace);
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        storageElementTable = ndStorageElementTable.changeMemSpace(memSpace);
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::changeStrides(StridesRef strides) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    auto data = ndData;
    if (getSeAttr() != nullptr) {
        // If SEAttr is set then this method works with effective data and it does not support
        // non compact strides for now
        const auto compact = StrideReqs::compact(ndData.getRank());
        const auto effectiveData =
                mlir::cast<NDTypeInterface>(VPU::getEffectiveSparseOutputType(*this)).changeStrides(strides);
        VPUX_THROW_WHEN(compact.checkStrides(effectiveData) == false,
                        "If SEAttr is set then then only compact input supported, got {0}", effectiveData);
    } else {
        data = ndData.changeStrides(strides);
    }
    return VPUIP::SparseBufferType::get(data, getSparsityMap(), getStorageElementTable(), getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::changeTypeComponents(const vpux::TypeComponents& typeComponents) const {
    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());

    Shape newInputDataShape(shape);
    if (auto seAttr = getSeAttr()) {
        newInputDataShape = seAttr.backInferInputShape(shape);
    }
    TypeComponents dataTypeComponents(typeComponents);
    const auto newData = ndData.changeTypeComponents(dataTypeComponents.setShape(newInputDataShape));

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        auto smTypeComponents = TypeComponents().setMemSpace(memSpace);

        if (getIsWeights() == nullptr) {
            smTypeComponents = smTypeComponents.setShape(shape).setDimsOrder(dimsOrder);
        } else {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(shape);
            smTypeComponents = smTypeComponents.setShape(newSMShape);
        }
        sparsityMap = ndSparsityMap.changeTypeComponents(smTypeComponents);
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);

        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];

        const auto SETComponents = TypeComponents().setShape(seTableShape).setMemSpace(memSpace);
        storageElementTable = ndStorageElementTable.changeTypeComponents(SETComponents);
    }

    return VPUIP::SparseBufferType::get(newData, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::extractDenseTile(ShapeRef tileOffsets, ShapeRef tileShape) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    Shape inputTileShape(tileShape.raw());
    Shape inputTileStart(tileOffsets.raw());
    auto seAttr = getSeAttr();
    if (seAttr != nullptr) {
        seAttr = seAttr.extractTile(tileOffsets, tileShape, ndData.getShape(), inputTileStart, inputTileShape);
    }
    const auto data = ndData.extractDenseTile(inputTileStart, inputTileShape);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(tileShape);
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.extractDenseTile(tileOffsets, tileShape);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableTileOffsets = Shape(tileOffsets.size(), 0);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];

        if (seTableTileShape[Dims4D::Act::C] != 1 && seTableTileShape[Dims4D::Act::C] != tileShape[Dims4D::Act::C]) {
            const auto seSize = ndData.getShape()[Dims4D::Act::C] / seTableTileShape[Dims4D::Act::C];
            seTableTileOffsets[Dims4D::Act::C] = tileOffsets[Dims4D::Act::C] / seSize;
            seTableTileShape[Dims4D::Act::C] = tileShape[Dims4D::Act::C] / seSize;
        }

        storageElementTable = ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(), sparsityCompression,
                                        seAttr);
}

NDTypeInterface VPUIP::SparseBufferType::extractViewTile(ShapeRef tileOffsets, ShapeRef tileShape,
                                                         ShapeRef tileElemStrides) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    Shape inputTileShape(tileShape.raw());
    Shape inputTileStart(tileOffsets.raw());
    auto seAttr = getSeAttr();
    if (seAttr != nullptr) {
        if (!tileElemStrides.empty()) {
            const auto strided = std::any_of(tileElemStrides.begin(), tileElemStrides.begin(), [](auto val) {
                return val != 1;
            });
            VPUX_THROW_WHEN(strided, "Extracting view tile with non dense strides is not supported if SEAttr is set");
        }
        seAttr = seAttr.extractTile(tileOffsets, tileShape, ndData.getShape(), inputTileStart, inputTileShape);
    }
    const auto data = ndData.extractViewTile(inputTileStart, inputTileShape, tileElemStrides);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(tileShape);
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.extractViewTile(tileOffsets, tileShape, tileElemStrides);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableTileOffsets = Shape(tileOffsets.size(), 0);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.extractDenseTile(seTableTileOffsets, seTableTileShape);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(), sparsityCompression,
                                        seAttr);
}

NDTypeInterface VPUIP::SparseBufferType::eraseTiledInfo() const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    const auto data = ndData.eraseTiledInfo();

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        sparsityMap = ndSparsityMap.eraseTiledInfo();
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        storageElementTable = ndStorageElementTable.eraseTiledInfo();
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

NDTypeInterface VPUIP::SparseBufferType::pad(ShapeRef padBefore, ShapeRef padAfter) const {
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());
    auto data = ndData.pad(padBefore, padAfter);

    Shape paddedOutputShape(data.getShape().toValues());
    if (auto seAttr = getSeAttr()) {
        paddedOutputShape = Shape(ndData.changeShape(getShape()).pad(padBefore, padAfter).getShape().raw());
        const auto inferredShape = seAttr.backInferInputShape(paddedOutputShape);
        data = data.changeShape(inferredShape);
    }

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto ndSparsityMap = mlir::cast<vpux::NDTypeInterface>(sparsityMap);
        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(paddedOutputShape);
            sparsityMap = ndSparsityMap.changeShape(newSMShape);
        } else {
            sparsityMap = ndSparsityMap.changeShape(paddedOutputShape);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = paddedOutputShape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = paddedOutputShape[Dims4D::Act::W];
        storageElementTable = ndStorageElementTable.changeShape(seTableShape);
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

//
// DistributedTypeInterface
//

bool VPUIP::SparseBufferType::containsDistributedTypes() const {
    // If the data is a distributed type, the metadata will be as well
    return mlir::isa<vpux::VPUIP::DistributedBufferType>(getData());
}

SmallVector<mlir::Type> VPUIP::SparseBufferType::getDistributedTypes() const {
    SmallVector<mlir::Type> distributedTypes;
    if (mlir::isa<vpux::VPUIP::DistributedBufferType>(getData())) {
        distributedTypes.push_back(getData());
    }
    if (getSparsityMap() != nullptr && mlir::isa<vpux::VPUIP::DistributedBufferType>(getSparsityMap())) {
        distributedTypes.push_back(getSparsityMap());
    }
    if (getStorageElementTable() != nullptr &&
        mlir::isa<vpux::VPUIP::DistributedBufferType>(getStorageElementTable())) {
        distributedTypes.push_back(getStorageElementTable());
    }
    return distributedTypes;
}

// @brief Change the type's shapes when the DistributedType has an explicit DistributionInfoAttr.
// Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
// for its components, based on the one that was passed as an arg.
NDTypeInterface VPUIP::SparseBufferType::changeShapeForExplicitDistribution(
        ShapeRef shape, VPU::DistributionInfoAttr distributedAttr) const {
    return changeShapeElemTypeForExplicitDistribution(shape, getElementType(), distributedAttr);
}

// @brief Change the type's shapes and elem type when the DistributedType has an explicit DistributionInfoAttr.
// Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
// for its components, based on the one that was passed as an arg.
NDTypeInterface VPUIP::SparseBufferType::changeShapeElemTypeForExplicitDistribution(
        ShapeRef shape, mlir::Type elemType, VPU::DistributionInfoAttr distributedAttr) const {
    const auto distributedData = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(getData());
    VPUX_THROW_WHEN(distributedData == nullptr,
                    "Calling changeShapeElemTypeForExplicitDistribution on SparseBuffer that does not have "
                    "DistributedBuffer data, {0}",
                    getData());
    auto ctx = getContext();

    Shape inputDataShape(shape.toValues());
    if (auto seAttr = getSeAttr()) {
        inputDataShape = seAttr.backInferInputShape(shape);
    }
    auto dataDistributedAttr =
            VPU::getExplicitDistrAttrForSparseData(distributedAttr, inputDataShape, getSeAttr(), ctx);
    const auto data =
            distributedData.changeShapeElemTypeForExplicitDistribution(inputDataShape, elemType, dataDistributedAttr);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto distributedSparsityMap = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(sparsityMap);
        VPUX_THROW_WHEN(distributedSparsityMap == nullptr,
                        "Calling changeShapeElemTypeForExplicitDistribution on SparseBuffer that does not have "
                        "DistributedBuffer sparsity map, {0}",
                        getSparsityMap());
        Shape newSMShape = shape.toValues();
        if (getIsWeights() != nullptr) {
            newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(shape);
        }

        auto smDistributedAttr =
                VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, newSMShape, getIsWeights(), ctx);

        sparsityMap = distributedSparsityMap.changeShapeForExplicitDistribution(newSMShape, smDistributedAttr);
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto distributedSETable = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(storageElementTable);
        VPUX_THROW_WHEN(distributedSETable == nullptr,
                        "Calling changeShapeElemTypeForExplicitDistribution on SparseBuffer that does not have "
                        "DistributedBuffer Storage Element Table, {0}",
                        getStorageElementTable());

        auto seTableShape = Shape(distributedSETable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];

        const auto prevSeSize = getShape()[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];
        const auto newSeSize = shape[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];

        VPUX_THROW_WHEN(prevSeSize != newSeSize && seTableShape[Dims4D::Act::C] != 1,
                        "After changeShapeElemTypeForExplicitDistribution, StorageElementTable does not have the same "
                        "seSize: sparseBuffer = {0}, new shape = {1}",
                        *this, shape);

        auto seTableDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, newSeSize, ctx);
        storageElementTable =
                distributedSETable.changeShapeForExplicitDistribution(seTableShape, seTableDistributedAttr);
    }

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

// @brief Change the type's components when the DistributedType has an explicit DistributionInfoAttr.
// Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
// for its components, based on the one that was passed as an arg.
NDTypeInterface VPUIP::SparseBufferType::changeTypeComponentsForExplicitDistribution(
        const TypeComponents& typeComponents, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return changeTypeComponents(typeComponents);
    }

    auto ctx = getContext();

    const auto shape = typeComponents.shape.value_or(Shape(getShape().toValues()));
    const auto dimsOrder = typeComponents.dimsOrder.value_or(getDimsOrder());
    const auto memSpace = typeComponents.memSpace.value_or(getMemSpace());

    Shape newInputDataShape(shape);
    if (auto seAttr = getSeAttr()) {
        newInputDataShape = seAttr.backInferInputShape(shape);
    }

    auto dataDistributedAttr =
            VPU::getExplicitDistrAttrForSparseData(distributedAttr, newInputDataShape, getSeAttr(), ctx);

    TypeComponents dataTypeComponents(typeComponents);
    const auto clusteredData = mlir::cast<vpux::ClusterTypeInterface>(getData());
    const auto newData = clusteredData.changeTypeComponentsForExplicitDistribution(
            dataTypeComponents.setShape(newInputDataShape), dataDistributedAttr);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto clusteredSparsityMap = mlir::cast<vpux::ClusterTypeInterface>(sparsityMap);
        auto smTypeComponents = TypeComponents().setMemSpace(memSpace);

        if (getIsWeights() == nullptr) {
            smTypeComponents = smTypeComponents.setShape(shape).setDimsOrder(dimsOrder);
        } else {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(shape);
            smTypeComponents = smTypeComponents.setShape(newSMShape);
        }

        auto smDistributedAttr = VPU::getExplicitDistrAttrForSparsityMap(
                distributedAttr, smTypeComponents.shape.value(), getIsWeights(), ctx);
        sparsityMap =
                clusteredSparsityMap.changeTypeComponentsForExplicitDistribution(smTypeComponents, smDistributedAttr);
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableShape = Shape(ndStorageElementTable.getShape().raw());
        seTableShape[Dims4D::Act::H] = shape[Dims4D::Act::H];
        seTableShape[Dims4D::Act::W] = shape[Dims4D::Act::W];

        const auto prevSeSize = getShape()[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];
        const auto newSeSize = shape[Dims4D::Act::C] / seTableShape[Dims4D::Act::C];

        VPUX_THROW_WHEN(prevSeSize != newSeSize && seTableShape[Dims4D::Act::C] != 1,
                        "After changeTypeComponentsForExplicitDistribution, StorageElementTable does not have the same "
                        "seSize: sparseBuffer = {0}, new shape = {1}",
                        *this, shape);

        auto seTableDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, newSeSize, ctx);

        const auto clusterSETable = mlir::cast<vpux::ClusterTypeInterface>(storageElementTable);
        const auto SETComponents = TypeComponents().setShape(seTableShape).setMemSpace(memSpace);
        storageElementTable =
                clusterSETable.changeTypeComponentsForExplicitDistribution(SETComponents, seTableDistributedAttr);
    }

    return VPUIP::SparseBufferType::get(newData, sparsityMap, storageElementTable, getIsWeights(),
                                        getSparsityCompression(), getSeAttr());
}

// @brief Extract a dense tile from a DistributedType with an explicit DistributionInfoAttr.
// Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
// for its components, based on the one that was passed as an arg.
NDTypeInterface VPUIP::SparseBufferType::extractDenseTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractDenseTile(tileOffsets, tileShape);
    }

    auto ctx = getContext();

    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());

    Shape inputTileShape(tileShape.raw());
    Shape inputTileStart(tileOffsets.raw());
    auto seAttr = getSeAttr();
    if (seAttr != nullptr) {
        seAttr = seAttr.extractTile(tileOffsets, tileShape, ndData.getShape(), inputTileStart, inputTileShape);
    }

    auto dataDistributedAttr = VPU::getExplicitDistrAttrForSparseData(distributedAttr, inputTileShape, seAttr, ctx);

    const auto data = mlir::cast<vpux::ClusterTypeInterface>(ndData).extractDenseTileForExplicitDistribution(
            inputTileStart, inputTileShape, dataDistributedAttr);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto clusteredSparsityMap = mlir::cast<vpux::ClusterTypeInterface>(sparsityMap);

        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(tileShape);

            auto smDistributedAttr =
                    VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, newSMShape, getIsWeights(), ctx);

            sparsityMap = clusteredSparsityMap.changeShapeForExplicitDistribution(newSMShape, smDistributedAttr);
        } else {
            auto smDistributedAttr =
                    VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, tileShape, getIsWeights(), ctx);
            sparsityMap = clusteredSparsityMap.extractDenseTileForExplicitDistribution(tileOffsets, tileShape,
                                                                                       smDistributedAttr);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);

        auto seTableTileOffsets = Shape(tileOffsets.size(), 0);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];

        const auto prevSeSize = getShape()[Dims4D::Act::C] / seTableTileShape[Dims4D::Act::C];
        const auto newSeSize = tileShape[Dims4D::Act::C] / seTableTileShape[Dims4D::Act::C];

        VPUX_THROW_WHEN(prevSeSize != newSeSize && seTableTileShape[Dims4D::Act::C] != 1,
                        "After extractDenseTileForExplicitDistribution, StorageElementTable does not have the same "
                        "seSize: sparseBuffer = {0}, new shape = {1}",
                        *this, tileShape);

        auto seTableDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, newSeSize, ctx);

        storageElementTable = mlir::cast<vpux::ClusterTypeInterface>(ndStorageElementTable)
                                      .extractDenseTileForExplicitDistribution(seTableTileOffsets, seTableTileShape,
                                                                               seTableDistributedAttr);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(), sparsityCompression,
                                        seAttr);
}

NDTypeInterface VPUIP::SparseBufferType::extractViewTileForExplicitDistribution(
        vpux::ShapeRef tileOffsets, vpux::ShapeRef tileShape, vpux::ShapeRef tileElemStrides,
        VPU::DistributionInfoAttr distributedAttr) const {
    if (distributedAttr == nullptr) {
        return extractViewTile(tileOffsets, tileShape, tileElemStrides);
    }

    auto ctx = getContext();
    const auto ndData = mlir::cast<vpux::NDTypeInterface>(getData());

    Shape inputTileShape(tileShape.raw());
    Shape inputTileStart(tileOffsets.raw());
    auto seAttr = getSeAttr();
    if (seAttr != nullptr) {
        if (!tileElemStrides.empty()) {
            const auto strided = std::any_of(tileElemStrides.begin(), tileElemStrides.begin(), [](auto val) {
                return val != 1;
            });
            VPUX_THROW_WHEN(strided, "Extracting view tile with non dense strides is not supported if SEAttr is set");
        }
        seAttr = seAttr.extractTile(tileOffsets, tileShape, ndData.getShape(), inputTileStart, inputTileShape);
    }

    auto dataDistributedAttr = VPU::getExplicitDistrAttrForSparseData(distributedAttr, inputTileShape, seAttr, ctx);

    const auto data = mlir::cast<vpux::ClusterTypeInterface>(ndData).extractViewTileForExplicitDistribution(
            inputTileStart, inputTileShape, tileElemStrides, dataDistributedAttr);

    auto sparsityMap = getSparsityMap();
    if (sparsityMap != nullptr) {
        const auto clusteredSparsityMap = mlir::cast<vpux::ClusterTypeInterface>(sparsityMap);

        if (getIsWeights() != nullptr) {
            auto newSMShape = VPU::NCESparsity::inferWeightsSparsityMapShape(tileShape);

            auto smDistributedAttr =
                    VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, newSMShape, getIsWeights(), ctx);
            sparsityMap = clusteredSparsityMap.changeShapeForExplicitDistribution(newSMShape, smDistributedAttr);
        } else {
            auto smDistributedAttr =
                    VPU::getExplicitDistrAttrForSparsityMap(distributedAttr, tileShape, getIsWeights(), ctx);
            sparsityMap = clusteredSparsityMap.extractViewTileForExplicitDistribution(
                    tileOffsets, tileShape, tileElemStrides, smDistributedAttr);
        }
    }

    auto storageElementTable = getStorageElementTable();
    if (storageElementTable != nullptr) {
        const auto ndStorageElementTable = mlir::cast<vpux::NDTypeInterface>(storageElementTable);
        auto seTableTileOffsets = Shape(tileOffsets.size(), 0);
        seTableTileOffsets[Dims4D::Act::H] = tileOffsets[Dims4D::Act::H];
        seTableTileOffsets[Dims4D::Act::W] = tileOffsets[Dims4D::Act::W];
        auto seTableTileShape = Shape(ndStorageElementTable.getShape().raw());
        seTableTileShape[Dims4D::Act::H] = tileShape[Dims4D::Act::H];
        seTableTileShape[Dims4D::Act::W] = tileShape[Dims4D::Act::W];

        const auto prevSeSize = getShape()[Dims4D::Act::C] / seTableTileShape[Dims4D::Act::C];
        const auto newSeSize = tileShape[Dims4D::Act::C] / seTableTileShape[Dims4D::Act::C];

        VPUX_THROW_WHEN(prevSeSize != newSeSize && seTableTileShape[Dims4D::Act::C] != 1,
                        "After extractViewTileForExplicitDistribution, StorageElementTable does not have the same "
                        "seSize: sparseBuffer = {0}, new shape = {1}",
                        *this, tileShape);

        auto seTableDistributedAttr = VPU::getExplicitDistrAttrForSETable(distributedAttr, newSeSize, ctx);

        storageElementTable = mlir::cast<vpux::ClusterTypeInterface>(ndStorageElementTable)
                                      .extractDenseTileForExplicitDistribution(seTableTileOffsets, seTableTileShape,
                                                                               seTableDistributedAttr);
    }

    const auto sparsityCompression = VPUIP::tileSparsityCompression(getSparsityCompression(), tileOffsets, tileShape);

    return VPUIP::SparseBufferType::get(data, sparsityMap, storageElementTable, getIsWeights(), sparsityCompression,
                                        seAttr);
}
