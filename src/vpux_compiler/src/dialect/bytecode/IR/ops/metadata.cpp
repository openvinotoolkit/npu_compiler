//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/metadata.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include "npu_bytecode_utils/network_description.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"

#include <mlir/IR/Builders.h>

#include <cstddef>
#include <cstdint>

namespace {

void serializeDescriptorMetadata(vpux::bytecode::BytecodeWriter& writer, mlir::Operation* op,
                                 intel_npu::vm::MetadataRecordKind kind, mlir::SymbolRefAttr nameRef,
                                 mlir::SymbolRefAttr precisionRef, mlir::SymbolRefAttr shapeRef,
                                 uint32_t indexUsedByDriver, bool hasDynamicStrides, mlir::ArrayAttr tensorNames,
                                 mlir::SymbolRefAttr shapeFromIRModelRef, mlir::SymbolRefAttr nodeFriendlyNameRef) {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp != nullptr, "Expected metadata op to be inside a module");

    using namespace vpux;
    const auto nameIndexOpt = bytecode::getIndex<bytecode::StringSectionOp, bytecode::StringOp>(nameRef, moduleOp);
    VPUX_THROW_UNLESS(nameIndexOpt.has_value(), "Failed to resolve metadata name symbol {0}", nameRef);
    const auto typeIndexOpt = bytecode::getIndex<bytecode::TypeSectionOp, bytecode::TypeOp>(precisionRef, moduleOp);
    VPUX_THROW_UNLESS(typeIndexOpt.has_value(), "Failed to resolve metadata precision symbol {0}", precisionRef);
    const auto shapeIndexOpt =
            bytecode::getIndex<bytecode::ConstantSectionOp, bytecode::ConstantOp>(shapeRef, moduleOp);
    VPUX_THROW_UNLESS(shapeIndexOpt.has_value(), "Failed to resolve metadata shape symbol {0}", shapeRef);

    const auto nameIndex = nameIndexOpt.value();
    const auto typeIndex = typeIndexOpt.value();
    const auto shapeIndex = shapeIndexOpt.value();
    const auto dynamicStrides = static_cast<uint8_t>(hasDynamicStrides);

    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), kind);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), nameIndex);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), typeIndex);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), shapeIndex);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), indexUsedByDriver);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), dynamicStrides);
    // tensor names
    const uint8_t tensorNameCount = (tensorNames != nullptr) ? vpux::checked_cast<uint8_t>(tensorNames.size()) : 0;
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), tensorNameCount);
    if (tensorNameCount > 0) {
        for (auto tensorNameAttr : tensorNames) {
            auto symRefAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(tensorNameAttr);
            VPUX_THROW_UNLESS(symRefAttr != nullptr, "Expected tensor name to be a SymbolRefAttr");
            const auto tensorNameIndex =
                    vpux::bytecode::getStringIndex(symRefAttr.getLeafReference().getValue(), moduleOp);
            intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), tensorNameIndex);
        }
    }

    // shape from IR node
    uint8_t hasShapeFromIRModel = (shapeFromIRModelRef != mlir::SymbolRefAttr()) ? 1 : 0;
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), hasShapeFromIRModel);
    if (hasShapeFromIRModel != 0) {
        const auto shapeFromIRModelIndex =
                vpux::bytecode::getConstantIndex(shapeFromIRModelRef.getLeafReference().getValue(), moduleOp);
        intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), shapeFromIRModelIndex);
    }

    // friendly names
    uint8_t hasNodeFriendlyName = (nodeFriendlyNameRef != mlir::SymbolRefAttr()) ? 1 : 0;
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), hasNodeFriendlyName);
    if (hasNodeFriendlyName) {
        const auto nodeFriendlyNameIndex =
                vpux::bytecode::getStringIndex(nodeFriendlyNameRef.getLeafReference().getValue(), moduleOp);
        intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), nodeFriendlyNameIndex);
    }
}

constexpr size_t DESCRIPTOR_METADATA_BINARY_BASE_SIZE = sizeof(uint8_t) +   // kind
                                                        sizeof(uint64_t) +  // nameIndex
                                                        sizeof(uint64_t) +  // typeIndex
                                                        sizeof(uint64_t) +  // shapeIndex
                                                        sizeof(uint32_t) +  // indexUsedByDriver
                                                        sizeof(uint8_t) +   // hasDynamicStrides
                                                        sizeof(uint8_t) +   // tensorNameCount
                                                        sizeof(uint8_t) +   // hasShapeFromIRModel
                                                        sizeof(uint8_t);    // hasNodeFriendlyName

constexpr size_t NETWORK_METADATA_BINARY_SIZE = sizeof(uint8_t) +   // kind
                                                sizeof(uint64_t) +  // nameIndex
                                                sizeof(uint64_t) +  // numStreams
                                                sizeof(uint64_t);   // numCmdlists

}  // namespace

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/metadata.cpp.inc>

//
// InputMetadataOp
//
template <typename T>
void serialize(vpux::bytecode::BytecodeWriter& writer, T op, intel_npu::vm::MetadataRecordKind kind) {
    mlir::ArrayAttr tensorNames =
            op->hasAttr(bytecode::BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES)
                    ? mlir::dyn_cast<mlir::ArrayAttr>(op->getAttr(bytecode::BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES))
                    : mlir::ArrayAttr();
    mlir::SymbolRefAttr nodeFriendlyName =
            op->hasAttr(bytecode::BYTECODE_IO_DESC_NODE_FRIENDLY_NAME)
                    ? mlir::dyn_cast<mlir::SymbolRefAttr>(op->getAttr(bytecode::BYTECODE_IO_DESC_NODE_FRIENDLY_NAME))
                    : mlir::SymbolRefAttr();
    mlir::SymbolRefAttr shapeFromIRModel =
            op->hasAttr(bytecode::BYTECODE_IO_DESC_SHAPE_FROM_IR_MODEL)
                    ? mlir::dyn_cast<mlir::SymbolRefAttr>(op->getAttr(bytecode::BYTECODE_IO_DESC_SHAPE_FROM_IR_MODEL))
                    : mlir::SymbolRefAttr();

    serializeDescriptorMetadata(writer, op.getOperation(), kind, op.getName(), op.getPrecision(), op.getShape(),
                                checked_cast<uint32_t>(op.getIndexUsedByDriver()), op.getHasDynamicStrides(),
                                tensorNames, shapeFromIRModel, nodeFriendlyName);
}
template <typename T>
size_t getSizeInBytes(T op, size_t baseSizeInBytes) {
    mlir::ArrayAttr tensorNames =
            op->hasAttr(bytecode::BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES)
                    ? mlir::dyn_cast<mlir::ArrayAttr>(op->getAttr(bytecode::BYTECODE_IO_DESC_OUTPUT_TENSOR_NAMES))
                    : mlir::ArrayAttr();
    return baseSizeInBytes + sizeof(uint64_t) * ((tensorNames != nullptr) ? tensorNames.size() : 0) +
           (op->hasAttr(bytecode::BYTECODE_IO_DESC_NODE_FRIENDLY_NAME) ? sizeof(uint64_t) : 0) +
           (op->hasAttr(bytecode::BYTECODE_IO_DESC_SHAPE_FROM_IR_MODEL) ? sizeof(uint64_t) : 0);
}

void bytecode::InputMetadataOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    ::serialize<bytecode::InputMetadataOp>(writer, *this, intel_npu::vm::MetadataRecordKind::Input);
}

size_t bytecode::InputMetadataOp::getBinarySize() {
    return ::getSizeInBytes(*this, DESCRIPTOR_METADATA_BINARY_BASE_SIZE);
}

//
// OutputMetadataOp
//

void bytecode::OutputMetadataOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    ::serialize<bytecode::OutputMetadataOp>(writer, *this, intel_npu::vm::MetadataRecordKind::Output);
}

size_t bytecode::OutputMetadataOp::getBinarySize() {
    return ::getSizeInBytes(*this, DESCRIPTOR_METADATA_BINARY_BASE_SIZE);
}

//
// ProfilingOutputMetadataOp
//

void bytecode::ProfilingOutputMetadataOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    ::serialize<bytecode::ProfilingOutputMetadataOp>(writer, *this, intel_npu::vm::MetadataRecordKind::ProfilingOutput);
}

size_t bytecode::ProfilingOutputMetadataOp::getBinarySize() {
    return ::getSizeInBytes(*this, DESCRIPTOR_METADATA_BINARY_BASE_SIZE);
}

//
// NetworkMetadataOp
//

void bytecode::NetworkMetadataOp::serialize(vpux::bytecode::BytecodeWriter& writer) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp != nullptr, "Expected network metadata op to be inside a module");

    const auto nameIndexOpt = bytecode::getIndex<bytecode::StringSectionOp, bytecode::StringOp>(getName(), moduleOp);
    VPUX_THROW_UNLESS(nameIndexOpt.has_value(), "Failed to resolve network metadata name symbol {0}", getName());
    const auto nameIndex = nameIndexOpt.value();
    const auto numStreams = static_cast<uint64_t>(getNumStreams());
    const auto numCmdLists = static_cast<uint64_t>(getNumCmdlists());

    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), intel_npu::vm::MetadataRecordKind::Network);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), nameIndex);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), numStreams);
    intel_npu::vm::appendValueTo(writer.getBytecodeBuffer(), numCmdLists);
}

size_t bytecode::NetworkMetadataOp::getBinarySize() {
    return NETWORK_METADATA_BINARY_SIZE;
}
