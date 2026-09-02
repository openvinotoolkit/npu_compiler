//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/version.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/bytecode/utils/serialization.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

using namespace vpux;

namespace {

uint64_t countGeneralRegisters(bytecode::FuncOp funcOp) {
    auto registerOps = funcOp.getOps<bytecode::GeneralRegisterOp>();
    std::optional<int64_t> maxRegNum;
    for (auto regOp : registerOps) {
        if (!maxRegNum.has_value() || regOp.getRegNum() > maxRegNum.value()) {
            maxRegNum = regOp.getRegNum();
        }
    }
    const auto numGeneralRegisters = maxRegNum.has_value() ? maxRegNum.value() + 1 : 0;
    return numGeneralRegisters;
}

// Parse a function section and create a corresponding section header
intel_npu::vm::SectionHeader parseFunctionSectionHeader(bytecode::FuncSectionOp funcSection,
                                                        bytecode::TypeSectionOp typeSection) {
    intel_npu::vm::SectionHeader header{};
    header.type = intel_npu::vm::SectionType::FuncSection;
    header.nameIndex = 0;  // Placeholder for actual name index calculation
    header.offset = 0;     // Placeholder for actual offset calculation
    header.size = funcSection.getBinarySize();

    auto typeIndexMap = bytecode::buildTypeIndexMap(typeSection);

    intel_npu::vm::details::FunctionSectionInfo funcInfo;
    funcInfo.entrypointFunctionIndex = 0;  // Placeholder for actual entry point index calculation
    auto functionOps = funcSection.getContent().getOps<bytecode::FuncOp>();
    funcInfo.numFunctions = std::distance(functionOps.begin(), functionOps.end());
    size_t bodyOffset = 0;
    for (auto funcOp : functionOps) {
        auto typeRefName = funcOp.getFunctionTypeRefAttr().getLeafReference().getValue();
        auto it = typeIndexMap.find(typeRefName);
        VPUX_THROW_WHEN(it == typeIndexMap.end(),
                        "Failed to resolve function type reference '@{0}' in the type section", typeRefName);

        intel_npu::vm::details::FunctionSectionInfo::FunctionInfo info{};
        const auto nameIndexOpt = bytecode::getIndex<bytecode::StringSectionOp, bytecode::StringOp>(
                funcOp.getFuncNameAttr(), getModuleOp(funcOp));
        VPUX_THROW_UNLESS(nameIndexOpt.has_value(), "Failed to resolve function name reference {0}",
                          funcOp.getFuncNameAttr());
        info.nameIndex = nameIndexOpt.value();
        info.functionTypeIndex = it->second;
        info.numGeneralRegisters = countGeneralRegisters(funcOp);
        info.bodyOffset = bodyOffset;
        info.bodySize = funcOp.getBinarySize();
        funcInfo.functionInfos.push_back(info);
        bodyOffset += info.bodySize;
    }
    header.info = std::make_unique<intel_npu::vm::details::FunctionSectionInfo>(funcInfo);
    return header;
}

// Parse a section that contains only data (i.e. offsets and sizes) and create a corresponding section header
// This is used for ConstantSection, StringSection and TypeSection
template <typename DataSectionOp, typename DataOp>
intel_npu::vm::SectionHeader parseDataSectionHeader(DataSectionOp sectionOp, intel_npu::vm::SectionType sectionType) {
    intel_npu::vm::SectionHeader header{};
    header.type = sectionType;
    header.nameIndex = 0;  // Placeholder for actual name index calculation
    header.offset = 0;     // Placeholder for actual offset calculation
    header.size = sectionOp.getBinarySize();

    intel_npu::vm::details::DataSectionInfo dataInfo;
    auto dataOps = sectionOp.getContent().template getOps<DataOp>();
    dataInfo.numData = std::distance(dataOps.begin(), dataOps.end());
    size_t dataOffset = 0;
    for (auto dataOp : dataOps) {
        intel_npu::vm::details::DataSectionInfo::DataInfo info{};
        info.offset = dataOffset;
        info.size = dataOp.getBinarySize();
        dataInfo.dataInfos.push_back(info);
        dataOffset += info.size;
    }
    header.info = std::make_unique<intel_npu::vm::details::DataSectionInfo>(dataInfo);
    return header;
}

intel_npu::vm::SectionHeader parseMetadataSectionHeader(bytecode::MetadataSectionOp metadataSection) {
    intel_npu::vm::SectionHeader header{};
    header.type = intel_npu::vm::SectionType::MetadataSection;
    header.nameIndex = 0;
    header.offset = 0;
    header.size = metadataSection.getBinarySize();

    intel_npu::vm::details::DataSectionInfo dataInfo;
    auto metadataOps = metadataSection.getContent().getOps<bytecode::SerializableOpInterface>();
    dataInfo.numData = std::distance(metadataOps.begin(), metadataOps.end());
    size_t dataOffset = 0;
    for (auto metadataOp : metadataOps) {
        intel_npu::vm::details::DataSectionInfo::DataInfo info{};
        info.offset = dataOffset;
        info.size = metadataOp.getBinarySize();
        dataInfo.dataInfos.push_back(info);
        dataOffset += info.size;
    }
    header.info = std::make_unique<intel_npu::vm::details::DataSectionInfo>(dataInfo);
    return header;
}

// Iterate through the IR and find the minimum required bytecode version for all versioned ops in the module. This is
// used to determine the target bytecode version for serialization
intel_npu::vm::Version getMinBytecodeVersion(mlir::ModuleOp moduleOp) {
    auto minVersion = intel_npu::vm::Version::getMinSupportedVersion();
    moduleOp.walk([&](bytecode::VersionedOpInterface versionedOp) {
        const auto requiredVersion = versionedOp.getMinVersion();
        if (requiredVersion > minVersion) {
            minVersion = requiredVersion;
        }
    });
    return minVersion;
}

}  // namespace

bytecode::BytecodeWriter::BytecodeWriter(mlir::ModuleOp moduleOp)
        : _moduleOp(moduleOp), _bytecodeBuffer(), _sectionHeaderTable() {
    prepareSectionHeaderTable();
}

std::vector<uint8_t>& bytecode::BytecodeWriter::getBytecodeBuffer() {
    return _bytecodeBuffer;
}

void bytecode::BytecodeWriter::prepareSectionHeaderTable() {
    // Enforce exactly one type section when function sections are present
    auto typeSectionOps = _moduleOp.getOps<bytecode::TypeSectionOp>();
    auto numTypeSections = std::distance(typeSectionOps.begin(), typeSectionOps.end());
    VPUX_THROW_UNLESS(numTypeSections <= 1, "Expected at most one TypeSectionOp in the module, but found {0}",
                      numTypeSections);

    _moduleOp.walk([&](mlir::Operation* op) {
        llvm::TypeSwitch<mlir::Operation*>(op)
                .Case<bytecode::FuncSectionOp>([&](bytecode::FuncSectionOp funcSection) {
                    VPUX_THROW_WHEN(numTypeSections == 0,
                                    "FuncSectionOp requires a TypeSectionOp for function type resolution");
                    _sectionHeaderTable.addSectionHeader(
                            parseFunctionSectionHeader(funcSection, *typeSectionOps.begin()));
                })
                .Case<bytecode::ConstantSectionOp>([&](bytecode::ConstantSectionOp constantSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<ConstantSectionOp, ConstantOp>(
                            constantSection, intel_npu::vm::SectionType::ConstantSection));
                })
                .Case<bytecode::KernelSectionOp>([&](bytecode::KernelSectionOp kernelSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<KernelSectionOp, KernelOp>(
                            kernelSection, intel_npu::vm::SectionType::KernelSection));
                })
                .Case<bytecode::StringSectionOp>([&](bytecode::StringSectionOp stringSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<StringSectionOp, StringOp>(
                            stringSection, intel_npu::vm::SectionType::StringSection));
                })
                .Case<bytecode::TypeSectionOp>([&](bytecode::TypeSectionOp typeSection) {
                    _sectionHeaderTable.addSectionHeader(parseDataSectionHeader<TypeSectionOp, TypeOp>(
                            typeSection, intel_npu::vm::SectionType::TypeSection));
                })
                .Case<bytecode::MetadataSectionOp>([&](bytecode::MetadataSectionOp metadataSection) {
                    _sectionHeaderTable.addSectionHeader(parseMetadataSectionHeader(metadataSection));
                })
                .Default([](mlir::Operation*) {});
    });

    _sectionHeaderTable.computeOffsets();
}

void bytecode::BytecodeWriter::appendFileHeader() {
    intel_npu::vm::MagicNumber magicNumber(intel_npu::vm::MAGIC_NUMBER);
    magicNumber.appendTo(_bytecodeBuffer);

    auto version = getMinBytecodeVersion(_moduleOp);
    version.appendTo(_bytecodeBuffer);

    _sectionHeaderTable.appendTo(_bytecodeBuffer);
}

void bytecode::BytecodeWriter::appendSections() {
    _moduleOp.walk([&](bytecode::FuncSectionOp funcSection) {
        funcSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::ConstantSectionOp constantSection) {
        constantSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::KernelSectionOp kernelSection) {
        kernelSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::StringSectionOp stringSection) {
        stringSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::TypeSectionOp typeSection) {
        typeSection.serialize(*this);
    });
    _moduleOp.walk([&](bytecode::MetadataSectionOp metadataSection) {
        metadataSection.serialize(*this);
    });
}

void bytecode::BytecodeWriter::appendInstruction(uint16_t opcode, ArrayRef<int16_t> operands) {
    intel_npu::vm::appendValueTo(_bytecodeBuffer, opcode);

    if (operands.empty()) {
        return;
    }
    const auto operandsData = reinterpret_cast<const uint8_t*>(operands.data());
    _bytecodeBuffer.insert(_bytecodeBuffer.end(), operandsData, operandsData + operands.size() * sizeof(int16_t));
}

void bytecode::BytecodeWriter::appendInstruction(uint16_t opcode, ArrayRef<uint8_t> binaryOperands) {
    intel_npu::vm::appendValueTo(_bytecodeBuffer, opcode);

    _bytecodeBuffer.insert(_bytecodeBuffer.end(), binaryOperands.begin(), binaryOperands.end());
}

void bytecode::BytecodeWriter::appendRawData(const uint8_t* data, size_t size) {
    _bytecodeBuffer.insert(_bytecodeBuffer.end(), data, data + size);
}

void bytecode::BytecodeWriter::cacheOffsets(mlir::Region& body) {
    _blockOffsets.clear();
    _opOffsets.clear();
    size_t runningOffset = 0;
    for (auto& block : body) {
        _blockOffsets[&block] = runningOffset;
        for (auto& op : block) {
            if (auto sOp = mlir::dyn_cast<bytecode::SerializableOpInterface>(&op)) {
                _opOffsets[&op] = runningOffset;
                runningOffset += sOp.getBinarySize();
            }
        }
    }
}

int64_t bytecode::BytecodeWriter::getRelativeOffset(mlir::Operation* jumpOp, mlir::Block* destBlock) {
    auto destIt = _blockOffsets.find(destBlock);
    VPUX_THROW_UNLESS(destIt != _blockOffsets.end(), "Jump destination block not found in block offset map");
    auto posIt = _opOffsets.find(jumpOp);
    VPUX_THROW_UNLESS(posIt != _opOffsets.end(), "Jump op position not found in op offset map");
    return checked_cast<int64_t>(destIt->second) - checked_cast<int64_t>(posIt->second);
}

void bytecode::BytecodeWriter::writeTo(llvm::raw_ostream& os) {
    os.write(reinterpret_cast<const char*>(_bytecodeBuffer.data()), _bytecodeBuffer.size());
    os.flush();
}
