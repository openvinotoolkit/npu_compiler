//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "function.hpp"
#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/type_section.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

intel_npu::vm::Function::Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint,
                                  std::vector<FuncParamResType> paramTypes, std::vector<FuncParamResType> resultTypes,
                                  std::vector<uint8_t> body)
        : _name(std::move(name)),
          _numGeneralRegisters(numGeneralRegisters),
          _isEntrypoint(isEntrypoint),
          _paramTypes(std::move(paramTypes)),
          _resultTypes(std::move(resultTypes)),
          _body(std::move(body)) {
}

std::string intel_npu::vm::Function::getName() const {
    return _name;
}

uint64_t intel_npu::vm::Function::getNumGeneralRegisters() const {
    return _numGeneralRegisters;
}

bool intel_npu::vm::Function::isEntrypoint() const {
    return _isEntrypoint;
}

const std::vector<intel_npu::vm::FuncParamResType>& intel_npu::vm::Function::getParamTypes() const {
    return _paramTypes;
}

const std::vector<intel_npu::vm::FuncParamResType>& intel_npu::vm::Function::getResultTypes() const {
    return _resultTypes;
}

const std::vector<uint8_t>& intel_npu::vm::Function::getBody() const {
    return _body;
}

std::optional<intel_npu::vm::FuncParamResType> intel_npu::vm::extractFuncParamResType(
        intel_npu::vm::BytecodeReader& reader, size_t typeIndex) {
    auto typeEntry = reader.getDataSectionEntry(intel_npu::vm::SectionType::TypeSection, typeIndex);
    if (typeEntry.empty()) {
        NPU_VM_LOG_ERROR("Failed to retrieve type entry for type index {}", typeIndex);
        return std::nullopt;
    }

    const auto typeCode = static_cast<intel_npu::vm::TypeCode>(typeEntry[0]);
    if (typeCode == intel_npu::vm::TypeCode::INTEGER) {
        intel_npu::vm::IntegerType intType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!intType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse IntegerType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{intType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::FLOAT) {
        intel_npu::vm::FloatType floatType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!floatType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse FloatType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{floatType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::OPAQUE) {
        intel_npu::vm::OpaqueType opaqueType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!opaqueType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse OpaqueType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{opaqueType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::BUFFER) {
        intel_npu::vm::BufferType bufferType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!bufferType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse BufferType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{bufferType}, static_cast<int64_t>(typeIndex)};
    } else {
        NPU_VM_LOG_ERROR("Unsupported type code {} for type index {}", static_cast<uint64_t>(typeCode), typeIndex);
        return std::nullopt;
    }
}
