//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_export.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace intel_npu::vm {

struct NPU_VM_EXPORT FuncParamResType {
    intel_npu::vm::Type type{};
    int64_t typeSectionIndex{};
};

class NPU_VM_EXPORT Function {
    std::string _name;                           // Human-readable function name
    uint64_t _numGeneralRegisters;               // Number of general-purpose registers used by the function
    bool _isEntrypoint;                          // True if this function is the bytecode entry point
    std::vector<FuncParamResType> _paramTypes;   // Types of the function parameters
    std::vector<FuncParamResType> _resultTypes;  // Types of the function result values
    std::vector<uint8_t> _body;                  // Raw instruction bytes representing the function body

public:
    Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint,
             std::vector<FuncParamResType> paramTypes, std::vector<FuncParamResType> resultTypes,
             std::vector<uint8_t> body);

    /// Get the name of the function
    std::string getName() const;

    /// Get the number of general-purpose registers that the function uses
    uint64_t getNumGeneralRegisters() const;

    /// Returns true if this function is the bytecode entry point (i.e. the function that should be called externally to
    /// execute the bytecode)
    bool isEntrypoint() const;

    /// Get the parameter types of the function
    const std::vector<FuncParamResType>& getParamTypes() const;

    /// Get the result types of the function
    const std::vector<FuncParamResType>& getResultTypes() const;

    /// Get the raw instruction bytes representing the function body
    const std::vector<uint8_t>& getBody() const;
};

// Extracts a function parameter or result type from the type section given its index. The type is expected to be a
// primitive type (integer, float, opaque or buffer). The return value contains the extracted type and its original
// index in the type section
std::optional<FuncParamResType> NPU_VM_EXPORT extractFuncParamResType(intel_npu::vm::BytecodeReader& reader,
                                                                      size_t typeIndex);

}  // namespace intel_npu::vm
