//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/section_header_table.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpux::bytecode {

// Serializes an MLIR ModuleOp into a flat binary bytecode format.
// The writer builds an in-memory buffer containing the file header and serialized section payloads, then writes the
// complete buffer to an output stream.
class BytecodeWriter {
    mlir::ModuleOp _moduleOp;
    std::vector<uint8_t> _bytecodeBuffer;
    intel_npu::vm::SectionHeaderTable _sectionHeaderTable;
    llvm::DenseMap<mlir::Block*, size_t> _blockOffsets;
    llvm::DenseMap<mlir::Operation*, size_t> _opOffsets;

    void prepareSectionHeaderTable();

public:
    // Construct a BytecodeWriter for the given module.
    // The section header table is prepared eagerly during construction.
    explicit BytecodeWriter(mlir::ModuleOp moduleOp);

    // Returns a reference to the internal bytecode buffer being built up by the writer for serialization
    std::vector<uint8_t>& getBytecodeBuffer();

    // Append the file header (magic number, version, section header table) to the internal bytecode buffer
    void appendFileHeader();

    // Serialize every section body (functions, constants, strings, types)
    // and append the resulting bytes to the internal bytecode buffer
    void appendSections();

    /// Encode a single instruction and append it to the bytecode buffer.
    /// @param opcode         base opcode value
    /// @param operands       variable-length operand list
    void appendInstruction(uint16_t opcode, ArrayRef<int16_t> operands);

    /// Encode a single instruction and append it to the bytecode buffer.
    /// @param opcode         base opcode value
    /// @param binaryOperands binary representation of the operands to append directly to the instruction (used for
    /// instructions whose operands are not only 16-bit integers)
    void appendInstruction(uint16_t opcode, ArrayRef<uint8_t> binaryOperands);

    /// Append raw binary data to the bytecode buffer
    /// @param data pointer to the beginning of the data
    /// @param size size of the data in bytes
    void appendRawData(const uint8_t* data, size_t size);

    // Pre-compute block and op byte-offsets (relative to function body start) for
    // all serializable ops in `body`. Must be called before getRelativeOffset().
    void cacheOffsets(mlir::Region& body);

    // Return the signed PC-relative byte offset from jumpOp to destBlock.
    // Requires cacheOffsets() to have been called for the enclosing region.
    int64_t getRelativeOffset(mlir::Operation* jumpOp, mlir::Block* destBlock);

    // Flush the accumulated bytecode buffer to the provided output stream
    void writeTo(llvm::raw_ostream& os);
};

}  // namespace vpux::bytecode
