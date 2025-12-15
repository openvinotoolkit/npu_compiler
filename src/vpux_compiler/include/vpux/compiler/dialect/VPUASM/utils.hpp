//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"

namespace vpux {
namespace VPUASM {

struct SparsityMap {
    uint32_t tileSelectMaskForBuffer;
    uint32_t size;
};

vpux::VPURT::BufferSection getBufferLocation(mlir::Operation* symTableOp, mlir::SymbolRefAttr symRef);
vpux::VPURT::BufferSection getBufferLocation(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);
vpux::VPUASM::BufferType getBufferType(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);
bool isWorkLoadManagementDMA(mlir::Operation* op);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer);
uint32_t getActCompressionEntryTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap);
SparsityMap getSparsityMapBuffTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap);

void setResourceRequirement(mlir::ModuleOp moduleOp, elf::NetworkMetadata& metadata);

void insertBinaryDimsIntoVector(SmallVector<uint8_t>& dimsVector, vpux::NDTypeInterface ndType);
void insertBinaryStridesIntoVector(SmallVector<uint8_t>& stridesVector, vpux::NDTypeInterface ndType);

}  // namespace VPUASM
}  // namespace vpux
