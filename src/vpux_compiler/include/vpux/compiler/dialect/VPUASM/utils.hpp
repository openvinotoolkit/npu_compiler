//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"

namespace vpux {
namespace VPUASM {

// DMA from SHV
//
// Release mechanism is used on platforms <= NPU5 which has a maximum of
// 2 DMA engines and 2 channels per engine.
//
// Since the maximum number of available DMA engines is 2, and each engine has
// 2 channels, we can support up to 4 skip DMA chains. Each skip DMA chain has
// its own release descriptor.
//
// Therefore, 4 is the upper bound on the number of release descriptors that may
// need to be resolved and assigned.
constexpr size_t maxNumReleaseDesc = 4;

struct SparsityMap {
    uint32_t tileSelectMaskForBuffer;
    uint32_t size;
};

vpux::VPURT::BufferSection getBufferLocation(mlir::Operation* symTableOp, mlir::SymbolRefAttr symRef);
vpux::VPURT::BufferSection getBufferLocation(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);
vpux::VPUASM::BufferType getBufferType(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);
mlir::MemRefType getMinMaxDataType(VPUASM::DPUInvariantOp invOp, ELF::SymbolReferenceMap& symRefMap);
bool isWorkLoadManagementDMA(mlir::Operation* op);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer);
uint32_t getActCompressionEntryTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap);
uint32_t getDynamicSequenceLengthBuffTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap);
SparsityMap getSparsityMapBuffTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap);

void setResourceRequirement(mlir::ModuleOp moduleOp, elf::NetworkMetadata& metadata);
SmallVector<uint32_t> getCMXStackFrames(mlir::ModuleOp moduleOp);

void insertBinaryDimsIntoVector(SmallVector<uint8_t>& dimsVector, vpux::NDTypeInterface ndType);
void insertBinaryStridesIntoVector(SmallVector<uint8_t>& stridesVector, vpux::NDTypeInterface ndType,
                                   bool normalizeUnitDimStrides = true);

}  // namespace VPUASM
}  // namespace vpux
