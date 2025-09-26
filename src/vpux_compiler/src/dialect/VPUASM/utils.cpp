//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"

namespace vpux {
namespace VPUASM {

vpux::VPURT::BufferSection getBufferLocation(mlir::Operation* symTableOp, mlir::SymbolRefAttr symRef) {
    VPUX_THROW_UNLESS(symTableOp->hasTrait<mlir::OpTrait::SymbolTable>(),
                      "The symTableOp parameter must have the SymbolTable trait");
    auto symTable = mlir::SymbolTable(symTableOp);

    auto referencedOp = symTable.lookupSymbolIn(symTableOp, symRef);
    if (auto logicalSec = referencedOp->getParentOfType<ELF::LogicalSectionOp>()) {
        return logicalSec.getSecLocation();
    } else if (auto dataSec = referencedOp->getParentOfType<ELF::DataSectionOp>()) {
        return dataSec.getSecLocation();
    }
    VPUX_THROW("BufferLocation can not be retrieved!");
}

vpux::VPURT::BufferSection getBufferLocation(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef) {
    auto referencedOp = symRefMap.lookupSymbol(symRef);

    if (auto logicalSec = referencedOp->getParentOfType<ELF::LogicalSectionOp>()) {
        return logicalSec.getSecLocation();
    } else if (auto dataSec = referencedOp->getParentOfType<ELF::DataSectionOp>()) {
        return dataSec.getSecLocation();
    }
    VPUX_THROW("BufferLocation can not be retrieved!");
}

BufferType getBufferType(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef) {
    auto referencedOp = symRefMap.lookupSymbol(symRef);

    if (auto bufferOp = mlir::dyn_cast<VPUASM::DeclareBufferOp>(referencedOp)) {
        return bufferOp.getBufferType();
    } else if (auto constantOp = mlir::dyn_cast<VPUASM::ConstBufferOp>(referencedOp)) {
        return constantOp.getBufferType();
    }
    VPUX_THROW("SymRef {0} does not point to a VPUASM::BufferType buffer", symRef.getLeafReference().getValue());
}

bool isWorkLoadManagementDMA(mlir::Operation* op) {
    return mlir::isa<VPUASM::DPUInvariantOp, VPUASM::DPUVariantOp, VPUIPDPU::DPUInvariantOp, VPUIPDPU::DPUVariantOp,
                     VPUASM::ActKernelInvocationOp, VPUASM::ActKernelRangeOp, VPUASM::DeclareTaskBufferOp>(op);
}

uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer) {
    auto bufferLocation = buffer.getBufferType().getLocation();
    if (bufferLocation.getSection() != VPURT::BufferSection::CMX_NN) {
        return 0;
    }

    return VPUMI40XX::generateTileMask({static_cast<uint32_t>(bufferLocation.getSectionIndex())});
}

uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer) {
    return VPUMI40XX::generateTileMask({static_cast<uint32_t>(taskBuffer.getTileIndex())});
}

uint32_t getActCompressionEntryTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap) {
    auto actCompressionSizeEntry = dmaOp.getActCompressionSizeEntry();
    if (actCompressionSizeEntry.has_value()) {
        auto actCompBufferRef = symRefMap.lookupSymbol(actCompressionSizeEntry.value());
        VPUX_THROW_UNLESS(actCompBufferRef, "Could not find symbol name entry for {0} of {1}",
                          actCompressionSizeEntry.value(), dmaOp);

        if (mlir::isa<VPUASM::DeclareBufferOp>(actCompBufferRef)) {
            auto actCompBuffer = mlir::cast<VPUASM::DeclareBufferOp>(actCompBufferRef);
            return getTileSelectMaskForBuffer(actCompBuffer);
        }
    }
    return 0;
}

SparsityMap getSparsityMapBuffTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap) {
    auto sparsityMapBuffer = dmaOp.getActCompressionSparsityMap();
    SparsityMap sparsityMap{};

    if (sparsityMapBuffer.has_value()) {
        auto sparsityMapBufferRef = symRefMap.lookupSymbol(sparsityMapBuffer.value());
        VPUX_THROW_UNLESS(sparsityMapBufferRef, "Could not find symbol name entry for {0} of {1}",
                          sparsityMapBuffer.value(), dmaOp);

        if (auto buffer = mlir::dyn_cast_if_present<VPUASM::DeclareBufferOp>(sparsityMapBufferRef)) {
            sparsityMap.tileSelectMaskForBuffer = getTileSelectMaskForBuffer(buffer);
            sparsityMap.size = buffer.getBinarySize(config::getArch(dmaOp));
        }
    }
    return sparsityMap;
}

void setResourceRequirement(mlir::ModuleOp moduleOp, elf::NetworkMetadata& metadata) {
    metadata.mResourceRequirements.nn_slice_count_ = VPUIP::getNumTilesUsed(moduleOp);
    uint32_t workspace_offset = 0;
    metadata.mResourceRequirements.nn_slice_length_ =
            workspace_offset +
            checked_cast<uint32_t>(config::getAvailableMemory(moduleOp, vpux::VPU::MemoryKind::CMX_NN).getByteSize());
}

}  // namespace VPUASM
}  // namespace vpux
