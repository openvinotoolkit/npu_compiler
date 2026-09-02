//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/platform_resources.hpp"

#include "vpux/compiler/dialect/ELF/IR/attributes.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"

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

mlir::MemRefType getMinMaxDataType(VPUASM::DPUInvariantOp invOp, ELF::SymbolReferenceMap& symRefMap) {
    mlir::MemRefType minMaxDataType;
    if (auto maxPerXy = invOp.getMaxPerXy()) {
        minMaxDataType = getBufferType(symRefMap, maxPerXy.value()).getMemref();
    } else if (auto minPerXy = invOp.getMinPerXy()) {
        minMaxDataType = getBufferType(symRefMap, minPerXy.value()).getMemref();
    } else if (auto minMaxPerTensor = invOp.getMinMaxPerTensor()) {
        auto minMaxRefArray = mlir::dyn_cast<mlir::ArrayAttr>(minMaxPerTensor.value());
        auto minMaxRef = mlir::dyn_cast<mlir::SymbolRefAttr>(minMaxRefArray[0]);
        minMaxDataType = getBufferType(symRefMap, minMaxRef).getMemref();
    }

    return minMaxDataType;
}

bool isWorkLoadManagementDMA(mlir::Operation* op) {
    return mlir::isa<VPUASM::DPUInvariantOp, VPUASM::DPUVariantOp, VPUIPDPU::DPUInvariantOp, VPUIPDPU::DPUVariantOp,
                     VPUASM::ActKernelInvocationOp, VPUASM::ActKernelRangeOp, VPUASM::DeclareTaskBufferOp,
                     VPUASM::NNDMAOp>(op);
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

uint32_t getDynamicSequenceLengthBuffTileMask(VPUASM::NNDMAOp dmaOp, ELF::SymbolReferenceMap& symRefMap) {
    if (auto dynSeqLenBuff = dmaOp.getDynamicSequenceLengthBuff().value_or(nullptr)) {
        auto dynSeqLenRef = symRefMap.lookupSymbol(dynSeqLenBuff);
        VPUX_THROW_UNLESS(dynSeqLenRef, "Could not find symbol name entry for {0} of {1}", dynSeqLenBuff, dmaOp);

        if (mlir::isa<VPUASM::DeclareBufferOp>(dynSeqLenRef)) {
            auto dynSeqLenBuffer = mlir::cast<VPUASM::DeclareBufferOp>(dynSeqLenRef);
            return getTileSelectMaskForBuffer(dynSeqLenBuffer);
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
    uint32_t reserved_memory = 0;
    if (config::getArch(moduleOp) == config::ArchKind::NPU40XX) {
        reserved_memory += 2 * static_cast<uint32_t>(CMX_SHAVE_STACK_SIZE.count());
        reserved_memory += static_cast<uint32_t>(HW_RESERVED_CMX.count());
        reserved_memory += static_cast<uint32_t>(CMX_METADATA_SIZE.count());
    }
    metadata.mResourceRequirements.nn_slice_length_ =
            checked_cast<uint32_t>(config::getAvailableMemory(moduleOp, vpux::VPU::MemoryKind::CMX_NN).getByteSize()) -
            reserved_memory;
}

SmallVector<uint32_t> getCMXStackFrames(mlir::ModuleOp moduleOp) {
    auto tileOp = config::getTileExecutor(moduleOp);
    auto tileCount = checked_cast<uint32_t>(tileOp.getCount());
    auto shvPerTile = checked_cast<uint32_t>(tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount());
    // First two stacks reserved at the beginning of the CMX space
    const size_t defaultStacksNum = 2;

    SmallVector<uint32_t> stacksOffsets;
    // SHAVE stacks grows backwards!
    // Set the address to the end of the allocated section so it does not override
    // outside of its buffer
    auto stackSize = static_cast<uint32_t>(CMX_SHAVE_STACK_SIZE.count());

    auto shaveStacksMem = config::getCMXStackFramesReservedMemory(moduleOp);
    VPUX_THROW_WHEN(shaveStacksMem == nullptr, "Missing reserved CMX memory for shave stack frames");
    auto shaveStacksMemOffset = shaveStacksMem.getOffset();
    VPUX_THROW_WHEN(shaveStacksMemOffset == std::nullopt, "No address allocated for shave stack frames in CMX");
    auto shaveStacksMemSize = checked_cast<uint32_t>(shaveStacksMem.getByteSize());
    VPUX_THROW_WHEN(shaveStacksMemSize < defaultStacksNum * stackSize,
                    "Insufficient memory allocated for shave stack frames in CMX");

    for (auto stackIdx : irange(defaultStacksNum)) {
        stacksOffsets.push_back(shaveStacksMemOffset.value() + (stackIdx + 1) * stackSize);
    }

    if (shvPerTile > defaultStacksNum) {
        auto additionalStacksNum = shvPerTile - defaultStacksNum;
        auto additionalShaveStacksMem = config::getCMXAdditionalStackFramesReservedMemory(moduleOp);
        VPUX_THROW_WHEN(additionalShaveStacksMem == nullptr,
                        "Missing reserved CMX memory for additional shave stack frames");
        auto additionalShaveStacksMemOffset = additionalShaveStacksMem.getOffset();
        VPUX_THROW_WHEN(additionalShaveStacksMemOffset == std::nullopt,
                        "No address allocated for additional shave stack frames in CMX");
        auto additionalShaveStacksMemSize = checked_cast<uint32_t>(additionalShaveStacksMem.getByteSize());
        VPUX_THROW_WHEN(additionalShaveStacksMemSize < additionalStacksNum * stackSize,
                        "Insufficient memory allocated for additional shave stack frames in CMX");

        for (auto stackIdx : irange(additionalStacksNum)) {
            stacksOffsets.push_back(additionalShaveStacksMemOffset.value() + (stackIdx + 1) * stackSize);
        }
    }

    SmallVector<uint32_t> stackFrameAddrs(static_cast<size_t>(tileCount) * shvPerTile);
    for (auto tileIdx : irange(tileCount)) {
        for (auto offset : llvm::enumerate(stacksOffsets)) {
            // Combine base address with offset to point inside reserved CMX memory
            stackFrameAddrs[static_cast<size_t>(tileIdx) * shvPerTile + offset.index()] =
                    offset.value() | CMX_BASE_ADDR;
        }
    }
    return stackFrameAddrs;
}

void insertBinaryDimsIntoVector(SmallVector<uint8_t>& dimsVector, vpux::NDTypeInterface ndType) {
    auto shape = ndType.getShape();
    const auto dimsOrder = ndType.getDimsOrder();
    const auto memShape = dimsOrder.toMemoryOrder(shape);

    for (auto& memDim : memShape | reversed) {
        auto dim = checked_cast<int32_t>(memDim);
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));
        dimsVector.insert(dimsVector.end(), valueAsArray.begin(), valueAsArray.end());
    }
}

void insertBinaryStridesIntoVector(SmallVector<uint8_t>& stridesVector, vpux::NDTypeInterface ndType,
                                   bool normalizeUnitDimStrides) {
    const auto strides = ndType.getMemStrides();
    const auto memShape = ndType.getMemShape();
    const auto rank = static_cast<int64_t>(strides.size());

    // A size-1 dimension's stride may be inherited from a parent subview and
    // exceed the compact value expected by SW kernels. Because dim==1, the
    // stride is never multiplied by a loop index, so normalizing it to compact
    // form is safe and cannot misrepresent non-contiguous layouts.
    // Normalize each unit-dim stride to: normalized_inner_stride * inner_dim_size.
    // Process from innermost to outermost so each outer level builds on the
    // already-normalized inner result.
    //
    // Example:
    //  Cluster shape 1x1x1x16384xf16 (NHWC) split across 2 ACT-SHAVEs; each SHAVE
    //  operates on a subview of shape 1x1x1x8192xf16 (NHWC).
    //
    //  mem-shape    (N, H, W, C): [1, 1, 8192, 1]
    //  raw strides  (N, H, W, C): [16384, 16384, 1, 1] elems
    //  norm strides (N, H, W, C): [8192,  8192,  1, 1] elems
    //
    SmallVector<Bit> normalizedStrides(rank);
    for (int64_t i = rank - 1; i >= 0; --i) {
        const auto md = MemDim(i);
        if (normalizeUnitDimStrides && memShape[md] == 1 && i + 1 < rank) {
            const auto innerMd = MemDim(i + 1);
            normalizedStrides[i] = normalizedStrides[i + 1] * memShape[innerMd];
        } else {
            normalizedStrides[i] = strides[md];
        }
    }

    for (auto&& stride : normalizedStrides | reversed) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
        stridesVector.insert(stridesVector.end(), valueAsArray.begin(), valueAsArray.end());
    }
}

}  // namespace VPUASM
}  // namespace vpux
