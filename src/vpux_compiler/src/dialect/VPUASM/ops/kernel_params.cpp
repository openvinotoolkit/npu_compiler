//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/SymbolTable.h>
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <kernels/inc/common_types.h>

#include <cstdint>

using namespace vpux;

//
// KernelParamsOp
//

vpux::NDTypeInterface getNdTypeFromBufferSymRef(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef) {
    auto buffOp = symRefMap.lookupSymbol(symRef);

    if (mlir::isa<VPUASM::DeclareBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(buffOp);
        return mlir::cast<vpux::NDTypeInterface>(buffer.getBufferType().getMemref());
    } else if (mlir::isa<VPUASM::ConstBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(buffOp);
        return mlir::cast<vpux::NDTypeInterface>(buffer.getBufferType().getMemref());
    }
    VPUX_THROW("Could not find symbol name entry for input {0}", symRef);
}

vpux::NDTypeInterface getNdTypeFromBufferSymRef(mlir::Operation* op, mlir::SymbolRefAttr symRef) {
    auto buffOp = mlir::SymbolTable::lookupNearestSymbolFrom(op, symRef);

    if (mlir::isa<VPUASM::DeclareBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(buffOp);
        return mlir::cast<vpux::NDTypeInterface>(buffer.getBufferType().getMemref());
    } else if (mlir::isa<VPUASM::ConstBufferOp>(buffOp)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(buffOp);
        return mlir::cast<vpux::NDTypeInterface>(buffer.getBufferType().getMemref());
    }
    VPUX_THROW("Could not find symbol name entry for input {0}", symRef);
}

void vpux::VPUASM::KernelParamsOp::serializeCached(elf::writer::BinaryDataSection<uint8_t>& binDataSection,
                                                   ELF::SymbolReferenceMap& symRefMap) {
    const auto params = getProperties().kernel_params.getStorage();

    // serialize pre-computed kernel params structs
    // will either be sw_params::MemRefData for pre-compiled kernels
    // or LLVM depictions of MemRef args for ShaveCodeGen
    binDataSection.appendData(params.data(), params.size());

    // E#81501: this logic should be present either at VPUMI lowering or VPUASM lowering... doing it here temporarily
    // for pre-compiled kernels, we need to also serialize dims & strides vectors as trailing data to the MemRefData
    // structs
    if (!getIsJitCompiled()) {
        std::vector<uint8_t> inputDimsVector, outputDimsVector;
        std::vector<uint8_t> inputStridesVector, outputStridesVector;

        auto inputBuffs = getInputs();
        auto outputBuffs = getOutputs();

        auto insertDimsIntoVector = [](std::vector<uint8_t>& dimsVector, vpux::NDTypeInterface ndType) {
            auto shape = ndType.getShape();
            const auto dimsOrder = ndType.getDimsOrder();
            const auto memShape = dimsOrder.toMemoryOrder(shape);

            for (auto& memDim : memShape | reversed) {
                auto dim = checked_cast<int32_t>(memDim);
                ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&dim), sizeof(dim));
                dimsVector.insert(dimsVector.end(), valueAsArray.begin(), valueAsArray.end());
            }
        };

        auto insertStridesIntoVector = [](std::vector<uint8_t>& stridesVector, vpux::NDTypeInterface ndType) {
            auto strides = ndType.getMemStrides();
            for (auto&& stride : strides | reversed) {
                ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&stride), sizeof(stride));
                stridesVector.insert(stridesVector.end(), valueAsArray.begin(), valueAsArray.end());
            }
        };

        // input Dims & Strides
        for (auto inputBuff : inputBuffs) {
            auto inputSymRef = mlir::cast<mlir::SymbolRefAttr>(inputBuff);
            auto inputNdType = getNdTypeFromBufferSymRef(symRefMap, inputSymRef);

            insertDimsIntoVector(inputDimsVector, inputNdType);
            insertStridesIntoVector(inputStridesVector, inputNdType);
        }

        // output Dims & Strides
        for (const auto outputBuff : outputBuffs) {
            auto outputSymRef = mlir::cast<mlir::SymbolRefAttr>(outputBuff);
            auto outputNdType = getNdTypeFromBufferSymRef(symRefMap, outputSymRef);

            insertDimsIntoVector(outputDimsVector, outputNdType);
            insertStridesIntoVector(outputStridesVector, outputNdType);
            if (getIsOutputBroadcasted()) {
                break;
            }
        }

        // serialize IO dims/strides
        binDataSection.appendData(inputDimsVector.data(), inputDimsVector.size());
        binDataSection.appendData(inputStridesVector.data(), inputStridesVector.size());
        binDataSection.appendData(outputDimsVector.data(), outputDimsVector.size());
        binDataSection.appendData(outputStridesVector.data(), outputStridesVector.size());
    }
}

size_t vpux::VPUASM::KernelParamsOp::getBinarySizeCached(ELF::SymbolReferenceMap& symRefMap, config::ArchKind) {
    auto actualParamsSize = getParamsStructSize();
    if (getIsJitCompiled()) {
        return actualParamsSize;
    }

    auto inputBuffs = getInputs();
    auto outputBuffs = getOutputs();

    size_t inputDimsSize = 0;
    size_t inputStridesSize = 0;

    for (const auto inputBuff : inputBuffs) {
        auto inputSymRef = mlir::cast<mlir::SymbolRefAttr>(inputBuff);
        auto ndType = getNdTypeFromBufferSymRef(symRefMap, inputSymRef);

        inputDimsSize += sizeof(int32_t) * ndType.getShape().size();
        inputStridesSize += sizeof(int64_t) * ndType.getMemStrides().size();
    }

    size_t outputDimsSize = 0;
    size_t outputStridesSize = 0;

    for (const auto outputBuff : outputBuffs) {
        auto outputSymRef = mlir::cast<mlir::SymbolRefAttr>(outputBuff);
        auto ndType = getNdTypeFromBufferSymRef(symRefMap, outputSymRef);

        outputDimsSize += sizeof(int32_t) * ndType.getShape().size();
        outputStridesSize += sizeof(int64_t) * ndType.getMemStrides().size();
    }

    return actualParamsSize + inputDimsSize + outputDimsSize + inputStridesSize + outputStridesSize;
}

size_t vpux::VPUASM::KernelParamsOp::getParamsStructSize() {
    const auto params = getProperties().kernel_params.getStorage();
    return params.size();
}

// The parameter structs for the sw layers must be 64Byte aligned as an ActShave requirement
size_t vpux::VPUASM::KernelParamsOp::getAlignmentRequirements(config::ArchKind) {
    return ELF::VPUX_DEFAULT_ALIGNMENT;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::KernelParamsOp::getPredefinedMemoryAccessors() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

vpux::ELF::SectionFlagsAttr vpux::VPUASM::KernelParamsOp::getMemoryAccessingProc() {
    return ELF::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
}

std::optional<ELF::SectionSignature> vpux::VPUASM::KernelParamsOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("shave", "params"), ELF::SectionFlagsAttr::SHF_ALLOC);
}

bool vpux::VPUASM::KernelParamsOp::hasMemoryFootprint() {
    return true;
}

// Kernel params does not get to be lowered to NPUReg due to code duplication and no requirements of HW specific
// descriptors. For this reason relocation info has not been moved to NPUReg dialect.
std::vector<ELF::RelocationInfo> vpux::VPUASM::KernelParamsOp::getRelocationInfo(ELF::SymbolReferenceMap& symRefMap) {
    std::vector<ELF::RelocationInfo> relocs;

    ELF::ElfSectionInterface targetSection = mlir::dyn_cast<ELF::ElfSectionInterface>(getOperation()->getParentOp());
    VPUX_THROW_UNLESS(targetSection, "The relocation info can be retrieved only if the op is included into a section");

    auto getLLVMMemrefStructSize = [](int64_t memrefRank) {
        return /* allocatedPointer */ sizeof(uint32_t) +
               /* alignedPointer */ sizeof(uint32_t) +
               /* offset */ sizeof(int32_t) +
               /* dimsArray */ sizeof(int32_t) * memrefRank +
               /* stridesArray */ sizeof(int32_t) * memrefRank;
    };

    auto getRelocationForType = [](const VPUASM::BufferType& bufferType) {
        auto relocType = bufferType.getLocation().getSection() == VPURT::BufferSection::CMX_NN
                                 ? ELF::RelocationType::R_VPU_32_BIT_OR_B21_B26_UNSET
                                 : ELF::RelocationType::R_VPU_32;
        return relocType;
    };

    auto kernelInputs = getInputs();
    for (auto input : kernelInputs | indexed) {
        auto inputSymRef = mlir::cast<mlir::SymbolRefAttr>(input.value());
        auto inputBufferType = VPUASM::getBufferType(symRefMap, inputSymRef);
        auto relocType = getRelocationForType(inputBufferType);

        size_t relocOffset = input.index() * sizeof(sw_params::MemRefData) + offsetof(sw_params::MemRefData, dataAddr);
        if (getIsJitCompiled()) {
            size_t llvmMemrefStructSize = getLLVMMemrefStructSize(inputBufferType.getMemref().getRank());
            relocOffset = input.index() * llvmMemrefStructSize + sizeof(uint32_t);
        }

        relocs.emplace_back(inputSymRef, targetSection, relocOffset, relocType,
                            ELF::getOffsetOfSymRef(symRefMap, inputSymRef),
                            "Input " + std::to_string(input.index()) + " (dataAddr) kernel params reloc");
    }

    auto kernelOutputs = getOutputs();
    for (auto output : kernelOutputs | indexed) {
        auto outputSymRef = mlir::cast<mlir::SymbolRefAttr>(output.value());
        auto outputBufferType = VPUASM::getBufferType(symRefMap, outputSymRef);
        auto relocType = getRelocationForType(outputBufferType);

        size_t relocOffset = (kernelInputs.size() + output.index()) * sizeof(sw_params::MemRefData) +
                             offsetof(sw_params::MemRefData, dataAddr);
        if (getIsJitCompiled()) {
            size_t llvmMemrefStructSize = getLLVMMemrefStructSize(outputBufferType.getMemref().getRank());
            relocOffset = (kernelInputs.size() + output.index()) * llvmMemrefStructSize + sizeof(uint32_t);
        }

        relocs.emplace_back(outputSymRef, targetSection, relocOffset, relocType,
                            ELF::getOffsetOfSymRef(symRefMap, outputSymRef),
                            "Output " + std::to_string(output.index()) + " (dataAddr) kernel params reloc");

        if (getIsOutputBroadcasted()) {
            break;
        }
    }

    if (!getIsJitCompiled()) {
        auto getNDTypeIfFromSymRef = [&symRefMap](mlir::SymbolRefAttr symRef) {
            auto memoryOp = symRefMap.lookupSymbol(symRef);
            auto bufferTypeAttr = memoryOp->getAttrOfType<mlir::TypeAttr>("buffer_type");
            VPUX_THROW_UNLESS(bufferTypeAttr, "Operation is not a memory-descriptive op");

            auto bufferType = mlir::cast<vpux::VPUASM::BufferType>(bufferTypeAttr.getValue());
            auto NDTypeIf = mlir::cast<vpux::NDTypeInterface>(bufferType.getMemref());
            return NDTypeIf;
        };

        auto baseOffset = getMemoryOffset();
        auto sizeOfParamsStruct = getParamsStructSize();
        auto addend = baseOffset + sizeOfParamsStruct;
        auto fullSourceSymRef = ELF::composeSectionObjectSymRef(targetSection, this->getOperation());

        const auto dynamicInputShapes = getDynamicInputShapes();
        const auto dynamicOutputShapes = getDynamicOutputShapes();

        auto checkDynamicShape = [&](const auto& shapes, size_t index) {
            bool isDynamic = false;
            mlir::SymbolRefAttr symbolRefAttr;
            if (!shapes.empty()) {
                auto element = shapes[index];
                symbolRefAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(element);
                isDynamic = symbolRefAttr && symbolRefAttr.getRootReference() != "placeholder_symbol";
            }
            return std::make_pair(isDynamic, symbolRefAttr);
        };

        for (auto kernelInputIt : kernelInputs | indexed) {
            auto [isDynamic, symbolRefAttr] = checkDynamicShape(dynamicInputShapes, kernelInputIt.index());

            auto inputSymRef = mlir::cast<mlir::SymbolRefAttr>(kernelInputIt.value());
            if (isDynamic && symbolRefAttr) {
                auto bufferType = VPUASM::getBufferType(symRefMap, inputSymRef);
                auto relocType = getRelocationForType(bufferType);

                relocs.emplace_back(symbolRefAttr, targetSection,
                                    kernelInputIt.index() * sizeof(sw_params::MemRefData) +
                                            offsetof(sw_params::MemRefData, dimsAddr),
                                    relocType, ELF::getOffsetOfSymRef(symRefMap, symbolRefAttr),
                                    "Input " + std::to_string(kernelInputIt.index()) +
                                            " dynamic dims (dimsAddr) kernel params reloc");
            } else {
                relocs.emplace_back(
                        fullSourceSymRef, targetSection,
                        kernelInputIt.index() * sizeof(sw_params::MemRefData) +
                                offsetof(sw_params::MemRefData, dimsAddr),
                        ELF::RelocationType::R_VPU_32, addend,
                        "Input " + std::to_string(kernelInputIt.index()) + " dims (dimsAddr) kernel params reloc");
            }

            addend += sizeof(int32_t) * getNDTypeIfFromSymRef(inputSymRef).getShape().size();
        }

        for (auto kernelInputIt : kernelInputs | indexed) {
            relocs.emplace_back(
                    fullSourceSymRef, targetSection,
                    kernelInputIt.index() * sizeof(sw_params::MemRefData) +
                            offsetof(sw_params::MemRefData, stridesAddr),
                    ELF::RelocationType::R_VPU_32, addend,
                    "Input " + std::to_string(kernelInputIt.index()) + " strides (stridesAddr) kernel params reloc");

            auto inputSymRef = mlir::cast<mlir::SymbolRefAttr>(kernelInputIt.value());
            addend += sizeof(int64_t) * getNDTypeIfFromSymRef(inputSymRef).getMemStrides().size();
        }

        for (auto kernelOutputIt : kernelOutputs | indexed) {
            auto [isDynamic, symbolRefAttr] = checkDynamicShape(dynamicOutputShapes, kernelOutputIt.index());

            auto outputSymRef = mlir::cast<mlir::SymbolRefAttr>(kernelOutputIt.value());
            if (isDynamic && symbolRefAttr) {
                auto bufferType = VPUASM::getBufferType(symRefMap, outputSymRef);
                auto relocType = getRelocationForType(bufferType);

                relocs.emplace_back(symbolRefAttr, targetSection,
                                    (kernelInputs.size() + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                                            offsetof(sw_params::MemRefData, dimsAddr),
                                    relocType, ELF::getOffsetOfSymRef(symRefMap, symbolRefAttr),
                                    "Output " + std::to_string(kernelOutputIt.index()) +
                                            " dynamic dims (dimsAddr) kernel params reloc");
            } else {
                relocs.emplace_back(
                        fullSourceSymRef, targetSection,
                        (kernelInputs.size() + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                                offsetof(sw_params::MemRefData, dimsAddr),
                        ELF::RelocationType::R_VPU_32, addend,
                        "Output " + std::to_string(kernelOutputIt.index()) + " dims (dimsAddr) kernel params reloc");
            }

            addend += sizeof(int32_t) * getNDTypeIfFromSymRef(outputSymRef).getShape().size();

            if (getIsOutputBroadcasted()) {
                break;
            }
        }

        for (auto kernelOutputIt : kernelOutputs | indexed) {
            relocs.emplace_back(
                    fullSourceSymRef, targetSection,
                    (kernelInputs.size() + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                            offsetof(sw_params::MemRefData, stridesAddr),
                    ELF::RelocationType::R_VPU_32, addend,
                    "Output " + std::to_string(kernelOutputIt.index()) + " strides (stridesAddr) kernel params reloc");

            auto outputSymRef = mlir::cast<mlir::SymbolRefAttr>(kernelOutputIt.value());
            addend += sizeof(int64_t) * getNDTypeIfFromSymRef(outputSymRef).getMemStrides().size();

            if (getIsOutputBroadcasted()) {
                break;
            }
        }
    }

    return relocs;
}

void vpux::VPUASM::KernelParamsOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::StringAttr symName,
                                         mlir::ArrayAttr inputs, mlir::ArrayAttr outputs,
                                         mlir::ArrayAttr dynamicInputShapes, mlir::ArrayAttr dynamicOutputShapes,
                                         mlir::StringAttr kernelType, KernelParamsProperty&& kernelParams,
                                         mlir::UnitAttr isOutputBroadcasted = nullptr,
                                         mlir::UnitAttr isJitCompiled = nullptr) {
    auto& props = state.getOrAddProperties<Properties>();
    props.sym_name = symName;
    props.kernel_params = std::move(kernelParams);
    props.inputs = inputs;
    props.outputs = outputs;
    props.dynamicInputShapes = dynamicInputShapes;
    props.dynamicOutputShapes = dynamicOutputShapes;
    props.kernel_type = kernelType;
    props.is_output_broadcasted = isOutputBroadcasted;
    props.isJitCompiled = isJitCompiled;
}
