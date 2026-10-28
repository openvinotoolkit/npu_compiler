//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELF/utils/reloc_manager.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_ADDRELOCATIONSFORDYNAMICSTRIDESDMAS
#define GEN_PASS_DEF_ADDRELOCATIONSFORDYNAMICSTRIDESDMAS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

namespace {

class AddRelocationsForDynamicStridesDmas :
        public ELF::impl::AddRelocationsForDynamicStridesDmasBase<AddRelocationsForDynamicStridesDmas> {
public:
    explicit AddRelocationsForDynamicStridesDmas(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    ELF::DmaSymbolOp createDynamicStridesDmaSymbol(ELF::RelocatableOpWithDynamicStridesInterface relocatableOp,
                                                   ELF::DmaSymbolSectionOp targetSection,
                                                   VPUASM::DeclareBufferOp ioDeclareOp,
                                                   VPUASM::DeclareBufferOp declareOp, mlir::StringAttr symbolName);

    using RelocItemVector = llvm::SmallVector<int64_t, elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS>;
};

ELF::DmaSymbolOp AddRelocationsForDynamicStridesDmas::createDynamicStridesDmaSymbol(
        ELF::RelocatableOpWithDynamicStridesInterface relocatableOp, ELF::DmaSymbolSectionOp targetSection,
        VPUASM::DeclareBufferOp ioDeclareOp, VPUASM::DeclareBufferOp declareOp, mlir::StringAttr symbolName) {
    auto ioIndex = declareOp.getBufferType().getLocation().getSectionIndex();

    auto dmaBufferType = mlir::cast<NDTypeInterface>(declareOp.getBufferType().getMemref());
    auto argumentBufferType = mlir::cast<NDTypeInterface>(ioDeclareOp.getBufferType().getMemref());
    auto dimsOrder = mlir::cast<NDTypeInterface>(declareOp.getBufferType().getMemref()).getDimsOrder();
    auto dmaBufferStrides = dmaBufferType.getMemStrides();
    auto dmaBufferShape = dimsOrder.toMemoryOrder(dmaBufferType.getShape());
    auto argBufferStrides = argumentBufferType.getMemStrides();
    auto argumentShape = argumentBufferType.getShape();
    int64_t dmaSize = Byte(dmaBufferType.getElemTypeSize()).count();

    for (auto& argBufferStride : argBufferStrides) {
        argBufferStride /= argumentBufferType.getElemTypeSize().count();
    }

    for (auto& dmaBufferStride : dmaBufferStrides) {
        dmaBufferStride /= dmaBufferType.getElemTypeSize().count();
    }

    RelocItemVector tileOffsets(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 0);
    if (auto offsetsAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(declareOp->getAttr(vpux::viewOffsetsAttrName))) {
        auto permutedOffsets = dimsOrder.toMemoryOrder(Shape(parseIntArrayAttr<int64_t>(offsetsAttr)));
        for (size_t idx = 0; idx < tileOffsets.size() && idx < permutedOffsets.size(); idx++) {
            tileOffsets[idx] = permutedOffsets[MemDim(permutedOffsets.size() - idx - 1)];
        }
    }

    RelocItemVector canonicalTensorShapes(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 1);
    RelocItemVector canonicalTensorStrides(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 0);
    RelocItemVector canonicalDmaShapes(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 1);
    RelocItemVector canonicalDmaStrides(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 0);
    RelocItemVector canonicalOffsets(elf::DMA_SYMBOL_MAX_TENSOR_DIMENSIONS, 0);

    ELF::getCanonicalDmaForm(dmaBufferShape, dmaBufferStrides, argumentShape, tileOffsets, canonicalDmaShapes,
                             canonicalDmaStrides, canonicalOffsets);

    for (size_t idx = 0; idx < argumentShape.size(); idx++) {
        canonicalTensorShapes[idx] = argumentShape[Dim(argumentShape.size() - 1 - idx)];
        canonicalTensorStrides[idx] = argBufferStrides[MemDim(argBufferStrides.size() - 1 - idx)].count();
    }

    auto dmaSymbolSectionBuilder = mlir::OpBuilder::atBlockEnd(targetSection.getBlock());
    auto ctx = declareOp->getContext();
    return ELF::DmaSymbolOp::create(dmaSymbolSectionBuilder, relocatableOp->getLoc(), symbolName, ioIndex,
                                    getIntArrayAttr(ctx, canonicalTensorShapes),
                                    getIntArrayAttr(ctx, canonicalTensorStrides),
                                    getIntArrayAttr(ctx, canonicalOffsets), getIntArrayAttr(ctx, canonicalDmaShapes),
                                    getIntArrayAttr(ctx, canonicalDmaStrides), dmaSize);
}

void AddRelocationsForDynamicStridesDmas::safeRunOnFunc() {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "No module op");

    auto inputBindings = to_small_vector(moduleOp.getRegion().getOps<VPUASM::InputBindingsOp>());
    auto outputBindings = to_small_vector(moduleOp.getRegion().getOps<VPUASM::OutputBindingsOp>());

    VPUX_THROW_UNLESS(inputBindings.size() == 1, "Expected exactly one InputBindings op");
    VPUX_THROW_UNLESS(outputBindings.size() == 1, "Expected exactly one OutputBindings op");

    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    auto elfBuilder = mlir::OpBuilder::atBlockBegin(elfMain.getBody());
    auto dmaInputSymbolSection = ELF::DmaSymbolSectionOp::create(
            elfBuilder, elfBuilder.getUnknownLoc(), "dmaInputSymbolsSection",
            ELF::SectionFlagsAttr::VPU_SHF_JIT | ELF::SectionFlagsAttr::VPU_SHF_USERINPUT);
    auto dmaOutputSymbolSection = ELF::DmaSymbolSectionOp::create(
            elfBuilder, elfBuilder.getUnknownLoc(), "dmaOutputSymbolsSection",
            ELF::SectionFlagsAttr::VPU_SHF_JIT | ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT);

    auto getDeclareOpForIo = [&](VPUASM::DeclareBufferOp declareOp) -> VPUASM::DeclareBufferOp {
        auto index = declareOp.getBufferType().getLocation().getSectionIndex();
        if (declareOp.getBufferType().getLocation().getSection() == VPURT::BufferSection::NetworkInput) {
            auto ioDeclares = to_small_vector(inputBindings[0].getRegion().getOps<VPUASM::DeclareBufferOp>());
            VPUX_THROW_UNLESS(index < ioDeclares.size(), "No such input");
            return ioDeclares[index];
        } else {
            auto ioDeclares = to_small_vector(outputBindings[0].getRegion().getOps<VPUASM::DeclareBufferOp>());
            VPUX_THROW_UNLESS(index < ioDeclares.size(), "No such output");
            return ioDeclares[index];
        }
    };

    ELF::RelocManager relocManager(elfMain);
    int inputRelocIdx = 0;
    int outputRelocIdx = 0;
    for (auto targetSection : elfMain.getOps<ELF::ElfSectionInterface>()) {
        // TODO:  E#195185 consider to add interface dedicated for sections that are intended to hold program data, aka
        // should be relocateable?
        if (!mlir::isa<ELF::DataSectionOp, ELF::LogicalSectionOp>(targetSection)) {
            continue;
        }

        auto block = targetSection.getBlock();
        for (auto relocatableOp : block->getOps<ELF::RelocatableOpWithDynamicStridesInterface>()) {
            std::vector<ELF::RelocationInfo> relocs;
            if (relocatableOp->getAttr(vpux::stridedInputAttrName)) {
                auto declareOp =
                        mlir::cast<VPUASM::DeclareBufferOp>(elfMain.lookupSymbol(relocatableOp.getInputSymbol()));
                auto ioDeclareOp = getDeclareOpForIo(declareOp);
                // Note that DMA can read from both input and output. Even though OV doesn't allow for network to read
                // from output node it is possible that, due to compiler optimizations, DMA reading from output buffer
                // was created. This seems to be especially popular in LLMs with KV cache
                ELF::DmaSymbolSectionOp section =
                        (declareOp.getBufferType().getLocation().getSection() == VPURT::BufferSection::NetworkInput)
                                ? dmaInputSymbolSection
                                : dmaOutputSymbolSection;
                std::string symbolName("INPUT_RELOC_");
                symbolName += std::to_string(inputRelocIdx);
                inputRelocIdx++;
                auto inputSymbol = createDynamicStridesDmaSymbol(relocatableOp, section, ioDeclareOp, declareOp,
                                                                 mlir::StringAttr::get(&getContext(), symbolName));
                auto newSymRef = mlir::SymbolRefAttr::get(&getContext(), inputSymbol.getSymName());
                auto inputSymbolName = mlir::SymbolRefAttr::get(&getContext(), section.getSymName(), {newSymRef});
                relocs.emplace_back(inputSymbolName, targetSection, 0, ELF::RelocationType::R_VPU_DMA_DESCRIPTOR_INPUT,
                                    0, "DMA descriptor relocation for strided input");
            }

            if (relocatableOp->getAttr(vpux::stridedOutputAttrName)) {
                auto declareOp =
                        mlir::cast<VPUASM::DeclareBufferOp>(elfMain.lookupSymbol(relocatableOp.getOutputSymbol()));
                auto ioDeclareOp = getDeclareOpForIo(declareOp);
                ELF::DmaSymbolSectionOp section =
                        (declareOp.getBufferType().getLocation().getSection() == VPURT::BufferSection::NetworkInput)
                                ? dmaInputSymbolSection
                                : dmaOutputSymbolSection;
                std::string symbolName("OUTPUT_RELOC_");
                symbolName += std::to_string(outputRelocIdx);
                outputRelocIdx++;
                auto outputSymbol = createDynamicStridesDmaSymbol(relocatableOp, section, ioDeclareOp, declareOp,
                                                                  mlir::StringAttr::get(&getContext(), symbolName));
                auto newSymRef = mlir::SymbolRefAttr::get(&getContext(), outputSymbol.getSymName());
                auto outputSymbolName = mlir::SymbolRefAttr::get(&getContext(), section.getSymName(), {newSymRef});
                relocs.emplace_back(outputSymbolName, targetSection, 0,
                                    ELF::RelocationType::R_VPU_DMA_DESCRIPTOR_OUTPUT, 0,
                                    "DMA descriptor relocation for strided output");
            }

            relocManager.createDynamicStridesDMARelocations(relocatableOp, relocs);
        }
    }

    if (dmaInputSymbolSection.getRegion().getOps<ELF::DmaSymbolOp>().empty()) {
        dmaInputSymbolSection.erase();
    }
    if (dmaOutputSymbolSection.getRegion().getOps<ELF::DmaSymbolOp>().empty()) {
        dmaOutputSymbolSection.erase();
    }
}

}  // namespace

//
// createRemoveEmptyELFSectionsPass
//

std::unique_ptr<mlir::Pass> vpux::ELF::createAddRelocationsForDynamicStridesDMAsPass(Logger log) {
    return std::make_unique<AddRelocationsForDynamicStridesDmas>(log);
}
