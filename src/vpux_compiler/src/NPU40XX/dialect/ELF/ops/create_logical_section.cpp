//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

using namespace vpux;

namespace {
bool hasIoSectionFlags(ELF::SectionFlagsAttr flags) {
    return (static_cast<bool>(flags & ELF::SectionFlagsAttr::VPU_SHF_USERINPUT) ||
            static_cast<bool>(flags & ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT) ||
            static_cast<bool>(flags & ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT));
}
}  // namespace

size_t ELF::LogicalSectionOp::getTotalSize(vpux::ELF::SymbolReferenceMap& symRefMap) {
    size_t totalSize = 0;
    auto calcSpan = [&](mlir::Operation& op) {
        auto binaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(&op);

        if (binaryOp) {
            // getting the BinarySize using VPU::ArchKind::UNKNOWN is OK at this point because the binaryOps should
            // already all be in their arch-specific form or are arch-independent
            auto span = binaryOp.getBinarySizeCached(symRefMap, VPU::ArchKind::UNKNOWN) + binaryOp.getMemoryOffset();
            totalSize = std::max(totalSize, span);
        }
    };
    llvm::for_each(getBody()->getOperations(), calcSpan);
    return totalSize;
}

void ELF::LogicalSectionOp::serialize(elf::Writer& writer, ELF::SectionMapType& sectionMap,
                                      ELF::SymbolMapType& symbolMap, ELF::SymbolReferenceMap&) {
    VPUX_UNUSED(writer);
    VPUX_UNUSED(sectionMap);
    VPUX_UNUSED(symbolMap);
}

void vpux::ELF::LogicalSectionOp::preserialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                               vpux::ELF::SymbolReferenceMap& symRefMap) {
    const auto name = getSymName().str();
    auto section = writer.addEmptySection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());
    section->setType(static_cast<elf::Elf_Word>(getSecType()));
    section->setSize(getTotalSize(symRefMap));

    sectionMap[getOperation()] = section;
}

ELF::SymbolSignature ELF::LogicalSectionOp::getSymbolSignature() {
    auto symName = ELF::SymbolOp::getDefaultNamePrefix() + getSymName();
    size_t symSize = 0;

    if (hasIoSectionFlags(getSecFlags())) {
        VPUASM::MemLocationType buffLoc;
        for (auto buff : getBody()->getOps<VPUASM::DeclareBufferOp>()) {
            buffLoc = buff.getBufferType().getLocation();
            break;
        }
        auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
        auto ioBindings = VPUASM::IOBindingsOp::getFromModule(moduleOp);

        auto section = buffLoc.getSection();
        auto index = buffLoc.getSectionIndex();

        ioBindings.walk([&symSize, &section, &index](VPUASM::DeclareBufferOp ioBuffer) {
            auto ioBuffLoc = ioBuffer.getBufferType().getLocation();
            if (ioBuffLoc.getSection() == section && ioBuffLoc.getSectionIndex() == index) {
                symSize = ioBuffer.getBinarySize(VPU::ArchKind::UNKNOWN);
            }
        });
    }
    return {mlir::SymbolRefAttr::get(getSymNameAttr()), symName.str(), ELF::SymbolType::STT_SECTION, symSize};
}
