//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

std::pair<const uint8_t*, size_t> vpux::ELFNPU37XX::getDataAndSizeOfElfSection(
        llvm::ArrayRef<uint8_t> elfBlob, const std::vector<std::string> possibleSecNames) {
    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    const uint8_t* secData = nullptr;
    uint32_t secSize = 0;

    bool secFound = false;

    for (size_t i = 0; i < elf_reader.getSectionsNum(); ++i) {
        auto section = elf_reader.getSection(i);
        const auto secName = section.getName();
        const auto sectionHeader = section.getHeader();

        for (auto possibleSecName : possibleSecNames) {
            if (strcmp(secName, possibleSecName.c_str()) == 0) {
                secSize = sectionHeader->sh_size;
                secData = section.getData<uint8_t>();
                secFound = true;
                break;
            }
        }
    }
    VPUX_THROW_UNLESS(secFound, "Section {0} not found in ELF", possibleSecNames);

    return {secData, secSize};
}

size_t vpux::ELFNPU37XX::getOffsetOfOpInSection(mlir::Value op, mlir::Value section, ELFNPU37XX::OffsetCache& cache) {
    // specific case of uninitialized buffer where the offset to it is actually specified as an op attribute and there
    // is no need for any computation
    auto actualOp = op.getDefiningOp();
    if (auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(actualOp)) {
        return declareBufferOp.getByteOffset();
    }

    auto& sectionCache = cache[section];
    if (sectionCache.empty()) {
        size_t totalOffset = 0;
        auto elfSection = mlir::cast<vpux::ELFNPU37XX::ElfSectionInterface>(section.getDefiningOp());
        for (auto& inner_op : elfSection.getBlock()->getOperations()) {
            if (!mlir::isa<vpux::ELFNPU37XX::BinaryOpInterface>(inner_op)) {
                continue;
            }

            if (auto putOpInSectionOp = mlir::dyn_cast<vpux::ELFNPU37XX::PutOpInSectionOp>(inner_op)) {
                sectionCache[putOpInSectionOp.getInputArg()] = totalOffset;
                auto binaryOp =
                        mlir::cast<vpux::ELFNPU37XX::BinaryOpInterface>(putOpInSectionOp.getInputArg().getDefiningOp());
                totalOffset += binaryOp.getBinarySize();
            } else {
                // E#105681: we cannot use getResult() if there are no results
                // in inner_op. how to properly fix this issue is unclear, so
                // for now just ignore the invalid operations
                if (inner_op.getNumResults() > 0) {
                    sectionCache[inner_op.getResult(0)] = totalOffset;
                }
                auto binaryOp = mlir::cast<vpux::ELFNPU37XX::BinaryOpInterface>(inner_op);
                totalOffset += binaryOp.getBinarySize();
            }
        }
    }

    VPUX_THROW_WHEN(op.getDefiningOp()->getResults().empty(),
                    "Cannot find the offset of an operation with no results {0}", op.getDefiningOp());
    VPUX_THROW_WHEN(sectionCache.count(op.getDefiningOp()->getResult(0)) == 0, "Unable to find {0} in {1}",
                    op.getDefiningOp()->getResult(0), section);
    return sectionCache[op.getDefiningOp()->getResult(0)];
}

size_t vpux::ELFNPU37XX::getOffsetOfOpInSection(mlir::Value& op) {
    auto actualOp = op.getDefiningOp();
    auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(actualOp);
    VPUX_THROW_UNLESS(
            declareBufferOp != nullptr && declareBufferOp.getMemorySpace() == vpux::VPURT::BufferSection::CMX_NN,
            "This version of getOffsetOfOpInSection() only works with CMX Buffers");

    auto type = mlir::dyn_cast<vpux::NDTypeInterface>(op.getType());

    auto tile = type.getMemSpace().getIndex().value_or(0);

    return tile * vpux::ELFNPU37XX::CMX_SLICE_SIZE + declareBufferOp.getByteOffset();
}

llvm::SmallString<128> vpux::ELFNPU37XX::getSwKernelArchString(config::ArchKind archKind) {
    VPUX_THROW_UNLESS(archKind == config::ArchKind::NPU37XX,
                      "The only supported architecture for sw kernels is 3720xx");
    return llvm::SmallString<128>("3720xx");
}

void vpux::ELFNPU37XX::RelocationManager::init(mlir::func::FuncOp funcOp) {
    funcOp_ = funcOp;
}

void vpux::ELFNPU37XX::RelocationManager::initCMXSymTab(ELFNPU37XX::CreateSymbolTableSectionOp cmxMappingSymTab) {
    VPUX_THROW_WHEN(cmxMappingSymTab_ != nullptr, "CMX Mapping SymTab is already init!");
    cmxMappingSymTab_ = cmxMappingSymTab;
}

ELFNPU37XX::ElfSectionInterface vpux::ELFNPU37XX::RelocationManager::getSection(mlir::Value value) {
    auto section = sectionMap_.find(value);
    if (section != sectionMap_.end()) {
        return section->second;
    }

    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    auto parent = value.getDefiningOp();
    if (parent->hasTrait<ELFNPU37XX::ElfSectionInterface::Trait>()) {
        auto parentInterface = mlir::cast<ELFNPU37XX::ElfSectionInterface>(parent);
        sectionMap_[value] = parentInterface;
        return parentInterface;
    }

    auto users = value.getUsers();
    ELFNPU37XX::PutOpInSectionOp placer;
    for (auto user : users) {
        if (mlir::isa<ELFNPU37XX::PutOpInSectionOp>(user)) {
            parent = user->getParentOp();
            auto parentInterface = mlir::cast<ELFNPU37XX::ElfSectionInterface>(parent);
            sectionMap_[value] = parentInterface;
            return parentInterface;
        }
    }

    VPUX_THROW("Cannot get the section of value {0}", value.getDefiningOp()->getName());
    return nullptr;
}

ELFNPU37XX::CreateSymbolTableSectionOp vpux::ELFNPU37XX::RelocationManager::getCMXSymTab() {
    VPUX_THROW_UNLESS(cmxMappingSymTab_ != nullptr, "Relocation Manager: CMX Mapping SymTab not init!");
    return cmxMappingSymTab_;
}

ELFNPU37XX::CreateSymbolTableSectionOp vpux::ELFNPU37XX::RelocationManager::getSymTab(mlir::Value value) {
    auto section = symTabMap_.find(value);
    if (section != symTabMap_.end()) {
        return section->second;
    }

    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    auto binaryOp = mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(value.getDefiningOp());

    if (binaryOp != nullptr) {
        auto memSpace = binaryOp.getMemorySpace();
        if (memSpace == VPURT::BufferSection::CMX_NN || memSpace == VPURT::BufferSection::Register) {
            VPUX_THROW_UNLESS(cmxMappingSymTab_ != nullptr, "RelocManager: cmxMappingSymTab is not init!");
            symTabMap_[value] = cmxMappingSymTab_;
            return cmxMappingSymTab_;
        }
    }

    auto lookupSymTab = [&](mlir::Value::user_range children) -> ELFNPU37XX::CreateSymbolTableSectionOp {
        for (auto child : children) {
            if (mlir::isa<ELFNPU37XX::SymbolOp>(child)) {
                if (mlir::isa<ELFNPU37XX::CreateSymbolTableSectionOp>(child->getParentOp())) {
                    auto symTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(child->getParentOp());
                    symTabMap_[value] = symTab;
                    return symTab;
                }

                auto grandChildren = child->getUsers();
                for (auto grandChild : grandChildren) {
                    if (mlir::isa<ELFNPU37XX::PutOpInSectionOp>(grandChild)) {
                        auto symTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(grandChild->getParentOp());
                        symTabMap_[value] = symTab;
                        return symTab;
                    }
                }
                VPUX_THROW("Symbol at {0} not in a section", child->getLoc());
            }
        }
        return nullptr;
    };

    auto children = value.getUsers();
    auto symTab = lookupSymTab(children);

    if (symTab) {
        return symTab;
    }

    // if OP has no symbol associated, then we will look at the Symtab that defines it's section symbol
    auto containerSection = getSection(value);
    children = containerSection.getOperation()->getResult(0).getUsers();
    symTab = lookupSymTab(children);

    if (symTab) {
        return symTab;
    }

    VPUX_THROW("Could not find symtab for op {0} at {1}", value.getDefiningOp()->getName(),
               value.getDefiningOp()->getLoc());

    return nullptr;
}

ELFNPU37XX::CreateRelocationSectionOp vpux::ELFNPU37XX::RelocationManager::getRelocSection(
        ELFNPU37XX::ElfSectionInterface section, ELFNPU37XX::CreateSymbolTableSectionOp symTab) {
    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    // for some reason, can't construct an interface object from opaque pointer. So will use the result as the key
    // for the map
    auto result = section.getOperation()->getResult(0);

    auto relocSection = relocMap_.find(std::make_pair(result, symTab));
    if (relocSection != relocMap_.end()) {
        return relocSection->second;
    }

    auto builder = mlir::OpBuilder::atBlockTerminator(&funcOp_.getBody().front());

    std::string secName = std::string(".rlt") + section.getName().str();
    auto newRelocation = builder.create<ELFNPU37XX::CreateRelocationSectionOp>(
            section.getLoc(),
            vpux::ELFNPU37XX::SectionType::get(builder.getContext()),  // mlir::Type
            llvm::StringRef(secName),                                  // llvm::StringRef secName
            symTab.getResult(),                                        // sourceSymbolTalbeSection
            section.getOperation()->getResult(0),                      // targetSection
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_INFO_LINK);

    relocMap_[std::make_pair(result, symTab)] = newRelocation;
    return newRelocation;
}

// TODO:
// create non-static getSymbol() method that returns a (generic or section) symbol for any mlir:Value (if it exists)
ELFNPU37XX::SymbolOp vpux::ELFNPU37XX::RelocationManager::getSymbol(ELFNPU37XX::ElfSectionInterface section) {
    mlir::Operation* op = section.getOperation();
    for (auto user : op->getUsers()) {
        if (mlir::isa<ELFNPU37XX::SymbolOp>(user)) {
            return mlir::cast<ELFNPU37XX::SymbolOp>(user);
        }
    }

    VPUX_THROW("No symbol op defined for section {0} at {1}", op->getName(), op->getLoc());
    return nullptr;
}

size_t vpux::ELFNPU37XX::math::gcd(size_t a, size_t b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

size_t vpux::ELFNPU37XX::math::lcm(size_t a, size_t b) {
    return (a / vpux::ELFNPU37XX::math::gcd(a, b)) * b;
}

//
// Platform Information
//

namespace {
const std::unordered_map<config::ArchKind, elf::platform::ArchKind> vpuToElfArchEnumMap = {
        {config::ArchKind::UNKNOWN, elf::platform::ArchKind::UNKNOWN},
        {config::ArchKind::NPU37XX, elf::platform::ArchKind::VPUX37XX},
        {config::ArchKind::NPU40XX, elf::platform::ArchKind::VPUX40XX},
};
}  // namespace

elf::platform::ArchKind vpux::ELFNPU37XX::mapVpuArchKindToElfArchKind(const config::ArchKind& archKind) {
    return vpuToElfArchEnumMap.at(archKind);
}
