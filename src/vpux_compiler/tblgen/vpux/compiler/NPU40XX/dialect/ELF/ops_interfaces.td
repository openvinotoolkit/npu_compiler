//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_ELF_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SymbolicallyRepresentedOpInterface
//

// For the moment, this interface will only be inherited by the SectionInterface
// If non-section symbols will be enabled, other entities could make use of this

def ELF_SymbolicallyRepresentedOpInterface : OpInterface<"SymbolicallyRepresentedOpInterface"> {
    let description = "ELF Interface for the operations that need to be represented by a symbol in a SymbolTable";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get the signature of the symbol, containing all the info necessary to construct it",
            "ELF::SymbolSignature",
            "getSymbolSignature", (ins)
        >
    ];
}

//
// ELFSectionInterface
//

def ELF_SectionInterface : OpInterface<"ElfSectionInterface"> {
    let description = "Interface for the ELF section";

    let cppNamespace = "vpux::ELF";

    let methods = [
        // make interface more modular and replace preserialize with getBinarySize
        // E#136377
        InterfaceMethod<
            "Serialization setup for the Op (elf sections creation, memory pre-allocation)",
            "void",
            "preserialize", (ins "elf::Writer&":$writer,
                                 "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap,
                                 "vpux::ELF::SymbolReferenceMap&":$symRefMap)
        >,
        InterfaceMethod<
            "Serialize the Section into an ELF entity",
            "void",
            "serialize", (ins "elf::Writer&":$writer,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap,
                              "vpux::ELF::SymbolReferenceMap&":$symRefMap)
        >,
        InterfaceMethod<
            "Get the block of the region",
            "mlir::Block*",
            "getBlock", (ins), [{}], [{
                return vpux::ELF::getSectionBlock($_op);
            }]
        >,
        InterfaceMethod<
            "Get the name of the section",
            "llvm::StringRef",
            "getSectionName", (ins), [{}], [{
                return $_op.getSymName();
            }]
        >,
        InterfaceMethod<
            "Get the type of the section",
            "std::optional<::vpux::ELF::SectionTypeAttr>",
            "getSectionType", (ins), [{}], [{
                auto op = this->getOperation();
                if (op->hasAttr("secType")) {
                    auto secTypeAttr = (op->getAttr("secType")).template cast<::vpux::ELF::SectionTypeAttrAttr>();
                    return std::optional<::vpux::ELF::SectionTypeAttr>(secTypeAttr.getValue());
                } else {
                    return std::nullopt;
                }
            }]
        >,
        InterfaceMethod<
            "Get the section flags",
            "::vpux::ELF::SectionFlagsAttr",
            "getSectionFlags", (ins), [{}], [{
                auto op = this->getOperation();
                auto flagsAttr = op->template getAttrOfType<ELF::SectionFlagsAttrAttr>("secFlags");
                return flagsAttr.getValue();
            }]
        >,
        InterfaceMethod<
            "Update the section flags with new flags",
            "::vpux::ELF::SectionFlagsAttr",
            "updateSectionFlags", (ins "::vpux::ELF::SectionFlagsAttr":$flags), [{}], [{
                auto op = this->getOperation();
                auto currFlags = this->getSectionFlags();
                if (::vpux::ELF::bitEnumContainsAll(currFlags, flags)) {
                    return currFlags;
                } else {
                    auto newFlags = currFlags | flags;
                    op->setAttr("secFlags", ::vpux::ELF::SectionFlagsAttrAttr::get(op->getContext(), newFlags));
                    return newFlags;
                }
            }]
        >,
        InterfaceMethod<
            "Get the symbol table where the symbol for the given section should be inserted",
            "ELF::SectionSignature",
            "getSymbolTableSectionSignature", (ins), [{}], [{
                auto op = this->getOperation();
                auto sectionFlagsAttr = op->template getAttrOfType<ELF::SectionFlagsAttrAttr>("secFlags");
                auto sectionFlags = sectionFlagsAttr ? sectionFlagsAttr.getValue() : ELF::SectionFlagsAttr::SHF_NONE;
                auto symTabFlags = ELF::SectionFlagsAttr::SHF_NONE;
                auto name = "symtab";
                if (static_cast<bool>(sectionFlags & ELF::SectionFlagsAttr::VPU_SHF_JIT)) {
                    name = "symtab.descriptor";
                    symTabFlags = ELF::SectionFlagsAttr::VPU_SHF_JIT;
                } else if (static_cast<bool>(sectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USERINPUT)) {
                    name = "symtab.io.NetworkInput";
                    symTabFlags = ELF::SectionFlagsAttr::VPU_SHF_USERINPUT | ELF::SectionFlagsAttr::VPU_SHF_JIT;
                } else if (static_cast<bool>(sectionFlags & ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT)) {
                    name = "symtab.io.NetworkOutput";
                    symTabFlags = ELF::SectionFlagsAttr::VPU_SHF_USEROUTPUT | ELF::SectionFlagsAttr::VPU_SHF_JIT;
                } else if (static_cast<bool>(sectionFlags & ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT)) {
                    name = "symtab.io.ProfilingOutput";
                    symTabFlags = ELF::SectionFlagsAttr::VPU_SHF_PROFOUTPUT | ELF::SectionFlagsAttr::VPU_SHF_JIT;
                }

                return ELF::SectionSignature(name, symTabFlags);
            }]
        >,
        InterfaceMethod<
            "Update the section address alignment if present",
            "void",
            "updateSectionAddressAlignment", (ins "mlir::IntegerAttr":$alignment), [{}], [{
                auto op = this->getOperation();
                if (!op->hasAttr("secAddrAlign")) {
                    return;
                }
                op->setAttr("secAddrAlign", alignment);
            }]
        >
    ];
}

def ELF_SymbolObjectInterface : OpInterface<"ElfSymbolObjectInterface"> {
    let description = "Interface for the ELF symbol objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF symbol",
            "void",
            "serialize", (ins "elf::writer::Symbol*":$symbol,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Section*>&":$sectionMap)
        >,
    ];
}

def ELF_RelocationObjectInterface : OpInterface<"ElfRelocationObjectInterface"> {
    let description = "Interface for the ELF relocation objects";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize an object as an ELF relocation object",
            "void",
            "serialize", (ins "elf::writer::Relocation*":$relocation,
                              "vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>&":$symbolMap)
        >,
    ];
}

//
// BinaryOpInterface
//

def ELF_BinaryOpInterface : OpInterface<"BinaryOpInterface"> {
    let description = "Interface for the ELF binary serialization of operations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Serialize the Op to the BLOB format",
            "void",
            "serialize", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection), [{}],
            /*defaultImplementation*/ [{
                VPUX_THROW("Unexpected call to interface implementation.");
            }]
        >,

        InterfaceMethod<
            // E#136377
            "Get the size in bytes of the serialized form of this object; it maybe not the same as sizeof for non-POD types",
            "size_t",
            "getBinarySize", (ins "::vpux::VPU::ArchKind":$arch), [{}],
            /*defaultImplementation*/ [{
                VPUX_THROW("Unexpected call to interface implementation.");
            }]
        >,

        InterfaceMethod<
            "Serialize the Op to the BLOB format, using a symbol reference map for faster look-up",
            "void",
            "serializeCached", (ins "elf::writer::BinaryDataSection<uint8_t>&":$binDataSection, "ELF::SymbolReferenceMap&":$symRefMap), [{}],
            /*defaultImplementation*/ [{
                $_op.serialize(binDataSection);
            }]
        >,
        InterfaceMethod<
            "Get the size in bytes of the serialized form of this object, using a symbol reference map for faster look-up",
            "size_t",
            "getBinarySizeCached", (ins "ELF::SymbolReferenceMap&":$symRefMap, "::vpux::VPU::ArchKind":$arch), [{}],
            /*defaultImplementation*/ [{
                return $_op.getBinarySize(arch);
            }]
        >,

        InterfaceMethod<
            "Get the alignment requirements for the serialization of this op",
            "size_t",
            "getAlignmentRequirements", (ins "::vpux::VPU::ArchKind":$arch), [{}],
            /*defaultImplementation*/ [{
                return ELF::VPUX_NO_ALIGNMENT;
            }]
        >,

        InterfaceMethod<
            "Set the offset of an operation inside an ELF section op",
            "void",
            "setMemoryOffset", (ins "mlir::IntegerAttr":$offset), [{}],
            /*defaultImplementation*/ [{
                this->getOperation()->setAttr(BinaryOpInterface::elfMemOffsetAttrName(), offset);
            }]
        >,

        InterfaceMethod<
            "Get the offset of an operation inside an ELF section op",
            "uint64_t",
            "getMemoryOffset", (ins), [{}],
            /*defaultImplementation*/ [{
                auto op = this->getOperation();
                if(op->hasAttr(BinaryOpInterface::elfMemOffsetAttrName())) {
                    auto attr = op->template getAttrOfType<mlir::IntegerAttr>(BinaryOpInterface::elfMemOffsetAttrName());
                    return attr.getUInt();
                }
                else {
                    return 0;
                }
            }]
        >
    ];

    let extraClassDeclaration = [{
        static llvm::StringLiteral elfMemOffsetAttrName() {
            return "elfMemOffsetAttrKey";
        }
    }];
}

//
// PredefinedPurposeMemoryOpInterface
//

def ELF_PredefinedPurposeMemoryOpInterface : OpInterface<"PredefinedPurposeMemoryOpInterface"> {
    let description = "ELF Interface for the operations that represent accessible memory";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get the flags representing the processors that are known to access the memory represented by the op",
            "::vpux::ELF::SectionFlagsAttr",
            "getPredefinedMemoryAccessors", (ins)
        >,
    ];
}

//
// WrappableOpInterface
//

def ELF_WrappableOpInterface : OpInterface<"WrappableOpInterface"> {
    let description = "Interface provided by ELF dialect that the dialect passes will interact with";

    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get the signature name for this op to be placed in",
            "std::optional<vpux::ELF::SectionSignature>",
            "getSectionSignature", (ins)
        >,

        InterfaceMethod<
            "Check if an op is allocatable in memory",
            "bool",
            "hasMemoryFootprint", (ins)
        >,
    ];
}

//
// RelocatableOpInterface
//

def ELF_RelocatableOpInterface : OpInterface<"RelocatableOpInterface"> {
    let description = "ELF Interface for the operations that need relocations";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get all necesarry information in order to construct the op's relocations",
            "std::vector<vpux::ELF::RelocationInfo>",
            "getRelocationInfo", (ins "ELF::SymbolReferenceMap&":$symRefMap)
        >
    ];
}

//
// DDRMemoryAccessingOpInterface
//

def ELF_DDRMemoryAccessingOpInterface : OpInterface<"DDRMemoryAccessingOpInterface"> {
    let description = "ELF Interface for the operations that imply access to DDR Memory";
    let cppNamespace = "vpux::ELF";

    let methods = [
        InterfaceMethod<
            "Get the Symbol names of all the sections that this Op has symbolic links to",
            "::llvm::SmallVector<::mlir::StringAttr>",
            "getAccessedSections", (ins), [{}],
            /*defaultImplementation*/ [{
                    llvm::SmallVector<mlir::StringAttr> accessedSections;
                    auto dictAttr = this->getOperation()->getAttrDictionary();

                    dictAttr.walk([&] (mlir::Attribute attribute) {
                        if (auto symRef = mlir::dyn_cast<mlir::SymbolRefAttr>(attribute)) {
                            if (!mlir::isa<mlir::FlatSymbolRefAttr>(symRef)) {
                                accessedSections.push_back(symRef.getRootReference());
                            }
                        }
                    });
                    return accessedSections;
            }]
        >,
        InterfaceMethod<
            "Get the memory accessing processor that this op implies in terms of SectionFlags",
            "::vpux::ELF::SectionFlagsAttr",
            "getMemoryAccessingProc", (ins)
        >,
    ];
}

#endif
