//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_OPS
#define VPUX_COMPILER_DIALECT_ELF_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/dialect.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/VPURT/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//
// Base class for the ELF dialect operations.
//

class ELF_Op<string mnemonic, list<Trait> traits = []> :
    Op<
        ELF_Dialect,
        mnemonic,
        traits
    >;

//
// CreateSectionOp
//

def DataSectionOp :
        ELF_Op<"CreateSection",
            [
                Symbol,
                SymbolContainer,
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>,
                DeclareOpInterfaceMethods<ELF_SymbolicallyRepresentedOpInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Representation of an ELF Section op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI64Attr:$secAddrAlign,
        DefaultValuedAttr<ELF_SectionTypeAttr,"SectionTypeAttr::SHT_PROGBITS">:$secType,
        DefaultValuedAttr<ELF_SectionFlagsAttr,"SectionFlagsAttr::SHF_ALLOC">:$secFlags,
        DefaultValuedAttr<VPURT_BufferSectionAttr,"VPURT::BufferSection::DDR">:$secLocation
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `aligned` `(` $secAddrAlign `)`
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        `secLocation` `(` $secLocation `)`
        $content
    }];
}

//
// CreateLogicalSectionOp :
//

def LogicalSectionOp :
        ELF_Op<"CreateLogicalSection",
            [
                Symbol,
                SymbolContainer,
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>,
                DeclareOpInterfaceMethods<ELF_SymbolicallyRepresentedOpInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Representation of an ELF Section, with no actual binary content in the ELF file ";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI64Attr:$secAddrAlign,
        DefaultValuedAttr<ELF_SectionTypeAttr,"SectionTypeAttr::SHT_NOBITS">:$secType,
        DefaultValuedAttr<ELF_SectionFlagsAttr,"SectionFlagsAttr::SHF_ALLOC">:$secFlags,
        VPURT_BufferSectionAttr:$secLocation
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `aligned` `(` $secAddrAlign `)`
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        `secLocation` `(` $secLocation `)`
        $content
    }];

    let extraClassDeclaration = [{
        size_t getTotalSize(vpux::ELF::SymbolReferenceMap&);
    }];
}

//
// CreateMetadataSectionOp
//

def CreateMetadataSectionOp :
        ELF_Op<"CreateMetadataSection",
            [
                Symbol,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Metadata Section";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI64Attr:$secAddrAlign,
        ELF_SectionFlagsAttr:$secFlags
        );

    let regions = (region
        SizedRegion<1>:$content
    );

    let extraClassDeclaration = [{
        void preserialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::ELF::SymbolReferenceMap& symRefMap);
        void serialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>& symbolMap, ::elf::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `aligned` `(` $secAddrAlign `)`
        `secFlags` `(` $secFlags `)`
        $content
    }];
}

//
// PerformanceMetrics
//

def PerformanceMetricsOp :
        ELF_Op<"PerformanceMetricsSection",
            [
                Symbol,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>
            ]
        > {
    let summary = "Create ELF PerformanceMetrics Section";

    let arguments = (ins
        SymbolNameAttr:$sym_name
        );

    let assemblyFormat = [{
        attr-dict
        $sym_name
    }];

    let builders = [
      OpBuilder<(ins)>
      ];
}

//
// CreateProfilingSectionOp
//

def CreateProfilingSectionOp :
        ELF_Op<"CreateProfilingSection",
            [
                Symbol,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Profiling Metadata Section";

    let description = [{
        This operation represents an ELF section specifically intended for the storage of profiling metadata. The
        metadata within this op will be save to the ".profiling" section of the resulting ELF file. The serialized
        profiling metadata will be used by profiling post-processing to generate human-readable profiling reports.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UI64Attr:$secAddrAlign,
        ELF_SectionFlagsAttr:$secFlags
        );

    let regions = (region
        SizedRegion<1>:$content
    );

    let extraClassDeclaration = [{
        void preserialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::ELF::SymbolReferenceMap& symRefMap);
        void serialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>& symbolMap);
    }];

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `aligned` `(` $secAddrAlign `)`
        `secFlags` `(` $secFlags `)`
        $content
    }];
}

//
// CreateSymbolTableSectionOp
//

def CreateSymbolTableSectionOp :
        ELF_Op<"CreateSymbolTableSection",
            [
                Symbol,
                SymbolContainer,
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
                      SymbolNameAttr:$sym_name,
                      ELF_SectionFlagsAttr:$secFlags
                    );

    let regions = (region
        SizedRegion<1>:$symbols
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `secFlags` `(` $secFlags `)`
        $symbols
    }];
}

//
// CreateRelocationSectionOp
//

def CreateRelocationSectionOp :
        ELF_Op<"CreateRelocationSection",
            [
                Symbol,
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELF_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Relocation Section";

    let arguments = (ins
                      SymbolNameAttr:$sym_name,
                      FlatSymbolRefAttr:$targetSection,
                      FlatSymbolRefAttr:$sourceSymbolTableSection,
                      ELF_SectionFlagsAttr:$secFlags
                    );

    let regions = (region
        SizedRegion<1>:$relocations
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `target` `(` $targetSection `)`
        `symtab` `(` $sourceSymbolTableSection `)`
        `secFlags` `(` $secFlags `)`
        $relocations
    }];
}


//
// SymbolOp
//

def SymbolOp :
        ELF_Op<"Symbol",
            [
                Symbol,
                DeclareOpInterfaceMethods<ELF_SymbolObjectInterface>
            ]
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
          SymbolNameAttr:$sym_name,
          SymbolRefAttr:$reference,
          DefaultValuedAttr<ELF_SymbolTypeAttr,"ELF::SymbolType::STT_OBJECT">:$type,
          OptionalAttr<UI64Attr>:$size,
          OptionalAttr<UI64Attr>:$value
        );

    let extraClassDeclaration = [{
        static llvm::StringLiteral getDefaultNamePrefix() {
            return "elfsym.";
        }
    }];

    let assemblyFormat = [{
        $sym_name
        `of` `(` $reference `)`
        `type` `(` $type `)`
        (`size` `(` $size^ `)`)?
        (`value` `(` $value^ `)`)?
        attr-dict
    }];

    let builders = [
        OpBuilder<(ins "vpux::ELF::SymbolSignature&":$signature)>,
        OpBuilder<(ins "::llvm::StringRef":$symName, "::mlir::SymbolRefAttr":$reference, "vpux::ELF::SymbolType":$type)>
    ];
}

//
// RelocOps
//

def RelocOp :
    ELF_Op<"Reloc",
        [
            DeclareOpInterfaceMethods<ELF_RelocationObjectInterface>
        ]
    > {
    let summary = "Operation that describes a relocation";

    let description = [{
        The operation describes an ELF Relocation, having the following components:
          * offset
          * source symbol
          * relocation type
          * addend
          * description - optional stringAttr which can be used to provide meta information about the relocation,
                          such as specifying the goal of one specific relocation
    }];

    let arguments = (ins
          IntAttr:$offset,
          SymbolRefAttr:$sourceSymbol,
          ELF_RelocationTypeAttr:$relocationType,
          IntAttr:$addend,
          OptionalAttr<StrAttr>:$description
        );

    let assemblyFormat = [{
        `offset` `(` $offset `)`
        `sourceSym` `(` $sourceSymbol `)`
        `relocType` `(` $relocationType `)`
        `addend` `(` $addend `)`
        (`(` `description` `:` $description^ `)`)?
        attr-dict
    }];

    let builders = [
        OpBuilder<(ins "int64_t":$offset, "mlir::SymbolRefAttr":$sourceSymbol, "vpux::ELF::RelocationType":$relocationType, "int64_t":$addend, "llvm::StringRef":$description)>,
        OpBuilder<(ins "int64_t":$offset, "mlir::SymbolRefAttr":$sourceSymbol, "vpux::ELF::RelocationType":$relocationType, "int64_t":$addend)>
    ];

}

//
// PadOp
//

def PadOp :
        ELF_Op<"Pad",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize"]>,
            ]
        > {
    let summary = "Padding for inner section alignment";

    let arguments = (ins
          UI32Attr:$paddingSize,
          OptionalAttr<UI8Attr>:$paddingValue
        );

    let assemblyFormat = [{
        `size` `(` $paddingSize `)`
        (`value` `(` $paddingValue^ `)`)?
        attr-dict}];
}

//
// ABIVersionOp
//

def ABIVersionOp :
        ELF_Op<"ABIVersion",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                DeclareOpInterfaceMethods<ELF_WrappableOpInterface>,
                Symbol
            ]
        > {
    let summary = "Represents a versioning mechanism for the ELF Container and ABI";

    let arguments = (ins
          SymbolNameAttr:$sym_name,
          UI32Attr:$major,
          UI32Attr:$minor,
          UI32Attr:$patch
        );

    let assemblyFormat = [{
        `(` $major `_` $minor `_` $patch `)`
        attr-dict}];

    let builders = [
        OpBuilder<(ins "uint32_t":$major, "uint32_t":$minor, "uint32_t":$patch)>
        ];
}

def MainOp :
        ELF_Op<"Main",
            [
                Symbol,
                SymbolTable,
                IsolatedFromAbove
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Main ELF op encapsulating the ELF contents";

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$content
    );

    let assemblyFormat = [{
        attr-dict
        $sym_name
        $content
    }];

}

#endif
