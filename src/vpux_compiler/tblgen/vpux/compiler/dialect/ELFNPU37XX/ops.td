//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELFNPU37XX_OPS
#define VPUX_COMPILER_DIALECT_ELFNPU37XX_OPS

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/ELFNPU37XX/attributes.td"
include "vpux/compiler/dialect/ELFNPU37XX/dialect.td"
include "vpux/compiler/dialect/ELFNPU37XX/types.td"
include "vpux/compiler/dialect/VPUMI37XX/dialect.td"
include "vpux/compiler/dialect/VPUMI37XX/types.td"
include "mlir/IR/RegionKindInterface.td"
include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPURT/types.td"
include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"

include "mlir/Interfaces/SideEffectInterfaces.td"


//
// Base class for the ELF dialect operations.
//

class ELFNPU37XX_Op<string mnemonic, list<Trait> traits = []> :
    Op<
        ELFNPU37XX_Dialect,
        mnemonic,
        traits
    >;

//
// CreateSectionOp
//

def CreateSectionOp :
        ELFNPU37XX_Op<"CreateSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Section";

    let arguments = (ins
        StrAttr:$secName,
        ELFNPU37XX_SectionTypeAttr:$secType,
        ELFNPU37XX_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateLogicalSectionOp :
//

def CreateLogicalSectionOp :
        ELFNPU37XX_Op<"CreateLogicalSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {

    let summary = "Create an ELF Logical Section, with no actual binary content in the ELF file ";

    let arguments = (ins
        StrAttr:$secName,
        ELFNPU37XX_SectionTypeAttr:$secType,
        ELFNPU37XX_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$declaredOps
    );

    let assemblyFormat = [{
        `secType` `(` $secType `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $declaredOps
    }];
}

//
// CreateMetadataSectionOp
//

def CreateMetadataSectionOp :
        ELFNPU37XX_Op<"CreateMetadataSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Metadata Section";

    let arguments = (ins
        StrAttr:$secName,
        ELFNPU37XX_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let extraClassDeclaration = [{
        void serialize(elf::Writer& writer, vpux::DenseMap<mlir::Operation*, elf::writer::Section*>& sectionMap, vpux::DenseMap<mlir::Operation*, elf::writer::Symbol*>& symbolMap, ::elf::NetworkMetadata& metadata);
    }];

    let assemblyFormat = [{
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateProfilingSectionOp
//

def CreateProfilingSectionOp :
        ELFNPU37XX_Op<"CreateProfilingSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Profiling Metadata Section";

    let arguments = (ins
        StrAttr:$secName,
        ELFNPU37XX_SectionFlagsAttr:$secFlags,
        IntAttr:$secInfo,
        IntAttr:$secAddrAlign
        );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];

    let hasVerifier = 1;
}

//
// CreateSymbolTableSectionOp
//

def CreateSymbolTableSectionOp :
        ELFNPU37XX_Op<"CreateSymbolTableSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
                      StrAttr:$secName,
                      ELFNPU37XX_SectionFlagsAttr:$secFlags,
                      UnitAttr:$isBuiltin
                    );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secName` `(` $secName `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}

//
// CreateRelocationSectionOp
//

def CreateRelocationSectionOp :
        ELFNPU37XX_Op<"CreateRelocationSection",
            [
                NoRegionArguments,
                DeclareOpInterfaceMethods<ELFNPU37XX_SectionInterface>
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Create ELF Relocation Section";

    let arguments = (ins
                      StrAttr:$secName,
                      ELFNPU37XX_Section:$sourceSymbolTableSection,
                      ELFNPU37XX_Section:$targetSection,
                      ELFNPU37XX_SectionFlagsAttr:$secFlags
                    );

    let results = (outs ELFNPU37XX_Section:$section);

    let regions = (region
        SizedRegion<1>:$aRegion
    );

    let assemblyFormat = [{
        `secName` `(` $secName `)`
        `sourceSymbolTableSection` `(` $sourceSymbolTableSection `)`
        `targetSection` `(` $targetSection `)`
        `secFlags` `(` $secFlags `)`
        attr-dict
        `->` type(results)
        $aRegion
    }];
}


//
// SymbolOp
//

def SymbolOp :
        ELFNPU37XX_Op<"Symbol",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_SymbolObjectInterface>
            ]
        > {
    let summary = "Create ELF Symbol Table Section";

    let arguments = (ins
          AnyType:$inputArg,
          UnitAttr:$isBuiltin,
          OptionalAttr<StrAttr>:$name,
          OptionalAttr<ELFNPU37XX_SymbolTypeAttr>:$symType,
          OptionalAttr<UI64Attr>:$size,
          OptionalAttr<UI64Attr>:$value
        );

    let results = (outs ELFNPU37XX_Symbol:$symbol);

    let assemblyFormat = [{
        $inputArg
        (`name` `(` $name^ `)`)?
        (`type` `(` $symType^ `)`)?
        (`size` `(` $size^ `)`)?
        attr-dict
        `:` type($inputArg)
    }];
}

//
// RelocOp
//

def RelocOp :
    ELFNPU37XX_Op<"Reloc",
        [
            DeclareOpInterfaceMethods<ELFNPU37XX_RelocationObjectInterface>
        ]
    > {
    let summary = "Operation that describes a relocation";

    let description = [{
        From a funtional point of view, the operation describes an ELF Relocation, having the following components:
          * offset
          * source symbol
          * relocation type
          * addend

        To enhance the ease of use, it also specifies 2 optional arguments:
          * baseOpO - if provided, the specified numerical offset is considered to be with respect to
                      the start of the binary representation of the baseOpOffset points to
                    - if not provided, the numerical offset is considered to be with respect to the start
                      of the target section of the relocation
          * description - optional stringAttr which can be used to provide meta information about the relocation,
                          such as specifying the goal of one specific relocation
    }];

    let arguments = (ins
          Optional<VPURegMapped_IndexType>:$baseOp,
          IntAttr:$offset,
          ELFNPU37XX_RelocationTypeAttr:$relocationType,
          ELFNPU37XX_Symbol:$sourceSymbol,
          IntAttr:$addend,
          OptionalAttr<StrAttr>:$description
        );

    let results = (outs);

    let assemblyFormat = [{
        (`baseOp` `(` $baseOp^ `:` type($baseOp) `)`)?
        `offset` `(` $offset `)`
        $relocationType $sourceSymbol $addend attr-dict}];

    let builders = [
        OpBuilder<(ins "mlir::Value":$baseOp, "int64_t":$offset, "vpux::ELFNPU37XX::RelocationType":$relocationType, "mlir::Value":$sourceSymbol, "int64_t":$addend, "llvm::StringRef":$description)>,
        OpBuilder<(ins "mlir::Value":$baseOp, "int64_t":$offset, "vpux::ELFNPU37XX::RelocationType":$relocationType, "mlir::Value":$sourceSymbol, "int64_t":$addend)>
    ];
    }

//
// PutOpInSectionOp
//

def PutOpInSectionOp :
        ELFNPU37XX_Op<"PutOpInSection",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Put the Argument Op in the ELF Section";

    let arguments = (ins
          AnyType:$inputArg
        );

    let results = (outs);

    let assemblyFormat = "$inputArg attr-dict `:` type($inputArg)";
}

//
// PadOp
//

def PadOp :
        ELFNPU37XX_Op<"Pad",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Padding for inner section alignment";

    let arguments = (ins
          UI32Attr:$paddingSize,
          OptionalAttr<UI8Attr>:$paddingValue
        );

    let results = (outs);

    let assemblyFormat = [{
        `size` `(` $paddingSize `)`
        (`value` `(` $paddingValue^ `)`)?
        attr-dict}];
}

//
// ABIVersionOp
//

def ABIVersionOp :
        ELFNPU37XX_Op<"ABIVersion",
            [
                DeclareOpInterfaceMethods<ELFNPU37XX_BinaryOpInterface>
            ]
        > {
    let summary = "Represents a versioning mechanism for the ELF Container and ABI";

    let arguments = (ins
          UI32Attr:$major,
          UI32Attr:$minor,
          UI32Attr:$patch
        );

    let results = (outs);

    let assemblyFormat = [{
        `(` $major `_` $minor `_` $patch `)`
        attr-dict}];

    let builders = [
        OpBuilder<(ins)>
        ];
}

#endif
