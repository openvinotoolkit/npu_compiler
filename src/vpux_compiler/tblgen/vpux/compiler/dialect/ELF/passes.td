//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_ELF_PASSES
#define VPUX_COMPILER_DIALECT_ELF_PASSES

include "mlir/Pass/PassBase.td"

//
// CreateSymbolTable
//

def AddELFSymbolTable : PassBase<"create-elf-symbol-table", "vpux::FunctionPass"> {
    let summary = "Create ELF symbol table section";

    let description = [{
        Each ELF section that holds program data needs to have an associated symbol op.
        For each DataSection and LogicalSection operation, an associated ELF symbol op will be added to the SymbolTableSection.
    }];

    let constructor = "vpux::ELF::createAddELFSymbolTablePass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

//
// CreateELFRelocations
//

def AddELFRelocations : PassBase<"create-elf-relocations", "vpux::FunctionPass"> {
    let summary = "Create ELF Relocation sections and relocation ops";

    let description = [{
        In the ELF dialect we represent addressable relationships via SymbolicReferences. Within the ELF flow these
        relationships are materialized via ELF Relocations.
        This pass will go over all sections and all OPS within those sections, and for each of the symbolic reference
        for an OP will add a relocation (if required).
    }];

    let constructor = "vpux::ELF::createAddELFRelocationsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

def AddRelocationsForDynamicStridesDmas : PassBase<"add-relocations-for-dynamic-strides-dmas", "vpux::FunctionPass"> {
    let summary = "Create relocations for DMAs supporting dynamic strides";

    let description = [{
        For each NNDMA op that has "stridedInput" or "stridedOutput" attribute set this pass will create
        a corresponding DMASymbolOp and RelocOp.
    }];

    let constructor = "vpux::ELF::createAddRelocationsForDynamicStridesDMAsPass()";
}

//
// SetElfOpOffsets
//

def SetOpOffsets : PassBase<"set-elf-op-offsets", "vpux::FunctionPass"> {
    let summary = "Set binary offsets of ELF operations";

    let description = [{
        Operations inside ELF sections have inate memory placements. This pass will iterate over them, and sets
        each ops offset from the start of the section, based on their binary size.
    }];

    let constructor = "vpux::ELF::createSetOpOffsetsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

//
// SetEntryPoint
//

def SetEntryPoint : PassBase<"set-elf-entrypoint", "vpux::FunctionPass"> {
    let summary = "Set the ELF entryPoint";

    let description = [{
        Any ELF file needs an entryPoint. Traditionally there is a dedicated entry field in the ELF header for this
        but our flow cannot yet support that(TBD).
        Our current mechanism is to set a dedicated symbol with a dedicated type to mark the address that is the entry.
    }];

    let constructor = "vpux::ELF::createSetEntryPointPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
        "vpux::VPUASM::VPUASMDialect"
    ];
}

//
// AddNetworkMetadata
//

def AddNetworkMetadata : PassBase<"add-network-metadata", "vpux::FunctionPass"> {
    let summary = "Create ELF metadata section with NetworkMetadataOp";

    let description = [{
        AI-generated summary

        Creates an ELF MetadataSection inside the ELF.MainOp and inserts a
        VPUASM.NetworkMetadataOp into it. After inserting the metadata op, the
        section alignment is updated to match the metadata op's alignment
        requirements for the target architecture.
    }];

    let constructor = "vpux::ELF::createAddNetworkMetadataPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
        "vpux::VPUASM::VPUASMDialect"
    ];
}

//
// UpdateELFSectionFlags
//

def UpdateELFSectionFlags : PassBase<"update-ELF-section-flags", "vpux::ModulePass"> {
    let summary = "Update ELF section flags";

    let description = [{
        The pass is part of the ELF pipeline.

        The pass will update the ELF section flags (secFlags attribute) for each relevant section based on which
        processors will access the given section at run time. This information is retrieved from the ops contained
        within the section via the ELF_WrappableOpInterface::getAccessingProcs interface method.

        A relevant section in the context of this pass is one that requires allocation and/or loading by the loader.
    }];

    let constructor = "vpux::ELF::createUpdateELFSectionFlagsPass()";

    let options = [
        Option<"enableShaveDDRAccess", "enable-shave-ddr-access", "bool", "true",
               "Enable SHAVE processor DDR access flag updates for ELF sections">
    ];

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

//
// Cleanup ELF Sections
//

def RemoveEmptyELFSections : PassBase<"remove-empty-ELF-sections", "vpux::FunctionPass"> {
    let summary = "Cleans up empty sections in ELF";

    let description = [{
        The pass is part of the ELF pipeline.

        Current lowering logic may pre-create sections, without yet knowing if any op will eventually land in it. This
        causes some empty sections to be placed in the ELF. This pass should clean them up.
    }];

    let constructor = "vpux::ELF::createRemoveEmptyELFSectionsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

//
// HandleAlignmentRequirements
//

def HandleAlignmentRequirements : PassBase<"handle-alignment-requirements", "vpux::FunctionPass"> {
    let summary = "Update ELF section alignment and add inner-section Pad ops ";

    let description = [{
        The pass that updates the section alignment requirements and ensures that - by adding ELF.PadOps in between the ops -
        alignment is met within the section.
    }];

    let constructor = "vpux::ELF::createHandleAlignmentRequirementsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

def AddABIVersion : PassBase<"add-abi-version", "vpux::FunctionPass"> {
    let summary = "Add ABI version op to ELF";

    let description = [{
        The pass adds ELF.ABIVersionOp to IR. Major, minor and patch version numbers are passed as arguments.
    }];

    let constructor = "vpux::ELF::createAddABIVersionPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect",
    ];
}

//
// SetCMXSymbolValue
//

def SetCMXSymbolValue : PassBase<"set-cmx-symbol", "vpux::ModulePass"> {
    let summary = "Set cmx symbol value pass";

    let description = [{
        Set CMX symbol value that are known at compile time. Note that the values used
        in this pass are also used in HostParsedInference_4000 (from the elf library). Any change
        to the values from this pass or values from the elf library should be duplicated in order to
        ensure the same behavior. The symbols will be later used in another pass to apply relocations for known symbols.
    }];

    let constructor = "vpux::ELF::createSetCMXSymbolValuePass()";

        let options = [
        Option<
            "workspaceAddr", "cmx-workspace-addr",
            "uint64_t", "0",
            "CMX workspace start address."
        >,
        Option<
            "workspaceSize", "cmx-workspace-size",
            "uint64_t", "0",
            "CMX workspace size."
        >,
        Option<
            "metadataAddr", "cmx-metadata-addr",
            "uint64_t", "0",
            "CMX metadata start address."
        >,
        Option<
            "metadataSize", "cmx-metadata-size",
            "uint64_t", "0",
            "CMX metadata size."
        >
    ];
}

def FinalizeSkipDmaChains : PassBase<"finalize-skip-dma-chains", "vpux::FunctionPass"> {
    let summary = "Patches the fetch DMAs for skip DMAs";

    let description = [{
        This pass updates the Fetch DMA operations to correctly reference the corresponding
        Skip DMA.

        The pass performs the following key steps:

        1. Collect all DMA operations:
           - Builds mappings from DMA symbols to ELF sections.
           - Maps Fetch DMA descriptors by descriptor ID.
           - Maps Skip DMA descriptors by descriptor ID
           - Maps KernelParams by skip descriptor ID

        2. Resolve release descriptors for Skip DMAs:
           - Follows the chain of Skip DMAs to find the first DMA without a taskLocation.
           - Updates the KernelParams's release descriptor attribute to point to the corresponding DMA

        3. Close Skip DMA loops across tiles:
           - Ensures sequential Skip DMA chains within the same logical task are properly linked.
           - Wraps the last Skip DMA back to the first in the group to form a circular chain.

        4. Patch Fetch DMAs:
           - Connects Fetch DMAs to the correct Skip DMA descriptors for data prefetching.

        Note: Each skipDescId is used to locate its corresponding release DMA descriptor.
        The reservation for release descriptors is fixed by the implementation.
        The pass reserves maxNumReleaseDesc * 8 bytes, which corresponds to 4 release
        descriptors at 8 bytes per descriptor, regardless of the number of skipDescIds.
    }];

    let constructor = "vpux::ELF::createFinalizeSkipDmaChainsPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

#endif
