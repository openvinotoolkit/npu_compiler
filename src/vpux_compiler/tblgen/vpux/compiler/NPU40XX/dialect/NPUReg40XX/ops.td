//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops_interfaces.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUASM/attributes.td"

include "mlir/IR/SymbolInterfaces.td"

class NPUReg40XX_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            NPUReg40XX_Dialect,
            mnemonic,
            traits
        >;

class NPUReg40XX_ExecutableTaskOp<string mnemonic, list<Trait> traits = []> :
        NPUReg40XX_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                ]
            )
        > {
}

class NPUReg40XX_ExecutableTaskOpWithProperties<string mnemonic, list<Trait> traits = [], string propertyName, string descriptorType> :
        NPUReg40XX_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                ]
            )
        > {
    let extraClassDeclaration = [{
    static mlir::ParseResult parse}] # propertyName # [{(
            mlir::AsmParser& parser, }] # descriptorType # [{& descriptor);

    static void print}] # propertyName # [{(mlir::AsmPrinter& printer, const }] # mnemonic # [{Op&,
                            const }] # descriptorType # [{& descriptor);
    }];

    let extraClassDefinition = [{
    mlir::ParseResult $cppClass::parse}] # propertyName # [{(
            mlir::AsmParser& parser, }] # descriptorType # [{& descriptor) {
        auto maybeDescriptor = descriptor.parse(parser);
        VPUX_THROW_UNLESS(maybeDescriptor.has_value(), "Something went wrong parsing }] # descriptorType # [{");
        descriptor = std::move(maybeDescriptor.value());
        return mlir::success(maybeDescriptor.has_value());
    }

    void $cppClass::print}] # propertyName # [{(mlir::AsmPrinter& printer, const $cppClass&,
                        const }] # descriptorType # [{& descriptor) {
        descriptor.print(printer);
    }
    }];

    let assemblyFormat = [{
        `descriptor` `=` custom<}] # propertyName # [{>($descriptor)
        attr-dict
    }];

}

//
// NNDMAOp
//

def NPUReg40XX_NNDMAOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"NNDMA",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
            ],
            "DmaDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::DMARegister"
        > {
    let summary = "NN DMA task";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DmaDescriptorProp:$descriptor,
        SymbolRefAttr:$input,
        SymbolRefArrayAttr:$output_buffs,
        OptionalAttr<SymbolRefAttr>:$next_link,
        OptionalAttr<SymbolRefAttr>:$act_compression_size_entry,
        OptionalAttr<SymbolRefAttr>:$indices,
        OptionalAttr<VPUIP_GatherAddressingMode>:$addressing_mode
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg40XX::Descriptors::DMARegister&&":$descriptor,
            "mlir::SymbolRefAttr":$input,
            "mlir::ArrayAttr":$output_buffs,
            "mlir::SymbolRefAttr":$next_link,
            "mlir::SymbolRefAttr":$act_compression_size_entry,
            "mlir::SymbolRefAttr":$indices,
            "VPUIP::GatherAddressingModeAttr":$addressing_mode
        )>
    ];
}

//
// MappedInferenceOp
//

def NPUReg40XX_MappedInferenceOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"MappedInference",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
                DeclareOpInterfaceMethods<VPURegMapped_SerializedVersionInterface>
            ],
            "MIDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuMappedInference"
        > {
    let summary = "Task representing the MappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64ArrayOfArraysAttr:$dmaCount,
        I64ArrayAttr:$dmaDDRCount,
        I64ArrayAttr:$dmaCMXCount,
        I64ArrayAttr:$invariantCount,
        I64ArrayAttr:$variantCount,
        I64ArrayAttr:$actKernelRangesCount,
        I64ArrayAttr:$actKernelInvocationsCount,
        I64Attr:$mediaCount,
        I64Attr:$barrierCount,
        SymbolRefAttr:$mappedInferenceVersion,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg,
        OptionalAttr<SymbolRefAttr>:$managedMappedInference,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefArrayAttr>:$invariantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$variantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelRanges,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$mediaTasks,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        MIDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::ArrayAttr":$dmaCount,
            "mlir::ArrayAttr":$dmaDDRCount,
            "mlir::ArrayAttr":$dmaCMXCount,
            "mlir::ArrayAttr":$invariantCount,
            "mlir::ArrayAttr":$variantCount,
            "mlir::ArrayAttr":$actKernelRangesCount,
            "mlir::ArrayAttr":$actKernelInvocationsCount,
            "mlir::IntegerAttr":$mediaCount,
            "mlir::IntegerAttr":$barrierCount,
            "mlir::SymbolRefAttr":$mappedInferenceVersion,
            "mlir::SymbolRefAttr":$actShaveRt,
            "mlir::ArrayAttr":$actShaveStacks,
            "mlir::SymbolRefAttr":$dmaHwpBase,
            "mlir::SymbolRefAttr":$hwpWorkpointCfg,
            "mlir::SymbolRefAttr":$managedMappedInference,
            "mlir::ArrayAttr":$dmaTasks,
            "mlir::ArrayAttr":$invariantTasks,
            "mlir::ArrayAttr":$variantTasks,
            "mlir::ArrayAttr":$actKernelRanges,
            "mlir::ArrayAttr":$actKernelInvocations,
            "mlir::SymbolRefAttr":$mediaTasks,
            "mlir::SymbolRefAttr":$barrierTasks,
            "vpux::NPUReg40XX::Descriptors::VpuMappedInference&&":$descriptor

        )>
    ];
}

//
// ConfigureBarrierOp
//

def NPUReg40XX_ConfigureBarrierOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ConfigureBarrierDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuBarrierCountConfig"
        > {
    let summary = "A task to configure the setup for a barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        ConfigureBarrierDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg40XX::Descriptors::VpuBarrierCountConfig&&":$descriptor
        )>
    ];
}

//
// ActShaveRtOp
//

def NPUReg40XX_ActShaveRtOp :
        NPUReg40XX_ExecutableTaskOp<"ActShaveRt",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol
            ]
        > {
    let summary = "Act Shave Management Kernel";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        StrAttr:$kernel_path
    );

    let extraClassDeclaration = [{
        uint32_t getKernelEntry();
        uint32_t getVersion();
    }];
}

//
// ActKernelRangeOp
//

def NPUReg40XX_ActKernelRangeOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"ActKernelRange",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ActKernelRangeDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuActKernelRange"
        > {
    let summary = "Activation Kernel Range";
    let arguments = (ins
        SymbolNameAttr:$sym_name,

        ActKernelRangeDescriptorProp:$descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$kernel_text,
        OptionalAttr<SymbolRefAttr>:$kernel_entry
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg40XX::Descriptors::VpuActKernelRange&&":$descriptor,
            "mlir::SymbolRefAttr":$task_location,
            "mlir::SymbolRefAttr":$kernel_text,
            "mlir::SymbolRefAttr":$kernel_entry
        )>
    ];
}

//
// ActKernelInvocationOp
//

def NPUReg40XX_ActKernelInvocationOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"ActKernelInvocation",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ActKernelInvocationDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuActKernelInvocation"
        > {
    let summary = "Activation Kernel Invocation";
    let arguments = (ins
        SymbolNameAttr:$sym_name,

        ActKernelInvocationDescriptorProp:$descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        OptionalAttr<SymbolRefAttr>:$next_link,
        SymbolRefAttr:$kernel_range,
        OptionalAttr<SymbolRefAttr>:$kernel_data,
        SymbolRefAttr:$kernel_params,
        OptionalAttr<SymbolRefAttr>:$profiling_data
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg40XX::Descriptors::VpuActKernelInvocation&&":$descriptor,
            "mlir::SymbolRefAttr":$task_location,
            "mlir::SymbolRefAttr":$next_link,
            "mlir::SymbolRefAttr":$kernel_range,
            "mlir::SymbolRefAttr":$kernel_data,
            "mlir::SymbolRefAttr":$kernel_params,
            "mlir::SymbolRefAttr":$profiling_data

        )>
    ];
}

//
// DPU Invariant Op
//

def NPUReg40XX_DPUInvariantOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"DPUInvariant", [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
            ],
            "DPUInvariantDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::DpuInvariantRegister"
        > {

    let summary = "DPU Invariant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPUASM_IndexTypeAttr:$task_index,

        DPUInvariantDescriptorProp:$descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$input,
        OptionalAttr<SymbolRefAttr>:$input_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$input_storage_element_table,

        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weights_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$weight_table,
        OptionalAttr<SymbolRefAttr>:$spr_lookup_table,

        OptionalAttr<SymbolRefAttr>:$output,
        OptionalAttr<SymbolRefAttr>:$output_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$profiling_data,
        UnitAttr:$is_zero_offset_weights_table,

        VPUIP_NCETaskTypeAttr:$nce_task_type,
        UnitAttr:$is_continued
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::TypeAttr":$task_index,
            "vpux::NPUReg40XX::Descriptors::DpuInvariantRegister&&":$descriptor,
            "mlir::SymbolRefAttr":$task_location,
            "mlir::SymbolRefAttr":$input,
            "mlir::SymbolRefAttr":$input_sparsity_map,
            "mlir::SymbolRefAttr":$input_storage_element_table,
            "mlir::SymbolRefAttr":$weights,
            "mlir::SymbolRefAttr":$weights_sparsity_map,
            "mlir::SymbolRefAttr":$weight_table,
            "mlir::SymbolRefAttr":$spr_lookup_table,
            "mlir::SymbolRefAttr":$output,
            "mlir::SymbolRefAttr":$output_sparsity_map,
            "mlir::SymbolRefAttr":$profiling_data,
            "mlir::UnitAttr":$is_zero_offset_weights_table,
            "VPUIP::NCETaskTypeAttr":$nce_task_type,
            "mlir::UnitAttr":$is_continued
        )>
    ];
}

//
// DPU Variant Op
//

def NPUReg40XX_DPUVariantOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"DPUVariant", [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "DPUVariantDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::DpuVariantRegister"
        > {

    let summary = "DPU Variant Op";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<SymbolRefAttr>:$next_link,
        VPUASM_IndexTypeAttr:$task_index,

        DPUVariantDescriptorProp:$descriptor,

        OptionalAttr<SymbolRefAttr>:$task_location,
        SymbolRefAttr:$invariant_task_location,
        OptionalAttr<SymbolRefAttr>:$weights,
        OptionalAttr<SymbolRefAttr>:$weight_table,
        VPUIP_NCETaskTypeAttr:$nce_task_type,
        OptionalAttr<IntAttr>:$workload_id
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::SymbolRefAttr":$next_link,
            "mlir::TypeAttr":$task_index,
            "vpux::NPUReg40XX::Descriptors::DpuVariantRegister&&":$descriptor,
            "mlir::SymbolRefAttr":$task_location,
            "mlir::SymbolRefAttr":$invariant_task_location,
            "mlir::SymbolRefAttr":$weights,
            "mlir::SymbolRefAttr":$weight_table,
            "VPUIP::NCETaskTypeAttr":$nce_task_type,
            "mlir::IntegerAttr":$workload_id
        )>
    ];
}


//
// NNRTConfigOp
//

def NPUReg40XX_NNrtConfigOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"NNrtConfig",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "NNRTCfgDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuNNRTConfig"
        > {
    let summary = "Task representing the NNRTConfig structure inside managed mapped inference";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UnitAttr:$isActKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$actShaveRt,
        OptionalAttr<SymbolRefArrayAttr>:$actShaveStacks,
        OptionalAttr<SymbolRefAttr>:$dmaHwpBase,
        OptionalAttr<SymbolRefAttr>:$hwpWorkpointCfg,
        NNRTCfgDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::UnitAttr":$isActKernelInvocations,
            "mlir::SymbolRefAttr":$actShaveRt,
            "mlir::ArrayAttr":$actShaveStacks,
            "mlir::SymbolRefAttr":$dmaHwpBase,
            "mlir::SymbolRefAttr":$hwpWorkpointCfg,
            "vpux::NPUReg40XX::Descriptors::VpuNNRTConfig&&":$descriptor
        )>
    ];
}

//
// ManagedMappedInferenceOp
//

def NPUReg40XX_ManagedMappedInferenceOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"ManagedMappedInference",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
                DeclareOpInterfaceMethods<VPURegMapped_SerializedVersionInterface>
            ],
            "MMIDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuManagedMappedInference"
        > {
    let summary = "Task representing the ManagedMappedInference structure";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$nnrtConfig,
        SymbolRefAttr:$mappedInferenceVersion,
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefAttr>:$workItems,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$bootstrapBarriers,
        OptionalAttr<SymbolRefAttr>:$barrierConfigurationTasks,
        OptionalAttr<SymbolRefAttr>:$numOfBarrierReprogrammings,
        MMIDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "mlir::SymbolRefAttr":$nnrtConfig,
            "mlir::SymbolRefAttr":$mappedInferenceVersion,
            "mlir::ArrayAttr":$dmaTasks,
            "mlir::SymbolRefAttr":$workItems,
            "mlir::SymbolRefAttr":$barrierTasks,
            "mlir::SymbolRefAttr":$bootstrapBarriers,
            "mlir::SymbolRefAttr":$barrierConfigurationTasks,
            "mlir::SymbolRefAttr":$numOfBarrierReprogrammings,
            "vpux::NPUReg40XX::Descriptors::VpuManagedMappedInference&&":$descriptor
        )>
    ];
}

//
// ManagedBarrierOp
//

def NPUReg40XX_ManagedBarrierOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"ManagedBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "TaskBarrierMapDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap"
        > {
    let summary = "A task to configure the setup for a managed barrier";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TaskBarrierMapDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap&&":$descriptor
        )>
    ];
}

//
// WorkItemOp
//

def NPUReg40XX_WorkItemOp :
        NPUReg40XX_ExecutableTaskOpWithProperties<"WorkItem",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "WorkItenDescriptorProp",
            "vpux::NPUReg40XX::Descriptors::WorkItem"
        > {
    let summary = "A task to configure the work item";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        VPURegMapped_TaskTypeAttr:$task_type,
        SymbolRefAttr:$first_task,
        WorkItenDescriptorProp:$descriptor
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "VPURegMapped::TaskTypeAttr":$task_type,
            "mlir::SymbolRefAttr":$first_task,
            "vpux::NPUReg40XX::Descriptors::WorkItem&&":$descriptor
        )>
    ];
}

//
// MappedInferenceVersionOp
//

def MappedInferenceVersionOp :
        NPUReg40XX_Op<"MappedInferenceVersion",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_SerializedVersionInterface>
            ]
        > {
    let summary = "Represents a versioning mechanism for Mapped Inference";

    let arguments = (ins
          SymbolNameAttr:$sym_name,
          UI32Attr:$major,
          UI32Attr:$minor,
          UI32Attr:$patch
        );

    let assemblyFormat = [{
        `(` $major `_` $minor `_` $patch `)`
        attr-dict}];
}
