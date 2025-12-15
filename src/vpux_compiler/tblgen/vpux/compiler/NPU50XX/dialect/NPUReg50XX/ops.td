//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/ops_interfaces.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/attributes.td"
include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPUASM/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPUASM/attributes.td"

include "mlir/IR/SymbolInterfaces.td"

class NPUReg50XX_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            NPUReg50XX_Dialect,
            mnemonic,
            traits
        >;

class NPUReg50XX_ExecutableTaskOp<string mnemonic, list<Trait> traits = []> :
        NPUReg50XX_Op<
            mnemonic,
            !listconcat(
                traits,
                [
                    DeclareOpInterfaceMethods<ELF_BinarySizeOpInterface, ["getBinarySize", "getAlignmentRequirements"]>,
                ]
            )
        > {
}

class NPUReg50XX_ExecutableTaskOpWithProperties<string mnemonic, list<Trait> traits = [], string propertyName, string descriptorType> :
        NPUReg50XX_Op<
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

def NPUReg50XX_NNDMAOp : NPUReg50XX_ExecutableTaskOpWithProperties<"NNDMA", [
            DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
            DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
            Symbol,
            DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
            IsolatedFromAbove
        ],
        "DmaDescriptorProp",
        "vpux::NPUReg50XX::Descriptors::DMARegister"
        >{
    let summary = "NN DMA task";
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DmaDescriptorProp:$descriptor,
        SymbolRefAttr:$input,
        SymbolRefArrayAttr:$output_buffs,
        OptionalAttr<SymbolRefAttr>:$next_link,
        OptionalAttr<SymbolRefAttr>:$act_compression_size_entry,
        OptionalAttr<SymbolRefAttr>:$act_compression_sparsity_map,
        OptionalAttr<SymbolRefAttr>:$indices,
        OptionalAttr<VPUIP_GatherAddressingMode>:$addressing_mode
    );

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::StringAttr":$sym_name,
            "vpux::NPUReg50XX::Descriptors::DMARegister&&":$descriptor,
            "mlir::SymbolRefAttr":$input,
            "mlir::ArrayAttr":$output_buffs,
            "mlir::SymbolRefAttr":$next_link,
            "mlir::SymbolRefAttr":$act_compression_size_entry,
            "mlir::SymbolRefAttr":$act_compression_sparsity_map,
            "mlir::SymbolRefAttr":$indices,
            "VPUIP::GatherAddressingModeAttr":$addressing_mode
        )>
    ];
}

//
// MappedInferenceOp
//
def NPUReg50XX_MappedInferenceOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"MappedInference",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
                DeclareOpInterfaceMethods<VPURegMapped_SerializedVersionInterface>
            ],
            "MIDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuMappedInference"
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
        OptionalAttr<SymbolRefArrayOfArraysAttr>:$dmaTasks,
        OptionalAttr<SymbolRefArrayAttr>:$invariantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$variantTasks,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelRanges,
        OptionalAttr<SymbolRefArrayAttr>:$actKernelInvocations,
        OptionalAttr<SymbolRefAttr>:$mediaTasks,
        OptionalAttr<SymbolRefAttr>:$barrierTasks,
        OptionalAttr<SymbolRefAttr>:$managedMappedInference,
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
            "mlir::ArrayAttr":$dmaTasks,
            "mlir::ArrayAttr":$invariantTasks,
            "mlir::ArrayAttr":$variantTasks,
            "mlir::ArrayAttr":$actKernelRanges,
            "mlir::ArrayAttr":$actKernelInvocations,
            "mlir::SymbolRefAttr":$mediaTasks,
            "mlir::SymbolRefAttr":$barrierTasks,
            "mlir::SymbolRefAttr":$managedMappedInference,
            "vpux::NPUReg50XX::Descriptors::VpuMappedInference&&":$descriptor

        )>
    ];
}

//
// ConfigureBarrierOp
//

def NPUReg50XX_ConfigureBarrierOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"ConfigureBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ConfigureBarrierDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig"
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
            "vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig&&":$descriptor
        )>
    ];
}

//
// ActShaveRtOp
//

def NPUReg50XX_ActShaveRtOp :
        NPUReg50XX_ExecutableTaskOp<"ActShaveRt",
            [
                    DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                    Symbol
            ]> {
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

def NPUReg50XX_ActKernelRangeOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"ActKernelRange",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ActKernelRangeDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuActKernelRange"
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
            "vpux::NPUReg50XX::Descriptors::VpuActKernelRange&&":$descriptor,
            "mlir::SymbolRefAttr":$task_location,
            "mlir::SymbolRefAttr":$kernel_text,
            "mlir::SymbolRefAttr":$kernel_entry
        )>
    ];
}

//
// ActKernelInvocationOp
//

def NPUReg50XX_ActKernelInvocationOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"ActKernelInvocation",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "ActKernelInvocationDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation"
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
            "vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation&&":$descriptor,
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

def NPUReg50XX_DPUInvariantOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"DPUInvariant", [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "DPUInvariantDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::DpuInvariantRegister"
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
            "vpux::NPUReg50XX::Descriptors::DpuInvariantRegister&&":$descriptor,
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

def NPUReg50XX_DPUVariantOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"DPUVariant", [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                Symbol,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "DPUVariantDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::DpuVariantRegister"
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
            "vpux::NPUReg50XX::Descriptors::DpuVariantRegister&&":$descriptor,
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
// ManagedMappedInferenceOp
//

def NPUReg50XX_ManagedMappedInferenceOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"ManagedMappedInference",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>,
                DeclareOpInterfaceMethods<VPURegMapped_SerializedVersionInterface>
            ],
            "MMIDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuManagedMappedInference"
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
            "vpux::NPUReg50XX::Descriptors::VpuManagedMappedInference&&":$descriptor
        )>
    ];
}

//
// ManagedBarrierOp
//

def NPUReg50XX_ManagedBarrierOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"ManagedBarrier",
            [
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "TaskBarrierMapDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuTaskBarrierMap"
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
            "vpux::NPUReg50XX::Descriptors::VpuTaskBarrierMap&&":$descriptor
        )>
    ];
}

//
// WorkItemOp
//

def NPUReg50XX_WorkItemOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"WorkItem",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "WorkItenDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::WorkItem"
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
            "vpux::NPUReg50XX::Descriptors::WorkItem&&":$descriptor
        )>
    ];
}

//
// NNRTConfigOp
//

def NPUReg50XX_NNrtConfigOp :
        NPUReg50XX_ExecutableTaskOpWithProperties<"NNrtConfig",
            [
                DeclareOpInterfaceMethods<ELF_RelocatableOpInterface>,
                DeclareOpInterfaceMethods<ELF_BinaryOpInterface, ["serialize"]>,
                DeclareOpInterfaceMethods<VPURegMapped_NPURegDescriptorOpInterface>
            ],
            "NNRTCfgDescriptorProp",
            "vpux::NPUReg50XX::Descriptors::VpuNNRTConfig"
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
            "vpux::NPUReg50XX::Descriptors::VpuNNRTConfig&&":$descriptor
        )>
    ];
}

//
// MappedInferenceVersionOp
//

def MappedInferenceVersionOp :
        NPUReg50XX_Op<"MappedInferenceVersion",
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
