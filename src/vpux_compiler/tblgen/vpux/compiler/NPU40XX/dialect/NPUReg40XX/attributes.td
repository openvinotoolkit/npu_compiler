//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg40XX_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_NPUReg40XX_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/attr_interfaces.td"
include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.td"

//
// Base classes
//

class NPUReg40XX_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::NPUReg40XX";
    let genSpecializedAttr = 0;
}

class NPUReg40XX_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<NPUReg40XX_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class NPUReg40XX_Attr<string name, list<Trait> traits = []> :
        AttrDef<NPUReg40XX_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// TaskType
//

def NPUReg40XX_TaskType :
        NPUReg40XX_I64EnumAttr<
            "TaskType",
            "The type of the relevant processing code of the VPU for the task",
            [
                I64EnumAttrCase<"DMA",                 0>,
                I64EnumAttrCase<"ActKernelInvocation", 1>,
                I64EnumAttrCase<"DPUInvariant",        2>,
                I64EnumAttrCase<"Generic",             3>,
            ]
        > {
}

def NPUReg40XX_TaskTypeAttr : NPUReg40XX_EnumAttr<NPUReg40XX_TaskType, "task_type">;

def DMADescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::DMARegister", "DMARegister"> {}
def DmaDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::DMARegister", "DMARegisterAttr", "NPUReg40XX"> {}

def DPUInvariantDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegister"> {}
def DPUInvariantDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegisterAttr", "NPUReg40XX"> {}

def DPUVariantDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::DpuVariantRegister", "DpuVariantRegister"> {}
def DPUVariantDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::DpuVariantRegister", "DpuVariantRegisterAttr", "NPUReg40XX"> {}

def ActKernelInvocationDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocation"> {}
def ActKernelInvocationDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocationAttr", "NPUReg40XX"> {}

def ActKernelRangeDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuActKernelRange", "VpuActKernelRange"> {}
def ActKernelRangeDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuActKernelRange", "VpuActKernelRangeAttr", "NPUReg40XX"> {}

def ConfigureBarrierDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfig"> {}
def ConfigureBarrierDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfigAttr", "NPUReg40XX"> {}

def TaskBarrierMapDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMap"> {}
def TaskBarrierMapDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMapAttr", "NPUReg40XX"> {}

def ManagedMappedInferenceDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInference"> {}
def MMIDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInferenceAttr", "NPUReg40XX"> {}

def WorkItemDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::WorkItem", "WorkItem"> {}
def WorkItemDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::WorkItem", "WorkItemAttr", "NPUReg40XX"> {}

def MappedInferenceDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuMappedInference", "VpuMappedInference"> {}
def MIDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuMappedInference", "VpuMappedInferenceAttr", "NPUReg40XX"> {}

def NNRTCfgDescriptorAttr : DescriptorAttrBase<NPUReg40XX_Dialect, "vpux::NPUReg40XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfig"> {}
def NNRTCfgDescriptorProp : PropertyBase<"vpux::NPUReg40XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfigAttr", "NPUReg40XX"> {}

#endif  // VPUX_COMPILER_DIALECT_NPUReg40XX_ATTRIBUTES
