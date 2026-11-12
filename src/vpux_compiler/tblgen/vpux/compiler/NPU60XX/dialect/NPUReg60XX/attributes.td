//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg60XX_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_NPUReg60XX_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/attr_interfaces.td"
include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/dialect.td"

//
// Base classes
//

class NPUReg60XX_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::NPUReg60XX";
    let genSpecializedAttr = 0;
}

class NPUReg60XX_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<NPUReg60XX_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class NPUReg60XX_Attr<string name, list<Trait> traits = []> :
        AttrDef<NPUReg60XX_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// TaskType
//

def NPUReg60XX_TaskType :
        NPUReg60XX_I64EnumAttr<
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

def NPUReg60XX_TaskTypeAttr : NPUReg60XX_EnumAttr<NPUReg60XX_TaskType, "task_type">;

def DMADescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::DMARegister", "DMARegister"> {}
def DmaDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::DMARegister", "DMARegisterAttr", "NPUReg60XX"> {}

def DPUInvariantDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegister"> {}
def DPUInvariantDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegisterAttr", "NPUReg60XX"> {}

def DPUVariantDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::DpuVariantRegister", "DpuVariantRegister"> {}
def DPUVariantDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::DpuVariantRegister", "DpuVariantRegisterAttr", "NPUReg60XX"> {}

def ActKernelInvocationDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocation"> {}
def ActKernelInvocationDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocationAttr", "NPUReg60XX"> {}

def ActKernelRangeDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuActKernelRange", "VpuActKernelRange"> {}
def ActKernelRangeDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuActKernelRange", "VpuActKernelRangeAttr", "NPUReg60XX"> {}

def ConfigureBarrierDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfig"> {}
def ConfigureBarrierDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfigAttr", "NPUReg60XX"> {}

def TaskBarrierMapDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMap"> {}
def TaskBarrierMapDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMapAttr", "NPUReg60XX"> {}

def ManagedMappedInferenceDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInference"> {}
def MMIDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInferenceAttr", "NPUReg60XX"> {}

def WorkItemDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::WorkItem", "WorkItem"> {}
def WorkItemDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::WorkItem", "WorkItemAttr", "NPUReg60XX"> {}

def NNRTCfgDescriptorAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfig"> {}
def NNRTCfgDescriptorProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfigAttr", "NPUReg60XX"> {}

def VpuDpuPVPAttr : DescriptorAttrBase<NPUReg60XX_Dialect, "vpux::NPUReg60XX::Descriptors::VpuDpuPVP", "VpuDpuPVP"> {}
def VpuDpuPVPProp : PropertyBase<"vpux::NPUReg60XX::Descriptors::VpuDpuPVP", "VpuDpuPVPAttr", "NPUReg60XX"> {}

#endif  // VPUX_COMPILER_DIALECT_NPUReg60XX_ATTRIBUTES
