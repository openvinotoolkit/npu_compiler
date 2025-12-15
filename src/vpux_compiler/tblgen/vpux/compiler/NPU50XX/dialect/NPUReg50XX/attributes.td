//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUREG50XX_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_NPUREG50XX_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/attr_interfaces.td"
include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.td"

//
// Base classes
//

class NPUReg50XX_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::NPUReg50XX";
    let genSpecializedAttr = 0;
}

class NPUReg50XX_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<NPUReg50XX_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class NPUReg50XX_Attr<string name, list<Trait> traits = []> :
        AttrDef<NPUReg50XX_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

//
// TaskType
//

def NPUReg50XX_TaskType :
        NPUReg50XX_I64EnumAttr<
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

def NPUReg50XX_TaskTypeAttr : NPUReg50XX_EnumAttr<NPUReg50XX_TaskType, "task_type">;

def DMADescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::DMARegister", "DMARegister"> {}
def DmaDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::DMARegister", "DMARegisterAttr", "NPUReg50XX"> {}


def DPUInvariantDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegister"> {}
def DPUInvariantDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::DpuInvariantRegister", "DpuInvariantRegisterAttr", "NPUReg50XX"> {}


def DPUVariantDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::DpuVariantRegister", "DpuVariantRegister"> {}
def DPUVariantDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::DpuVariantRegister", "DpuVariantRegisterAttr", "NPUReg50XX"> {}


def ActKernelInvocationDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocation"> {}
def ActKernelInvocationDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation", "VpuActKernelInvocationAttr", "NPUReg50XX"> {}

def ActKernelRangeDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuActKernelRange", "VpuActKernelRange"> {}
def ActKernelRangeDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuActKernelRange", "VpuActKernelRangeAttr", "NPUReg50XX"> {}


def M2IDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuMediaTask", "VpuMediaTask"> {}
def M2IDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuMediaTask", "VpuMediaTaskAttr", "NPUReg50XX"> {}


def ConfigureBarrierDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfig"> {}
def ConfigureBarrierDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig", "VpuBarrierCountConfigAttr", "NPUReg50XX"> {}

def TaskBarrierMapDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMap"> {}
def TaskBarrierMapDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuTaskBarrierMap", "VpuTaskBarrierMapAttr", "NPUReg50XX"> {}


def ManagedMappedInferenceDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInference"> {}
def MMIDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuManagedMappedInference", "VpuManagedMappedInferenceAttr", "NPUReg50XX"> {}


def WorkItemDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::WorkItem", "WorkItem"> {}
def WorkItenDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::WorkItem", "WorkItemAttr", "NPUReg50XX"> {}

def MappedInferenceDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuMappedInference", "VpuMappedInference"> {}
def MIDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuMappedInference", "VpuMappedInferenceAttr", "NPUReg50XX"> {}

def NNRTCfgDescriptorAttr : DescriptorAttrBase<NPUReg50XX_Dialect, "vpux::NPUReg50XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfig"> {}
def NNRTCfgDescriptorProp : PropertyBase<"vpux::NPUReg50XX::Descriptors::VpuNNRTConfig", "VpuNNRTConfigAttr", "NPUReg50XX"> {}

#endif  // VPUX_COMPILER_DIALECT_NPUREG50XX_ATTRIBUTES
