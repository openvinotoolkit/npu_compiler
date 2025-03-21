//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/dialect.td"

//
// Base classes
//

class VPURegMapped_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPURegMapped";
    let genSpecializedAttr = 0;
}

class VPURegMapped_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPURegMapped_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class VPURegMapped_Attr<string name, list<Trait> traits = []> : AttrDef<VPURegMapped_Dialect, name, traits> {
    let mnemonic = name;
}

//
// RegFieldDataType
//

def VPURegMapped_RegFieldDataType :
    VPURegMapped_I64EnumAttr<
        "RegFieldDataType",
        "RegFieldDataType that the regField supports",
        [
            I64EnumAttrCase<"UINT", 0>,
            I64EnumAttrCase<"SINT", 1>,
            I64EnumAttrCase<"FP",   2>,
            I64EnumAttrCase<"BF",   3>,
        ]
    > {
}

def VPURegMapped_RegFieldDataTypeAttr : VPURegMapped_EnumAttr<VPURegMapped_RegFieldDataType, "reg_field_data_type">;

//
// RegisterFieldAttr
//

def VPURegMapped_RegisterFieldAttr : VPURegMapped_Attr<"RegisterField"> {
    let description = "This object represents closely a RegisterField Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegFieldType":$regField);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $regField `>`";
}

//
// RegisterFieldArrayAttr
//

def VPURegMapped_RegisterFieldArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterFieldAttr,
    "array of RegisterFields">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterAttr
//

def VPURegMapped_RegisterAttr : VPURegMapped_Attr<"Register"> {
    let description = "This object represents closely a Register Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegisterType":$reg);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $reg `>`";
}

//
// RegisterArrayAttr
//

def VPURegMapped_RegisterArrayAttr : TypedArrayAttrBase<VPURegMapped_RegisterAttr,
    "array of Registers">{
        string cppType = "::mlir::ArrayAttr";
    }

//
// RegisterMappedAttr
//

def VPURegMapped_RegisterMappedAttr : VPURegMapped_Attr<"RegisterMapped"> {
    let description = "This object represents closely a RegisterMapped Attr";
    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins "vpux::VPURegMapped::RegMappedType":$regMapped);

    let genVerifyDecl = 1;
    let assemblyFormat = "`<` $regMapped `>`";
}

def VPURegMapped_IndexTypeAttr : VPURegMapped_Attr<"IndexType"> {
    let cppNamespace = "vpux::VPURegMapped";
    let parameters = (ins "vpux::VPURegMapped::IndexType":$index);
    let assemblyFormat = "`<` $index `>`";
}

//
// TaskType
//

def VPURegMapped_TaskType :
    VPURegMapped_I64EnumAttr<
        "TaskType",
        "The type of the relevant processing code of the VPU for the task",
        [
            I64EnumAttrCase<"DMA",                 0>,
            I64EnumAttrCase<"ActKernelInvocation", 1>,
            I64EnumAttrCase<"ActKernelRange",      2>,
            I64EnumAttrCase<"DPUInvariant",        3>,
            I64EnumAttrCase<"DPUVariant",          4>,
            I64EnumAttrCase<"M2I",                 5>
        ]
    > {
}

def VPURegMapped_TaskTypeAttr : VPURegMapped_EnumAttr<VPURegMapped_TaskType, "task_type">;

//
// TaskGroupAttr
//

def VPURegMapped_TaskGroupAttr : VPURegMapped_Attr<"TaskGroup"> {
    let description = [{
        This object represents a task group (tasks of the same type) mapped to the memory layout.
        The task group is described using the following parameters:
            * realSize - the actual (total) count of tasks of a specific type for a specific tile & list
            * declaredSize - the expected count of tasks (=capacity of list of tasks in memory), either default-valued or computed by compiler
            * offset - numeric offset with respect to the start of CMX area that stores the tasks for that specific tile
            * binaryElementSize - size in bytes of a specific task

        Having all this info, the offset of any task (with any index) can be computed statically.

        Example IR representation:

        ```mlir
        #VPURegMapped.TaskGroup<dynamicTaskListSize(512 : ui64), staticTaskListSize(64 : ui64), startOffset(53760 : ui64), binaryTaskSize(96 : ui64)>
        ```
    }];

    let cppNamespace = "vpux::VPURegMapped";

    let parameters = (ins
        "mlir::IntegerAttr":$dynamicSize,
        "mlir::IntegerAttr":$staticSize,
        "mlir::IntegerAttr":$offset,
        "mlir::IntegerAttr":$binaryElementSize
    );

    let assemblyFormat = "`<` `dynamicTaskListSize` `(` $dynamicSize `)` `,` `staticTaskListSize` `(` $staticSize `)` `,` `startOffset` `(` $offset `)` `,` `binaryTaskSize` `(` $binaryElementSize `)` `>`";
}

#endif
