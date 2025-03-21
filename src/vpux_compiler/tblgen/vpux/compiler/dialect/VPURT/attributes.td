//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPURT_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURT/dialect.td"

//
// Base classes
//

class VPURT_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPURT";
    let genSpecializedAttr = 0;
}

class VPURT_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPURT_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

//
// BufferSection
//

def VPURT_BufferSection :
        VPURT_I64EnumAttr<
            "BufferSection",
            "Values indicating which section of BLOB the buffer resides in",
            [
                I64EnumAttrCase<"NetworkInput",     0>,
                I64EnumAttrCase<"NetworkOutput",    1>,
                I64EnumAttrCase<"ProfilingOutput",  2>,
                I64EnumAttrCase<"Constant",         3>,
                I64EnumAttrCase<"SW_KernelText",    4>,
                I64EnumAttrCase<"DDR",              5>,
                I64EnumAttrCase<"CSRAM",            6>,
                I64EnumAttrCase<"CMX_NN",           7>,
                I64EnumAttrCase<"Register",         8>,
                I64EnumAttrCase<"MAC_Accumulators", 9>,
                I64EnumAttrCase<"FunctionInput",    10>,
                I64EnumAttrCase<"FunctionOutput",   11>,
            ]
        > {
}

def VPURT_BufferSectionAttr : VPURT_EnumAttr<VPURT_BufferSection, "buffer_section">;

#endif
