//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/config/dialect.td"

//
// Base classes
//

class Config_Attr<string name, list<Trait> traits = []> : AttrDef<Config_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

class Config_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::config";
    let genSpecializedAttr = 0;
}

class Config_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Config_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

//
// CompilationMode
//

def Config_CompilationMode  :
        Config_I64EnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                I64EnumAttrCase<"ReferenceSW",  0>,
                I64EnumAttrCase<"DefaultHW",    1>,
                I64EnumAttrCase<"HostCompile",  2>,
                I64EnumAttrCase<"WSMonolithic", 3>,
                I64EnumAttrCase<"WSInit",       4>
            ]
        > {
}

def Config_CompilationModeAttr : Config_EnumAttr<Config_CompilationMode, "compilation_mode">;

//
// ArchKind
//

def Config_ArchKind :
        Config_I64EnumAttr<
            "ArchKind",
            "Represents VPU architecture generation",
            [
                I64EnumAttrCase<"UNKNOWN", 0>,
                I64EnumAttrCase<"NPU37XX", 3>,
                I64EnumAttrCase<"NPU40XX", 4>,
            ]
        > {
}

def Config_ArchKindAttr : Config_EnumAttr<Config_ArchKind, "arch_kind">;

//
// RevisionID
//

def Config_RevisionID  :
        Config_I64EnumAttr<
            "RevisionID",
            "Revision ID",
            [
                I64EnumAttrCase<"REVISION_A0",  0>,
                I64EnumAttrCase<"REVISION_A1",  1>,
                I64EnumAttrCase<"REVISION_A3",  2>,
                I64EnumAttrCase<"REVISION_B",   3>,
                I64EnumAttrCase<"REVISION_C",   4>,
                I64EnumAttrCase<"REVISION_D",   5>,
                I64EnumAttrCase<"REVISION_K",   6>,
                I64EnumAttrCase<"REVISION_NONE",   7>
            ]
        > {
}

def Config_RevisionIDAttr : Config_EnumAttr<Config_RevisionID, "revision_id">;

#endif
