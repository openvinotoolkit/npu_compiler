//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONFIG_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/config/dialect.td"

//
// Base classes
//

class Config_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::config";
    let genSpecializedAttr = 0;
}

class Config_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Config_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class Config_Attr <string name, list <Trait> traits = []> :
        AttrDef<Config_Dialect, name, traits>;

//
// CompilationMode
//

def Config_CompilationMode  :
        Config_I64EnumAttr<
            "CompilationMode",
            "Compilation Mode",
            [
                I64EnumAttrCase<"ReferenceSW",           0>,
                I64EnumAttrCase<"DefaultHW",             1>,
                I64EnumAttrCase<"HostCompile",           2>,
                I64EnumAttrCase<"WSInit",                3>,
                I64EnumAttrCase<"WSMain",                4>,
                I64EnumAttrCase<"HostCompile_JIT",        5>,
                I64EnumAttrCase<"HostCompile_Interpreter", 6>,
            ]
        > {
}

def Config_CompilationModeAttr : Config_EnumAttr<Config_CompilationMode, "compilation_mode">;

//
// Platform
//

def Config_Platform :
        Config_I64EnumAttr<
            "Platform",
            "NPU target platform",
            [
                I64EnumAttrCase<"NPU3720", 3720>,
                I64EnumAttrCase<"NPU4000", 4000>,
                I64EnumAttrCase<"NPU5010", 5010>,
                I64EnumAttrCase<"NPU5020", 5020>,
            ]
        > {
}

def Config_PlatformAttr : Config_EnumAttr<Config_Platform, "platform">;

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
                I64EnumAttrCase<"NPU50XX", 5>,
            ]
        > {
}

def Config_ArchKindAttr : Config_EnumAttr<Config_ArchKind, "arch_kind">;

//
// Version
//

def Config_VersionAttr : Config_Attr<"Version"> {
    let summary = "Represents a semantic version";
    let mnemonic = "version";
    let parameters = (ins
        "uint32_t":$major,
        "uint32_t":$minor,
        "uint32_t":$patch
    );
    let assemblyFormat = "`<`$major`:`$minor`:`$patch`>`";
    let extraClassDeclaration = [{
        Version getVersion() const {
            return {getMajor(), getMinor(), getPatch()};
        }
    }];
    let builders = [
        AttrBuilder<(ins "const Version &":$value), [{
            return $_get($_ctxt, value.major, value.minor, value.patch);
        }]>
    ];
}

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

//
// ExecutorKind
//

def Config_ExecutorKind :
        Config_I64EnumAttr<
            "ExecutorKind",
            "Representd hardware executror resources",
            [
                I64EnumAttrCase<"DMA_NN",    0>,
                I64EnumAttrCase<"NCE",       1>,
                I64EnumAttrCase<"DPU",       2>,
                I64EnumAttrCase<"SHAVE_NN",  3>,
                I64EnumAttrCase<"SHAVE_ACT", 4>,
                I64EnumAttrCase<"M2I",       5>,
                I64EnumAttrCase<"UNKNOWN",   6>,
            ]
        > {
}

def Config_ExecutorKindAttr : Config_EnumAttr<Config_ExecutorKind, "executor_kind">;

#endif
