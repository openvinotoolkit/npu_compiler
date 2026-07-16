//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_SHAVE_ATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "vpux/compiler/dialect/Shave/dialect.td"

//
// Base classes
//

class Shave_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::Shave";
    let genSpecializedAttr = 0;
}

class Shave_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<Shave_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class Shave_Attr<string name, list<Trait> traits = []> :
        AttrDef<Shave_Dialect, name, traits> {
    let mnemonic = name;
    let assemblyFormat = "`<` struct(params) `>`";
}

#endif
