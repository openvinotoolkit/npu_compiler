//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_VPUMI37XX_ATTRIBUTES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPUMI37XX/dialect.td"

//
// Base classes
//

class VPUMI37XX_I64EnumAttr <string name, string summary, list<I64EnumAttrCase> cases> :
        I64EnumAttr<name, summary, cases> {
    let cppNamespace = "vpux::VPUMI37XX";
    let genSpecializedAttr = 0;
}

class VPUMI37XX_EnumAttr <EnumAttrInfo enumInfo, string name = "", list <Trait> traits = []> :
        EnumAttr<VPUMI37XX_Dialect, enumInfo, name, traits> {
    let assemblyFormat = "`<`$value`>`";
}

class VPUMI37XX_Attr<string name, list<Trait> traits = []> :
        AttrDef<VPUMI37XX_Dialect, name, traits> {
    let mnemonic = name;
}

//
// PPEType
//

def VPUMI37XX_PPELayerType :
    VPUMI37XX_I64EnumAttr<
        "PPELayerType",
        "Post Processing Element Type",
        [
            // Low-level instructions
            I64EnumAttrCase<"STORE", 0>,
            I64EnumAttrCase<"LOAD",  1>,
            I64EnumAttrCase<"CLEAR", 2>,
            I64EnumAttrCase<"NOOP",  3>,
            I64EnumAttrCase<"HALT",  4>,

            // Element-Wise Operations
            I64EnumAttrCase<"ADD",     5>,
            I64EnumAttrCase<"SUB",     6>,
            I64EnumAttrCase<"MULT",    7>,
            I64EnumAttrCase<"MAXIMUM", 8>,
            I64EnumAttrCase<"MINIMUM", 9>,
            I64EnumAttrCase<"AND",     10>,
            I64EnumAttrCase<"OR",      11>,
            I64EnumAttrCase<"XOR",     12>,

            // Activations
            I64EnumAttrCase<"LRELU",   13>,
            I64EnumAttrCase<"LRELUX",  14>,
            I64EnumAttrCase<"LPRELU",  15>,
            I64EnumAttrCase<"CEIL",    16>,
            I64EnumAttrCase<"FLOOR",   17>,
            I64EnumAttrCase<"POW",     18>,
            I64EnumAttrCase<"EXP",     19>,
            I64EnumAttrCase<"SIGMOID", 20>,
            I64EnumAttrCase<"TANH",    21>,
            I64EnumAttrCase<"SQRT",    22>,
            I64EnumAttrCase<"RSQRT",   23>,
            I64EnumAttrCase<"FLEXARB", 24>,
            I64EnumAttrCase<"NOT",     25>,
            I64EnumAttrCase<"ABS",     26>,
            I64EnumAttrCase<"NEG",     27>
        ]
    > {
}

def VPUMI37XX_PPELayerTypeAttr : VPUMI37XX_EnumAttr<VPUMI37XX_PPELayerType, "ppe_layer_type">;

#endif
