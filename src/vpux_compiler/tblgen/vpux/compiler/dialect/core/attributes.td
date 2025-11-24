//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_CORE_ATTRIBUTES
#define VPUX_COMPILER_CORE_ATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "vpux/compiler/dialect/core/attr_interfaces.td"

def IntAttr : Attr<CPred<"$_self.isa<mlir::IntegerAttr>()">, "Integer attribute"> {
    let storageType = "mlir::IntegerAttr";
    let valueType = I64;
    let returnType = "int64_t";
    let convertFromStorage = "$_self.getValue().getSExtValue()";
    let constBuilderCall = "$_builder.getI64IntegerAttr($0)";
}

def I64ArrayOfArraysAttr : TypedArrayAttrBase<I64ArrayAttr, "array of 64-bit integer arrays">;

def SymbolRefArrayOfArraysAttr : TypedArrayAttrBase<SymbolRefArrayAttr, "array of symbol ref arrays">;

def UI32ArrayAttr : TypedArrayAttrBase<UI32Attr,
                                      "Unsigned 32-bit integer array attribute">;

def UI8ArrayAttr : TypedArrayAttrBase<UI8Attr,
                                      "Unsigned 8-bit integer array attribute">;
def IndexedSymbolAttr :
        Attr<
            CPred<"$_self.isa<vpux::IndexedSymbolAttr>()">,
            "Symbol name with optional index"
        > {
    let storageType = "vpux::IndexedSymbolAttr";
    let returnType = "vpux::IndexedSymbolAttr";
    let convertFromStorage = "$_self";
    let constBuilderCall = "$0";
}

def TensorAttr :
        Attr<
            CPred<"$_self.isa<vpux::TensorAttr>()">,
            "Symbol name with optional index"
        > {
    let storageType = "vpux::TensorAttr";
    let returnType = "vpux::TensorAttr";
    let convertFromStorage = "$_self";
    let constBuilderCall = "$0";
}

class InlinerDispatchAttr<Dialect dialect, string name> : AttrDef<dialect, name, [DeclareAttrInterfaceMethods<InlinerDispatchAttrInterface>]> {
    let mnemonic = name;
    let description = "Dialects can implement this class when they want to support custom inlining semantics for 'func' operations";
    let summary = [{
        Dialects can implement this class when they want to support custom inlining semantics for 'func' operations.

        ```
        def MyInlinerDispatchAttr : InlinerDispatchAttr<My_Dialect, "MyInlinerDispatch">
        ```

        In IR:
        ```MLIR
        func.call {inliner_dispatch = #MyDialect.MyInlinerDispatchAttr} @someFunction() -> ()
        ```
    }];
}

def SmallVectorU8Prop: Property<"SmallVector<uint8_t>">{
    let hashProperty = "hash_value($_storage)";
    let storageType = "SmallVector<uint8_t>";
    let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;
    let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
}

#endif
