//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CORE_OPS
#define VPUX_COMPILER_DIALECT_CORE_OPS

include "vpux/compiler/dialect/core/dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//
// Base classes
//

class Core_Op<string mnemonic, list<Trait> traits = []> :
    Op<Core_Dialect, mnemonic, traits>;

//
// NestedCallOp
//

def Core_NestedCallOp : Core_Op<"NestedCall", [
      CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "An operation that is similar to `func.call` but supports nested symbols";

    let description = [{
        This operation fulfills the same purpose as `func.call`, except for that it *only* supports nested symbols. The verifier checks if the symbol points to a `func.func` operation with a matching type signature.

        Example:

        ```MLIR
        module @subModule {
            func.func @foo() -> () { ... }
        }

        func.func @bar() {
            Core.NestedCall @subModule::@foo()
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$arg_operands, SymbolRefAttr:$callee);

    let results = (outs Variadic<AnyType>);

    // implementation of CallOpInterface
    let extraClassDeclaration = [{
        /// Return the callee of this operation.
        mlir::CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
            setCalleeAttr(mlir::cast<mlir::SymbolRefAttr>(callee));
        }
    }];

    let builders = [
        OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::TypeRange":$results, CArg<"mlir::ValueRange", "{}">:$operands)>
    ];

    let assemblyFormat = [{
        $callee `(` $arg_operands `)` attr-dict `:` functional-type($arg_operands, results)
    }];
}

def Core_ReinterpretCastOp : Core_Op<"ReinterpretCast", [Pure, ViewLikeOpInterface]> {
    let summary = "This operation overrides all type details of a given tensor / memref.";
    let description = [{
        This operation is a highly specialized type-obfuscating facility that is
        *prohibited* from being used in general contexts. By itself, it does not
        result in any operation being generated, only serves as glue code to
        preserve legal IR.

        The operation imposes minimal requirements to guarantee the basic
        correctness of the transformation: dimensionality of the data must be
        preserved, byte size of the data must be preserved, with the exception
        of the dynamic dimensions, which can be changed freely.
    }];

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$input
    );
    let results = (outs
        AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$output
    );
    let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands) `->` type(results)";
    let hasVerifier = 1;
    let hasFolder = 1;

    let extraClassDeclaration = [{
        mlir::Value getViewSource() { return getInput(); }
    }];
}

#endif
