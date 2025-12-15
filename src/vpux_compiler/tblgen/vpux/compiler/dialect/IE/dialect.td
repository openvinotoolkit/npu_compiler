//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE
#define VPUX_COMPILER_DIALECT_IE

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Quant/IR/QuantBase.td"

def IE_Dialect : Dialect {
    let summary = "InferenceEngine IR Dialect";

    let description = [{
The **IE Dialect** represents InferenceEngine/nGraph IR in terms of MLIR framework.

It has the following properties:

* Describes network topology without HW details (memory hierarchy, memory allocation, scheduling).
* Represents the latest nGraph opset and in addition some portion of legacy IE opset (for convenience).
* Works with MLIR Tensor Types as atomic Values (no memory effects), all operations are pure.
* Performs high level transformations/optimizations, that doesn't need low level details (memory buffers, layouts, scheduling).

Some of the layer operations in the **IE Dialect** defines Canonicalization hooks to simplify IR for further optimizations:

* Remove redundant Operations (same type `Reshape`/`Tile`, `Add` with 0, etc.).
* Apply Lazy Constant Folding.
* Replace Constant Values with Attributes (more linear graph).
* Fuse common patterns (for example, `Mul+Add => ScaleShift`, `Convolution+Bias`).
* Use more convinient Operations (for example, `MatMul => FullyConnected`).

Quantization parameters are stored as a part of tensor/buffer element type (`QuantizedType` from **Quant Dialect**).

The network topology (nGraph) is represented as a MLIR Function, which works with `tensor` types.

```MLIR
func.func @main(%input: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %output = IE.SoftMax(%input) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    return %output
}
```

The network inputs and outputs information (names, precision, layout) is held in separate Operation - `net.NetworkInfo`.

```MLIR
net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : memref<1x3x400x400xf32>
    }
    outputsInfo : {
        DataInfo "output" : memref<1x1000xf32>
    }
```
[./IE/_ops_interfaces.md]
    }];

    let name = "IE";

    let cppNamespace = "vpux::IE";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::Core::CoreDialect",
        "vpux::config::ConfigDialect",
        "vpux::net::NetDialect",
        "mlir::func::FuncDialect",
        "mlir::tensor::TensorDialect",
        "mlir::quant::QuantDialect"
    ];

    let extraClassDeclaration = [{
        void registerAttributes();
    }];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;

    let usePropertiesForAttributes = 1;
}

#endif
