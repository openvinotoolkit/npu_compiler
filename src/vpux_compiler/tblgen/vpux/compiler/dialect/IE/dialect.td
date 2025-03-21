//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE
#define VPUX_COMPILER_DIALECT_IE

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Quant/QuantOpsBase.td"

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

The network inputs and outputs information (names, precision, layout) is held in separate Operation - `IE.CNNNetwork`.

```MLIR
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "input" : memref<1x3x400x400xf32>
    }
    outputsInfo : {
        IE.DataInfo "output" : memref<1x1000xf32>
    }
```

The **IE Dialect** provides separate operations to describe the available and used run-time resources.
It deals with the following resource types:

* Memory space.
* Executor (CPU, HW module, DMA).

```MLIR
IE.TileResource 4 of @NCE {
    IE.ExecutorResource 5 of @DPU
    IE.ExecutorResource 5 of @SHAVE_NN
    IE.ExecutorResource 1 of @SHAVE_ACT
    IE.MemoryResource 1048576 bytes of @CMX_NN {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
}
IE.ExecutorResource 1 of @DMA_NN

IE.MemoryResource 31457280 bytes of @DDR {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
```

The `IE.ExecutorResource` and `IE.MemoryResource` are added by underlying low-level dialect to provide information about HW-specific resources.

[./IE/_ops_interfaces.md]
    }];

    let name = "IE";

    let cppNamespace = "vpux::IE";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::Core::CoreDialect",
        "mlir::func::FuncDialect",
        "mlir::tensor::TensorDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerAttributes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;

    let usePropertiesForAttributes = 1;
}

#endif
