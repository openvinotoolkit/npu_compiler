//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST
#define VPUX_COMPILER_DIALECT_CONST

include "mlir/IR/OpBase.td"

def Const_Dialect : Dialect {
    let summary = "Const Dialect";

    let description = [{
The **VPUX NN Compiler** uses lazy constant folding approach to reduce memory footprint
for large constant values (like dense tensors).

The **Const Dialect** provides utilities for that lazy constant folding support.

It defines a special attribute `Const::ContentAttr` to hold original constant data
and the transformations applied to it.
The `Const::ContentAttr` provides an API to apply the transformations on-the-fly,
when accessing the data.

The **Const Dialect** supports the following transformations:

* Broadcast
* Precision conversion
* Quantization types casting from raw storage
* Dequantization
* Rescale
* Add
* Reshape
* Reverse
* Reorder
* Padding
* SubView
* Transpose
* ExpandDilated
* GetSparsityMap
* Sparsify

The transformations are stored as separate attributes, which implemented specific Attribute Interface.
The interface allows to extend the set of transformations outside of the **Const Dialect** implementation.

Initial non-transformed constant should be created via `Const::DeclareOp` and `Const::ContentAttr`:

```C++
mlir::ElementsAttr baseValue = ...;
Const::ConstOp constOp = builder.create<Const::DeclareOp>(loc, resultType, Const::ContentAttr::get(baseValue));
```

In assembly code it will be represented as:

```MLIR
%0 = const.Declare tensor<1x2x3x4xf32> = dense<...> : tensor<1x2x3x4xf32>>
```

The content is accessed from `Const::DeclareOp` via `content()` method.
The method returns special `Const::Content` object, which allows to access underlying values as range.
The `Const::Content` object must be stored in separate variable, since it might contain temporal buffer,
which will be freed at object destruction.

```C++
Const::DeclareOp constOp = ...;
Const::Content content = constOp.content(); // This call will apply all transformations
const auto valsRange = content.getValues<float>(); // Access the values via range-like class
for (auto val : valsRange) { ... }
```

Note: the `getValues` allows to specify desired type for elements and will perform conversion from underlying storage type on-the-fly.

To perform tranformations on constant data, the new `Const::ContentAttr` must be created on top of existed with specified transformation:

```C++
Const::DeclareOp origConstOp = ...;
Const::ConstContentAttr origConstAttr = origConstOp.getContentAttr();
Const::ConstContentAttr newConstAttr = origConstAttr.castElemType(mlir::Float16Type::get(ctx));
Const::ConstOp newConstOp = builder.create<Const::DeclareOp>(loc, newConstAttr.getType(), std::move(newConstAttr));
```

In assembly code it will be represented as:

```MLIR
%0 = const.Declare tensor<1x2x3x4xf16> = dense<...> : tensor<1x2x3x4xf32>, [#const.CastElemType<f16>]
```

The `Const::ConstContentAttr` might hold a list of transformations, they will be applied in the order:

```MLIR
%0 = const.Declare memref<1x2x3x4xf16, #NHWC, #strides> =
        dense<...> : tensor<2x3x4xf32>,
        [
            #const.CastElemType<f16>,
            #const.Reshape<[1, 2, 3, 4]>,
            #const.Reorder<#NHWC>
        ]
```

Note: the tensor-related type attributes (shape, layout, element type) for the `Const::DeclareOp` result value
and for `Const::ConstContentAttr` final infered type must match.

The quantized constant is represented via `#const.CastElemType` transformation,
which casts raw integer storage representation to quantized type with
quantization parameters:

```MLIR
%0 = const.Declare tensor<1x16x1x1x!quant.uniform<u8:f16, ...>> =
        dense<...> : tensor<1x16x1x1xui8>,
        [
            #const.CastElemType<!quant.uniform<u8:f16, ...>>
        ]
```

Some transformations have a requirement to be applied before or after other transformations.
To ensure this requirement is met, each transformation has a `PositionRequirement` attached.
The available PositionRequirements are:

```
- NONE            // can be anywhere in the list
- PREFERRED_LAST  // will be last unless a transformation with LAST requirement is present
- LAST            // will be the last transformation in the list
```

The position of the transformations in the list will be:

```
[NONE]* -> [PREFERRED_LAST]* -> LAST
```

No two transformations with the LAST requirement can exist. The order of transformations with the same requirement is stable
(i.e. the insertion order is maintained).

To set the requirement for a transformation, override the default implementation for the `getPositionRequirement` method for
`Const_TransformAttrInterface`:

```
def Const_SomeAttr : Const_Attr<"SomeAttr",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
```

In the source file for your transformation, implement the `getPositionRequirement` method:

```C++
//
// SomeAttr::getPositionRequirement
//

Const::details::PositionRequirement Const::SomeAttr::getPositionRequirement() const {
    return Const::details::PositionRequirement::LAST;
}
```

[./const/_attr_interfaces.md]
    }];

    let name = "const";

    let cppNamespace = "vpux::Const";

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
