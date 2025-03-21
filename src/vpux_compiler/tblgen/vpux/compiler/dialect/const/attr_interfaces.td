//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_CONST_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// TransformAttrInterface
//

def Const_TransformAttrInterface : AttrInterface<"TransformAttrInterface"> {
    let description = [{
        The interface for Attributes, which holds information about lazy constant folding operation.
    }];

    let cppNamespace = "vpux::Const";

    let methods = [
        InterfaceMethod<
            "Infer output type",
            "vpux::NDTypeInterface", "inferOutputType",
            (ins "vpux::NDTypeInterface":$input)
        >,

        StaticInterfaceMethod<
            "Tell whether the data will be splat after this transformation",
            "bool", "inferOutputSplat",
            (ins "bool":$inputIsSplat, "vpux::NDTypeInterface":$input)
        >,

        InterfaceMethod<
            "Transform the constant content",
            "vpux::Const::Content", "transform",
            // HACK: it should be rvalue-reference, but can't use it due to auto-generated code limitation
            (ins "vpux::Const::Content&":$input)
        >,

        InterfaceMethod<
            "Get transformation name",
            "llvm::StringLiteral", "getTransformationName",
            (ins), [{}], [{
                return $_attr.getMnemonic();
            }]
        >,

        InterfaceMethod<
            "Get transformation location requirement",
            "vpux::Const::details::PositionRequirement", "getPositionRequirement",
            (ins), [{}], [{
                return vpux::Const::details::PositionRequirement::NONE;
            }]
        >,

        InterfaceMethod<
            "Supports sub byte storage type",
            "bool", "supportsSubByteStorageType",
            (ins), [{}], [{
                return false;
            }]
        >,

        InterfaceMethod<
            "In contrast to Attribute::hash_value, this method returns the same value for different contexts(e.g. in case of different runs)",
            "llvm::hash_code", "getStableHashValue",
            (ins)
        >
    ];
}

#endif
