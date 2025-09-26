//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_CONFIG_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// DefinedInArch op trait
//

class DefinedInArch<string arch>
    : ParamNativeOpTrait<"DefinedInArch", arch>  {
    let cppNamespace = "vpux::config";
}

//
// LimitedToArch op trait
//

class LimitedToArch<list<string> archs>
    : ParamNativeOpTrait<"LimitedToArch", !interleave(archs, ", ")>  {
    let cppNamespace = "vpux::config";
}

//
// ResourceOpBase
//

def Config_ResourceOpInterface : OpInterface<"ResourceOpInterface"> {
    let description = "Base interface for a meta-like operation containing information about a runtime resource";

    let cppNamespace = "vpux::config";

    let methods = [
        InterfaceMethod<
            "Get resource kind as StringAttr",
            "mlir::StringAttr", "getKind", (ins),
            [{}],
            [{
                return $_op.getSymNameAttr();
            }]
        >,
    ];
}

//
// ComputeResourceOp base
//

def Config_ComputeResourceOpInterface : OpInterface<"ComputeResourceOpInterface"> {
    let description = "Base interface for a meta-like operation containing information about a runtime compute resource";

    let cppNamespace = "vpux::config";

    let methods = [
        InterfaceMethod<
            "Get number of instantiated compute ops.",
            "mlir::IntegerAttr", "getCountAttr", (ins),
            [{}],
            [{
                return $_op.countAttr();
            }]
        >,
    ];
}


#endif
