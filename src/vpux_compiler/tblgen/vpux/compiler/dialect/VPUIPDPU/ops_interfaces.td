//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPDPU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIPDPU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def VPUIPDPU_MPECfgOpInterface : OpInterface<"MPECfgOpInterface"> {
    let description = "Interface for MPECfgOp arch specific verifier";

    let cppNamespace = "vpux::VPUIPDPU";

    let methods = [
        InterfaceMethod<
            "Verify inner ops",
            "mlir::LogicalResult", "verifyInnerOps"
        >
    ];
}

def VPUIPDPU_ODUCfgOpInterface : OpInterface<"ODUCfgOpInterface"> {
    let description = "Interface for ODUCfgOp arch specific verifier";

    let cppNamespace = "vpux::VPUIPDPU";

    let methods = [
        InterfaceMethod<
            "Verify inner ops",
            "mlir::LogicalResult", "verifyInnerOps"
        >
    ];
}

def VPUIPDPU_DPUVariantOpInterface : OpInterface<"DPUVariantOpInterface"> {
    let description = "Interface for DPUVariantOp arch specific verifier";

    let cppNamespace = "vpux::VPUIPDPU";

    let methods = [
        InterfaceMethod<
            "Verify inner ops",
            "mlir::LogicalResult", "verifyInnerOps"
        >
    ];
}

#endif
