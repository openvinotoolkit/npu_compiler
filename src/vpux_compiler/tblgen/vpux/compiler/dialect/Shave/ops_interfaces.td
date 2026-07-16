//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_SHAVE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def Shave_KernelWrapperOpInterface : OpInterface<"KernelWrapperOpInterface"> {
    let description = "Interface for kernel wrapper ops";

    let cppNamespace = "vpux::Shave";
}

#endif
