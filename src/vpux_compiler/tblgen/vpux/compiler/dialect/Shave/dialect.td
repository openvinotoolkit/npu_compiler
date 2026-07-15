//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE
#define VPUX_COMPILER_DIALECT_SHAVE

include "mlir/IR/OpBase.td"

def Shave_Dialect : Dialect {
    let summary = "Shave Dialect";

    let description = [{
        The `Shave` dialect provides abstractions and constructs for representing operations/concepts connected to the Shave.
    }];

    let name = "Shave";

    let cppNamespace = "vpux::Shave";

    let extraClassDeclaration = [{
        void registerAttributes();
    }];

}

#endif
