//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CORE
#define VPUX_COMPILER_DIALECT_CORE

include "mlir/IR/OpBase.td"

def Core_Dialect : Dialect {
    let summary = "Core Dialect";

    let description = [{
        The `Core` dialect contains operations, attributes and types that are common across different dialects.
    }];

    let name = "Core";

    let cppNamespace = "vpux::Core";
}

#endif
