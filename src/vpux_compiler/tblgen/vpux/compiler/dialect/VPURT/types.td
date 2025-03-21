//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_TYPES
#define VPUX_COMPILER_DIALECT_VPURT_TYPES

include "vpux/compiler/dialect/VPURT/dialect.td"
include "vpux/compiler/dialect/core/type_interfaces.td"

include "mlir/IR/AttrTypeBase.td"

//
// Base classes
//

class VPURT_Type<string name, list<Trait> traits = []> :
        TypeDef<
            VPURT_Dialect,
            name,
            traits
        > {
    let mnemonic = name;
}

//
// Barrier
//

def VPURT_Barrier : VPURT_Type<"Barrier"> {
    let summary = "VPUIP Barrier Type";

    let description = "This object represents closely a Barrier in the device";
}

#endif
