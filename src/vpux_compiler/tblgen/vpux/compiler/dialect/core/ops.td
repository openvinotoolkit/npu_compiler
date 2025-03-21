//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CORE_OPS
#define VPUX_COMPILER_DIALECT_CORE_OPS

include "vpux/compiler/dialect/core/dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//
// Base classes
//

class Core_Op<string mnemonic, list<Trait> traits = []> :
    Op<Core_Dialect, mnemonic, traits>;

#endif
