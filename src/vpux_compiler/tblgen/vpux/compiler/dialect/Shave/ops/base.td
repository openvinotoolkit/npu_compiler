//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_OPS_BASE
#define VPUX_COMPILER_DIALECT_SHAVE_OPS_BASE

include "mlir/IR/OpBase.td"
include "vpux/compiler/dialect/Shave/dialect.td"

//
// Shave Op Base
//

class Shave_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            Shave_Dialect,
            mnemonic,
            traits
        >;

//
// Base for Shave Meta-ops
//

class Shave_MetaOp<string mnemonic, list<Trait> traits = []> :
        Shave_Op<
            mnemonic,
            [
            ] # traits
        >;

#endif
