//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg60XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_NPUReg60XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SingleOutputAsIndexOp
//

def NPUReg60XX_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::NPUReg60XX";
}

#endif  // VPUX_COMPILER_DIALECT_NPUReg60XX_OPS_INTERFACES
