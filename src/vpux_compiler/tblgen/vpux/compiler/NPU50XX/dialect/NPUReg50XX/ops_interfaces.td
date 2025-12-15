//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUREG50XX_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_NPUREG50XX_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SingleOutputAsIndexOp
//

def NPUReg50XX_SingleOutputAsIndexOp : NativeOpTrait<"SingleOutputAsIndexOp"> {
    let cppNamespace = "vpux::NPUReg50XX";
}

#endif  // VPUX_COMPILER_DIALECT_NPUREG50XX_OPS_INTERFACES
