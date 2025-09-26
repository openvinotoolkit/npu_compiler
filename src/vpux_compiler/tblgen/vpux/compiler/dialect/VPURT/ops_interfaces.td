//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPURT_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// SerializeInterface
//

def VPURT_SerializeInterface : OpInterface<"SerializeInterface"> {
    let description = "Interface for blob serialization";

    let cppNamespace = "vpux::VPURT";
}

//
// BarrierOpInterface
//

def VPURT_BarrierOpInterface : OpInterface<"BarrierOpInterface"> {
    let description = "Interface for accessing barrier type ops";

    let cppNamespace = "vpux::VPURT";

    let methods = [
        InterfaceMethod<
            "Get barrier produced by this op",
            "mlir::Value", "getBarrier", (ins),
            [{}],
            [{
                return $_op.getBarrier();
            }]
        >,
        InterfaceMethod<
            "Check if barrier is a final barrier",
            "bool", "getIsFinalBarrier", (ins),
            [{}],
            [{
                return $_op.getIsFinalBarrier();
            }]
        >,
        InterfaceMethod<
            "Get WLM Page attribute of this barrier",
            "std::optional<int>", "getWlmPage", (ins),
            [{}],
            [{
                return $_op.getWlmPage();
            }]
        >,
        InterfaceMethod<
            "Get the barrier index",
            "std::optional<int64_t>", "getBarrierIndex", (ins)
        >,
        InterfaceMethod<
            "Set the barrier index",
            "void", "setBarrierIndex", (ins "std::optional<int64_t>":$value)
        >
    ];
}

#endif
