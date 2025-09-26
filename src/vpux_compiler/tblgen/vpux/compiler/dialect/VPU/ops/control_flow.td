//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_CONTROL_FLOW
#define VPUX_COMPILER_DIALECT_VPU_OPS_CONTROL_FLOW

include "vpux/compiler/dialect/core/types.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"


def VPU_LoopSelectOp :
        VPU_LayerOp<
            "LoopSelect"
        > {
    let summary = "Select a slice of the input according to collected execution conditions, helping implement Loop Op.";

    let arguments = (ins
        1DTensorOf<[Bool8, SI8]>:$initExecCond,
        1DTensorOf<[Bool8, SI8]>:$execConds,
        AnyRankedTensor:$input,

        BoolAttr:$do_concat,
        IntAttr:$axis,
        IntAttr:$stride
    );

    let results = (outs
        AnyRankedTensor:$output
    );

    let hasVerifier = 1;
}


def VPU_YieldOp :
        VPU_Op<
            "Yield",
            [
                HasParent<"VerticalFusionOp">,
                DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for wrapping operation";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor]>>:$operands
    );

    let assemblyFormat = [{
        $operands
        custom<OptionalTypes>(type($operands)) ``
        attr-dict
    }];

    let hasVerifier = 1;
}


#endif
