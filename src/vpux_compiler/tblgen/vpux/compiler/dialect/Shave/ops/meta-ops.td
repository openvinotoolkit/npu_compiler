//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_SHAVE_OPS_META
#define VPUX_COMPILER_DIALECT_SHAVE_OPS_META

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "vpux/compiler/dialect/Shave/ops/base.td"
include "vpux/compiler/dialect/Shave/attributes.td"
include "vpux/compiler/dialect/Shave/ops_interfaces.td"

//
// CustomOps
//

//
// ExternalKernelOp
//

def Shave_ExternalKernelOp :
        Shave_Op<
            "ExternalKernel",
            [
                Shave_KernelWrapperOpInterface
            ]
        > {
    let summary = "Represents a kernel whose details/implementation are defined externally";

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        DictionaryAttr:$attrDict,
        StrAttr:$unique_id,
        StrAttr:$kernel_path
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        $unique_id
        `at_path` `(` $kernel_path `)`
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        `attrs` `(` $attrDict `)`
        attr-dict
        `->` type(results)
    }];
}

def Shave_OutSliceInfoOp :
        Shave_Op<
            "OutSliceInfo",
            [
                Pure,
                AttrSizedResultSegments,
            ]
        > {
    let hasVerifier = 1;

    let summary = "Pseudo-op, provides symbolic size/offsets values for a slice";

    let description = [{
        Provided values represent the sizes and offsets of a slice in an N
        dimensional iteration space (where N is equal to the number of sizes
        or offsets returned by this op, and the number of sizes equals the
        number of offsets).
    }];

    let arguments = (ins
       Index:$loopId,
       Index:$tileNum
    );

    let results = (outs
        Variadic<Index>:$sizes,
        Variadic<Index>:$offsets
    );

    let hasCustomAssemblyFormat = 1;
}

def Shave_LoopTripCountOp :
        Shave_Op<
            "LoopTripCount",
            [
                Pure,
            ]
        > {
    let summary = "Pseudo-op, provides a symbolic value for the number of tiles";

    let arguments = (ins
       Index:$loopId
    );

    let results = (outs
        Index:$output
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

def Shave_TilingRootOp :
        Shave_Op<
            "TilingRoot",
            [
                Pure,
                DeclareOpInterfaceMethods<TilingInterface,
                    ["getLoopIteratorTypes",
                     "getIterationDomain",
                     "getTiledImplementation",
                     "getResultTilePosition",
                    ]
                >
            ]
        > {
    let hasVerifier = 1;

    let summary = "Pseudo-op used as the tiling root for parametric tiling";

    let description = [{
        This operation is a pseudo-op used to start the parametric tiling process.
        The semantics of the TilingRootOp operation is to propagate all the values
        received in inputs to its outputs.
        Tiling is performed starting from the first result along the dimensions
        indicated by the tilingRanks attribute.
    }];

    let arguments = (ins
        Index:$loopId,
        Variadic<AnyRankedTensor>:$inputs,
        I64ArrayAttr:$tilingRanks
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}

#endif
