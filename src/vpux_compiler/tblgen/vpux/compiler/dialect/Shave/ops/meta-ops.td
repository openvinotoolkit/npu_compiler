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

    let description = [{
        Represents an invocation of a custom SHAVE kernel whose implementation is not
        known to the compiler and is instead supplied externally by the user, alongside
        a Custom Layer Config XML describing it (parsed into a CustomOpConfigOp identified
        by the same unique_id).
    }];

    let arguments = (ins
        Variadic<AnyRankedTensor>:$inputs,
        DictionaryAttr:$attrDict,
        StrAttr:$unique_id
    );

    let results = (outs
        Variadic<AnyRankedTensor>:$outputs
    );

    let assemblyFormat = [{
        $unique_id
        (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
        `attrs` `(` $attrDict `)`
        attr-dict
        `->` type(results)
    }];
}

//
// CustomOp Config ops
//

def Shave_CustomOpConfigCollectionOp :
        Shave_MetaOp<
            "CustomOp.ConfigCollection",
            [
                HasParent<"mlir::ModuleOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                SymbolTable,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Collection of CustomOp configs";

    let description = [{
        Top-level container for all Custom Layer configs parsed from a Custom Layer
        Config XML file. Holds a symbol table region with one CustomOpConfigOp per
        custom layer defined in the collection, keyed by its unique layer name.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$configs
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `configs` `:` $configs
    }];
}

def Shave_CustomOpConfigOp :
        Shave_MetaOp<
            "CustomOp.Config",
            [
                HasParent<"Shave::CustomOpConfigCollectionOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                Symbol
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Configuration for a single CustomOp custom layer";

    let description = [{
        Represents the configuration for one custom layer, corresponding to a
        "CustomLayer" node in the Custom Layer Config XML. sym_name is the unique
        layer name, matched against the unique_id of the ExternalKernelOp that this
        config applies to.

        The nested region holds the specific config sub-ops for this layer: exactly
        one CustomOpKernelConfigOp, and optionally CustomOpIOConfigOp,
        CustomOpHeapConfigOp, and CustomOpTilingConfigOp.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let regions = (region
        SizedRegion<1>:$configs
    );

    let assemblyFormat = [{
        $sym_name
        attr-dict
        `configs` `:` $configs
    }];
}

def Shave_CustomOpIOConfigOp :
        Shave_MetaOp<
            "CustomOp.IOConfig",
            [
                HasParent<"Shave::CustomOpConfigOp">,
                IsolatedFromAbove,
                NoRegionArguments
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "IO (tensor and scratch buffer) configuration for a CustomOp";

    let description = [{
        Groups the per-argument IO configuration for a custom layer, corresponding to
        the "Buffers" node in the Custom Layer Config XML. The nested region holds one
        CustomOpIOTensorConfigOp per kernel tensor argument and one
        CustomOpIOScratchConfigOp per scratch buffer required by the kernel.
    }];

    let regions = (region
        SizedRegion<1>:$configs
    );

    let assemblyFormat = [{
        attr-dict
        `configs` `:` $configs
    }];
}

def Shave_CustomOpIOTensorConfigOp :
        Shave_MetaOp<
            "CustomOp.IOConfig.Tensor",
            [
                HasParent<"Shave::CustomOpIOConfigOp">
            ]
        > {
    let summary = "IO configuration for a single CustomOp tensor argument";

    let description = [{
        Declares the compiler-facing constraints for a single kernel tensor argument
        (input or output), identified by arg_index, corresponding to a "Tensor" node
        in the Custom Layer Config XML "Buffers" section.

        supported_rank/supported_dtype list the ranks and data types the kernel
        accepts for this argument. enforced_rank/enforced_dtype/enforced_layout are
        optional and, when set, command the compiler to convert the argument to that
        exact rank/data type/layout before the kernel executes.
    }];

    let arguments = (ins
        UI32Attr:$arg_index,
        Shave_IOTypeAttr:$ioType,
        Shave_TensorRankArrayAttr:$supported_rank,
        Shave_TensorDataTypeArrayAttr:$supported_dtype,
        OptionalAttr<Shave_TensorRankAttr>:$enforced_rank,
        OptionalAttr<Shave_TensorDataTypeAttr>:$enforced_dtype,
        OptionalAttr<Shave_TensorLayoutAttr>:$enforced_layout
    );

    let extraClassDeclaration = [{
        // Get the supported ranks as enum values
        llvm::SmallVector<Shave::TensorRank> getSupportedRankValues();

        // Get the supported data types as enum values
        llvm::SmallVector<Shave::TensorDataType> getSupportedDtypeValues();
    }];

    let assemblyFormat = [{
        attr-dict
    }];
}

def Shave_CustomOpIOScratchConfigOp :
        Shave_MetaOp<
            "CustomOp.IOConfig.Scratch",
            [
                HasParent<"Shave::CustomOpIOConfigOp">
            ]
        > {
    let summary = "Scratch buffer configuration for a CustomOp";

    let description = [{
        Declares a scratch buffer required by the kernel, identified by arg_index,
        corresponding to a "Scratch" node in the Custom Layer Config XML "Buffers"
        section. size is the buffer size in bytes, and location is the memory space
        the buffer must be allocated in.
    }];

    let arguments = (ins
        UI32Attr:$arg_index,
        UI32Attr:$size,
        Shave_ScratchLocationAttr:$location
    );

    let assemblyFormat = [{
        attr-dict
    }];
}

def Shave_CustomOpKernelConfigOp :
        Shave_MetaOp<
            "CustomOp.KernelConfig",
            [
                HasParent<"Shave::CustomOpConfigOp">
            ]
        > {
    let summary = "Kernel implementation configuration for a CustomOp";

    let description = [{
        Holds the kernel implementation details for a custom layer, corresponding to
        the "Kernel" node (and its "Source"/"CompilerOptions" children) in the Custom
        Layer Config XML.

        source is the path to the kernel source/binary, kernel_type selects how it
        should be built/loaded (e.g. prebuilt binary), and the
        optional compiler_config is forwarded to the kernel compiler as-is. uses_dma
        and uses_dpu indicate whether the kernel makes use of the DMA/DPU engines.
    }];

    let arguments = (ins
        StrAttr:$source,
        Shave_KernelTypeAttr:$kernel_type,
        OptionalAttr<StrAttr>:$compiler_config,
        UnitAttr:$uses_dma,
        UnitAttr:$uses_dpu
    );

    let assemblyFormat = [{
        attr-dict
    }];
}

def Shave_CustomOpHeapConfigOp :
        Shave_MetaOp<
            "CustomOp.HeapConfig",
            [
                HasParent<"Shave::CustomOpConfigOp">
            ]
        > {
    let summary = "Scratch heap configuration for a CustomOp";

    let description = [{
        Declares the size (in bytes) of the scratch heap the kernel requires at
        runtime, corresponding to the "Heap" node in the Custom Layer Config XML.
    }];

    let arguments = (ins
        UI32Attr:$size
    );

    let assemblyFormat = [{
        attr-dict
    }];
}

def Shave_CustomOpTilingConfigOp :
        Shave_MetaOp<
            "CustomOp.TilingConfig",
            [
                HasParent<"Shave::CustomOpConfigOp">
            ]
        > {
    let summary = "Tiling strategy configuration for a CustomOp";

    let description = [{
        Declares the tiling strategy to apply to the custom layer, corresponding to
        the "Tiling" node in the Custom Layer Config XML. mode selects whether the
        kernel runs untiled (OFF) or is repeated across multiple SHAVE tiles
        (REPEATED).
    }];

    let arguments = (ins
        Shave_TilingModeAttr:$mode
    );

    let assemblyFormat = [{
        attr-dict
    }];
}

def Shave_OutSliceInfoOp :
        Shave_Op<
            "OutSliceInfo",
            [
                Pure,
                AttrSizedResultSegments
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
                Pure
            ]
        > {
    let summary = "Pseudo-op, provides a symbolic value for the number of tiles";

    let description = [{
        Provides a symbolic value for the total number of tiles that the loop
        identified by loopId will iterate over. Used together with OutSliceInfoOp
        during parametric tiling, before the actual tile count is known.
    }];

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

//
// KernelRegionOp
//

def Shave_KernelRegionOp :
        Shave_Op<
            "KernelRegion",
            [
                IsolatedFromAbove,
                SingleBlock
            ]
        > {
    let summary = "Isolated region encapsulating a Shave kernel computation";

    let description = [{
        Encapsulates a body region of operations computing Shave kernel results.
        The region is isolated from above; all required values must be passed
        explicitly through $inputs. Results are produced by a terminating
        Shave.KernelRegion.Yield op inside the body.

        Example:
        ```mlir
        %out = Shave.KernelRegion(%val : f32) -> f32 {
        ^bb0(%arg0: f32):
            Shave.KernelRegion.Yield %arg0 : f32
        }
        ```
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let regions = (region SizedRegion<1>:$body);

    let hasVerifier = 1;
    let assemblyFormat = [{
        (`(` $inputs^ `:` type($inputs) `)`)? (`->` type($outputs)^)? attr-dict-with-keyword $body
    }];
}

//
// KernelRegionYieldOp
//

def Shave_KernelRegionYieldOp :
        Shave_Op<
            "KernelRegion.Yield",
            [
                HasParent<"KernelRegionOp">,
                Pure,
                Terminator
            ]
        > {
    let summary = "Terminator for Shave.KernelRegion that yields results";

    let description = [{
        Terminates and yields values from a Shave.KernelRegion body region.
        The number and types of yielded values must match the result types
        of the enclosing KernelRegionOp.
    }];

    let arguments = (ins
        Variadic<AnyType>:$values
    );

    let assemblyFormat = [{
        attr-dict ($values^ `:` type($values))?
    }];
}

#endif
