//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_INTERNAL
#define VPUX_COMPILER_DIALECT_VPU_OPS_INTERNAL

include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"



def VPU_GenericSwLayerOp :
        VPU_Op<
            "GenericSwLayer",
            [
                Pure,
                DeclareOpInterfaceMethods<ArgAndResultAttrsOpInterface, ["getArgAttrsAttr", "getResAttrsAttr", "setArgAttrsAttr", "setResAttrsAttr",
                                                                         "removeArgAttrsAttr", "removeResAttrsAttr"]>,
                DeclareOpInterfaceMethods<CallOpInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>,
                DeclareOpInterfaceMethods<VPU_SWOpInterface>,
                DeclareOpInterfaceMethods<VPU_TilingBuilderOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface>,
                DeclareOpInterfaceMethods<VPU_AuxiliaryBufferOpInterface>,
                AttrSizedOperandSegments
            ]
        > {
    let summary = "VPU Op that describes a generic SW Layer";

    let description = [{
        Represents an invocation of a generated software layer. The operation
        contains optional tiling metadata which provides the iteration-space
        slice computed by this op instance. Kernel-level tiling metadata
        (tilingInfoFunc, tilingAxes, numSlicedInputs) is stored in
        `VPU::KernelInfoAttr` on the callee function.

        `inputs` - generic inputs to the kernel.

        `scratch_inputs` - auxiliary buffers.

        `tilingProperties` - optional attribute encoding the iteration-space
        slice: static_sizes and static_offsets (one entry per tiling axis).
        Present only for tileable kernels. The number of entries equals the
        number of tiling axes recorded in `VPU::KernelInfoAttr` on the
        callee.

        `dynamic_sizes` / `dynamic_offsets` - dynamic sizes/offsets in the
        iteration space. Should only be present if the tiling attribute is also
        present and are interpreted in conjunction with the static sizes/offsets
        from the tiling attribute.

        `returnTypes` - inferred attribute storing the result types of the
        operation. Not set by callers; populated automatically by the builder
        from the operation output types. Used internally to implement
        `InferTypeOpInterface`. Omitted from the textual assembly form.

        Syntax:

        ```
        VPU.GenericSwLayer(<inputs> : <types>)
            [scratch(<scratch_inputs> : <types>)]
            <callee>
            [tiling(sizes = [<mixed_list>], offsets = [<mixed_list>])]
            [attr-dict]
            -> <result_types>
        ```

        Example:

        ```mlir
        %0 = VPU.GenericSwLayer(%arg0 : tensor<1x16x4000x200xf32>)
            @generated_0
            tiling(sizes = [%c5, 4000, 256], offsets = [0, %c3, 0])
            -> tensor<1x16x4000x256xf32>

        %scratch = VPU.Empty : tensor<1x256xf32>
        %1 = VPU.GenericSwLayer(%arg0 : tensor<1x256xf32>)
            scratch(%scratch : tensor<1x256xf32>)
            @generated_1
            -> tensor<1x256xf32>
        ```
    }];

    let arguments = (ins
        SymbolRefAttr:$callee, // Generated sw layer function

        Variadic<AnyType>:$inputs,
        Variadic<AnyType>:$scratch_inputs,
        OptionalAttr<VPU_JITTilingAttr>:$tilingProperties,
        Variadic<Index>:$dynamic_sizes,
        Variadic<Index>:$dynamic_offsets,
        ArrayAttr:$returnTypes
    );

    let results = (outs
        Variadic<AnyShaped>:$outputs
    );

    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
        // Builder automatically creates returnTypes from resultTypes
        // Usage: builder.create<VPU::GenericSwLayerOp>(loc, resultTypes, callee, inputs,
        //            [tilingProperties], [dynamic_sizes], [dynamic_offsets], [scratch])
        OpBuilder<(ins "mlir::TypeRange":$resultTypes,
                       "mlir::SymbolRefAttr":$callee,
                       "mlir::ValueRange":$inputs,
                       CArg<"mlir::Attribute", "{}">:$tilingProperties,
                       CArg<"mlir::ValueRange", "{}">:$dynamic_sizes,
                       CArg<"mlir::ValueRange", "{}">:$dynamic_offsets,
                       CArg<"mlir::ValueRange", "{}">:$scratch_inputs)>
    ];

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
            return vpux::areTypesCompatible(lhs, rhs, vpux::IE::TypeComparisonMode::STRICT_EQUAL,
                                            /*checkDimsOrder=*/true, /*checkMemSpace=*/false);
        }

        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem);
        bool fitIntoCMX(::llvm::ArrayRef<vpux::NDTypeInterface> buffers);
    }];
}

def VPU_UnrolledTypeOp :
        VPU_Op<
            "UnrolledType",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "This layer mediate between unrolled distributed tensor type and usual type";

    let arguments = (ins
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];

    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def VPU_EmptyOp :
        VPU_Op<
            "Empty",
            [
                Pure
            ]
        > {
    let summary = "Empty tensor operation";

    let description = [{
        `VPU.empty` is an operation that defines a tensor of a particular shape.
        The contents of the tensor are unspecified and the only purpose of the
        op result is to materialize the specified shape in IR.

        The operation can be lowered to a buffer allocation operation, such as
        `memref.alloc` or `VPURT.AllocDistributed`.
    }];


    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$result
    );

    let assemblyFormat = "attr-dict `:` type($result)";

    let hasCanonicalizer = 1;
}

def VPU_VerticalFusionOp :
        VPU_Op<
            "VerticalFusion",
            [
                Pure,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands",
                                                                    "getSuccessorInputs",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of VF subgraph";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands,
        I64ArrayAttr:$tilingStrategy,
        OptionalAttr<VPU_VFScenarioAttr>:$scenario,
        UnitAttr:$isManualConfigured
    );

    let results = (outs
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$results
    );

    let regions = (region SizedRegion<1>:$ops);

    let hasVerifier = 1;

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder,
            "mlir::ArrayAttr":$tilingInfo)>,
        OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::ValueRange":$operands,
            "llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>":$bodyBuilder,
            "mlir::ArrayAttr":$tilingInfo, "mlir::UnitAttr":$isManualConfigured)>
    ];

    let extraClassDeclaration = [{
        using BodyBuilderFn =
            llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;

        mlir::Operation* getFirstInnerTaskOp();

        void print(::mlir::OpAsmPrinter& p);
        static ::mlir::ParseResult parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result);
    }];
}


def VPU_DistributedCastOp :
        VPU_Op<
            "DistributedCast",
            [
                VPU_ViewLikeOpInterface
            ]
        > {
    let summary = "Operation that casts one DistributedTensor type to another.";

    let description = [{
        Used to cast one DistributedTensor type to another and help with NNCMX retention
        of data.

        Currently following distribution mode pairs are compatible:

        DUPLICATED|SEGMENTED -> DUPLICATED ## needed for K cluster tiling
    }];

    let arguments = (ins
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$input
    );

    let results = (outs
        AnyTypeOf<[VPU_DistributedTensor, VPU_SparseTensor]>:$output
    );

    let assemblyFormat = [{
        `(` $input `:` qualified(type($input)) `)`
        attr-dict
        `->` qualified(type($output))
    }];

    let hasFolder = 1;

    let hasVerifier = 1;
}


def VPU_GroupSparseTensorOp :
        VPU_Op<
            "GroupSparseTensor",
            [
                Pure,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
                AttrSizedOperandSegments,
                VPU_GroupedViewLikeOpInterface,
                DeclareOpInterfaceMethods<VPU_TilingViewLikeOpInterface>
            ]
        > {
    let summary = "Groups sparse data and metadata into a single value";

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_DistributedTensor]>:$data,
        Optional<AnyTypeOf<[I1Tensor, VPU_DistributedTensor]>>:$sparsityMap,
        Optional<AnyTypeOf<[I32Tensor, VPU_DistributedTensor]>>:$storageElementTable,

        UnitAttr:$is_weights,
        OptionalAttr<VPU_SparsityCompressionAttr>:$sparsity_compression,

        OptionalAttr<VPU_SEAttrInterface>:$seAttr
    );

    let results = (outs
        AnyTypeOf<[VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );

    let builders = [
        OpBuilder<
            (ins "mlir::Value":$data,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"bool", "{}">:$is_weights, CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsity_compression)
        >,
        OpBuilder<
            (ins "mlir::Value":$data, "mlir::Value":$sparsityMap, "mlir::Value":$storageElementTable,
                CArg<"VPU::SEAttr", "{}">:$seAttr)
        >
    ];

    let assemblyFormat = [{
        `(` $data
            (`,` $sparsityMap^ `` custom<OptionalTypes>(type($sparsityMap)))?
            (`,` $storageElementTable^ `` custom<OptionalTypes>(type($storageElementTable)))?
        `)`
        attr-dict
        `` custom<OptionalTypes>(type($data))
        `->` type(results)
    }];

    let hasCanonicalizer = 1;
}



def VPU_UngroupSparseTensorOp :
        VPU_Op<
            "UngroupSparseTensor",
            [
                Pure,
                AttrSizedResultSegments,
                DeclareOpInterfaceMethods<MultiViewOpInterface>,
                DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
            ]
        > {
    let summary = "Ungroups sparse data and metadata into multiple values";

    let arguments = (ins
        VPU_SparseTensor:$input
    );

    let results = (outs
        AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>:$data,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$sparsityMap,
        Optional<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor]>>:$storageElementTable
    );

    let assemblyFormat = [{
        `(` $input `)`
        attr-dict
        `` custom<OptionalTypes>(type($input))
        `->` type(results)
    }];
}

#endif
