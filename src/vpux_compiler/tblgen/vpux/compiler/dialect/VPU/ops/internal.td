//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_INTERNAL
#define VPUX_COMPILER_DIALECT_VPU_OPS_INTERNAL

include "vpux/compiler/dialect/core/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/ops/base.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/ops_interfaces.td"
include "vpux/compiler/dialect/VPU/types.td"

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"



def VPU_GenericSwLayerOp :
        VPU_Op<
            "GenericSwLayer",
            [
                Pure,
                DeclareOpInterfaceMethods<CallOpInterface>,
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>
            ]
        > {
    let summary = "VPU Op that describes a generic SW Layer";

    let arguments = (ins
        SymbolRefAttr:$callee, // Generated sw layer function

        Variadic<AnyType>:$inputs
    );

    let results = (outs
        Variadic<AnyShaped>:$outputs
    );

    let assemblyFormat = [{
        `(` operands `)` attr-dict `:` type(operands) `->` type(results)
    }];
}


def VPU_ExternalKernelOp :
        VPU_Op<
            "ExternalKernel",
            [
                DeclareOpInterfaceMethods<VPU_LayerOpInterface>
            ]
        > {
    let summary = "Represents a kernel whose details/implementation are defined externally";

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




def VPU_VerticalFusionOp :
        VPU_Op<
            "VerticalFusion",
            [
                Pure,
                IsolatedFromAbove,
                DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorOperands",
                                                                    "areTypesCompatible"]>,
                SingleBlockImplicitTerminator<"YieldOp">
            ]
        > {
    let summary = "Operation that encapsulates details of VF subgraph";

    let arguments = (ins
        Variadic<AnyTypeOf<[AnyRankedTensor, VPU_DistributedTensor, VPU_SparseTensor]>>:$operands,
        I64ArrayAttr:$tilingStrategy,
        OptionalAttr<VPU_VFScenarioAttr>:$scenario
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

        OptionalAttr<VPU_SEAttr>:$seAttr
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
