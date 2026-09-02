//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NET_OPS
#define VPUX_COMPILER_DIALECT_NET_OPS

include "vpux/compiler/dialect/net/dialect.td"
include "vpux/compiler/dialect/core/attributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class Net_Op<string mnemonic, list<Trait> traits = []> :
    Op<Net_Dialect, mnemonic, traits>;

def Net_NetworkInfoOp :
        Net_Op<
            "NetworkInfo",
            [
                DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                HasParent<"mlir::ModuleOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Neural Network description";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about the neural network:

          * Precision and layout for user-provided inputs.
          * Precision and layout for user-provided outputs.
          * Layout for output profiling data(optional).
          * Entry point (Function name) for the network inference.
          * Model inference time by DPU cycle unit.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$entryPoint,
        OptionalAttr<IntAttr>:$inferenceTiming
    );

    let regions = (region
        SizedRegion<1>:$inputs_info,
        SizedRegion<1>:$outputs_info,
        VariadicRegion<SizedRegion<1>>:$profiling_outputs_info
    );

    let extraClassDeclaration = [{
        size_t getNetInputsCount();
        vpux::SmallVector<vpux::net::DataInfoOp, 1> getInputsDataInfo();

        size_t getNetOutputsCount();
        vpux::SmallVector<vpux::net::DataInfoOp, 1> getOutputsDataInfo();

        size_t getProfilingOutputsCount();
        vpux::SmallVector<vpux::net::DataInfoOp, 1> getProfilingOutputsDataInfo();

        static void getFromModule(
                mlir::ModuleOp module,
                vpux::net::NetworkInfoOp& netInfo,
                mlir::func::FuncOp& netFunc);

        static mlir::StringRef getDefaultDialect() {
            return "net";
        }
    }];

    let builders = [
        OpBuilder<
            (ins "mlir::FlatSymbolRefAttr":$entryPoint, "bool":$withProfiling)
        >
    ];

    let assemblyFormat = [{
        attr-dict
        `entryPoint` `:` $entryPoint
        `inputsInfo` `:` $inputs_info
        `outputsInfo` `:` $outputs_info
        (`profilingOutputsInfo` `:` $profiling_outputs_info^)?
    }];

    let hasVerifier = 1;
}

def Net_DataInfoOp :
        Net_Op<
            "DataInfo",
            [
                HasParent<"vpux::net::NetworkInfoOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Information about the input/output Data object of the neural network";

    let description = [{
        This operation is bound to the `net.NetworkInfo` Operation and holds information about the Data object:

          * Name - primary name of Data object
          * Actual shape - OV node output tensor shape after batching
          * Original shape - OV node output tensor shape before batching
          * Friendly name - OV node friendly name
          * Input name - legacy name of Parameter or Result tensor
          * Tensor names - OV node tensor names
          * User-defined shape
          * User-defined precision (element type)
          * User-defined layout
    }];

    let arguments = (ins
        StrAttr:$name,
        TypeAttr:$userType,
        OptionalAttr<TypeAttr>:$originalShape,
        OptionalAttr<StrAttr>:$friendlyName,
        OptionalAttr<StrAttr>:$inputName,
        OptionalAttr<ArrayAttr>:$tensorNames
    );

    let regions = (region
        VariadicRegion<SizedRegion<1>>:$sections
    );

    let extraClassDeclaration = [{
        vpux::DimsOrder getDimsOrder();
    }];

    let assemblyFormat = [{
        $name (`friendlyName` `=` $friendlyName^)? (`inputName` `=` $inputName^)? (`tensorNames` `=` $tensorNames^)? ($sections^)? `:` $userType (`originalShape` `=` $originalShape^)?
        attr-dict
    }];

    let hasVerifier = 1;

    let builders = [
        // Additional simplified builder used internally within the compiler.
        OpBuilder<(ins "mlir::StringRef":$name, "mlir::Type":$userType)>
    ];
}

def Net_SparsityStatisticsOp :
        Net_Op<
            "SparsityStatistics",
            [
                HasParent<"mlir::ModuleOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Sparsity statistics for inputs of operations";

    let description = [{
        This operation is bound to MLIR Module and holds extra information about input sparsity ratios collected from dataset:

          * NGraph layer name.
          * Input id.
          * Sparsity ratio.
    }];

    let regions = (region
        SizedRegion<1>:$sparsityInfo
    );

    let assemblyFormat = [{
        attr-dict
        `sparsityInfo` `:` $sparsityInfo
    }];

    let hasVerifier = 0;
}

def Net_SparsityInfoOp :
        Net_Op<
            "SparsityInfo",
            [
                HasParent<"vpux::net::SparsityStatisticsOp">,
                IsolatedFromAbove
            ]
        > {
    let summary = "Information about estimated sparsity ratio for input of specific layer";

    let description = [{
        This operation is bound to the `net.SparsityStatistics` Operation and holds information about the SparsityInfo object:

          * Node name
          * Input ID
          * Sparsity ratio
    }];

    let arguments = (ins
        StrAttr:$name,
        IntAttr:$inputId,
        F64Attr:$ratio
    );

    let assemblyFormat = [{
        $ratio `at` `input` $inputId `of` $name
        attr-dict
    }];

    let hasVerifier = 0;
}

def Net_PrecisionRequirementOp :
        Net_Op<
            "PrecisionRequirement",
            [
                HasParent<"mlir::ModuleOp">,
                IsolatedFromAbove,
                NoRegionArguments,
                OpAsmOpInterface
            ]
            # GraphRegionNoTerminator.traits
        > {
    let summary = "Precision sensitiveness requirement per model layer";

    let description = [{
        This operation is bound to MLIR Module and holds information if a model layer is precision sensitive.
    }];

    let regions = (region
        SizedRegion<1>:$precisionInfo
    );

    let assemblyFormat = [{
        attr-dict
        `precisionInfo` `:` $precisionInfo
    }];

    let hasVerifier = 0;
}

def Net_PrecisionInfoOp :
        Net_Op<
            "PrecisionInfo",
            [
                HasParent<"vpux::net::PrecisionRequirementOp">,
                IsolatedFromAbove
            ]
        > {
    let summary = "Information whether a specific layer is required to keep its precision setting.";

    let description = [{
        This operation is bound to `net.PrecisionRequirement` operation and holds information about the PrecisionInfo object.
        Any operation in the model that has conversion to f16 disabled via the DisablePrecisionConversion RT attribute will be added here.

          * operation name
          * operation type
    }];

    let arguments = (ins
        StrAttr:$opName,
        StrAttr:$opType
    );

    let assemblyFormat = [{
        $opName $opType
        attr-dict
    }];

    let hasVerifier = 0;
}

#endif
