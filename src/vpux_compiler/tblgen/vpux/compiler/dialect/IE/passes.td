//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// Precisions and Layouts
//=================================================================================

//
// Outliner
//

def Outliner : PassBase<"outliner", "vpux::ModulePass"> {
    let summary = "Extracts function based on result from IR analysis";

    let description = [{
        This pass is used for parallel compilation. It outlines functions, it does this in
        a similar way to how inlining a function works but other way around. It breaks a large
        function into multiple functions which the original function calls.

        Below is an example of what the outlier does on the function main:
        ```
        func.func @main(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
            %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
            %0 = IE.Convolution(%arg0, %cst) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
            %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
            %2 = IE.Add(%0, %1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
            return %0, %2: tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
        }
        ```
        It splits the function by the number of splits passed as a pass option `num-parts`.
        In the example below the number of parts are two:
        ```
        func.func private @main_part1(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
            %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
            %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
            %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
            return %0, %1 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
        }
        func.func private @main_part2(%arg0: tensor<1x48x60x60xf32>, %arg1: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
            %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
            return %0 : tensor<1x48x60x60xf32>
        }
        func.func @main(%arg0: tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) {
            %0:2 = call @main_part1(%arg0) : (tensor<1x3x62x62xf32>) -> (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>)
            %1 = call @main_part2(%0#0, %0#1) : (tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
            return %0#0, %1 : tensor<1x48x60x60xf32>, tensor<1x48x60x60xf32>
        }
        ```

        # Configuration

        Available modes and their parameters are:
          naive
            num-parts          - the number of parts to split the IR into
          repeating-blocks
            max-num-iterations - the maximum number of iterations
            min-ops-in-block   - the minimum number of operations allowed in a blocks

        Example:
            vpux-opt --outliner="function-outlining='repeating-blocks='ax-num-iterations=30 min-ops-in-block=16, naive=num-parts=2'"
    }];

    let constructor = "vpux::IE::createOutlinerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "functionOutlining", "function-outlining",
            "std::string", "\"naive\"",
            "Define a list of outlining modes and their parameters where the next outlining mode is the fallback mode of the previous one. See pass description for more information."
        >
    ];
}

//
// DuplicateFQAcrossFunctionCalls
//

def DuplicateFQAcrossFunctionCalls : PassBase<"duplicate-fq-across-function-calls", "vpux::ModulePass"> {
    let summary = "Duplicates FakeQuantize operations across function calls";

    let description = [{
        Identifies cases where FakeQuantzie operations should be duplicated inside or outside functions.

        For example, for the following IR:
            func @function(%arg) {
                %fq1 = FakeQuantize(%arg)
                %0 = Op(%fq1)
                return %0
            }
            func @main(%arg) {
                %0 = Op(%arg)
                %call = call @function(%0)
                %fq2 = FakeQuantize(%call)
                %1 = Op(%fq2)
                return %1
            }
        The %fq1 operation will be duplicated outside the function and will be the parent operation for the call op.
        Similarly, %fq2 will be duplicated inside the function and will be the parent for the return op.
    }];

    let constructor = "vpux::IE::createDuplicateFQAcrossFunctionCallsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// Debatcher
//

def Debatcher : PassBase<"debatcher", "vpux::FunctionPass"> {
    let summary = "Downcast input batched tensor arguments of a `main`-function to single batch tensors, which is supposed to unblock further transformations dedicated to support batched models compilation. . This is a frontend of 'Debatcher-Outliner-DeDebatcher' approach";

    let description = [{
        This pass is used together the Outliner pass, simplifying outlining routine
        by eliminating necessity in debatching every input & output of outlined operations.
        Instead, it just debatches the input tensor rather than intruding in outliner logic

    }];

    let constructor = "vpux::IE::createDebatcherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "extraArgs", "extra-args",
            "std::string", [{""}],
            "Debatcher extra arguments"
        >
    ];

}

//
// DeDebatcher
//

def DeDebatcher : PassBase<"de-debatcher", "vpux::FunctionPass"> {
    let summary = "Rollback DeDebatcher tensors downcasting operations from N->1 by adding a body function N-respective repetitions";

    let description = [{
        This pass is used together with the Debatcher-Outliner pass. It finalizes batching tensors compilation routine
        by adding repetitions an outlined function, which representing an IR body.
        The total number of repetitions is the initial N dimension value, which was previously downcasted into N
    }];

    let constructor = "vpux::IE::createDeDebatcherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "debatcherMethod", "debatching-inlining-method",
            "std::string", [{"naive"}],
            "Propagate relevant debatcher inlining method"
        >
    ];

}

//
// OverrideTileExecutorNum
//

def OverrideTileExecutorNum : PassBase<"override-tile-executor-num", "vpux::ModulePass"> {
    let summary = "Overrides the tile executor count to optimize memory resource allocation.";

    let description = [{
        This pass is used together with or after the DeDebatcher pass, to simplify the process of
        managing memory resources by avoiding complex recalculations of shapes, offsets, and other attributes.

        Currently supports two modes:
        - `apply`: Adjusts the tile count to match the number of tiles per batch.
        - `revert`: Reverts the tile count to the value before overriding.

        Note:
        - This pass does not allow multiple overrides without a revert to ensure consistency.
        - This pass relies on DebatchedCallOpAttributeView information to track and manage the tile count.
    }];

    let constructor = "vpux::IE::createOverrideTileExecutorNumPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "overrideToTilesPerBatchMode", "override-to-tiles-per-batch-mode",
            "std::string", [{"apply"}],
            "Selects the overriding mode: `apply` or `revert`"
        >
    ];
}

//
// UseUserPrecision
//

def UseUserPrecision : PassBase<"use-user-precision", "vpux::ModulePass"> {
    let summary = "Use user precisions for entry point function prototype";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass updates the CNNNetwork entry point function prototype and use user-provided precisions for its operands and results.
        The pass inserts Convert operations from/to topology precisions.
    }];

    let constructor = "vpux::IE::createUseUserPrecisionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustLayouts
//

def AdjustLayouts : PassBase<"adjust-layouts", "vpux::FunctionPass"> {
    let summary = "Adjust required layouts for all layers";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass adds the required layouts instead of the default one
        depending on the layer specification from underlying Dialect.
    }];

    let constructor = "vpux::IE::createAdjustLayoutsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// FuseReshapeMvnPass
//

def FuseReshapeMvn : PassBase<"fuse-reshape-mvn", "vpux::FunctionPass"> {
    let summary = "Fuse Reshape->MVN->Reshape (back to initial shape) into MVN with internal-reshape";

    let description = [{
        The pass is a part of `IECommon` pipeline.
        Can only succeed in NHWC layout, for large tensors that require MVN-decomposition.
    }];

    let constructor = "vpux::IE::createFuseReshapeMvnPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseRMSNorm
//
def FuseRMSNorm : PassBase<"fuse-rmsnorm", "vpux::FunctionPass"> {
    let summary = "Fuse Power-ReduceMean-Add-Sqrt-Divide-Multiply-Multiply to RMSNorm";
    let description = [{
        fuse a sequence of ops to one RMSNorm.
    }];
    let constructor = "vpux::IE::createFuseRMSNormPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseRoPE
//
def FuseRoPE : PassBase<"fuse-rope", "vpux::FunctionPass"> {
    let summary = "Fuse Multiply-StridedSlice-Multiply-StridedSlice-Concat-Multiply-Add to RoPE";
    let description = [{
        fuse a sequence of ops to one RoPE.
    }];
    let constructor = "vpux::IE::createFuseRoPEPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeReorders
//

def OptimizeReorders : PassBase<"optimize-reorders", "vpux::FunctionPass"> {
    let summary = "Optimize extra Reorder operations";

    let description = [{
        The pass is a part of `IECommon` pipeline.

        This pass tries to optimize out Reorder operations for common cases
        by propagating them from inputs to outputs and merging into layers.
    }];

    let constructor = "vpux::IE::createOptimizeReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// OptimizeTileOp
//

def OptimizeTileOp : PassBase<"optimize-tile-op", "vpux::FunctionPass"> {
    let summary = "Optimize tile ops";

    let description = [{
        The pass removes the useless tile op if the user is sw which supports broadcast
    }];

    let constructor = "vpux::IE::createOptimizeTileOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizePrecisionAcrossFunctionCalls
//

def OptimizePrecisionAcrossFunctionCalls : PassBase<"optimize-precision-across-function-calls", "vpux::ModulePass"> {
    let summary = "Optimizes conversion / quantization operations across function calls";

    let description = [{
        Tries to optimize out Convert->Convert / Dequantize->Quantize operations that are found at the boundaries of functions,
        if these pairs of operations end up producing the same element type.
    }];

    let constructor = "vpux::IE::createOptimizePrecisionAcrossFunctionCallsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeReordersAcrossFunctionCalls
//

def OptimizeReordersAcrossFunctionCalls : PassBase<"optimize-reorders-across-function-calls", "vpux::ModulePass"> {
    let summary = "Optimizes Reorder operations across function calls";

    let description = [{
        Tries to optimize out Reorder operations that are found at the boundaries of functions, when possible.

        The Reorders found at the boundaries of a function (i.e. the users of the block arguments or the producers
        of the returned values) are indirectly connected with other operations outside the function, via the call
        operations of the function. If these Reorders are found to be optimizable due to these connections, they
        will be removed and the function signature updated.
    }];

    let constructor = "vpux::IE::createOptimizeReordersAcrossFunctionCallsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// ConvertSplitConcatToTranspose
//

def ConvertSplitConcatToTranspose : PassBase<"convert-split-concat-to-transpose", "vpux::FunctionPass"> {
    let summary = "Convert the pattern {Split -> AffineReshape -> Concat} to Transpose";

    let description = [{
        This pass replaces the pattern {Split -> AffineReshape -> Concat} with Transpose operation.
    }];

    let constructor = "vpux::IE::createConvertSplitConcatToTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//=================================================================================
// AdjustForVPU
//=================================================================================

//
// ConvertAssignReadValueToReturnsAndInputs
//

def ConvertAssignReadValueToReturnsAndInputs : PassBase<"convert-assign-read-value", "vpux::FunctionPass"> {
    let summary = "Convert assign to returns and read value to inputs";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Assign` operations with main function returns and
        `ReadValue` operations with main function inputs.
    }];

    let constructor = "vpux::IE::createConvertAssignReadValueToReturnsAndInputs()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToSpatialOp
//

def ConvertToSpatialOp : PassBase<"convert-to-spatial-op", "vpux::FunctionPass"> {
    let summary = "Insert Transpose around operations in case that they have non-spatial axes";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass inserts `Transpose` operations around dedicated operations to get spatial axes.
        `Interpolate` and `Roll` are supported so far.
    }];

    let constructor = "vpux::IE::createConvertToSpatialOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "m2iEnabled", "m2i-enabled",
            "bool", "false",
            "Flag which identifies whether M2I is enabled. If no, the conversion can be applied to Interpolate"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies whether SE Roll is enabled. If yes, the conversion can be applied to Roll"
        >
    ];
}

//
// ConvertNearestToStridedConcat
//

def ConvertNearestToStridedConcat : PassBase<"convert-nearest-to-broadcast-or-strided-concat", "vpux::FunctionPass"> {
    let summary = "Convert nearest interpolate op to broadcast or strided concat ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Nearest Interpolate` operations with `Broadcast` or `Concat` operations with strides.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to `Broadcast` or `Concat`.
    }];

    let constructor = "vpux::IE::createConvertNearestToBroadCastOrStridedConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "interpolateAsSEOp", "interpolate-as-se-op",
            "bool", "false",
            "Flag which identifies whether an Interpolate operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// SplitBilinerIntoHAndW
//

def SplitBilinerIntoHAndW : PassBase<"split-bilinear-into-H-and-W", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate on H and W to slice, concat, convolution and interpolate on H";

    let description = [{
        This pass convert `Bilinear Interpolate` operations to interpolate on H and interpolate on W, and
        the interpolate on W will convert to slice, concat and convolution.

        This pass is enabled when both `interpolateAsSEOp` and `SplitBilinerIntoHAndW` options are set to true.
    }];

    let constructor = "vpux::IE::createSplitBilinerIntoHAndWPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

}

//
// ConvertBilinearToStridedConcatAndConv
//

def ConvertBilinearToStridedConcatAndConv : PassBase<"convert-bilinear-to-strided-concat-and-conv", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, MaxPool and some depthwise convolution Ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Bilinear Interpolate` operations with `Concat` operations with strides,
        MaxPool and some `depthwise` convolutions.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats & NCE ops.
    }];

    let constructor = "vpux::IE::createConvertBilinearToStridedConcatAndConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "interpolateAsSEOp", "interpolate-as-se-op",
            "bool", "false",
            "Flag which identifies whether an Interpolate operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// ConvertScatterNDUpdateToStridedConcat
//

def ConvertScatterNDUpdateToStridedConcat : PassBase<"convert-scatterndupdate-to-strided-concat", "vpux::FunctionPass"> {
    let summary = "Convert ScatterNDUpdate op to strided concat ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `ScatterNDUpdate` operations with `Concat` operations with strides.
    }];

    let constructor = "vpux::IE::createConvertScatterNDUpdateToStridedConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPrecisionToFP16
//

def ConvertPrecisionToFP16 : PassBase<"convert-precision-to-fp16", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        This pass replaces all FP32 tensors with FP16.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToFP16Pass()";

    let options = [
        Option<
            "computeLayersWithHigherPrecision", "compute-layers-with-higher-precision",
            "std::string", [{""}],
            "Keep the specified FP32 layer(s) unchanged during the conversion to FP16"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPrecisionToI32
//

def ConvertPrecisionToI32 : PassBase<"convert-precision-to-i32", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from I64 to I32";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.
        This pass replaces all I64 tensors with I32.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToI32Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustSoftwareOpsPrecision
//

def AdjustSoftwareOpsPrecision : PassBase<"adjust-software-ops-precision", "vpux::ModulePass"> {
    let summary = "Adjust precision of software ops to satisfy kernel implementation";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        Some kernel implementations only support specific precisions. To satisfy this requirement,
        such ops are surrounded by conversion layers.
    }];

    let constructor = "vpux::IE::createAdjustSoftwareOpsPrecisionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustNCEOpsWithI32Inputs
//

def AdjustNCEOpsWithI32Inputs : PassBase<"adjust-nce-ops-with-i32-inputs", "vpux::ModulePass"> {
    let summary = "Adjust precision for some NCE ops with i32 inputs";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        Currently NCE ops only support f16 or quantized inputs. In some cases, such ops with i32 inputs need conversion layers.
    }];

    let constructor = "vpux::IE::createAdjustNCEOpsWithI32InputsPass()";

    let options = [
        Option<
            "enableConvertFCToConv", "convert-fc-to-conv",
            "bool", "true",
            "Specifies whether IE.FullyConnected will be converted to Conv and thus will need their inputs adjusted"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDepth2SpaceToTransposedConv
//

def ConvertDepth2SpaceToTransposedConv : PassBase<"convert-d2s-to-transposed-conv", "vpux::FunctionPass"> {
    let summary = "Convert D2S layers to transposed convolution";

    let description = [{
        Convert D2S layers to a transposed convolution so we can execute them on DPU
        rather than using a DMA.
    }];

    let constructor = "vpux::IE::createConvertDepth2SpaceToTransposedConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDepth2SpaceLayer
//

def ConvertDepth2SpaceLayer : PassBase<"convert-depthToSpace", "vpux::FunctionPass"> {
    let summary = "Convert DepthToSpace layer to {reshape -> transpose -> reshape} subgraph";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `DepthToSpace` operations with {reshape -> transpose -> reshape} subgraph.
    }];

    let constructor = "vpux::IE::createConvertDepth2SpaceLayerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSpace2DepthLayer
//

def ConvertSpace2DepthLayer : PassBase<"convert-spaceToDepth", "vpux::FunctionPass"> {
    let summary = "Convert SpaceToDepth layer to {reshape -> transpose -> reshape} pattern";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `SpaceToDepth` operations with {reshape -> transpose -> reshape} pattern.
    }];

    let constructor = "vpux::IE::createConvertSpace2DepthLayerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGatherToSlice
//

def ConvertGatherToSlice : PassBase<"convert-gather-to-slice", "vpux::FunctionPass"> {
    let summary = "Convert Gather operation to Slice operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces legal `Gather` operations with `Slice` operations.
    }];

    let constructor = "vpux::IE::createConvertGatherToSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertScalarToTensor
//

def ConvertScalarToTensor : PassBase<"convert-scalar-to-tensor", "vpux::FunctionPass"> {
    let summary = "Convert a scalar input to tensor";

    let description = [{
        This pass checks the operands/results rank for any operation and if it is a scalar(its rank is 0), it will be converted into a tensor with one element.
    }];

    let constructor = "vpux::IE::createConvertScalarToTensorPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertMinMaxToClamp
//

def ConvertMinMaxToClamp : PassBase<"convert-min-max-to-clamp", "vpux::FunctionPass"> {
    let summary = "Convert Min and Max to Clamp";

    let description = [{
        This pass replaces MinimumOp and MaximumOp, having one input as tensor(2D,3D,4D) and the other one as scalar, with ClampOp.
    }];

    let constructor = "vpux::IE::createConvertMinMaxToClampPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertShapeTo4D
//

def ConvertShapeTo4D : PassBase<"convert-shape-to-4d", "vpux::FunctionPass"> {
    let summary = "Convert tensors shapes to 4D";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces ND tensor with 4D analogues for layers, which has such limitations on VPUIP level.
        Also this pass replaces ND network inputs and outputs with 4D analogues to overcome runtime limitations.
    }];

    let constructor = "vpux::IE::createConvertShapeTo4DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPaddingsToFloorMode
//

def ConvertPaddingsToFloorMode : PassBase<"convert-paddings-to-floor-mode", "vpux::FunctionPass"> {
    let summary = "Convert Convolution and Pooling layers paddings to FLOOR rouding mode";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass updates padding attributes for Convolution and Pooling layers.
        It switches layer rounding mode to FLOOR and updates paddings to satisfy output shape.
    }];

    let constructor = "vpux::IE::createConvertPaddingsToFloorModePass()";
}

//
// ConvertFCToConv
//

def ConvertFCToConv : PassBase<"convert-fc-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert FullyConnected op to Convolution operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `FullyConnected` operations with `Convolution` operation.
        It inserts extra `Reshape` operations to satisfy `Convolution` specification.
    }];

    let constructor = "vpux::IE::createConvertFCToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertShuffleChannels
//

def ConvertShuffleChannels : PassBase<"convert-shuffle-channels", "vpux::FunctionPass"> {
    let summary = "Convert ShuffleChannels to Reshape->Transpose->Reshape";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        Converts ShuffleChannels to Reshape->Transpose->Reshape.
    }];

    let constructor = "vpux::IE::createConvertShuffleChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MatMulInputsTo2d
//

def MatMulInputsTo2d : PassBase<"matmul-inputs-to-2d", "vpux::FunctionPass"> {
    let summary = "Convert MatMul inputs to 2d";

    let description = [{
        This pass converts `MatMul` inputs to 2d.

        For example, `MatMul` input with 4x1x64 geometry will be split to four inputs with 1x64 dimensions.
        Resulting inputs with filters go to `MatMul` operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::createMatMulInputsTo2dPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableGroupedMatMul", "enable-grouped-matmul",
            "bool", "false",
            "Flag to enable or disable grouped MatMul execution"
        >
    ];
}

//
// ConvertNonConstantPadToSliceAndConcat
//

def ConvertNonConstantPadToSliceAndConcat : PassBase<"convert-non-constant-pad-to-slice-and-concat", "vpux::FunctionPass"> {
    let summary = "Convert non constant pad to slice and concat";

    let description = [{
        // FIXME: #-108139
        In Pad Operation, when pad_mode is not "constant", the padding operation will create a non constant padding according
        to padding mode.

        The Pad Operation can be transformed into multiple slice and concat operations.
    }];

    let constructor = "vpux::IE::createConvertNonConstantPadToSliceAndConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableSEPPad", "enable-sep-pad",
            "bool", "false",
            "Flag which identifies whether an Pad operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// ConvertBatchedLayerTo1N
//

def ConvertBatchedLayerTo1N : PassBase<"convert-batched-layer-to-1n", "vpux::FunctionPass"> {
    let summary = "Convert layer with batched input to new one with batch equal to 1";

    let description = [{
        This pass inserts Transpose to convert batched input to new one with batch equal to 1

        Original operation:
            Activation: 4x16x1x1 ->
                                    Conv -> 4x5x1x1
            Weights:    5x16x1x1 ->

        New subgraph:
            Activation:4x16x1x1     Weights:5x16x1x1
                |                       |
            Transpose:1x16x4x1          |
                |                       |
                |                       |
                        Conv:1x5x4x1
                            |
                     Transpose:4x5x1x1
    }];

    let constructor = "vpux::IE::createConvertBatchedLayerTo1NPass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def ConvertDivideToMultiply : PassBase<"convert-divide-to-multiply", "vpux::FunctionPass"> {
    let summary = "Converts IE.Divide to IE.Multiply";

    let description = [{
        Converts suitable IE.Divide with a constant to IE.Multiply with a
        reciprocal(constant). Indirectly, this reduces the amount of complex
        patterns as one does not need to differentiate between division and
        multiplication after this pass.

        See also ConvertSubtractToAdd for similar idea concerning IE.Subtract ->
        IE.Add conversion.

        See also ConvertToScaleShift for the logic of converting IE.Multiply
        into an (eventually) HW operation.
    }];

    let constructor = "vpux::IE::createConvertDivideToMultiplyPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertMatMulToConv
//

def ConvertMatMulToConv : PassBase<"convert-matmul-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert MatMul with 2d 'weights' to convolution";

    let description = [{
        This pass replaces 2d `Matmul` operations with `Convolution` .
    }];

    let constructor = "vpux::IE::createConvertMatMulToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptDynamicEltwiseWithShapeOf
//

def OptDynamicEltwiseWithShapeOf : PassBase<"opt-dynamic-eltwise-shapeof", "vpux::FunctionPass"> {
    let summary = "Optimize the `dynamic eltwise` with the `shapeOf` pattern by folding the eltwise operation or converting the eltwise to `dynamicReshape`";

    let description = [{
        In the `dynamic eltwise` followed by `shapeOf` pattern, numerical operations are unnecessary for determining the output shape.
        If the eltwise has a single input or the input shape matches the output shape, the eltwise operand can be folded.
        In some cases, the eltwise can be replaced by `dynamicReshape`, where the `dynamicReshape` - `shapeOf` pattern provides the final shape.
    }];

    let constructor = "vpux::IE::createOptDynamicEltwiseWithShapeOfPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertConvBackpropDataToTransposedConv
//

def ConvertConvBackpropDataToTransposedConv : PassBase<"convert-convbackpropdata-to-transposedconv", "vpux::FunctionPass"> {
    let summary = "Convert (Group)ConvolutionBackPropData to (Group)TransposedConvolution";

    let description = [{
        Converts (Group)ConvolutionBackPropData operations to (Group)TransposedConvolution.

        This is done since the two operations have different ordering for the weights:
        - (Group)ConvolutionBackPropData: [(GROUPS,) C_IN, C_OUT, Z, Y, X]
        - (Group)TransposedConvolution:   [(GROUPS,) C_OUT, C_IN, Z, Y, X]

        The order used by (Group)TransposedConvolution is aligned with the order used by
        all other convolution operations in the compiler, making it simple to work with
        this operation and convert between transposed and forward convolutions.
    }];

    let constructor = "vpux::IE::createConvertConvBackpropDataToTransposedConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustGroupConvShape
//

def AdjustGroupConvShape: PassBase<"adjust-groupconv-shape", "vpux::FunctionPass"> {
    let summary = "Adjust Goupconvolution input shape and kernel shape to avoid or reduce expand size for channel align request";

    let description = [{
        This pass adjusts Groupconvolution input shape and kernel shape to get a better performance

        For the shape can reserve a channel aligned shape:
        Original operation:
            Activation: 1x2x64x512 ->
                                    Conv -> 1x2x64x512
            Weights:    2x1x1x1 ->

        New subgraph:
            Activation: 1x2x64x512     Weights:16x1x1x1
                |                       |
            Reshape: 1x16x8x512         |
                |                       |
                |                       |
                |                       |
                     Conv: 1x16x8x512
                            |
                     Reshape: 1x2x64x512
                            |

        For the shape can's reserve a channel aligned shape:
            Original operation:
                Activation: 1x1x289x289 ->
                                        Conv -> 1x1x289x289
                Weights:    1x1x1x1     ->

            New subgraph:
                Activation: 1x1x289x289     Weights:289x1x1x1
                    |                       |
                ShapeCast: 1x289x17x17      |
                    |                       |
                    |                       |
                     \                     /
                        Conv: 1x289x17x17
                                |
                    ShapeCast: 1x1x289x289
                                |
    }];

    let constructor = "vpux::IE::createAdjustGroupConvShapePass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustConvolutionWeights
//

def AdjustConvolutionWeights: PassBase<"adjust-convolution-weights", "vpux::FunctionPass"> {
    let summary = "Adjust Convolution input shape and kernel shape to avoid slice-concat-expand pattern for channel align request";

    let description = [{
        This pass adjusts Convolution input shape and kernel shape to get a better performance
    }];

    let constructor = "vpux::IE::createAdjustConvolutionWeightsPass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustConvolutionShape
//

def AdjustConvolutionShape: PassBase<"adjust-convolution-shape", "vpux::FunctionPass"> {
    let summary = "Adjust Convolution input shape and kernel shape to avoid or reduce expand size for channel align request";

    let description = [{
        This pass adjusts Convolution input shape and kernel shape to get a better performance
    }];

    let constructor = "vpux::IE::createAdjustConvolutionShapePass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustConvolutionInputShape
//

def AdjustConvolutionInputShape: PassBase<"adjust-convolution-input-shape", "vpux::FunctionPass"> {
    let summary = "Adjust convolution input shape for better hardware utilization";

    let description = [{
        This pass adjusts convolution input shape for better hardware utilization

        Original operation:
            Activation: 1x32x256x1 ->
                                    Conv -> 1x16x256x1
            Weights:    16x32x1x1 ->

        New subgraph:
            Activation: 1x32x256x1     Weights:16x32x1x1
                |                       |
            Reshape: 1x32x64x4          |
                |                       |
                |                       |
                |                       |
                     Conv: 1x16x64x4
                            |
                     Reshape: 1x16x256x1
                            |
    }];

    let constructor = "vpux::IE::createAdjustConvolutionInputShapePass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustMaxPoolInputShape
//

def AdjustMaxPoolInputShape: PassBase<"adjust-maxpool-input-shape", "vpux::FunctionPass"> {
    let summary = "Adjust maxpool input shape for better hardware utilization and Less splitting when handle large kernel";

    let description = [{
        Adjust maxpool input shape for better hardware utilization and Less splitting when handle large kernel.

        Convert maxpool from
                                      reshape
                                         |
           input                       input
       [1, C, H, 1]         =>    [1, C, H/Int, Int]
            or                          or
       [1, C, 1, W]         =>    [1, C, Int, W/Int]
            |                            |
          maxpool                     maxpool
            |                            |
          output                      output
        [1, C, 1, 1]                 [1, C, 1, 1]
    }];

    let constructor = "vpux::IE::createAdjustMaxPoolInputShapePass()";
        let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToScaleShift
//

def ConvertToScaleShift : PassBase<"convert-to-scale-shift", "vpux::FunctionPass"> {
    let summary = "Convert Add and Multiply operations to ScaleShift operations";

    let description = [{
        This pass replaces suitable `Add` and `Multiply` operations with `ScaleShift` operations.
    }];

    let constructor = "vpux::IE::createConvertToScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeLSTMSequence
//

def DecomposeLSTMSequence : PassBase<"decompose-lstm-sequence", "vpux::FunctionPass"> {
    let summary = "Decompose LSTMSequenceOp";

    let description = [{
        This pass:
        - Extracts the matrix multiplication of inputData and weights, and the
          addition of biases from LSTMSequence to allow them to run on the DPU.
        - If the operation is not supported by the VPU::LSTMSequenceOp, it
          decomposes the bidirectional LSTMSequence into one forward and one
          reverse operator, and then unrolls all LSTMSequenceOps to LSTMCellOps.
    }];

    let constructor = "vpux::IE::createDecomposeLSTMSequencePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeLSTMCell
//

def DecomposeLSTMCell : PassBase<"decompose-lstm-cell", "vpux::FunctionPass"> {
    let summary = "Replace LSTMCell operation with a subgraph of smaller operations";

    let description = [{
        Decomposes `LSTMCell` operation into smaller `DPU` friendly operations, followed by a single `LSTMGates` operation which computes activation functions.
    }];

    let constructor = "vpux::IE::createDecomposeLSTMCellPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeGRUCell
//

def DecomposeGRUCell : PassBase<"decompose-gru-cell", "vpux::FunctionPass"> {
    let summary = "Replace GRUCell operation with a subgraph of smaller operations";

    let description = [{
        Decomposes `GRUCell` operation into smaller `DPU` friendly operations and activation functions.
    }];

    let constructor = "vpux::IE::createDecomposeGRUCellPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
//  DecomposeNormalizeL2
//

def DecomposeNormalizeL2 : PassBase<"decompose-normalize-l2", "vpux::FunctionPass"> {
    let summary = "Decompose NormalizeL2Op into separate eltwise operations";

    let description = [{
        NormalizeL2Op will be decomposed into separate eltwise operations if axes parameter has all dimensions
    }];

    let constructor = "vpux::IE::createDecomposeNormalizeL2Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DilatedConvConvert
//

def DilatedConvConvert : PassBase<"dilated-conv-convert", "vpux::FunctionPass"> {
    let summary = "Replace a subgraph to Dilated Convolution(GroupConvolution)";

    let description = [{
        Transformation replaces following graph:
            SpaceToBatch -> FakeQuantize(optional) -> Convolution / GroupConvolution -> BatchToSpace
        to a single Convolution(GroupConvolution) node with updated pads and dilations:
            FakeQuantize(optional) -> Dilated Convolution / Dilated GroupConvolution
    }];

    let constructor = "vpux::IE::createDilatedConvConvertPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertBroadcastToTile
//

def ConvertBroadcastToTile : PassBase<"convert-broadcast-to-tile", "vpux::FunctionPass"> {
    let summary = "Convert Broadcast operation to Tile operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Broadcast` operation with `Tile` operation.
    }];

    let constructor = "vpux::IE::createConvertBroadcastToTilePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGRNToNormalizeL2
//

def ConvertGRNToNormalizeL2 : PassBase<"convert-grn-to-normalizel2", "vpux::FunctionPass"> {
    let summary = "Convert GRN operation to Normalize_L2 operation";

    let description = [{
        This pass replaces `GRN` with `Normalize_L2` operation. `GRN` is a `Normalize_L2` operation with specified axes input and processing mode whose values are fixed in all possible cases, also there is no dedicated kernel for  `GRN` operation.
    }];

    let constructor = "vpux::IE::createConvertGRNToNormalizeL2Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSubtractToAdd
//

def ConvertSubtractToAdd : PassBase<"convert-subtract-to-add", "vpux::FunctionPass"> {
    let summary = "Convert Subtract operation to Add with either Negative or DW Conv operations";

    let description = [{
        This pass replaces `Subtract` operation with `Add` with `DW Conv` operations.
    }];

    let constructor = "vpux::IE::createConvertSubtractToAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertScaleShiftToDW
//

def ConvertScaleShiftToDW : PassBase<"convert-scale-shift-depthwise", "vpux::FunctionPass"> {
    let summary = "Convert Scale-Shift operation to Depthwise Convolution";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        Convert Scale-Shift operation to Depthwise convolution.
    }];

    let constructor = "vpux::IE::createConvertScaleShiftToDWPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseActivationOps
//

def FuseActivationOps : PassBase<"fuse-activation-ops", "vpux::FunctionPass"> {
    let summary = "Fuse activation functions and/or Clamp with tasks that support post-processing";

    let description = [{
        The pass is a part of `OptimizeActivations` pipeline.

        1. Fuse activation functions (e.g. ReLU, leaky ReLU) with tasks that support post-processing
        depending on the compilation mode
        2. Fuse clamp with tasks that support post-processing depending on the compilation mode
    }];

    let constructor = "vpux::IE::createFuseActivationOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableFuseClamp", "enable-fuse-clamp",
            "bool", "false",
            "Flag to enable or disable fuse clamp"
        >
    ];
}

//
// FusePermuteQuantize
//

def FusePermuteQuantize : PassBase<"fuse-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute combination in 1 common operation";

    let description = [{
        Converts Quantize-MemPermute combination in 1 common operation.
    }];

    let constructor = "vpux::IE::createFusePermuteQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "dpuOnly", "dpu-only",
            "bool", "false",
            "[Optional] Set to true when target platform does not have software PermuteQuantize layer"
        >,
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// LegalizeDilatedConvolution
//

def LegalizeDilatedConvolution : PassBase<"legalize-dilated-conv", "vpux::FunctionPass"> {
    let summary = "Handle dilated convolutions";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass expands filter of dilated convolution so that they are able to be infered
            on dpu because of hardware limitation.
    }];

    let constructor = "vpux::IE::createLegalizeDilatedConvolutionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableSEPDilatedGroupConv", "enable-sep-dilated-group-conv",
            "bool", "false",
            "Flag which identifies whether an GroupConvolution operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// ResolveStridedSlice
//

def ResolveStridedSlice : PassBase<"resolve-strided-slice", "vpux::FunctionPass"> {
    let summary = "Decouple strided slice to slice + reshape";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        It adjusts the attributes' rank of IE::StridedSlice to be same with input's rank.
        It replaces IE::StridedSlice with non zero masks to a simpler IE::StridedSlice with zero masks + IE::Reshape
        It replaces IE::StridedSlice with dense<1> strides strides with a simple IE::Slice operation
    }];

    let constructor = "vpux::IE::createResolveStridedSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertStridedSlice2Conv
//

def ConvertStridedSlice2Conv : PassBase<"convert-strided-slice-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert strided slice to dwconv";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        The pass replaces `StridedSlice` with strides > 1 operations with `Convolution` operation.
    }];

    let constructor = "vpux::IE::createConvertStridedSlice2ConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// RunF16ToF32ConvertOnDPU
//

def RunF16ToF32ConvertOnDPU : PassBase<"run-f16-to-f32-convert-on-dpu", "vpux::FunctionPass"> {
    let summary = "Run F16 -> F32 Convert layer on DPU";

    let description = [{
        DPU can output an FP32 tensor and can be used to perform fp16 -> fp32 conversions, instead of SW Convert.

        If an IE.ConvertOp with f16 input, f32 output has a future DPU op as parent, this pass sets f32 to parent output
        and removes the original IE.ConvertOp
    }];

    let constructor = "vpux::IE::createRunF16ToF32ConvertOnDPUPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeSliceWithStride
//

def OptimizeSliceWithStride : PassBase<"optimize-slice-with-stride", "vpux::FunctionPass"> {
    let summary = "Optimize slice with stride";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        The pass replace(fuse) 'Slice' with(into) 'Convolution' which meet the requirements:
        1. Cannot optimize in previous passes
        2. DDR->DDR stride copy

        So that op will be replaced with new Conv when:
        - Can use 'ShapeCast' for channel alignment instead of expand

        Or op will be fused into previous 'Convolution' when:
        - If there is a 'Concat' concatenate slice's output with a constant, and use 'Add' to replace the 'Concat'
    }];

    let constructor = "vpux::IE::createOptimizeSliceWithStridePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ResolveScatterUpdateByTranspose
//

def ResolveScatterUpdateByTranspose : PassBase<"resolve-scatter-update-by-transpose", "vpux::FunctionPass"> {
    let summary = "Resovle ScatterUpdate operation by Transpose Operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        Only axis == 0 is supported in SWkernel.
        The pass replaces ScatterUpdate(axis!=0) with `IE::Transpose-IE::ScatterUpdate(axis=0)-IE::Transpose` pipeline.

    }];

    let constructor = "vpux::IE::createResolveScatterUpdateByTransposePass()";

    let dependentDialects = [
    "vpux::IE::IEDialect"
    ];
}


//
// ConvertNceOpsTo4D
//

def ConvertNceOpsTo4D : PassBase<"convert-nce-ops-to-4d", "vpux::FunctionPass"> {
    let summary = "Convert non 4D NCE tasks to its 4D variance";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Extends input, filter and output tensors with height = 1.
        [N, C, W] -> [N, C, 1, W]
        strides:    {2} -> strides:    {1, 2}
        pads_begin: {2} -> pads_begin: {0, 2}
        pads_end:   {2} -> pads_end:   {0, 2}
        dilations:  {2} -> dilations:  {1, 2}
    }];

    let constructor = "vpux::IE::createConvertNceOpsTo4DPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGroupConvToConv
//

def ConvertGroupConvToConv : PassBase<"convert-groupconv-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert GroupConvolution to Convolution";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces some `GroupConvolution` operations with `Convolution` operation.
        There are two kinds of converter:
        1. Convert to multiple Convolutions. It is a general converter.
        It inserts extra `Slice` and `Concat` operations to satisfy `Convolution` specification.
        2. Convert to single Convolution. It is only for GroupConv with constant weights.
        It reconstruct `Weights` to satisfy `Convolution` specification.
    }];

    let constructor = "vpux::IE::createConvertGroupConvToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertLargeConvToMultiConvWithAdd
//

def ConvertLargeConvToMultiConvWithAdd : PassBase<"convert-large-conv-to-multi-conv-with-add", "vpux::FunctionPass"> {
    let summary = "Convert large Convolution to multi Convolution with Add Op";

    let description = [{
        This pass is part of the `AdjustForVPU` pipeline.

        It converts large `Convolution` Op into multiple smaller `Convolution` Op followed by an `Add` Op.

        The purpose is to optimize the `Convolution` by splitting it into smaller pieces along the input channels.
        There are two performance benefits:
        1. Reduces the overlapped input data, thereby decreasing the DMA size
        2. Reduces the number of tiles, preventing excessive tiling and improving workload efficiency
    }];

    let constructor = "vpux::IE::createConvertLargeConvToMultiConvWithAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollConv3dToConv2d
//

def UnrollConv3dToConv2d : PassBase<"unroll-conv3d-to-conv2d", "vpux::FunctionPass"> {
    let summary = "Handle 3D convolutions";

    let description = [{
        The pass is a part of `DefaultHW` pipeline.

        This pass unrolls 3D convolution to 2D convolution, so that they are able to be infered
            on dpu because of hardware don't support 3D convolution natively.
    }];

    let constructor = "vpux::IE::createUnrollConv3dToConv2dPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UniquifyOps
//

def UniquifyOps : PassBase<"uniquify-ops", "vpux::FunctionPass"> {
    let summary = "Remove duplicating operations with a common producer Value";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass merges operations that are identical to each other, combining consumers.
    }];

    let constructor = "vpux::IE::createUniquifyOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMemPermute
//

def ConvertToMemPermute : PassBase<"convert-to-mem-permute", "vpux::FunctionPass"> {
    let summary = "Convert Reorder and Transpose ops to MemPermute operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces all `Reorder` and `Transpose` operations with `MemPermute` operation.
    }];

    let constructor = "vpux::IE::createConvertToMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollBatch
//

def UnrollBatch : PassBase<"unroll-batch", "vpux::FunctionPass"> {
    let summary = "Split inputs of NCE tasks when their batch size is greater than 1";

    let description = [{
        This pass splits inputs of NCE tasks by batch.

        For example:
        * `FullyConnected` input with 2x64 geometry will be split by two inputs with 1x64 dimensions.
        * `Convolution` input 3x16x32x64 will be split into three 1x16x32x64 inputs.
        Resulting tensors go to corresponding operations and the outputs are concatenated.
    }];

    let constructor = "vpux::IE::createUnrollBatchPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseMvn6ScaleBiasPass
//

def FuseMvn6ScaleBias : PassBase<"fuse-mvn6-scale-bias", "vpux::FunctionPass"> {
    let summary = "Fuse subsequent Mul, Add into MVN6";

    let description = [{
        The pass is a part of `InitialTransformations` pipeline.
    }];

    let constructor = "vpux::IE::createFuseMvn6ScaleBiasPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertMVN6ToMVN1Pass
//

def ConvertMVN6ToMVN1 : PassBase<"convert-mvn6-to-mvn1", "vpux::FunctionPass"> {
    let summary = "Convert MVN6 ops to MVN1 operation";

    let description = [{
        The pass is a part of `InitialTransformations` pipeline.

        This pass replaces `MVN6` operations with `MVN1` operation, for specific cases.
    }];

    let constructor = "vpux::IE::createConvertMVN6ToMVN1Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleU16FakeQuantize
//

def HandleU16FakeQuantize : PassBase<"handle-u16-fake-quantize", "vpux::FunctionPass"> {
    let summary = "Handle U16 FakeQuantize";

    let description = [{
        The pass is a part of `InitialTransformations` pipeline.

        This pass do the following transformations to U16 FakeQuantize:
        1. In case enableU16FQToScaleShiftConversion is true and the FakeQuantize has in_low != out_low or in_high != out_high it is replaced with a ScaleShift op.
        2. In case enableU16FQToScaleShiftConversion is false and the FakeQuantize is per tensor and the input and output low is equal to 0 it is replaced with a ReLu activation function.
        3. Otherwise the FakeQuantize is completely removed.
    }];

    let constructor = "vpux::IE::createHandleU16FakeQuantizePass()";

    let options = [
        Option<
            "enableU16FQToScaleShiftConversion", "enable-u16-fake-quantize-to-scale-shift-conversion",
            "bool", "false",
            "Enable U16 FakeQuantize to ScaleShift conversion"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// SwapMemPermuteAndExpand
//

def SwapMemPermuteAndExpand : PassBase<"swap-mem-permute-and-expand", "vpux::FunctionPass"> {
    let summary = "Swap MemPermute and Expand operation";

    let description = [{
        The pass is a part of `MemPermute processing` pipeline.
        This pass swap Reorder-like `MemPermute` and `Expand` operation order for optimization.
        For subgraph MemPermute -> Expand, it will be converted as Expand -> MemPermute,
        which will be further optimized in later pass with single DMA op.

    }];

    let constructor = "vpux::IE::createSwapMemPermuteAndExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LegalizeNDMemPermute
//

def LegalizeNDMemPermute : PassBase<"legalize-nd-mem-permute", "vpux::FunctionPass"> {
    let summary = "Legalize MemPermute operation with input rank > 4";

    let description = [{
        This pass tries to legalize MemPermute operations by merging dims that are adjacent before and after the permutation.
        Applied only for VPUX.37XX because SW Kernel Tiling is limited to 4D.
    }];

    let constructor = "vpux::IE::createLegalizeNDMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SplitConvWithMultipleFQ
//

def SplitConvWithMultipleFQ : PassBase<"split-conv-with-multiple-fq", "vpux::FunctionPass"> {
    let summary = "Splits Convolution for multiple FakeQuantize";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It splits `Convolution` operation with multiple consumers with `FakeQuantize` operations,
        into multiple `Convolution` operations, one for each consumer. This transformation is needed to be
        able to quantize convolution and fuse bias and post-processing operations.
    }];

    let constructor = "vpux::IE::createSplitConvWithMultipleFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertTransposedConvToConv
//

def ConvertTransposedConv2DToConv2D : PassBase<"convert-transposed-conv-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert TransposedConvolution 2D to Convolution 2D";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        Replaces transposed convolution by upsampling and convolution
    }];

    let constructor = "vpux::IE::createConvertTransposedConv2DToConv2DPass()";

    let options = [
        Option<
            "enableSEPTransposedConv", "enable-sep-transposed-conv",
            "bool", "false",
            "Flag which identifies whether Transposed Convolutions operation can be executed using the Storage Element hardware feature"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGroupTransposedConvToGroupConv
//

def ConvertGroupTransposedConvToGroupConv : PassBase<"convert-group-transposed-conv-to-groupconv", "vpux::FunctionPass"> {
    let summary = "Convert GroupTransposedConvolution to GroupConvolution";

    let description = [{
        The pass replaces IE::GroupTransposedConvolution by IE::Upsampling and IE::GroupConvolution
    }];

    let constructor = "vpux::IE::createConvertGroupTransposedConvToGroupConvPass()";

    let options = [
        Option<
            "enableSEPTransposedConv", "enable-sep-transposed-conv",
            "bool", "false",
            "Flag which identifies whether Transposed Convolutions operation can be executed using the Storage Element hardware feature"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGroupTransposedConvToTransposedConv
//

def ConvertGroupTransposedConvToTransposedConv : PassBase<"convert-group-transposed-conv-to-transposed-conv", "vpux::FunctionPass"> {
    let summary = "Convert GroupTransposedConvolution to TransposedConvolution";

    let description = [{
        The pass replaces IE::GroupTransposedConvolution by IE::TransposedConvolution.
        Only applicable for operations that can be executed using SEP.
    }];

    let constructor = "vpux::IE::createConvertGroupTransposedConvToTransposedConvPass()";

    let options = [
        Option<
            "enableSEPTransposedConv", "enable-sep-transposed-conv",
            "bool", "false",
            "Flag which identifies whether Transposed Convolutions operation can be executed using the Storage Element hardware feature"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleLargeStrides
//

def HandleLargeStrides : PassBase<"handle-large-strides", "vpux::FunctionPass"> {
    let summary = "Handle operations with large strides";

    let description = [{
        This pass splits operations with strides larger than supported on hardware.
    }];

    let constructor = "vpux::IE::createHandleLargeStridesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//=================================================================================
// LowPrecision
//=================================================================================

//
// WeightsDequantizeToFakeQuantize
//

def WeightsDequantizeToFakeQuantize : PassBase<"weights-dequantize-to-fake-quantize", "vpux::FunctionPass"> {
    let summary = "Replace weights dequantization with FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Replaces Constant (i8) -> Convert (to fp) -> Subtract (zp) -> Multiply (scale) -> with
        Constant (i8) -> Convert (to fp) -> FakeQuantize -> deducing levels and FakeQuantize limits according to actual values in the weights Constant
    }];

    let options = [
        Option<
            "enableWDBlockArgumentInput", "enable-wd-blockarg-input",
            "bool", "false",
            "Enable WeightsDequantizeToFakeQuantizePass on structures with BlockArgument input"
        >
    ];

    let constructor = "vpux::IE::createWeightsDequantizeToFakeQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConsolidateWeightsDequantization
//

def ConsolidateWeightsDequantization : PassBase<"consolidate-weights-dequantize", "vpux::FunctionPass"> {
    let summary = "Replace weights dequantization with DynamicDequantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Replaces
        INT4 weights as params -> Convert (to fp16) -> Subtract (w params) -> Multiply (w params) -> Conv with
        INT4 weights as params -> QuantCast (dummy quant zp=0 scale=1) -> DynamicDequantize (w scale and zp params) -> Conv
    }];

    let options = [
        Option<
            "enableWeightsDynamicDequantization", "enable-weights-dynamic-dequantization",
            "bool", "false",
            "Enable dynamic dequantization for weights as input"
        >
    ];

    let constructor = "vpux::IE::createConsolidateWeightsDequantizationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// SwapFakeQuantWithReshapeAndStridedSlice
//

def SwapFakeQuantWithReshapeAndStridedSlice : PassBase<"swap-fake-quant-with-reshape-and-strided-slice", "vpux::FunctionPass"> {
    let summary = "Swap FakeQuantize with Reshape and StridedSlice when required to void redundant expand and permute ops";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It matches pattern non-channel-aligned op -> optional Reshapes -> FQ -> Reshapes / StridedSlice -> channel-aligned op
        Move the FQ right before the channel-aligned op to avoid redundant expand and permute ops.
    }];

    let constructor = "vpux::IE::createSwapFakeQuantWithReshapeAndStridedSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SplitFakeQuant
//

def SplitFakeQuant : PassBase<"split-fake-quant", "vpux::FunctionPass"> {
    let summary = "Splits FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It splits `FakeQuantize` operations to `quant.qcast -> quant.dcast` pair.
    }];

    let constructor = "vpux::IE::createSplitFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// ConvertToDequantize
//

def ConvertToDequantize : PassBase<"convert-to-dequantize", "vpux::FunctionPass"> {
    let summary = "Convert ConvertOp to DequantizeOp";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It matches pattern non-const -> Convert -> ViewLikeOp/TransposeOp -> Convolution/GroupConvolution,
        and replaces ConvertOp with QuantizeCastOp -> DequantizeOp if the non-const input can be propagated to the
        filter of a Convolution/GroupConvolution. We expect that DequantizeOp will be propagated to the
        Convolution/GroupConvolution and then be optimized by mixed precision.

        This optimization supports Convert with si8 input and fp16 output.
        It improves the performance if there are converts for weights-as-input.
    }];

    let constructor = "vpux::IE::createConvertToDequantizePass()";

    let options = [
        Option<
            "enableWDBlockArgumentInput", "enable-wd-blockarg-input",
            "bool", "false",
            "Enable ConvertToDequantize on structures with BlockArgument input"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DequantizeConst
//

def DequantizeConst : PassBase<"dequantize-const", "vpux::FunctionPass"> {
    let summary = "Dequantize constant tensors";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It performs constant folding for `Constant -> quant.dcast` case.
        The pass is used as a fallback to FP16 computations for the cases, where quantized types where not used by layers.
    }];

    let constructor = "vpux::IE::createDequantizeConstPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableRuntimeDequant", "enable-runtime-dequant",
            "bool", "false",
            "Enable runtime dequantization for mixed precision weights"
        >,
        Option<
            "runtimeDequantizationLimit", "runtime-dequantization-limit",
            "int64_t", "524288",
            "Lower limit on weight size for runtime dequantization"
            "Weights smaller than the limit will be statically dequantized"
        >
    ];
}

//
// MergeFakeQuant
//

def MergeFakeQuant : PassBase<"merge-fake-quant", "vpux::FunctionPass"> {
    let summary = "Merge back to FakeQuantize";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        It merges pair `quant.qcast -> quant.dcast` into single `IE.FakeQuantize`.
        The pass is used as a fallback to FP16 computations for the cases, where quantized types where not used by layers.
    }];

    let constructor = "vpux::IE::createMergeFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateQuantizeDequantize
//

def PropagateQuantizeDequantize : PassBase<"propagate-quantize-dequantize", "vpux::FunctionPass"> {
    let summary = "Propagate Quantize/Dequantize through agnostic operations";

        let description = [{
        The pass is a part of LowPrecision pipeline.

        Quantize/Dequantize are propagated through operations
        }];

    let constructor = "vpux::IE::createPropagateQuantizeDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag which identifies whether an operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// FoldActivationBeforeFQ
//

def FoldActivationBeforeFQ : PassBase<"fold-activation-before-fq", "vpux::FunctionPass"> {
    let summary = "Delete ReLUOp if next Op is FakeQuantize with input_low > 0; Delete ClampOp if next Op is FakeQuantize with input ranges smaller then Clamp limits.";

    let description = [{
        The pass is a part of `InitialLowPrecision` pipeline.

        It deletes the ReLUOp from `ReLU -> FakeQuantize` if the FakeQuantizeOp has input_low > 0.
        It deletes the ClampOp from `Clamp -> FakeQuantize` if the FakeQuantizeOp has input ranges smaller then Clamp limits.
    }];

    let constructor = "vpux::IE::createFoldActivationBeforeFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseFQAndMul
//

def FuseFQAndMul : PassBase<"fuse-fq-and-mul", "vpux::FunctionPass"> {
    let summary = "Fuse Mul into FakeQuantize if it's a const weight";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Fuse Mul into FakeQuantize if it's a const weight.
    }];

    let constructor = "vpux::IE::createFuseFQAndMulPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// EltwiseFakeQuantizeFusion
//

def EltwiseFakeQuantizeFusion : PassBase<"eltwise-fake-quantize-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse Eltwise into FakeQuantize if one of Eltwise's input is a constant scalar (can be also a quantized constant)";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Fuse constant scalar Eltwise's input into FakeQuantize consumer input range.
    }];

    let constructor = "vpux::IE::createEltwiseFakeQuantizeFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapOperations
//

def SwapOperations : PassBase<"swap-operations", "vpux::FunctionPass"> {
    let summary = "Swap operations implemented ElemTypeInfoOpInterface interface with bias and activation";

        let description = [{
        In order to fuse the bias and activation functions into a main operation,
        intermediate operations will be moved after the fusable operation. For example:
          `Conv -> Reshape -> bias` will be converted to `Conv -> bias -> Reshape`
          `Conv -> Transpose -> ReLU` will be converted to `Conv -> ReLU -> Transpose
        Only operations that implement `IE_ElemTypeInfoOpInterface` are moved.
        Currently, operations are moved only through bias (Add), ReLU, Sigmoid, Tanh and Clamp.
        }];

    let constructor = "vpux::IE::createSwapOperationsPass()";

    let dependentDialects = [
            "vpux::IE::IEDialect"
            ];
            let options = [

    Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag which identifies whether an operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// SwapPadLayer
//

def SwapPadLayer : PassBase<"swap-pad-layer", "vpux::FunctionPass"> {
    let summary = "Swap pattern Pad -> Transpose to Transpose -> Pad";

        let description = [{

        In order to fuse Pad layer to Convolution swap Pad with operations between it and Convolution.
        For now only case Pad -> Transpose is supported

        }];

let constructor = "vpux::IE::createSwapPadLayerPass()";

let dependentDialects = [
        "vpux::IE::IEDialect"
        ];
}

//
// PropagateOpThroughBatchConcat
//

def PropagateOpThroughBatchConcat : PassBase<"propagate-op-through-batch-concat", "vpux::FunctionPass"> {
    let summary = "Propagate SW ops after batch unrolled matmul to enable vertical fusion";

        let description = [{

        Move ops after concat to place after each batch unrolled matmul.
        Currently only softmax is enabled.

        }];

let constructor = "vpux::IE::createPropagateOpThroughBatchConcatPass()";

let dependentDialects = [
        "vpux::IE::IEDialect"
        ];
}

//
// FuseQuantizedOps
//

def FuseQuantizedOps : PassBase<"fuse-quantized-ops", "vpux::FunctionPass"> {
    let summary = "Update quantize/dequantize ops";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
    }];

    let constructor = "vpux::IE::createFuseQuantizedOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag to identify whether operations that can be executed using the Storage Element hardware feature are enabled"
        >,
        Option<
            "seExperimentalOpsEnabled", "se-experimental-ops-enabled",
            "bool", "false",
            "This flag identifies operations that are still a work in progress and can be executed using the Storage Element hardware feature."
        >
    ];
}

//
// RemoveQuantDequantSeq
//

def RemoveQuantDequantSeq : PassBase<"remove-quantdequant-seq", "vpux::FunctionPass"> {
    let summary = "Removes quantize->dequantize ops sequence";

    let description = [{
        The optional pass in the `LowPrecision` pipeline.

        Pass detects pattern quantize -> dequantize and removes it
    }];

    let constructor = "vpux::IE::createRemoveQuantDequantSeqPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// OptimizeUnalignedQDQSeq
//

def OptimizeUnalignedQDQSeq : PassBase<"optimize-unaligned-qdq-seq", "vpux::FunctionPass"> {
    let summary = "Swaps AffineReshape->FakeQuantize sequence if channels become unaligned after AffineReshape";

    let description = [{
        Pass swaps order of AffineReshape->FakeQuantize sequence if channels become unaligned after AffineReshape
        Otherwise additionals ops are introduce in order to align channels which impacts performance.
    }];

    let constructor = "vpux::IE::createOptimizeUnalignedQDQSeqPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

//
// ConvertWeightsToU8
//

def ConvertWeightsToU8 : PassBase<"convert-weights-to-u8", "vpux::FunctionPass"> {
    let summary = "Shift data from a signed range to an unsigned one";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        This pass detects quantized convolution and shifts weights data from a signed range to an unsigned one.

        The weights in the OV .bin file are in Int8.
        The NPU hardware supports both UInt8 and Int8 computation.
        It has support for UInt8 with zero point. For Int8 the zero point is 0.

        It can be advantageous for accuracy to convert the weights to UInt8 with zero point as it enables asymetric quantization.
        When using asymmetric quantization, the quantized range is fully utilized. That is because one can exactly map the min/max values
        from the float range to the min/max of the quantized range. Using symmetric quantization, if the float range is biased towards
        one side, could result in a quantized range where a significant portion of the dynamic range is dedicated to values that will never appear.
        The most extreme example of this is after ReLU, where the entire tensor is positive.

        The OV IR has this pattern for constants
        Const --> Convert --> Subract (ZP) --> Multiply --> Convolution

        This pass converts the Int8 weights to UInt8 using this simple mapping:

        Int8   UInt8
        -128     0
        -127     1
        -126     2
        ....    ....
        126     254
        127     255

        But we also are required to change the Int8 ZP to a UInt8 zero point and provide that to the hardware.
        Assume ZP_I8 = 16 (In the Subtract node). The zero-point for UInt8 would be 16+128 = 134.
    }];

    let constructor = "vpux::IE::createConvertWeightsToU8Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertWeightsToI4
//

def ConvertWeightsToI4 : PassBase<"convert-weights-to-i4", "vpux::FunctionPass"> {
    let summary = "Shift data from an unsigned range to an signed one";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass detects quantized convolution and only shifts weights from U4 type with zero point of 8 to I4
    }];

    let constructor = "vpux::IE::createConvertWeightsToI4Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertQuantizeOpsToNceOps
//

def ConvertQuantizeOpsToNceOps : PassBase<"convert-quantize-ops-to-nce-ops", "vpux::FunctionPass"> {
    let summary = "Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Converts per-tensor Quantize/Dequantize to eltwise And mixed-precision operation
        where input2 is input1 to perform type conversion on DPU.
    }];

    let constructor = "vpux::IE::createConvertQuantizeOpsToNceOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapViewOpAndClamp
//

def SwapViewOpAndClamp : PassBase<"swap-viewop-and-clamp", "vpux::FunctionPass"> {
    let summary = "Swap ViewOp and Clamp operations";

    let description = [{
        After AlignScales pass we have an additional Clamp layers in IR.
        Therefore, we may get such subgraph:
        ARG -> FQ -> Clamp -> Concat

        Then after SplitFakeQuant and PropagateQuantizeDequantize we have:
        ARG -> Q -> Clamp -> D -> Concat

        Then after ConvertQuantizeOpsToNceOps we have:
        ARG -> Add -> QuantizeCast -> Clamp -> QuantizeCast -> Add -> Concat

        In order to fuse Clamp into eltwise Add we need to move QuantizeCast after Clamp.
        And we need to recalculate the clamp min max value according to QuantizeCast parameter

        The pass can also swap other viewop like Reshape, Permutecast and Slice with Clamp
    }];

    let constructor = "vpux::IE::createSwapViewOpAndClampPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseConvertWithQuantize
//

def FuseConvertWithQuantize : PassBase<"fuse-convert-with-quantize", "vpux::FunctionPass"> {
    let summary = "Fuse Convert with Quantize into QuantCast operation";

    let description = [{
        Pass detects pattern Convert(i8/ui8 -> FP16) -> Quantize(FP16 -> !quant.uniform<...>)
        and fuses it into single QuantCast(i8/ui8 -> !quant.uniform<...>) operation.
    }];

    let constructor = "vpux::IE::createFuseConvertWithQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertAvgPoolToDWConv
//

def ConvertAvgPoolToDWConv : PassBase<"convert-avg-pool-to-dw-conv", "vpux::FunctionPass"> {
    let summary = "Convert AvgPool op to GroupConvolution op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces suitable `AvgPool` operations with `GroupConvolution` operation.
    }];

    let constructor = "vpux::IE::createConvertAvgPoolToDWConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertReverseToDWConv
//

def ConvertReverseToDWConv : PassBase<"convert-reverse-to-dw-conv", "vpux::FunctionPass"> {
    let summary = "Convert Reverse op to GroupConvolution op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces suitable `Reverse` operations with `GroupConvolution` operation.
    }];

    let constructor = "vpux::IE::createConvertReverseToDWConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleAsymmetricStrides
//

def HandleAsymmetricStrides : PassBase<"handle-asymmetric-strides", "vpux::FunctionPass"> {
    let summary = "Handle operations with asymmetric strides";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass splits operations so that they are able to be infered with symmetric strides
            on dpu because of hardware limitation.
    }];

    let constructor = "vpux::IE::createHandleAsymmetricStridesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePadOps
//

def FusePadOps : PassBase<"fuse-pad-ops", "vpux::FunctionPass"> {
    let summary = "Fuse PadOp with CONSTANT model";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        PadOp with CONSTANT model, pad value is 0 and the padding is needed in H and W dimensions only.
        Merge [Pad] -> [Conv] into [Conv].
        Merge [Pad] -> [GroupConv] into [GroupConv].
        Merge [Pad] -> [MaxPool] into [MaxPool].
    }];

    let constructor = "vpux::IE::createFusePadOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

}

//
// ConvertPadToConcat
//

def ConvertPadToConcat : PassBase<"convert-pad-to-concat", "vpux::FunctionPass"> {
    let summary = "Convert Pad Ops to Concat with Constant";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        After FusePadOps pass, there are Pad Ops can not be fused.
        Replace `IE::PadOp` with `IE::ConcatOp` and `Const::DeclareOp`
        Only `IE::PadMode::CONSTANT` case is supported.
    }];

    let constructor = "vpux::IE::createConvertPadToConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UpstreamSlice
//

def UpstreamSlice : PassBase<"upstream-slice", "vpux::FunctionPass"> {
    let summary = "Optimization by upstreaming slice operations";

    let description = [{
        Optimizes scenarios of IE::StridedSlice and IE::SliceOp without neighboring operations.
        Moves the slice operations upwards through the graph, reducing both compute and memory usage.
        In some cases the slice operation may be safely removed from the graph, if the action of upstreaming it
            only adapts the operations constants.
    }];

    let constructor = "vpux::IE::createUpstreamSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleLargeKernels
//

def HandleLargeKernels : PassBase<"handle-large-kernels", "vpux::FunctionPass"> {
    let summary = "Handle large kernels ops";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces Pooling layers or Convolution layers that have kernels bigger than supported by hardware (11x11),
        with equivalent Pooling (approx equiv in case of prime kernel i.e. 13x13) or Convolution layers.

        For convolutioin, there are 2 consecutive rewriters to handle large kernel:
        1. ReshapeLargeConvRewriter
        If IC and height or width equals to 1 and KX or KY is bigger than MAX_KERNEL_SIZE. The utilization may be low if slicing it directly.
        Add AffineReshape to change the shape will not involve extra memory copy and increase utilization.
        After large height or width being split, we will tranpose lowest-dimension continuous data to IC to make result data correctness.
        Original shape only one dim is valid, after affineReshape and transpose, IC of activation is factors.second, so strides should div by factors.second.
        2. SliceLargeConvRewriter
        Slice large kernel to small kernel size for several convolutions, then add the results.
    }];

    let constructor = "vpux::IE::createHandleLargeKernelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertReduceSumToConv
//

def ConvertReduceSumToConv : PassBase<"convert-reduce-sum-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert ReduceSum to Convolution operation";

    let description = [{
        The pass is to convert ReduceSum operation into Convolution.
    }];

    let constructor = "vpux::IE::createConvertReduceSumToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertReduceToPooling
//

def ConvertReduceToPooling : PassBase<"convert-reduce-to-pooling", "vpux::FunctionPass"> {
    let summary = "Convert reduce to pooling ops";

    let description = [{
        The pass is to convert reduce operations (mean, max, sum, min) into pooling.
    }];

    let constructor = "vpux::IE::createConvertReduceToPoolingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollReduceMinAllAxes
//

def UnrollReduceMinAllAxes : PassBase<"unroll-reducemin-all-axes", "vpux::FunctionPass"> {
    let summary = "unroll reducemin on all axes which cannot be convert to efficient nce pooling ops";

    let description = [{
        The pass is to unroll reducemin operations.
    }];

    let constructor = "vpux::IE::createUnrollReduceMinAllAxesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleExcludePadForAvgPool
//

def HandleExcludePadForAvgPool : PassBase<"handle-exclude-pad-for-avg-pool", "vpux::FunctionPass"> {
    let summary = "Handle exclude-pad attribute for AvgPool operations";

    let description = [{
        This pass introduces exclude pad atribute handling for AvgPool operations, that have pad = stride = 1,
        by splitting operation in multiple AvgPool operations in order to handle this particular case.
    }];

    let constructor = "vpux::IE::createHandleExcludePadForAvgPoolPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// InsertReorderBetweenConcatAndLayers
//

def InsertReorderBetweenLayerAndConcat : PassBase<"layer-reorder-concat-pass", "vpux::FunctionPass"> {
    let summary = "Inserts Reorder operation between Transpose and Concat";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It inserts `Reorder` operation between layers `Transpose`, `AffineReshape` and `Concat` operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createInsertReorderBetweenLayerAndConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapTransposeWithFQ
//

def SwapTransposeWithFQ : PassBase<"swap-transpose-with-fq", "vpux::FunctionPass"> {
    let summary = "Swaps Transpose operation with FakeQuantize";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `Transpose` operation with per-tensor `FakeQuantize` operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapTransposeWithFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapConvertWithTransposeReshape
//

def SwapConvertWithTransposeReshape : PassBase<"swap-convert-with-transpose-reshape", "vpux::FunctionPass"> {
    let summary = "Swaps Transpose operation with Convert";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `Transpose` and 'Reshape' operations with Convert operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapConvertWithTransposeReshapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UniquifyBranches
//

def UniquifyBranches : PassBase<"uniquify-branches", "vpux::FunctionPass"> {
    let summary = "Eliminates redundant operations from multiple branches";

    let description = [{
        Supported cases:
        * Convert this subgraph:
                         -> Slice -> Layer -> Consumer
                Producer -> Slice -> Layer -> Consumer
                         -> Slice -> Layer -> Consumer

            into this:
                                  -> Slice -> Consumer
                Producer -> Layer -> Slice -> Consumer
                                  -> Slice -> Consumer

            Now at the place of "Layer" supported Reorder, Expand, Transpose, PermuteCast, AffineReshape and MemPermute operations.
            Most of the time, the conversion is valid in case Slice and Layer transform different axes.
            But there is no such restriction for the swap of Slice and Reorder.

        * Convert this subgraph:
                                  -> Reorder -> Consumer
                Producer -> Split -> Reorder -> Consumer
                                  -> Reorder -> Consumer

            into this:
                                             -> Consumer
                Producer -> Reorder -> Split -> Consumer
                                             -> Consumer

            in case the split axis will be in lower memory dim after Reorder.
    }];

    let constructor = "vpux::IE::createUniquifyBranchesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// TransposeToPermuteCast
//

def TransposeToPermuteCast : PassBase<"transpose-to-permute-cast", "vpux::FunctionPass"> {
    let summary = "Converts Transpose operation to PermuteCast with Reorder";

    let description = [{
        It is possible to replace a `Transpose` operation with a combination of `PermuteCast` and `Reorder`.
        To compute the permutation cast, which is required for the source tensor, one must inverse the
        affine map from the original `Transpose` operation. For example, consider the following transposition:
        `1x16x32x64 -> 1x64x16x32`, its affine map is: `(d0, d1, d2, d3) -> (d0, d3, d1, d2)`.
        The inverse will be:
        ```
            d0, d3, d1, d2   ->  d0, d1, d2, d3
            aN, aC, aH, aW   ->  aN, aH, aW, aC
        ```
        Which gives permutation cast into NHWC.
        In order to maintain the layout in data flow, `Reorder` must always rearrange `PermuteCast` result into the
        order of original `Transpose` operation.
    }];

    let constructor = "vpux::IE::createTransposeToPermuteCastPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateAffineReshape
//

def PropagateAffineReshape : PassBase<"propagate-affine-reshape", "vpux::FunctionPass"> {
    let summary = "Moves AffineReshape operation down";

    let description = [{
        Supported cases:
        * Move through Transpose
        * Move through Expand
        * Move through Concat // TODO: #-58713
            Before:
                AffineReshape ->
                AffineReshape -> Concat
                AffineReshape ->
            After:
                Concat -> AffineReshape
        * Move through Softmax
    }];

    let constructor = "vpux::IE::createPropagateAffineReshapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateShapeCast
//

def PropagateShapeCast : PassBase<"propagate-shape-cast", "vpux::FunctionPass"> {
    let summary = "Moves ops post ShapeCast operation";

    let description = [{
        Supported cases:
        * Move through Abs, Gelu, Swish, HSwish, Sigmoid, Tanh
    }];

    let constructor = "vpux::IE::createPropagateShapeCastPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateTranspose
//

def PropagateTranspose : PassBase<"propagate-transpose", "vpux::FunctionPass"> {
    let summary = "Moves Transpose operation down";

    let description = [{
        Supported cases:
        * Move through Softmax
    }];

    let constructor = "vpux::IE::createPropagateTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateFqThroughConcat
//

def PropagateFqThroughConcat : PassBase<"propagate-fq-through-concat", "vpux::FunctionPass"> {
    let summary = "Propagate FakeQuantize operation through Concat";

    let description = [{
        `ConvertPadToConcat` adds a `Concat` operation which does not propagate `FakeQuantize` operation.

        1. Check if such `Concat` operation has one and only one quantized input
        2. Fetch quantization parameters
        3. Apply them to every single `Concat` input and output
    }];

    let constructor = "vpux::IE::createPropagateFqThroughConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LoopOutliner
//

def LoopOutliner : PassBase<"loop-outliner", "vpux::ModulePass"> {
    let summary = "Extract loop body as a function and call it inside of loop";

    let description = [{
        This pass extracts a function from loop body.
        By calling it before unrolling, the compiled IR can be greatly reduced.
    }];

    let constructor = "vpux::IE::createLoopOutlinerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollTensorIterator
//

def UnrollTensorIterator : PassBase<"unroll-tensor-iterator", "vpux::FunctionPass"> {
    let summary = "Unroll tensor iterator op.";

    let description = [{
        This pass unrolls tensor iterator to realize the loop function.
        Will fill the inputs according to the attribute in each network copy.
    }];

    let constructor = "vpux::IE::createUnrollTensorIteratorPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PerAxisFQConcat
//

def PerAxisFQConcat : PassBase<"per-axis-fq-concat", "vpux::FunctionPass"> {
    let summary = "Supports Concat operation with per-axis FQ inputs";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It creates `FakeQuantize` operation, which combines per-channel quantization from `Concat` inputs,
        and places it after the `Concat` operation. For example:
        The following `Concat`:
        ```
            FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
                                  |
                                GroupConv 1x304x128x128
        ```
        will be transformed into:
        ```
            FQ 1x256x128x128 -> Concat <- FQ 1x48x128x128
                                  |
                                 FQ 1x304x128x128
                                  |
                                GroupConv 1x304x128x128
        ```
    }];

    let constructor = "vpux::IE::createPerAxisFQConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapTransposeConcat
//

def SwapTransposeConcat : PassBase<"swap-transpose-concat", "vpux::FunctionPass"> {
    let summary = "Swap Transpose and Concat operations";

    let description = [{
        Pass converts pattern from
        Transpose ->
        Transpose -> Concat
        Transpose ->

        to
        Concat -> Transpose
    }];

    let constructor = "vpux::IE::createSwapTransposeConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPowerToMult
//

def ConvertPowerToMult : PassBase<"convert-power-to-mult", "vpux::FunctionPass"> {
    let summary = "Convert power to multiply operation";

    let description = [{
        The pass converts power with single constant exponent value to multiplication.
    }];

    let constructor = "vpux::IE::createConvertPowerToMultPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertUpsamplingToStridedConcat
//

def ConvertUpsamplingToStridedConcat : PassBase<"convert-upsampling-to-strided-concat", "vpux::FunctionPass"> {
    let summary = "Convert upsampling op to strided concat op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Upsampling` operations with `Concat` operations with strides and a zero filled const.
    }];

    let constructor = "vpux::IE::createConvertUpsamplingToStridedConcatPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeOpSlice
//

def OptimizeOpSlice : PassBase<"optimize-op-slice", "vpux::FunctionPass"> {
    let summary = "Bypass concat if slice is the subtensor of one of concat inputs";

    let description = [{
        For the pattern ConcatOp->SliceOp, if SliceOp input is the subtensor of one of ConcatOp input,
        Bypass ConcatOp and ConcatOp would be removed if it has only one user.
        For the pattern Tile->(AffineReshape)->SliceOp, If Tile Axis and Slice Axis are same, and the Slice
        Axis value could be fused into the Tile Repeats, then remove Slice and update Tile Repeats attribute.
    }];

    let constructor = "vpux::IE::createOptimizeOpSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandActivationWidth
//

def ExpandActivationWidth : PassBase<"expand-activation-width", "vpux::FunctionPass"> {
    let summary = "Align input tensors shape of DPU operation with hardware requirements";

    let description = [{
        This pass processes operations, which can be compiled as DPU tasks and
        expands output width to the next number divisible by 16 when they don't
        meet hardware requirements.
        Applicable only for operations with NHWC input and NCHW output.
        For instance, consider convolution with 16x20x23 input, 16x18x21 output and 3x3 kernel.
        21 is not divisible by 16, so the output width must be expanded to 32: 16x18x32.
        In order to comply to the operation traits, input width must be expanded to 16x20x34.

        Supported operations:
        * IE.PermuteQuantize
            To avoid blocking optimization of permuteQuantize, place the pass after buildMemPermuteProcessingPipeline()
            for NPU37XX, NPU40XX.
    }];

    let constructor = "vpux::IE::createExpandActivationWidthPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// AdaptShapesForScaleShiftPass
//

def AdaptShapesForScaleShiftPass : PassBase<"adapt-shapes-for-scale-shift", "vpux::FunctionPass"> {
    let summary = "Adjusts 2-d and 3-d `IE.Add` and `IE.Multiply` for further conversion to `IE.ScaleShift`";

    let description = [{
        Converts these subgraphs:
        ```
            Input tensor<NxM> => IE.Add : tensor<NxM>, tensor<1xM> -> tensor<NxM>
            Input tensor<NxM> => IE.Multiply : tensor<NxM>, tensor<1xM> -> tensor<NxM>
            Input tensor<1xNxM> => IE.Add : tensor<1xNxM>, tensor<1x1xM> -> tensor<1xNxM>
            Input tensor<1xNxM> => IE.Multiply : tensor<1xNxM>, tensor<1x1xM> -> tensor<1xNxM>
        ```
        Into the following subgraphs respectively:
        ```
            Input tensor<NxM> => IE.Add : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<NxM>
            Input tensor<NxM> => IE.Multiply : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<NxM>
            Input tensor<1xNxM> => IE.Add : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<1xNxM>
            Input tensor<1xNxM> => IE.Multiply : tensor<1xMxNx1>, tensor<1xMx1x1> => tensor<1xNxM>
        ```
        The following shape transformations will be applied for 2-d case:
        ```
            Input NxM => Reshape 1xNxMx1 => Transpose 1xMxNx1 => Add => Transpose 1xNxMx1 => Reshape NxM
        ```
        For 3-d case:
        ```
            Input 1xNxM => Reshape 1xNxMx1 => Transpose 1xMxNx1 => Add => Transpose 1xNxMx1 => Reshape 1xNxM
        ```
        It is also possible to apply reshape to get `IE.Add : tensor<NxMx1x1>, tensor<1xMx1x1>`.
        However, such approach may lead to a big cluster of NCE tasks after `UnrollBatch` pass.
        The measurements show that transposition is more effective for this pass.
    }];

    let constructor = "vpux::IE::createAdaptShapesForScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustInputShape
//

def AdjustInputShape: PassBase<"adjust-input-shape", "vpux::FunctionPass"> {
    let summary = "Reshape the activation inputs to reduce channel alignment cost";

    let description = [{
        `IE.ExpandOp` is used to guarantee the channel alignment requirement for NCE ops. And this PR will try
        to reshape the nce ops to reduce the expand op's cost
        1. Insert ShapeCast ops to the inputs of eltwise ops when the input channels require alignment.
           Insert ShapeCast ops to the outputs before the next non-eltwise op.
        2. Insert ShapeCast op and Expand op to the input of pooling ops when the pooling has KY=1, SY=1.
           Insert ShapeCast op, Slice Op and Expand op to the outputs.
        Platforms NPU37XX, NPU40XX all benefit.
    }];

    let constructor = "vpux::IE::createAdjustInputShapePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MovePermutePostEltwise
//

def MovePermutePostEltwise : PassBase<"move-permute-post-eltwise", "vpux::FunctionPass"> {
    let summary = "Move the input Permute ops post Eltwise to reduce the number of Permute ops";

    let description = [{
        The layout does not matter for eltwise ops as long as the input and output layouts are the same.
        move the permute ops from the inputs of the eltwise to the output to reduce the number of permute ops.
    }];

    let constructor = "vpux::IE::createMovePermutePostEltwisePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// NormalizeL2Fusion
//

def NormalizeL2Fusion : PassBase<"normalizeL2-fusion", "vpux::FunctionPass"> {
    let summary = "Convert a subgraph to normalizeL2";

    let description = [{
        Convert this subgraph

            |
          /   \
         |     |
         |  ReduceL2
         |     |
         |   Clamp
          \   /
          Divide
            |
        to a single normalizeL2Op
    }];

    let constructor = "vpux::IE::createNormalizeL2FusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertExtractImagePatches
//

def ConvertExtractImagePatches : PassBase<"convert-extract-image-patches", "vpux::FunctionPass"> {
    let summary = "Converts subgraphs around ExtractImagePatches into some more optimal for VPU ones when the necessary conditions are met";

    let description = [{
        Converts these subgraphs:
        ```
            IE.ReduceSum -> IE.ExtractImagePatches -> IE.Transpose -> IE.ReduceSum
            IE.ReduceSum -> IE.ExtractImagePatches -> IE.ReduceSum
            IE.ExtractImagePatches -> IE.Transpose -> IE.AffineReshape
            IE.ExtractImagePatches -> IE.Transpose
            IE.ExtractImagePatches
        ```
        Into the following subgraphs respectively:
        ```
            IE.ReduceSum -> IE.Unsqueeze
            IE.ReduceSum -> IE.Unsqueeze
            N x IE.Slice -> IE.Concat
            N x IE.Slice -> IE.Concat -> IE.AffineReshape
            IE.Transpose
        ```
    }];

    let constructor = "vpux::IE::createConvertExtractImagePatchesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// BroadcastInputForAdd
//

def BroadcastInputForAdd : PassBase<"broadcast-input-for-add", "vpux::FunctionPass"> {
    let summary = "Broadcast input for Add op";

    let description = [{
        This pass broadcast input for AddOp when the input1's shape isn't equal to input2's shape which cannot convert to ScaleShift.
    }];

    let constructor = "vpux::IE::createBroadcastInputForAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// BroadcastInputForMultiply
//

def BroadcastInputForMultiply : PassBase<"broadcast-input-for-multiply", "vpux::FunctionPass"> {
    let summary = "Broadcast input for Multiply op";

    let description = [{
        This pass broadcasts input for MultiplyOp to execute the layer on DPU.
    }];

    let constructor = "vpux::IE::createBroadcastInputForMultiplyPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapMVNWithTranspose
//

def SwapMVNWithTranspose : PassBase<"swap-mvn-with-transpose", "vpux::FunctionPass"> {
    let summary = "Swaps MVN operation with parent Transpose";

    let description = [{
        The pass is a part of `HardwareMode` pipeline.

        It swaps `MVN` with Transpose operation when possible.
        This transormation reduces the number of `MemPermute` operations in resulting graph.
    }];

    let constructor = "vpux::IE::createSwapMVNWithTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


def AlignScales : PassBase<"align-scales", "vpux::FunctionPass"> {
    let summary = "Align FQ ranges around Concat layers.";

    let description = [{
        This pass aligns FQ ranges around Concat ops and inserts Clamp ops to keep the valuse in the original ranges.

        Original subgraph:
        Conv1                 ARG
          |                    |
        FQ1(range1)     FQ2(range2)
                      |
                    Concat
                      |
                    Conv2

        New subgraph:
        Conv1                                               ARG
          |                                                  |
        FQ(newRange)                                        FQ(newRange)
          |                                                  |
        Clamp(low and hight from original FQ)               Clamp(low and hight from original FQ)
                                                    |
                                                  Concat
                                                    |
                                                  Conv2
    }];


    let constructor = "vpux::IE::createAlignScalesPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "seOpsEnabled", "se-ops-enabled",
            "bool", "false",
            "Flag which identifies whether an operation can be executed using the Storage Element hardware feature"
        >
    ];
}

//
// ConvertReorderToPermuteQuantize
//

def ConvertReorderToPermuteQuantize : PassBase<"convert-reorder-to-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts IE.Reorder to DPU permute";

    let description = [{
        Converts IE.Reorder with float16 input and float16 output to DPU permute.
    }];

    let constructor = "vpux::IE::createConvertReorderToPermuteQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateMemPermuteThroughAdd
//

def PropagateMemPermuteThroughAdd : PassBase<"propagate-mem-permute-through-add", "vpux::FunctionPass"> {
    let summary = "Propagates IE.MemPermute through IE.Add when both inputs of IE.Add have IE.MemPermute";

    let description = [{
        Propagates last permute in the chain. Converts this subgraph
        ```
            IE.MemPermute -> IE.ShapeCast \
                                           IE.Add -> IE.ShapeCast -> IE.MemPermute
            IE.MemPermute -> IE.ShapeCast /
        ```
        into
        ```
            IE.MemPermute -> IE.MemPermute -> IE.ShapeCast \
                                                            IE.Add -> IE.ShapeCast
            IE.MemPermute -> IE.MemPermute -> IE.ShapeCast /
        ```
        Canonicalization may simplify this when IE.MemPermute operations cancel one another
    }];

    let constructor = "vpux::IE::createPropagateMemPermuteThroughAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustMemPermuteAroundOp
//

def AdjustMemPermuteAroundOp : PassBase<"adjust-mem-permute-around-op", "vpux::FunctionPass"> {
    let summary = "Adjust MemPermuteOps around to reduce number of real permutes";

    let description = [{
        Adjust MemPermuteOps around an operation to avoid unnecessary permutes
        For example, a possible conversion from
        ```
            IE.LayerOp -> IE.MemPermute \
                                         IE.Eltwise -> IE.MemPermute -> IE.LayerOp
                             IE.LayerOp /
        ```
        to
        ```
                              IE.LayerOp \
                                          IE.Eltwise -> IE.LayerOp
            IE.LayerOp -> IE.PermuteCast /
        ```
    }];

    let constructor = "vpux::IE::createAdjustMemPermuteAroundOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseMemPermutePass
//

def FuseMemPermutePass : PassBase<"fuse-mem-permute", "vpux::FunctionPass"> {
    let summary = "Fuses IE.MemPermute to previous NCE task as ODU permutation";

    let description = [{
        Converts these subgraphs:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> IE.MemPermute [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> IE.MemPermute [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> IE.MemPermute [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> IE.MemPermute [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> IE.MemPermute [NCHW]
        ```
        Into the following subgraphs respectively:
        ```
            Input [NHWC] -> IE.Convolution [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NCHW]
            Input [NHWC] -> IE.MaxPool [NCHW]
            Input [NHWC] -> IE.AvgPool [NCHW]
            Input [NHWC] -> IE.Add [NCHW]
        ```
    }];

    let constructor = "vpux::IE::createFuseMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// HandleLargePads
//

def HandleLargePads : PassBase<"handle-large-pads", "vpux::FunctionPass"> {
    let summary = "Handle operations with large pads";

    let description = [{
        This pass handle operations with pad larger than supported on hardware.
        It will move the pad from the layer parameter to it's input. for example,
        if one conv's top pad is 5, but what HW support is 2, we will set conv's pad
        to 2, and concat  the input with 3 line zero constant.
    }];

    let constructor = "vpux::IE::createHandleLargePadsPass()";

}

//
// PropagateMemPermuteBeforeOp
//

def PropagateMemPermuteBeforeOp : PassBase<"propagate-mem-permute-before-op", "vpux::FunctionPass"> {
    let summary = "Propagate IE.MemPermute through concrete op";

    let description = [{
        Propagates permute through concrete op.
        1. Converts this subgraph
        ```
            IE.AffineReshape -> IE.MemPermute
        ```
        into
        ```
            [IE.PermuteCast] -> IE.MemPermute -> IE.Reshape -> IE.PermuteCast
        ```
        2. Converts this subgraph
        ```
            IE.MVN -> IE.MemPermute
        ```
        into
        ```
            IE.MemPermute -> IE.MVN-> IE.PermuteCast
        ```

        Canonicalization may fuse IE.MemPermute with another one.
    }];

    let constructor = "vpux::IE::createPropagateMemPermuteBeforeOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeConcatWithConv
//

def OptimizeConcatWithConv : PassBase<"optimize-concat-with-conv", "vpux::FunctionPass"> {
    let summary = "Optimize IE.Concat with IE.Convolution op";

    let description = [{
        Optimize IE.Concat with IE.Convolution if the IE.Concat meets the following conditions:
        1. Layout is NCHW;
        2. There are two inputs with same shape like [1, C, 1, 1];
        3. Concat Axis is H;
        For example:
            Input0[1,HWC,1,1]\
                              Concat[1, HWC, 2, 1]
            Input1[1,HWC,1,1]/
        Converts to
            Input0[1,HWC,1,1]->Reshape[1,C,H,W]->LayoutCast[1,C,H,W]#NHWC\
                                                                         Conv[1,2C,H,W]#NHWC->LayoutCast[1,2C,H,W]->Reshape[1,HWC,2,1]
            Input0[1,HWC,1,1]->Reshape[1,C,H,W]->LayoutCast[1,C,H,W]#NHWC/

        The Convoluation has weight[2C, C, H+1, 1] with Pad[0, 0, 0, 0] and Strides[1, 1]
        The weights values are filled as follows:
                        0                 1          ...        C-1
        OC0:      [1, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
        OC1:      [0, 0,..., 0, 1], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
        OC2:      [0, 0,..., 0, 0], [1, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
        OC3:      [0, 0,..., 0, 0], [0, 0,..., 0, 1], ...,  [0, 0, ..., 0, 0]
        ...
        OC(2C-2): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [1, 0, ..., 0, 0]
        OC(2C-1): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 1]
    }];

    let constructor = "vpux::IE::createOptimizeConcatWithConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateMemPermuteThroughSoftMax
//

def PropagateMemPermuteThroughSoftMax : PassBase<"propagate-mem-permute-through-softmax", "vpux::FunctionPass"> {
    let summary = "Propagate IE.MemPermute through IE.SoftMaxOp";

    let description = [{
        Propagates permute through SoftMaxOp. Converts this subgraph
        ```
            IE.SoftMaxOp -> IE.MemPermute
        ```
        into
        ```
            IE.MemPermute -> IE.SoftMaxOp
        ```
        Canonicalization may fuse IE.MemPermute with another one.
    }];

    let constructor = "vpux::IE::createPropagateMemPermuteThroughSoftMaxPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeIdentityPool
//

def OptimizeIdentityPool : PassBase<"optimize-identity-pools", "vpux::FunctionPass"> {
    let summary = "Optimize identiy pools";

    let description = [{
        The pass removes the identity ops
        Because we have passes which introduce such identity pools, we can't have this
        as a folder/canonicalizer
        The pass fuse the Conv without postOp and the identity AvgPool with postOp
    }];

    let constructor = "vpux::IE::createOptimizeIdentityPoolPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertExpandToConvPass
//

def ConvertExpandToConvPass : PassBase<"convert-expand-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert IE.Expand to IE.Reshape -> IE.Convolution -> IE.Reshape";

    let description = [{
        Replace NHWC IE.Expand with IE.Convolution.
        1. Reshape input from [1, IC, H, W] to [1, IC * 16, H, W / 16]
        For example 1x3x480x640 becomes 1x48x480x40
        2. Compose IE.Convolution that has OC = (IC + padIC) * 16.
        Padded channels must be multiplied by zero.
        Activation channels must be multiplied by 1.
        3. Reshape output back to original shape [1, IC + padIC, H, W]
        For example when padIC = 13, OC = (3 + 13) * 16 = 256
        When padIC = 1, OC = (3 + 1) * 16 = 64
        Weights structure goes like this for 3 input channels and 4 output channels:
        | idx   |    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |  ... |   46 |   47 |
        | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
        | OC 0  |    1 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 1  |    0 |    1 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 2  |    0 |    0 |    1 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 3  |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 4  |    0 |    0 |    0 |    1 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 5  |    0 |    0 |    0 |    0 |    1 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 6  |    0 |    0 |    0 |    0 |    0 |    1 |    0 |    0 |  ... |    0 |    0 |
        | OC 7  |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
        | OC 8  |    0 |    0 |    0 |    0 |    0 |    0 |    1 |    0 |  ... |    0 |    0 |
        | OC 9  |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    1 |  ... |    0 |    0 |
        | ...   |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |  ... |
        | OC 62 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    1 |
        | OC 63 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |  ... |    0 |    0 |
    }];

    let constructor = "vpux::IE::createConvertExpandToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustScaleShiftForDWConv
//

def AdjustScaleShiftForDWConv : PassBase<"adjust-scale-shift-for-dw-conv", "vpux::FunctionPass"> {
    let summary = "Adjust ScaleShift for DW Convolution";

    let description = [{
        Adjust input N > 1 scaleShift to N = 1 by broadcast and reshape,
        for preventing the generation of the large number of DW convolution fragments.

        If the activation is Const, will adjust ScaleShift regardless of the size of N.

        If the activation is not Const, will adjust ScaleShift only when N > 16 (rough
        estimates obtained from experiments).
    }];

    let constructor = "vpux::IE::createAdjustScaleShiftForDWConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// RemoveViewLikeOpsChainPass
//

def RemoveViewLikeOpsChainPass : PassBase<"remove-view-like-ops-chain", "vpux::FunctionPass"> {
    let summary = "Remove View-Like ops chain if first view-like op's input is same as the last view like op's output";

    let description = [{
        Remove View-Like ops chain if first view-like op's input is same as the last view like op's output.

        Converts subgraph like:
        ```
        LayerOp -> PermuteCastOp1 -> AffineReshape -> PermuteCastOp2 -> LayerOp
        ```
        Into
        ```
        LayerOp -> LayerOp
        ```
        if PermuteCastOp1's input == PermuteCastOp2's output.
    }];

    let constructor = "vpux::IE::createRemoveViewLikeOpsChainPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LogOpOptimizations
//

def LogOpOptimizations : PassBase<"log-op-optimizations", "vpux::FunctionPass"> {
    let summary = "Identifies operations that can be optimized into the pass' logs";

    let description = [{
        Iterates over all operations in the IR and identifies those that can be optimized.
        At the moment, only operations that can be optimized using the Storage Element Pointer
        feature are identified.

        The identified operations are mentioned in the logs of the pass. In case the
        operations should already be optimized later in compilation, this will also
        be mentioned in the logs.
    }];

    let constructor = "vpux::IE::createLogOpOptimizationsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertMemPermuteToPoolPass
//

def ConvertMemPermuteToPoolPass : PassBase<"convert-mem-permute-to-pool", "vpux::FunctionPass"> {
    let summary = "Converts an `IE.MemPermute` operation to an `IE.MaxPool` operation";

    let description = [{
        Replace an `IE.MemPermute` operation with a chain of
        ```
            IE.ShapeCast (reinterpret mem_perm) -> IE.LayoutCast (to NHWC) -> IE.MaxPool (ODU permute) -> IE.LayoutCast (to dst_order) -> IE.ShapeCast
        ```
        Input IE.ShapeCast swaps the dimensions of the initial IE.MemPermute.
        Input IE.LayoutCast overrides the layout of an input with NHWC (required by NCE tasks).
        Output IE.LayoutCast is needed since dst_order may differ from the ODU permute order.
        Output IE.ShapeCast restores the shape of the original IE.MemPermute output.

        Only partial support is available:
        * IE.MemPermute (NCHW input, NCHW dst_order, NHCW mem_perm)
        * IE.MemPermute (NCHW input, NCHW dst_order, NHWC mem_perm)
        * IE.MemPermute (NHCW input, NCHW dst_order, NHCW mem_perm)
        * IE.MemPermute (NCWH input, NHWC dst_order, NWHC mem_perm)
    }];

    let constructor = "vpux::IE::createConvertMemPermuteToPoolPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AdjustNonZeroFakeQuant
//

def AdjustNonZeroFakeQuant : PassBase<"adjust-non-zero-fake-quant", "vpux::FunctionPass"> {
    let summary = "Adjust NonZero FakeQuant to have 0";

    let description = [{
        For FakeQuant Op, HW only support zero point between [0, 255], so if quantize range do not
        contain 0, the zero point is out of requirement, the pass will try to adjust the range. For
        example: the min/max range adjust from [0.1, 1] to [0, 1]
    }];

    let constructor = "vpux::IE::createAdjustNonZeroFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// FuseConvWithSlice
//

def FuseConvWithSlice : PassBase<"fuse-conv-with-slice", "vpux::FunctionPass"> {
    let summary = "Fuse conv with slice";

    let description = [{
        The pass convert IE.ConvolutionOp + IE.SliceOp To IE.ConvolutionOp in case that slice only
        happened in OC axis. For example:
        Input(1x16x8x8)    Filter (64x16x1x1)         Input(1x16x8x8)      Filter(32x16x1x1)
                \            /                                 \               /
                 Convolution(1x64x8x8)         To              Convolution(1x32x8x8)
                       |
                     Slice(1x32x8x8)
        Input(1x16x8x8)      Filter (64x16x1x1)
          \                   /
           Convolution(1x64x8x8)
               |             |
        Slice(1x32x8x8)    Slice(1x32x8x8)
        To
        Filter(32x16x1x1)            Input(1x16x8x8)         Filter(32x16x1x1)
                 \                 /        \               /
               Convolution(1x32x8x8)       Convolution(1x32x8x8)
    }];

    let constructor = "vpux::IE::createFuseConvWithSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwishFusion
//

def SwishFusion : PassBase<"swish-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse Sigmoid and Multiply to a Swish";

    let description = [{
        This pass fuse Sigmoid and Multiply to a Swish.

    }];

    let constructor = "vpux::IE::createSwishFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MVNFusion
//

def MVNFusion : PassBase<"mvn-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse some primary ops to a MVN";

    let description = [{
        This pass fuse ops to MVN following the equation below:
        ```
        (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2)) + eps)
        ```
        or
        ```
        (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean(x^2, axes) - (ReduceMean(x, axes) ^ 2) + eps))
        ```
    }];

    let constructor = "vpux::IE::createMVNFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MergeFullyConnected
//

def MergeFullyConnected : PassBase<"merge-fully-connected", "vpux::FunctionPass"> {
    let summary = "The pass merges the unrolled fullyConnected layers.";

    let description = [{
        For FullyConnected which has first dim = 1, e.g. KV cached Matmul, the performance bottleneck is caused by too much small DMA ops instead of compute ops.
        So this pass will try to merge the unrolled FullyConnected layers to improve performance by using more data calculation and less DMA ops.
    }];

    let constructor = "vpux::IE::createMergeFullyConnectedPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollGroupQuantize
//

def UnrollGroupQuantize : PassBase<"unroll-group-quantize", "vpux::FunctionPass"> {
    let summary = "The pass splits quantize operation with multiple axes into a group of quantize operations with one axis";

    let description = [{
        Let's suppose a topology features the following IE.FakeQuantize operation:
        ```
            %data: tensor<1x2x128x768xf32>
            %in_low = const.Declare tensor<1x2x1x768xf32>
            %in_high = const.Declare tensor<1x2x1x768xf32>
            %out_low = const.Declare tensor<1x2x1x768xf32>
            %out_high = const.Declare tensor<1x2x1x768xf32>
            %result = IE.FakeQuantize(%data, %in_low, %in_high, %out_low, %out_high) -> tensor<1x2x128x768xf32>
        ```
        Many compiler passes expect IE.FakeQuantize operations to have only one axis.
        This limitation arises because the scale table cannot fit a tensor with an arbitrary rank, as it only stores vectors.
        The operation must be adapted for further passes by unrolling the outermost axis.
        ```
            %data: tensor<1x2x128x768xf32>
            // First group:
            %data_0 = IE.Split(%data) -> tensor<1x1x128x768xf32>
            %i_lo_0 = const.Declare tensor<1x1x1x768xf32>
            %i_hi_0 = const.Declare tensor<1x1x1x768xf32>
            %o_lo_0 = const.Declare tensor<1x1x1x768xf32>
            %o_hi_0 = const.Declare tensor<1x1x1x768xf32>
            %fq_0 = IE.FakeQuantize(%data_0, %i_lo_0, %i_hi_0, %o_lo_0, %o_hi_0) -> tensor<1x1x128x768xf32>
            // Second group:
            %data_1 = IE.Split(%data) -> tensor<1x1x128x768xf32>
            %i_lo_1 = const.Declare tensor<1x1x1x768xf32>
            %i_hi_1 = const.Declare tensor<1x1x1x768xf32>
            %o_lo_1 = const.Declare tensor<1x1x1x768xf32>
            %o_hi_1 = const.Declare tensor<1x1x1x768xf32>
            %fq_1 = IE.FakeQuantize(%data_1, %i_lo_1, %i_hi_1, %o_lo_1, %o_hi_1) -> tensor<1x1x128x768xf32>
            %result = IE.Concat(%fq_0, %fq_1) -> tensor<1x2x128x768xf32>
        ```
    }];

    let constructor = "vpux::IE::createUnrollGroupQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollFullyConnected
//

def UnrollFullyConnected : PassBase<"unroll-fully-connected", "vpux::FunctionPass"> {
    let summary = "Split `IE.FullyConnected` with multiple `IE.FakeQuantize` by columns";

    let description = [{
        This transformation takes the following subgraph:
        ```
            IE.FQ [1, H, W]  IE.FQ [1, H, W] ... IE.FQ [1, H, W]
                           \    |               /
                            IE.Concat [G, H, W]
                                |
                            IE.AffineReshape [G * H, W]
                                |
                            IE.Transpose [W, G * H]
                                |
            Input [N, G * H] -> IE.FullyConnected -> [N, W]
        ```
        and splits the matrix multiplication by the number of groups as follows:
        ```
                                                       Input [N, G * H] -> Split - ... ----
                                                                             |            |
                                                                           [N, H]  ...  [N, H]
                                                                             |            |
            IE.FQ [1, H, W] -> IE.Reshape [H, W] -> IE.Transpose [W, H] -> IE.FC   ...  IE.FC
                                                                               \        /
                                                                             IE.Accumulate ([N, W], [N, W])
        ```
        1. Match `IE.FakeQuantize -> IE.Concat -> IE.AffineReshape -> IE.Transpose -> IE.FullyConnected` pattern.
        2. Reshape all `IE.FakeQuantize` operations from 1xHxW to HxW.
        3. Split `Input` by the number of groups (G).
        4. Build a `IE.MatMul` for each group using splits from step 3 and reshape results from step 2.
        5. Accumulate `IE.MatMul` results with `IE.Accumulate` operations.

        Motivation: this transformation eliminates `IE.Concat` from the subgraph.
        Such concat is problematic because its axis (d0) is different from the quantization axis (d2).
        `PropagateQuantizeDequantize` pass cannot handle the case when the axes differ.
        Therefore, a dequantize cannot be fused into a convolution.
        Without the `IE.Concat`, the fusion is possible.
    }];

    let constructor = "vpux::IE::createUnrollFullyConnectedPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "accumulateMatmulWithDPU", "accumulate-matmul-with-dpu",
            "bool", "false",
            "Flag to toggle unrolled Matmul results accumulation with DPU or SHAVE"
        >
    ];
}

//
// M2IBatchNormFusion
//

def M2IBatchNormFusion : PassBase<"m2i-batchnorm-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse a Mult + Add operation that follows or precedes an interpolate/CSC operation mappable to m2i into a IE.BatchNormInferenceOp operation, to be also mapped on m2i";

    let description = [{
        BatchNormInferenceOp is always decomposed into a Mult + Add operations by prepostprocessing API and later mapped to either DPU or Shave.
        Since M2I is able to compute an fp16 normalisation operation, in case the Mult + Add pair are preceded or followed by another m2i-mappable operation such as Interpolate or CSC,
        the two operations get fused back into a BatchNormInferenceOp, which is then mapped to m2i and lowered to VPU::M2INormOp.
        This then allows to fuse CSC/Interp/Permute/Norm patterns into a single M2ITask operation
    }];

    let constructor = "vpux::IE::createM2IBatchNormFusionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseScalesToAccumulate
//

def FuseScalesToAccumulate : PassBase<"fuse-scales-to-accumulate", "vpux::FunctionPass"> {
    let summary = "Fetch scales from `IE.FullyConnected` weights and fuse them into `IE.Accumulate`";

    let description = [{
        This transformation takes the following subgraph:
        ```
                                                                          producer ... producer
            scale = 2                                                        |            |          scale = 3
            IE.FQ [1, H, W] -> IE.Reshape [H, W] -> IE.Transpose [W, H] -> IE.FC   ...  IE.FC <- ... IE.FQ
                                                                               \       /
                                                                             IE.Accumulate ([N, W], [N, W], scale = 1, scale = 1)
        ```
        fetches the scales from the `IE.FakeQuantize` operation and fuses them into `IE.Accumulate`
        ```
                                                                          producer ... producer
            scale = 1                                                        |            |          scale = 1
            IE.FQ [1, H, W] -> IE.Reshape [H, W] -> IE.Transpose [W, H] -> IE.FC   ...  IE.FC <- ... IE.FQ
                                                                               \       /
                                                                             IE.Accumulate ([N, W], [N, W], scale = 2, scale = 3)
        ```
        1. Match `IE.FakeQuantize -> IE.Reshape -> IE.Transpose -> IE.MatMul -> IE.Accumulate` pattern.
        2. Fetch scales from `IE.FakeQuantize`
        3. Reset the scales of `IE.FakeQuantize` with 1
        4. Set scales of `IE.Accumulate` with the values fetched from `IE.FakeQuantize`

        Motivation: this transformation is necessary for runtime weight table population.
        The activation shave that populates the weight table does not set scales.
        The PPE of the convolution is configured to use per-tensor scale value equal to 1.
        `IE.Accumulate` must apply scales after the matrix multiplication to get accurate results.
    }];

    let constructor = "vpux::IE::createFuseScalesToAccumulatePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def MoveMultiplyPostOp : PassBase<"move-multiply-post-op", "vpux::FunctionPass"> {
    let summary = "Move`IE.Multiply` post op";

    let description = [{
        This transformation takes the following subgraph like move muliply post matmul:
        ```
                                        (1x32x1024x80)      (1x1x1x1)
                                            \               /
                        (1x32x1x80)   IE.Multiply (1x32x1024x80)
                            \            /
                            IE.Matmul(1x32x1x1024)
        ```
        To:
        ```
                        (1x32x1x80)    1x32x1024x80)
                            \            /
                            IE.Matmul(1x32x1x1024)     (1x1x1x1)
                                    \                   /
                                    IE.Multiply (1x32x1x1024)
        ```
        If matmul output size is smaller than input, after swap multiply with matmul, we will
        reduce the computation of multiply. For above case, 80 times saving.
        The pass also did some other optimization like move multiply post gather or concat for
        llm optimization.
    }];

    let constructor = "vpux::IE::createMoveMultiplyPostOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapD2SAndScaleShift
//

def SwapD2SAndScaleShift : PassBase<"swap-d2s-and-scale-shift", "vpux::FunctionPass"> {
    let summary = "Swap Depth2Space and ScaleShift";

    let description = [{
        Swap Depth2Space and ScaleShift in order to fuse ScaleShift to the parent op of Depth2Space if possible.
        This pass has a dependency to ConvertDepth2SpaceToTransposedConv pass,
        only those ops are not converted to TransposedConvolution will be propagated.
    }];

    let constructor = "vpux::IE::createSwapD2SAndScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertBranchesConcatToConv
//

def ConvertBranchesConcatToConv : PassBase<"convert-branches-concat-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert branches concatenation to Convolution";

    let description = [{
        1. OptimizeGroupConvConcat pattern
        Below pattern is not performant when concat inputs on channel dimension.

        Root(1x16x144x144)             Weights(16x1x3x3)
       /               \                  /
      |                GroupConv(1x16x144x144)
       \                    /
        Concat(1x32x144x144)

        The pattern creates a new Convolution to replace the pattern to avoid concat on the inner most
        dimension like below:

        Root(1x16x144x144)             Weights(32x1x3x3)
                    \                     /
                    Convolution(1x32x144x144)

        2. OptimizeConvConcat pattern
        Below pattern is not performant when concat inputs on channel dimension and all the parameters of convs are same except weights.

            Weights(3x16x1x1)   Root(1x16x144x144)      Weights(1x16x1x1)
                    \          /               \         /
                    Conv(1x3x144x144)        Conv(1x1x144x144)
                                \                /
                                Concat(1x4x144x144)

            The pattern creates a new Convolution to replace the pattern to avoid concat on the inner most
            dimension like below:

                Root(1x16x144x144)       Weights(4x16x1x1)
                    \                     /
                    Convolution(1x4x144x144)

        3. OptimizeSliceMultiplyConcat
        Create a new Convolution to replace below pattern to avoid Slice and Concat.
                Root(1x24x1x64)
                /               \
        Slice(1x24x1x32)    Slice(1x24x1x32)    Constant(1x1x1x1)
                |                   \               /
                |                   Multiply(1x24x1x32)
                \                   /
                Concat(1x24x1x64)

        Above patten is converted to:
            Root(1x24x1x64)
                    |
        PermuteCast(1x64x24x1@NHWC)     Weights(64x64x1x1)
                        \               /
                        Convolution(1x64x24x1@NHWC)
                                |
                            PermuteCast(1x24x1x64)
    }];

    let constructor = "vpux::IE::createConvertBranchesConcatToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeAvgPoolWithUnalignedChannels
//

def OptimizeAvgPoolWithUnalignedChannels : PassBase<"optimize-avg-pool-with-unaligned-channels", "vpux::FunctionPass"> {
    let summary = "Optimize Avg Pool with unaligned channels";

    let description = [{
        This pass converts IE.AvgPool with unaligned channels to IE.Convolution operation.
        Then adjust-convolution-shape pass can adjust shape for it to avoid expand & slice.

        Take the IE.AvgPool with below configuration as an example:
        - input shape [1, 3, 640, 640]
        - output shape [1, 3, 640, 640]
        - filter shape [2, 2]
        - strides [2, 2]

        This pass will create a new IE.Convolution with filter shape [3, 3, 2, 2].
        Some filter constant values will be assigned to 1 / ( KX * KY) to perform average pooling.
        Weights structure goes like below, value A represents 1 / ( KX * KY)
        | idx   |    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |    9 |   10 |   11 |
        | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
        | OC 0  |    A |    A |    A |    A |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |
        | OC 1  |    0 |    0 |    0 |    0 |    A |    A |    A |    A |    0 |    0 |    0 |    0 |
        | OC 2  |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    0 |    A |    A |    A |    A |

        In the end, the new IE.Convolution can be optimized by adjust-convolution-shape pass.
    }];

    let constructor = "vpux::IE::createOptimizeAvgPoolWithUnalignedChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def ReshapeMatMulInputs : PassBase<"reshape-matmul-inputs", "vpux::FunctionPass"> {
    let summary = "Reshape inputs of `IE.MatMul`";

    let description = [{
        1. Unsqueeze 1d inputs to 2d.
        2. Insert `IE.Transpose` if transpose_a is set. Transform
        ```
            IE.MatMul(16x32, 16x64) {transpose_a = true}
        ```
        into
        ```
            IE.Transpose(16x32 -> 32x16) -> IE.MatMul(16x32, 16x64) {transpose_a = false}
        ```
        3. Insert `IE.Transpose` if transpose_b is not set.
        ```
            IE.MatMul(16x32, 32x64) {transpose_b = false}
        ```
        into
        ```
            IE.Transpose(32x64 -> 64x32) -> IE.MatMul(16x32, 64x32) {transpose_b = true}
        ```
        4. Fold batch dimensions. 2x3x4x5x6 becomes 1x24x5x6.
        5. Collapse batch dimension into rows when the second input has 2d shape.
        ```
            IE.MatMul(1x4x32x64, 64x32) {transpose_b = true}
        ```
        becomes
        ```
            IE.Reshape(1x4x32x64 -> 128x64) -> IE.FullyConnected(128x64, 64x32)
        ```
    }];

    let constructor = "vpux::IE::createReshapeMatMulInputsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableGroupedMatMul", "enable-grouped-matmul",
            "bool", "false",
            "Flag to enable or disable grouped MatMul execution"
        >
    ];
}


def MergeTileWithSlice : PassBase<"merge-tile-with-slice", "vpux::FunctionPass"> {
    let summary = "Merge `IE.Tile` with `IE.Slice`";

    let description = [{
        This transformation takes the following subgraph:
        ```
                                Input(2x1x10x80)
                                        |
                                IE.Tile(2x2x10x80)
                                        |
                                IE.Reshape(1x4x10x80)
                                        |
             -----------------------------------------------------------
             |                   |                   |                 |
        IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)

        ```
        To:
        ```
                                Input(2x1x10x80)
                                        |
             -----------------------------------------------------------
             |                   |                   |                 |
        IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)
        ```
        The optimization is used for LLM GQA.
    }];

    let constructor = "vpux::IE::createMergeTileWithSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


def MergeParallelFullyConnected : PassBase<"merge-parallel-fully-connected", "vpux::FunctionPass"> {
    let summary = "Merge parallel 'IE.FullyConnectedOp'";

    let description = [{
        This transformation takes the following subgraph:
        ```
        cst_set1(2x3x2,1x1x1,1x1x1,2x1x2,2x1x2)    cst_set2(2x3x3,1x1x1,1x1x1,2x1x3,2x1x3)
                  \     |     |    /     /                   \     |    |     /    /
                IE.FakeQuantize (2x3x2)                 IE.FakeQuantize (2x3x3)
                        |                                        |
                IE.AffineReshape(6x2)                   IE.AffineReshape(6x3)
                        |                                        |
                IE.Transpose(2x6)                       IE.Transpose(3x6)
                       \                                   /
                        \         Input(1x6)              /
                         \       /           \           /
                IE.FullyConnected(1x2)    IE.FullyConnected(1x3)
        ```
        To:
        ```
        cst_set_concat(2x3x5,1x1x1,1x1x1,2x1x5,2x1x5)
                         \      |    |     /     /
                        IE.FakeQuantize (2x3x5)
                                |
                        IE.AffineReshape(6x5)
                                |
                        IE.Transpose(5x6)       Input(1x6)
                                \               /
                                 \             /
                              IE.FullyConnected(1x5)
                                /              \
                        IE.SLice(1x2)     IE.slice(1x3)
        ```
        The optimization is used for LLM GPTQ model. GPTQ MatMul is not efficient in LNL,
        especially for KV cache model, due to so many small ops increase the runtime idle
        time. Matmul will convert to fully connected layer. Assume originally two parallel
        MatMul unroll to 64 small MatMul, after merge, we only 32 MatMul, op reduce will
        benefit to runtime idle. Meanwhile, we increase the DMA copy size(reduce DMA copy number),
        which will also benefit to performance.
    }];

    let constructor = "vpux::IE::createMergeParallelFullyConnectedPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ReshapeMaxPool
//

def ReshapeMaxPool : PassBase<"reshape-max-pool", "vpux::FunctionPass"> {
    let summary = "Reshapes MaxPool op with large input";

    let description = [{
        The pass is an optimization which reshapes MaxPool op with input having channel larger than VPU_DIMENSION_LIMIT.
        Ex: MaxPool(1x42840x14x1) => MaxPool(1x6x7140x14)
        If channel % 16 = 0 then the pass will find a divisor that leads to aligned channels.
        Ex: MaxPool(1x46560x14x1) => MaxPool(1x16x2910x14)
    }];

    let constructor = "vpux::IE::createReshapeMaxPoolPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseOutstandingQuant
//

def FuseOutstandingQuant: PassBase<"fuse-outstanding-quant", "vpux::FunctionPass"> {
    let summary = "Fuse outstanding quantize before two-input Eltwise task";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass walks through quant-agnostic ops and removes quantize before
        two-input Eltwise task with quantized input.

        Converts :
        `f16_SWKernel_f16 -> f16_Quantize_u8 -> u8_ElemTypeInfoOp_u8 -> ... -> u8_Add_u8`
        To :
        `f16_SWKernel_f16 -> f16_ElemTypeInfoOp_f16 -> ... -> f16_Add_u8`
    }];

    let constructor = "vpux::IE::createFuseOutstandingQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def MergeWeightsSharedConv : PassBase<"merge-weights-shared-conv", "vpux::FunctionPass"> {
    let summary = "Merge Weights Shared 'IE.ConvolutionOp'";

    let description = [{
        This transformation takes the following subgraph:
        ```

        Input1(1x128x2x2)       Filter(16x128x1x2)     Input2(1x128x3x2)
                        \       /           \           /
                IE.Conv(1x16x2x1)         IE.Conv(1x16x3x1)
        ```
        To:
        ```
        Input1(1x128x2x2)       Input2(1x128x3x2)
                        \       /
                IE.Concat(1x128x5x2)       Filter(16x128x1x2)
                                \            /
                               IE.Conv(1x16x5x1)
                                /              \
                        IE.Slice(1x16x2x1)    IE.Slice(1x16x3x1)
        ```
        The optimization is used for LLM GQA. Assume queries(input) is 28, value or key(filter)
        is 4, that means each 7 queires share the same value or key(filter). Originally we have
        28 conv in total, after optimization, only have 4 convs. Reduce Op number will benefit to
        runtime idle, and also for some case convolution input H increase from 1 to 7 also benefit
        to HW efficient.
    }];

    let constructor = "vpux::IE::createMergeWeightsSharedConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def ShrinkMatmulGroups : PassBase<"shrink-matmul-groups", "vpux::FunctionPass"> {
    let summary = "Shrink Matmul groups number";

    let description = [{
        This transformation shrinks groups number of 3D Matmul when rhs is a tensor obtained through a BroadCastOp.
        Take below case as an example:
        Convert 3D Matmul with 24 groups:
                            RHS
                        1x8x1x1024x64
                            |
                        Broadcast
                            |
                        1x8x3x1024x64
                            |
                        AffineReshape
            LHS             |
        1x24x1x64       1x24x1024x64
            \               /
                MatMul

        to a new 3D Matmul with 8 groups:
            LHS             RHS
        1x24x1x64       1x8x1x1024x64
            |               |
        Reshape         Reshape
            |               |
        1x8x3x64        1x8x1024x64
            \               /
                MatMul

        The benefits are:
        1. The BroadCastOp can be eliminated.
        2. The converted 3D Matmul with less groups number can fit in CMX to benefit from NCE group Matmul.
    }];

    let constructor = "vpux::IE::createShrinkMatmulGroupsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagatePermuteCastThroughDequantize
//

def PropagatePermuteCastThroughDequantize : PassBase<"propagate-permute-cast-through-dequantize", "vpux::FunctionPass"> {
    let summary = "Propagates permute cast thorugh dequantizeop";

    let description = [{
        Converts :
        DequantizeOp -> PermuteCastOp
        to :
        PermuteCastOp -> DequantizeOp
        Target of this change is mostly ConstDeclare -> Dequantize -> PermuteCast pattern
        with the purpose of enabling easier strategy assignment logic and folding permuteCast
        to Const.
    }];

    let constructor = "vpux::IE::createPropagatePermuteCastThroughDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDynamicDequantizeToDequantize
//

def ConvertDynamicDequantizeToDequantize : PassBase<"convert-dynamic-dequantize-to-dequantize", "vpux::FunctionPass"> {
    let summary = "Convert 'IE.DynamicDequantize' to 'IE.Dequantize'";

    let description = [{
        This transformation takes the following subgraph:
        ```
                Input2(512x128xi4:f16)    Input3(512x1xf16)
                              \           /
        Input1(1x128xf16)     IE.DynamicDequantize(512x128xf16)
                        \       /
                    IE.FullyConnected(1x512xf16)
        ```
        To:
        ```
                Input2(512x128xi4:f16)
                              \
        Input1(1x128xf16)    IE.Dequantize(512x128xf16)
                        \       /                      Input3(512x1xf16)
                    IE.FullyConnected(1x512xf16)              |
                                    \                IE.Transpose(1x512xf16)
                                     \               /
                                 IE.Multiply(1x512xf16)
        ```
        When DynamicDequantize input parameter scale is 1.0f, zero point is 0, and is the
        input if FC, we could convert it to a normal Dequantize operation, and move the scale
        of DynamicDequantize to post FC by a Multiply operation.
    }];

    let constructor = "vpux::IE::createConvertDynamicDequantizeToDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MoveDynamicDequantizeToUser
//

def MoveDynamicDequantizeToUser : PassBase<"move-dynamic-dequantize-to-user", "vpux::FunctionPass"> {
    let summary = "Move Dynamic Dequantize operation to user";

    let description = [{
        Move DynamicDequantize after AffineReshape and PermuteCast
        to ease conversion of Dynamic Quantization subgraph to VPU dialect.
         DynamicDequantize               AffineReshape
               |                               |
         AffineReshape            ->      PermuteCast
               |                               |
          PermuteCast                  DynamicDequantize
    }];

    let constructor = "vpux::IE::createMoveDynamicDequantizeToUserPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def PopulateDynamicDimensionsHW : PassBase<"populate-dynamic-dimensions-hw", "vpux::FunctionPass"> {
    let summary = "Adapter for `mlir::bufferization::populateDynamicDimSizes` that targets HW operations";

    let description = [{
        1. Decide wether the operation is a tail of HW executed operation chain.
        2. Call `populateDynamicDimSizes` to insert `tensor.dim` operations into the graph.
        3. Concatenate values of static dimensions with dynamic dimensions.
           a. Cast `tensor.dim` to i64 via `mlir::arith::IndexCastOp`
           b. Cast i64 to tensor<1xi64> via `mlir::tensor::FromElementsOp`
           c. Cast tensor<1xi64> to tensor<1xsi64> with `mlir::tensor::BitcastOp`
           d. These steps are necessary because IE dialect cannot work with signless scalars.
        4. Create a `StridedSlice` operation to slice the output.
        5. Insert a `DynamicReshape` operation to to reconcile type differences.
           a. One of the inputs of the `StridedSlice` is the output of `Concat`
           b. `StridedSlice` infers tensor<?x?x?x?xf16> if any of its inputs are unknown at compile time.
           c. Function signature does not always set every dimension to ?, so the `DynamicReshape` is required.
    }];

    let constructor = "vpux::IE::createPopulateDynamicDimensionsHWPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def PopulateDynamicDimensionsGeneric : PassBase<"populate-dynamic-dimensions-generic", "vpux::FunctionPass"> {
    let summary = "Adapter for `mlir::bufferization::populateDynamicDimSizes` that targets ReifyRankedShapedTypeOpInterface operations";

    let description = [{
        Finds ReifyRankedShapedTypeOpInterface operations and populates output dimensions.
        Resulted static and dynamic dimensions are combined with ConcatOp.
        Result of ConcatOp is the new shape.
    }];

    let constructor = "vpux::IE::createPopulateDynamicDimensionsGenericPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LegalizeReifyResultShapesResiduals
//

def LegalizeReifyResultShapesResiduals : PassBase<"legalize-reify-result-shapes-residuals", "vpux::FunctionPass"> {
    let summary = "Legalize operations created by reifyResultShapes functions";

    let description = [{
        Convert tensor.dim to IE::ShapeOf->IE::SliceOp
        Convert arith.divsi to IE::DivideOp
    }];

    let constructor = "vpux::IE::createLegalizeReifyResultShapesResidualsPass()";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",

        "vpux::IE::IEDialect"
    ];
}

def PadDynamicInputs : PassBase<"pad-dynamic-inputs", "vpux::FunctionPass"> {
    let summary = "Pad dynamic inputs of a graph";

    let description = [{
        1. Look for `IE.StridedSlice` -> `IE.DynamicReshape` pattern created by `PopulateDynamicDimensions`
        2. Traverse dynamic graph towards block arguments and reset operations shapes with static values.
        3. Insert `IE.DynamicExpand` before the first operation to freeze input shape by upper bound.

        Example:
        ```
            BlockArg (1x2x3x?, bounds = [1, 2, 3, 10]) -> MaxPool -> StridedSlice (1x2x3x?, bounds = [1, 2, 3, 10])
        ```
        becomes:
        ```
            BlockArg (1x2x3x?, bounds = [1, 2, 3, 10]) -> DynamicExpand (1x2x3x10) -> MaxPool -> StridedSlice (1x2x3x?, bounds = [1, 2, 3, 10])
        ```
    }];

    let constructor = "vpux::IE::createPadDynamicInputsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertParallelSlicesToGather
//

def ConvertParallelSlicesToGather : PassBase<"convert-parallel-slices-to-gather", "vpux::FunctionPass"> {
    let summary = "Convert parallel Slice operations to single Gather operation";

    let description = [{
        When SliceOp branches share the same source and are finally concatenated, convert these SliceOp into single Gather operation.
        For NPU4000+, Gather operation will be mapped to GatherDMA.
        With this conversion, DMA workloads can be reduced and benefit performance.
    }];

    let constructor = "vpux::IE::createConvertParallelSlicesToGatherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapOperationWithGather
//

def SwapOperationWithGather : PassBase<"swap-operation-with-gather", "vpux::FunctionPass"> {
    let summary = "Move operation after GatherOp";

    let description = [{
        Move operation after GatherOp to reduce the tensor size to be processed
    }];

    let constructor = "vpux::IE::createSwapOperationWithGatherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseInputScaleShift
//

def FuseInputScaleShift : PassBase<"fuse-input-scale-shift", "vpux::FunctionPass"> {
    let summary = "Fuse input network ScaleShift into Convolution and Bias";

    let description = [{
        Looking for the following pattern:
             [input]        [Weights]
                |              |
            (Multiply)?        |
                |              |
              (Add)          (FQ2)
                |              |
              (FQ1)            |
                |              |
              (conv) --------- |
                |
              (Add) -------- [Bias]
                |
             [output]

        Disclamer: following is a rough description of the idea behind this transformation
        ConvInput =  [In] * scales + shifts
        ConvOutput = [ConvInput] * weights + biases
        =>
        ConvOutput = ([In] * scales + shifts) * weights + biases
        =>
        newWeighs = [In] * scales * weights
        newBias = shifts * weights + biases

        So the result is:
             [input]      [new Weights]
                |              |
             (new FQ1)      (new FQ2)
                |              |
              (conv) --------- |
                |
              (Add) -------- [new Bias]
                |
             [output]
    }];

    let constructor = "vpux::IE::createFuseInputScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateFQ
//

def PropagateFQ : PassBase<"propagate-fq", "vpux::FunctionPass"> {
    let summary = "Propagates FQ up and down through FQ-agnostic ops";

    let description = [{
        The pass is a part of `PostImportPipeline` pipeline.
        And was moved here from the nGraph transformations.

        Propagates FQ up and down through FQ-agnostic ops.
        This transformation helps to "quantize" as many HW ops as possible.
        Please note "propagate" does not just mean to "move" an operation, but also to "duplicate" it.
        For example:
        ```
            Transpose -> FQ -> MaxPool -> Squeeze -> Softmax
        ```
        To:
        ```
            Transpose -> FQ -> MaxPool -> FQ -> Squeeze -> FQ -> Softmax
        ```

        This is a temporary solution that should be replaced with PropagateQuantizeDequantize: #-152958
        This pass is paired with CleanupFQ pass.
    }];

    let constructor = "vpux::IE::createPropagateFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// CleanupFQ
//

def CleanupFQ : PassBase<"cleanup-fq", "vpux::FunctionPass"> {
    let summary = "Remove FQ operations with equal input and output low/high values.";

    let description = [{
        The pass is a part of `PostImportPipeline` pipeline.
        And was moved here from the nGraph transformations.

        Pass removes FQ operations where:
            - input low == output low && input high == output high
            - producer and all consumers are "view-like"(not "computational") operations e.g. TransposeOp, SplitOp, ReshapeOp, etc.

        Consider removing this pass: #-152958
        This pass is paired with PropagateFQ pass.
    }];

    let constructor = "vpux::IE::createCleanupFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertVariadicSplitToStridedSlice
//

def ConvertVariadicSplitToStridedSlice : PassBase<"convert-variadic-split-to-strided-slice", "vpux::FunctionPass"> {
    let summary = "Convert an IE.VariadicSplit operation to one or more IE.StridedSlice operations";
    let description = [{
        The pass is a part of `PostImportPipeline` pipeline.
        And was moved here from the nGraph transformations.

        This pass replaces every IE.VariadicSplit with one or more IE.StridedSlice operations that are semantically equivalent.
        This pass is designed to be called soon after the import stage. The goal is that IE.VariadicSplit is
        removed as early as possible from the IR and replaced by the more widely supported IE.StridedSlice operation.
    }];

    let constructor = "vpux::IE::createConvertVariadicSplitToStridedSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
