//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_PASSES
#define VPUX_COMPILER_DIALECT_IE_PASSES

include "mlir/Pass/PassBase.td"

//=================================================================================
// Precisions and Layouts
//=================================================================================

//
// MapBilinearInterpolateOnDPU
//

def MapBilinearInterpolateOnDPU : PassBase<"map-bilinear-interpolate-on-dpu", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate op to strided concat, AvgPool and some depthwise convolution Ops";

    let description = [{
        Purpose:
        This pass replaces `Bilinear Interpolate` for which tiling is required to fit in CMX
        with sequences of operation that can be mapped on DPU and DMA.

        How it works:
        The supported interpolation axis currently supported are H and W.
        For each of these axis the scaling is happening individually, first perform vertical scaling and after perform horizontal scaling.
        On each axis the processing is split in three main regions BEGIN, MIDDLE and END.
        These three regions refers to slices from the output tensor and are influenced by the coordinate transformation mode attribute.
        * BEGIN - refers to the slice from output for which it is only needed to duplicate the first line/column from input
        * MIDDLE - refers to the slice from output where:
                    * for each output line/column from the output it is required to take two consecutive lines/colums from the input
                    * based on the coordinate transformation mode attribute compute the weight coefficients each of two lines/columns
                      has on theresulting output line/column
                    * each output line/column is computed with a GroupConvolution operation for which the weights are obtained by expanding
                     the weight coefficients of the input lines/columns
        * END - refers to the slice from output for which it is only needed to duplicate the last line/column from input
        ```
        Vertical scaling:                    Horizontal scaling
        ________________________       ____________________________
        |        BEGIN         |       |   |                  |   |
        |______________________|       |   |                  |   |
        |                      |       | B |        M         |   |
        |                      |       | E |        I         | E |
        |        MIDDLE        |       | G |        D         | N |
        |                      |       | I |        D         | D |
        |______________________|       | N |        L         |   |
        |         END          |       |   |        E         |   |
        |______________________|       |___|__________________|___|
        ```

        The rewrite implemented per each region is described below:
         BEGIN region:
        ```         Input
                      |
                    Slice
               first line/column
                |    ...    |
            Identity       Identity
            AvgPool        AvgPool

         MIDDLE region
                         Input
                  ---------|---------
                 |                   |
             Slice        ...       Slice
         two lines/colums       two lines/colums
               |                        |
           GroupConv               GroupConv
         one output line/colum   one output line/colum

         END region:
                    Input
                      |
                    Slice
               last line/column
                |    ...     |
            Identity       Identity
            AvgPool        AvgPool
        ```
        At the end the results of all the operation resulted are concatenated together on the scaling axis.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats.
    }];

    let constructor = "vpux::IE::createMapBilinearInterpolateOnDPUPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// Outliner
//

def Outliner : PassBase<"outliner", "vpux::ModulePass"> {
    let summary = "Extracts function based on result from IR analysis";

    let description = [{
        AI-generated summary

        This pass is used for parallel compilation. It outlines functions, it does this in
        a similar way to how inlining a function works but other way around. It breaks a large
        function into multiple functions which the original function calls.

        Below is an example of what the outliner does on the function main:
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
          batching
            Used together with Debatcher/DeDebatcher for batched model compilation.
            Must not be combined with other outlining modes.

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

def Debatcher : PassBase<"debatcher", "vpux::ModulePass"> {
    let summary = "Downcast input batched tensor arguments of a `main`-function to single batch tensors, which is supposed to unblock further transformations dedicated to support batched models compilation. This is a frontend of 'Debatcher-Outliner-DeDebatcher' approach";

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
            "debatcherInputCoeffPartitions", "debatcher-input-coefficients-partitions",
            "std::string", [{""}],
            "Determines which dimension and what proportion debatching of input tensors should be done."
        >
    ];

}

//
// DeDebatcher
//

def DeDebatcher : PassBase<"de-debatcher", "vpux::ModulePass"> {
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

        This pass updates the NetworkInfo entry point function prototype and use user-provided precisions for its operands and results.
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
        AI-generated summary

        The pass is a part of `IECommon` pipeline.

        This pass adds the required layouts instead of the default one
        depending on the layer specification from underlying Dialect.

        Uses a LayerRewriter that matches any operation implementing the
        `IE::LayoutInfoOpInterface`. For each matched operation, it queries
        the interface for the required layout and inserts IE.Reorder operations
        on inputs and outputs when the existing layout does not match.
    }];

    let constructor = "vpux::IE::createAdjustLayoutsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseReshapeMvnPass
//

def FuseReshapeMvn : PassBase<"fuse-reshape-mvn", "vpux::FunctionPass"> {
    let summary = "Fuse Reshape->MVN->Reshape (back to initial shape) into MVN with internal-reshape";

    let description = [{
        AI-generated summary

        The pass is a part of `IECommon` pipeline.
        Can only succeed in NHWC layout, for large tensors that require MVN-decomposition.

        The supported base pattern is:
        Reorder(NCHW) -> Reshape -> Reorder(NHWC) -> MVN -> Reorder(NCHW) -> Reshape -> Reorder(NHWC),
        which is fused into a single MVN operation with an internal reshape.

        This pattern can be extended with several optional variants, for example:
        Reorder(NCHW) -> Reshape -> Reorder(NHWC) -> MVN -> AffineReshape -> GroupConv -> Reorder(NCHW) -> Reshape -> Reorder(NHWC).
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
    let summary = "Fuse Power-ReduceMean-Add-[optional ReLU]-Sqrt-Divide-Multiply-Multiply to RMSOp";
    let description = [{
        AI-generated summary

        Detects and fuses Root Mean Square Normalization patterns where squared input
        (Power(x,2) or Multiply(x,x)) is followed by ReduceMean or ReduceSum, optional epsilon
        addition/clamp, optional ReLU before Sqrt, and reciprocal square root
        (Sqrt->Divide or Power(-0.5)),
        into a single IE.RMS operation (RMSOp) with precomputed gamma and epsilon parameters.
        Supports multiple mathematical formulations including intermediate scale constants
        and dimension expansion operations.
    }];
    let constructor = "vpux::IE::createFuseRMSNormPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseNormalizeL2ToRMS
//
def FuseNormalizeL2ToRMS : PassBase<"fuse-normalize-l2-to-rms", "vpux::FunctionPass"> {
    let summary = "Fuse NormalizeL2 + Multiply into RMSOp";
    let description = [{
        Detects NormalizeL2 (eps_mode = ADD) reducing a single innermost axis
        (axes provided either via a constant tensor operand or via axes_value,
        and containing exactly one axis: [-1] or an equivalent positive axis
        for the input rank)
        followed by a Multiply with a constant scale, and fuses the pattern into
        a single IE.RMSOp with gamma = scale / sqrt(reduceSize).
    }];
    let constructor = "vpux::IE::createFuseNormalizeL2ToRMSPass()";
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
        AI-generated summary

        Identifies Rotary Position Embedding patterns involving input splitting via
        StridedSlice/Split operations, multiplication with precomputed cosine and sine
        embeddings, concatenation, and final addition, replacing the entire sequence
        with a dedicated IE.RoPE operation. Handles both standard and interleaved RoPE
        variants with optional reshape operations for shape compatibility.
    }];
    let constructor = "vpux::IE::createFuseRoPEPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseColorConversion
//
def FuseColorConversion : PassBase<"fuse-color-conversion", "vpux::FunctionPass"> {
    let summary = "Fuse YUV to RGB color conversion pattern into IE.YuvToRgb and IE.Transpose";
    let description = [{
        AI-generated summary

        Fuses a sequence of operations that implement YUV to RGB color conversion.
        The pattern is replaced with IE.YuvToRgb followed by IE.Transpose and IE.Multiply if needed.

        Registers two rewrite patterns anchored on the ops that terminate the
        color-conversion chain:
        - FuseColorConversionClampPattern: matches IE.ClampOp and walks backward
          through the YUV-to-RGB operation sequence.
        - FuseColorConversionFakeQuantizePattern: matches IE.FakeQuantizeOp and
          walks backward through the YUV-to-RGB operation sequence.
        Both patterns invoke a shared matcher that identifies the Add, Convolution,
        and input split structure characteristic of YUV-to-RGB conversion.
    }];

    let constructor = "vpux::IE::createFuseColorConversionPass()";

    let options = [
        Option<
            "enableYuvToRgbShaveScale", "yuv-to-rgb-shave-scale",
            "bool", "false",
            "Enable YUV to RGB SHAVE scale conversion"
        >
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FoldShapeOfGatherDynamicReshapeToTranspose
//
def FoldShapeOfGatherDynamicReshapeToTranspose :
        PassBase<"fold-shapeof-gather-dynamic-reshape-to-transpose", "vpux::FunctionPass"> {
    let summary = "Recover IE.Transpose from a ShapeOf/Gather/DynamicReshape decomposition";

    let description = [{
        Matches the following IR chain, which is emitted by the OpenVINO frontend when a
        Transpose is applied to a fully-dynamic tensor:

            %shape   = IE.ShapeOf(%x)
            %new_sh  = IE.Gather(%shape, %perm_const) {axis_value = 0, batch_dims = 0}
            %out     = IE.DynamicReshape(%x, %new_sh)

        and rewrites it back into a single `IE.Transpose(%x)` with the extracted
        permutation encoded in `order_value`. This is safe when `%perm_const` is a
        constant permutation of `[0..rank-1]` and the recorded `output_shape`/`output_bounds`
        of `IE.DynamicReshape` match the input shape/bounds permuted by that vector.

        The rewrite is required because the HostCompile output-shape prediction pipeline
        (OutlineDimOperations) relies on `ReifyRankedShapedTypeOpInterface`, which
        `IE.ShapeOf`, `IE.Gather` and `IE.DynamicReshape` do not implement, whereas
        `IE.Transpose` does.
    }];

    let constructor = "vpux::IE::createFoldShapeOfGatherDynamicReshapeToTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseSDPA
//
def FuseSDPA : PassBase<"fuse-sdpa", "vpux::FunctionPass"> {
    let summary = "Fuse MatMul-Divide-Add-Softmax-MatMul to SDPA";
    let description = [{
        SDPA (Scaled Dot-Product Attention) is a sequence of ops that implements
        the Self-Attention mechanism.
        See: https://github.com/uxlfoundation/oneDNN/blob/main/doc/graph/fusion_patterns/sdpa_with_compressed_kv.md#sdpa-pattern

        The pass detects and fuses multiple attention patterns into a single IE.SDPAOp.

        The general pattern is:
        Reshape -> FC(Q,K) -> Reshape -> [Scale] -> [Add(mask)] -> Softmax -> Reshape -> FC(.,V) -> Reshape
        All QKV inputs pass through Reshape operators before entering the FullyConnected ops.

        Supported pattern variants:
        - FullyConnected with Multiply: K is pre-multiplied with scale before entering the first FC.
        - FullyConnected with Divide: scale is applied as Div/Mul after the first FC.
        - AdaptiveStripping with Multiply: like the Multiply variant but includes a ReLU after Softmax.

        The pass also optimizes the V-input by transposing it so that K and V
        sequence dimensions align.
    }];
    let constructor = "vpux::IE::createFuseSDPAPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseSoftMaxConvert
//

def FuseSoftMaxConvert : PassBase<"fuse-softmax-convert", "vpux::FunctionPass"> {
    let summary = "Fuse SoftMax->Convert into SoftMax";

    let description = [{
        Fuses SoftMax->Convert into a single SoftMax and sets the dstElemType attr to the ConvertOp output type.
        This will result in better SoftMaxOp performance and accuracy.
    }];

    let constructor = "vpux::IE::createFuseSoftMaxConvertPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseLogSoftmaxVariants
//

def FuseLogSoftmaxVariants : PassBase<"fuse-logsoftmax-variants", "vpux::FunctionPass"> {
    let summary = "Fuse different LogSoftmax and TopK patterns into the same operation";
    let description = [{
        AI-generated summary

        Optimizes patterns such as:
            input -> LogSoftmax -> output0
              |
              | ---> TopK -> output1

              or

            input -> LogSoftmax -> Gather -> output0
              |                      ^
              | ---> TopK -----------|
                      | --> output1

        Registered rewrite patterns:
        - FuseLogSoftmaxPeakPattern: Detects LogSoftmax -> Gather <- TopK patterns
          and fuses them by rewriting Gather indices and output types.
        - FuseLogSoftmaxTopKPattern: Finds LogSoftmax and TopK sharing a common
          input through Expand (and optional AffineReshape) operations. Converts
          LogSoftmax to operate in f32 precision and replaces the expanded path
          with a fused LogSoftmax+TopK pair.
    }];

    let constructor = "vpux::IE::createFuseLogSoftmaxVariantsPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ExpandAttention
//
def ExpandAttention : PassBase<"expand-attention", "vpux::FunctionPass"> {
    let summary = "Pad shared Attention root sequence length before decomposition";
    let description = [{
        Pads the shared root tensor of IE.AttentionOp before decomposition when
        the sequence length is not channel-aligned. After Attention decomposition,
        the sequence length is mapped to the input and output channels of the
        Convolution after MatMul lowering. Padding those channels independently
        requires different operations (e.g. Expand and Concat), which makes later
        padding cleanup difficult, especially after SDPA unrolling.

        The pass traces Q/K/V inputs through supported view-like, fully-connected,
        and element-wise operations to find the common root tensor, and inserts one
        zero-padded Concat at it. FullyConnected weights may be constants directly
        or constants wrapped by FakeQuantize.
    }];
    let constructor = "vpux::IE::createExpandAttentionPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect",
    ];
}

//
// FuseAttention
//
def FuseAttention : PassBase<"fuse-attention", "vpux::FunctionPass"> {
    let summary = "Fuse SDPA like Patterns into Attention operator";
    let description = [{
        The pass performs two walks:
        1. Converts existing IE.SDPAOp operations into IE.SDPAExtendedOp,
           transposing the V-input so K and V sequence dimensions align,
           and extracting unbroadcasted inputs when possible.
        2. Detects custom attention patterns built from primitive ops:
           MatMul(Q,K) -> [Scale] -> [Add(mask)] -> [Add(bias)] -> Softmax -> MatMul(.,V)
           where layout/reshape operations between stages are skipped automatically.
           Extracts optional scale and bias operands, and fuses the entire
           sequence into a single IE.SDPAExtendedOp.

        Additionally, when the input batch dimension is > 1, the pass optimizes
        SDPA inputs by reshaping batch into the channel dimension (N*C) to
        improve DPU utilization, and reshapes the output back.

        Also SDPA has a Shave control for DPU execution
    }];
    let constructor = "vpux::IE::createFuseAttentionPass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeAttention
//
def DecomposeAttention : PassBase<"decompose-attention", "vpux::FunctionPass"> {
    let summary = "Decompose Attention operations into primitive IE operations";
    let description = [{
        This pass decomposes Attention operations into a sequence of IE operations
        following the scaled dot-product attention formula:
            Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k) + mask + bias) * V

        The decomposition follows this pipeline:
        1. Q*K^T: Apply IE.MatMul to compute query-key dot products
        2. Scale: Apply IE.Multiply to scale by 1/sqrt(d_k) or provided scale
        3. Add Mask: Apply IE.Add to add attention mask if provided
        4. Add Bias: Apply IE.Add to add bias if provided
        5. Append Sink: If a sink input is provided, IE.Concat it as an extra
           column on the last dimension so it joins the softmax normalization
        6. Softmax: Apply IE.SoftMax on the last dimension
        7. Drop Sink: If a sink input is provided, IE.Slice off the extra sink
           column added in step 5
        8. Output: Apply IE.MatMul to compute final attention output with values

        By default, Attention ops matching a legal SDPA configuration are left untouched so
        they can run on the native SDPA path. When `force-attention-decomposition` is set, every
        Attention op is decomposed. The softmax scale placement (Query, Key, or result) is chosen
        by shape heuristics. This is required on platforms without native SDPA execution (NPU40XX).
    }];
    let constructor = "vpux::IE::createDecomposeAttentionPass()";

    let options = [
        Option<"forceAttentionDecomposition", "force-attention-decomposition", "bool", /*default=*/"false",
               "Force decomposition of every Attention op">
    ];

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseDynamicQuantize
//
def FuseDynamicQuantize : PassBase<"fuse-dynamic-quantize", "vpux::FunctionPass"> {
    let summary = "Fuse DynamicQuantizeLinear decomposed by OV to DynamicQuantize op";
    let description = [{
        Fuse a sequence of ops to one DynamicQuantize.
        Supports scalar and broadcast-compatible min/max tensors when the decomposed graph computes
        matching scale and zero-point tensors.

        Supported decomposition variants:
        - Unsigned (ui8): scale = (max-min) / 255, clamp [0, 255]
        - Signed asymmetric (si8): scale = (max-min) / 254, clamp [-127, 127]

        Signed symmetric (no zero-point) decompositions are recognized by the matcher
        but their fusion is intentionally disabled (see "Skipping symmetric
        DynamicQuantize fusion" in fuse_dynamic_quantize.cpp) due to an observed
        performance regression on the symmetric kernel path. Support will be revisited
        once the kernel-side cost is addressed.

        Scale computation accepts:
        - Multiply variant: (max-min) * (1/quantSpan)
        - Divide variant: (max-min) / quantSpan

        Shape requirements:
        - min/max tensors must be broadcast-compatible with the input tensor
        - scale and zero-point shapes must match min/max shapes

        Optional bounding ops on ReduceMin/ReduceMax outputs:
        - Any ClampOp (walked through unconditionally)
        - MinimumOp with constant 0 (for min path)
        - MaximumOp with constant 0 (for max path)

        Also performs backward compatibility fixup for older model XMLs that produce
        signless i8 output types, converting them to signed si8 and removing redundant
        Convert ops.

        Searching for the following pattern (asymmetric variant):

        ```
                                  [input]
                            /        |      \
                    (ReduceMin) (ReduceMax) (MultiPly)
                          |          |         |
                       (Clamp/Min) (Clamp/Max) |
                       /     \       |         |
               (Subtract)    (Subtract)        |
                |             /        \      /
                |    (Multiply|Divide) (Divide)
                |     /    |            |
             (Divide)      |            |
                |          |           /
             (Clamp)    [scale]      /
                |    \            /
                |      \        /
                |         (Add)
                |           |
                |        (Clamp)
                |           |
            [zero-point]  [quant-output]
        ```

        And fuse the ops excluding `ReduceMin` and `ReduceMax`.

        ```
                       [input]
                    /     |    \
            (ReduceMin)   |  (ReduceMax)
                    \     |     /
                    (DynamicQuantize)
                    /     |      \
             [scale] [zero-point] [quant-output]
        ```
    }];

    let constructor = "vpux::IE::createFuseDynamicQuantizePass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSelectToEltwise
//
def ConvertSelectToEltwise : PassBase<"convert-select-to-eltwise", "vpux::FunctionPass"> {
    let summary = "Fuse Select(const or LogicalNot(const), zero, Broadcast(data)) to Multiply(data, float_mask)";
    let description = [{
        Fuses one of the following subgraphs:

        Pattern A (condition via LogicalNot):

        ```
            bool_const zero_const  data
                |         |         |
         (IE.LogicalNot)  |    IE.Broadcast
                \         |        /
                        IE.Select
                          |
                        output
        ```

        Pattern B (condition is a constant directly):

        ```
            bool_const zero_const  data
                |         |         |
                |         |    IE.Broadcast
                \         |        /
                        IE.Select
                          |
                        output
        ```

        Both patterns are fused into:

        ```
            float_mask  data
                |        |
                IE.Multiply
                    |
                  output
        ```

        where `float_mask` is derived from `bool_const` (inverted for Pattern A,
        direct for Pattern B) and materialized as a compile-time constant.
    }];
    let constructor = "vpux::IE::createConvertSelectToEltwisePass()";
    let dependentDialects = ["vpux::IE::IEDialect"];
}

//
// FuseReduceSquare
//
def FuseReduceSquare : PassBase<"fuse-reduce-square", "vpux::FunctionPass"> {
    let summary = "Fuse square-reduce-sqrt chains to ReduceSquare";
    let description = [{
        AI-generated summary

        Detects patterns where element-wise squared operations (Power(x,2) or
        Multiply(x,x)) immediately precede ReduceMean or ReduceSum, optionally followed
        by epsilon addition, and terminated by either Sqrt or Power(..., 0.5). Fuses
        these chains into a single ReduceSquare operation that computes the equivalent
        reduction of squares. Extracts epsilon values and preserves reduction axes and
        keep-dims attributes.
    }];
    let constructor = "vpux::IE::createFuseReduceSquarePass()";
    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeParallelLayers
//
def OptimizeParallelLayers : PassBase<"optimize-parallel-layers", "vpux::FunctionPass"> {
    let summary = "Optimize parallel layers";
    let description = [{
        AI-generated summary

        Moves parametric layers (Convolution, Add, Multiply) before Slice operations when
        a single source tensor is distributed across parallel identical slices without size
        variation or offset overlap, eliminating redundant computation by reducing parameters
        and slices to the minimum required tensor sizes. Validates legality of transformation
        before rewriting to ensure numerical equivalence.
    }];
    let constructor = "vpux::IE::createOptimizeParallelLayersPass()";
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
        AI-generated summary

        The pass is a part of `IECommon` pipeline.

        This pass tries to optimize out Reorder operations for common cases
        by propagating them from inputs to outputs and merging into layers.

        Registered rewrite patterns (applied in two phases):
        Phase 1 (greedy):
        - ReorderWithShapeChange: propagates Reorder through Reshape, AffineReshape,
          and ShapeCast operations.
        - ReorderWithSubView: moves Reorder past Slice operations.
        - ExpandWithLayer: swaps Expand with subsequent Reorder when beneficial.
        - ReorderWithSplit: propagates Reorder through Split operations.
        - ReorderWithConcat: propagates Reorder through Concat operations.
        - ReorderWithQuantCast: moves Reorder past QuantizeCast.
        - ReorderWithTile: propagates Reorder through Tile operations.
        - ReorderWithLayer: generic pattern for propagating through single-input layers.
        - ReorderWithPermuteCast: folds Reorder into PermuteCast.
        - ReorderWithHWEltwise: optimizes Reorder around Add+MVN and Add+Subtract pairs.
        - ReorderWithHWAddSlice: optimizes Reorder around Add+Slice.
        - ReorderWithGroupConv: optimizes Reorder around depthwise GroupConvolution.
        - ReorderWithAddAndGroupConv: replaces Reorder-Add-GroupConv(eltwise)-Reorder with
          zero-copy PermuteCast operations, adjusting the filter and groups count to match
          the permuted channel dimension.
        - ReorderWithEltwise: propagates Reorder through GeluOp.
        Phase 2 (cleanup):
        - ReorderWithConvert: merges adjacent Reorder and Convert.
        - ReorderWithExpandSlice: optimizes Reorder+Expand+Slice chains.
        - ReorderWithAffineReshapeTile: optimizes Reorder+AffineReshape+Tile chains.
    }];

    let constructor = "vpux::IE::createOptimizeReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeTileOp
//

def OptimizeTileOp : PassBase<"optimize-tile-op", "vpux::FunctionPass"> {
    let summary = "Optimize tile ops";

    let description = [{
        Optimizes IE.TileOp and related patterns via three rewrite patterns applied to
        IE.TileOp, IE.ConvertOp, and IE.AddOp respectively. All cases require the TileOp
        to have a single use and its consumer to be IE.MultiplyOp or IE.AddOp with matching
        input/output element types (no implicit precision conversion) and no PPE post-op
        (except spatial-broadcast cases).

        1. FoldTileOpRewriter -- four fold paths on IE.TileOp:
           - Scalar (totalSize==1): reshapes the scalar input to an all-ones shape matching
             the consumer's expected rank; the consumer applies NUMPY broadcast implicitly.
           - Large single-channel (4D, C==1, inputSize > total CMX): replaces the Tile with
             its pre-Tile input; the consumer eltwise broadcasts the single-channel dimension.
           - NxCx1x1 spatial-broadcast (4D, H==1 && W==1 on input, N/C unchanged, output
             not 1D, outputSize * numTiles > CMX): replaces the Tile with its pre-Tile input
             so the consumer's NUMPY broadcast handles H/W expansion without an explicit DMA.
           - Force-fold (isSpatialBroadcastFromGroupConv): same replacement regardless of
             size when the Tile's sole user is a bias-Add for a static-filter GroupConv with
             a static-constant bias -- bypasses the CMX threshold guard.

        2. FuseTileConvertRewrite -- fuses Convert(A->B) -> Tile(B) -> Convert(B->C) into
           Convert(A->C) -> Tile(C) when sizeof(C) <= sizeof(B) in bits, so the Tile DMA
           operates on smaller (or equal) data.

        3. FuseGroupConvWithBiasAdd -- folds IE.Add(GroupConv(x,w), bias_1xCx1x1) into a
           single GroupConvolutionOp with the bias in the weight-table slot, eliminating the
           Add and any preceding Tile on the bias path.
           Fires for: static-filter + static-bias (all platforms); dynamic-filter + static-bias
           when the platform supports the new weight-table bias format (the bias must remain
           static because downstream NCE lowering always materializes the weight-table bias
           from a compile-time constant).
           Skips: static-filter + dynamic-bias; dynamic-bias with dynamic-filter (regardless of
           platform); Add with output type != GroupConv output type; Add with input/output
           padding attrs; Add with a scale operand or static_scale attribute (would be dropped
           by fusion); non-float bias element type.
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
}

//
// ConvertSplitConcatToAffineReshape
//

def ConvertSplitConcatToAffineReshape : PassBase<"convert-split-concat-to-affinereshape", "vpux::FunctionPass"> {
    let summary = "Convert the pattern {Split -> Concat} to AffineReshape";

    let description = [{
        AI-generated summary

        Converts Split followed by Concat sequences (without intermediate reshape
        operations) into a single AffineReshapeOp using reassociation mapping that
        combines consecutive dimensions or splits them. Detects when concatenation
        occurs along compatible axes and validates that the split/concat sequence
        represents a pure reshape without shape expansion or contraction.
    }];

    let constructor = "vpux::IE::createConvertSplitConcatToAffineReshapePass()";

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

        Inserts `Transpose` operations around dedicated operations to move
        non-spatial axes into spatial positions.
        Supported ops: `Interpolate` (when M2I disabled), `Roll` (when SEP enabled).
        For RollOp, when transposing is not applied, it decomposes Roll into Slice+Concat
        as a lower-priority fallback.
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
}

//
// SplitBilinearIntoHAndW
//

def SplitBilinearIntoHAndW : PassBase<"split-bilinear-into-H-and-W", "vpux::FunctionPass"> {
    let summary = "Convert bilinear interpolate on H and W to slice, concat, convolution and interpolate on H";

    let description = [{
        This pass convert `Bilinear Interpolate` operations to interpolate on H and interpolate on W, and
        the interpolate on W will convert to slice, concat and convolution.

        This pass is enabled when both `interpolateAsSEOp` and `SplitBilinearIntoHAndW` options are set to true.
    }];

    let constructor = "vpux::IE::createSplitBilinearIntoHAndWPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

}

//
// SplitInterpolateAxes
//

def SplitInterpolateAxes : PassBase<"split-interpolate-axes", "vpux::FunctionPass"> {
    let summary = "Split interpolate axes to separate interpolate operations";

    let description = [{
        This pass is specifically designed for the 5D interpolation case, focusing on scaling across three axes. It transforms the interpolation operation by splitting it into two separate operations. Each operation targets different axes, allowing for independent scaling and processing of these axes (the first operation scales one axis, while the second operation scales the next two axes).
    }];

    let constructor = "vpux::IE::createSplitInterpolateAxesPass()";

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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `Bilinear Interpolate` operations with `Concat` operations with strides,
        MaxPool and some `depthwise` convolutions.

        Three converter patterns are registered with different priorities:
        - BilinearInterpolateOpConverterV2 (mid priority): handles general bilinear
          interpolation with coordinate transformation and weight computation.
        - BilinearInterpolateOpConverter (low priority): fallback converter for
          standard bilinear interpolation cases.
        - SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter: specialization
          for small-channel inputs with pytorch_half_pixel coordinate transformation,
          applied in a separate phase.

        In case the `interpolateAsSEOp` option is set to true, only cases that cannot be executed
        using the Storage Element hardware feature will be converted to concats & NCE ops.
    }];

    let constructor = "vpux::IE::createConvertBilinearToStridedConcatAndConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertScatter
//

def ConvertScatter : PassBase<"convert-scatter", "vpux::FunctionPass"> {
    let summary = "Optimize scatter related operations";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.
        This pass optimizes the following cases:
        Converts `ScatterNDUpdate` operations with `Concat` operations with strides.
        Converts `ScatterNDUpdate` operations with `Slice` and `Concat`.
        Splits `ScatterNDUpdate` operations with large number of updates into multiple `ScatterNDUpdate`
                    operations with smaller number of updates.
        Converts `ScatterElementsUpdate` to `Add` or `Multiply`
    }];

    let constructor = "vpux::IE::createConvertScatterPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertPrecisionToFP
//

def ConvertPrecisionToFP : PassBase<"convert-precision-to-fp", "vpux::ModulePass"> {
    let summary = "Convert tensors precision from FP32 to FP16";

    let description = [{
        The pass is a part of `AdjustPrecision` pipeline.

        This pass replaces all FP32 tensors with FP16.
        It updates both function bodies as well as Function signatures.
    }];

    let constructor = "vpux::IE::createConvertPrecisionToFPPass()";

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
        AI-generated summary

        The pass is a part of `AdjustPrecision` pipeline.

        Some kernel implementations only support specific precisions. To satisfy this requirement,
        such ops are surrounded by conversion layers.

        Registered rewrite patterns:
        - DequantizeConverter: wraps IE.DequantizeOp with Convert operations to
          ensure the dequantized output uses a supported precision.
        - DynamicDequantizeConverter: wraps IE.DynamicDequantizeOp with Convert
          operations similarly.
        - TopKConverter: inserts Convert around IE.TopKOp when its output
          element type does not match kernel requirements.
        - FlashSDPAConverter: adjusts IE.FlashSDPAOp input precision by inserting
          Convert operations when inputs are not f16.
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
        AI-generated summary

        The pass is a part of `AdjustPrecision` pipeline.

        Currently NCE ops only support f16 or quantized inputs. In some cases, such ops
        with i32 inputs need conversion layers.

        Handles the following NCE operations by inserting Convert(i32 -> f16) on inputs
        and Convert(f16 -> i32) on outputs:
        - IE.MatMulOp
        - IE.ConvolutionOp
        - IE.GroupConvolutionOp
        - IE.FullyConnectedOp (when convert-fc-to-conv option is enabled)
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
// ConvertGather
//

def ConvertGather : PassBase<"convert-gather", "vpux::FunctionPass"> {
    let summary = "Convert Gather operation to Slice or Reverse operation";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces legal `Gather` operations with simpler operations.

        Registered rewrite patterns:
        - GatherToSlice: converts Gather with scalar or 1D indices on the gather axis
          into a Slice operation when the indices represent a contiguous range.
        - GatherToReverse: converts Gather into a Reverse operation when the indices
          represent the axis elements in reversed order.
        - FuseBroadcastGatherND: fuses a GatherND whose input is a Broadcast into
          a simpler Broadcast operation, eliminating the GatherND.
        - GatherRepeatInterleaveToBroadcast: converts Gather with repeat_interleave
          indices pattern (indices[i] = i / repeatFactor) into a sequence of
          Reshape -> Broadcast -> Reshape, eliminating the costly gather operation.
    }];

    let constructor = "vpux::IE::createConvertGatherPass()";

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

    let options = [
        Option<
            "forceConvertGatherTo4D", "force-convert-gather-to-4d",
            "bool", "false",
            "For WS Init schedule, GatherOp is forced to 4D"
        >,
        Option<
            "enableShapeVerification", "enable-shape-verification",
            "bool", "false",
            "When set, marks the module to enable 4D shape verification after conversion"
        >
    ];
}

//
// ConvertPaddingsToFloorMode
//

def ConvertPaddingsToFloorMode : PassBase<"convert-paddings-to-floor-mode", "vpux::FunctionPass"> {
    let summary = "Convert Convolution and Pooling layers paddings to FLOOR rounding mode";

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
// DecomposeConcatMatMul
//

def DecomposeConcatMatMul : PassBase<"decompose-concat-matmul", "vpux::FunctionPass"> {
    let summary = "Decompose inefficient Concat+MatMul patterns";

    let description = [{
        This pass decomposes inefficient Concat+MatMul patterns where:
        1. Concat operates on the innermost dimension with low channel counts
        2. MatMul has unaligned input/output channels causing inefficient computation

        The transformation decomposes MatMul into element-wise Multiply and Add operations
        to avoid channel expansion on both input channels (IC) and output channels (OC),
        thereby improving computational efficiency.

        For example:
        IE.Concat(%input1, %input2, %input3, %input4) : 4x tensor<1x256x2048x1xf32> -> tensor<1x256x2048x4xf32>
        IE.MatMul(%concat, %weights) {transpose_b} : tensor<1x256x2048x4xf32>, tensor<1x256x2x4xf32> -> tensor<1x256x2048x2xf32>

        Is transformed to:
        Multiple Multiply and Add operations using the original inputs directly.
    }];

    let constructor = "vpux::IE::createDecomposeConcatMatMulPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertNegativePadToSlice
//

def ConvertNegativePadToSlice : PassBase<"convert-negative-pad-to-slice", "vpux::FunctionPass"> {
    let summary = "Convert Pad with negative padding values to Slice";

    let description = [{
        This pass handles Pad operations with negative padding values.
        According to OpenVINO spec, negative padding in Pad operation represents slicing.

        The pass performs the following transformations:
        1. If all pads are negative or zero: Pad -> Slice
        2. If mixed positive and negative pads: Pad -> Slice + Pad (with non-negative pads only)

        This pass should run before other Pad conversion passes to ensure they only
        see non-negative padding values.
    }];

    let constructor = "vpux::IE::createConvertNegativePadToSlicePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
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
}

//
// ConvertBatchedLayerTo1N
//

def ConvertBatchedLayerTo1N : PassBase<"convert-batched-layer-to-1n", "vpux::FunctionPass"> {
    let summary = "Convert layer with batched input to new one with batch equal to 1";

    let description = [{
        AI-generated summary

        Unrolls NCE operations with batch > 1 into batch=1 by transposing the batch
        dimension into a spatial dimension (H or W, whichever equals 1).
        WHY: DPU hardware processes batch=1 only; transposing batch into a spatial
        dimension keeps the operation DPU-executable.

        Supported operation types: Convolution, GroupConvolution, MaxPool, AvgPool, Add,
        Multiply, Subtract, Power, Sigmoid, and Convert.

        Example for Convolution:
            Activation: 4x16x1x1 ->
                                    Conv -> 4x5x1x1
            Weights:    5x16x1x1 ->

        New subgraph:
            Activation:4x16x1x1     Weights:5x16x1x1
                |                       |
            Transpose:1x16x4x1          |
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
        AI-generated summary

        Converts IE.Divide operations to IE.Multiply with reciprocal values.
        This reduces pattern complexity since downstream passes need not
        differentiate between division and multiplication.

        Registered patterns:
        - ConstDivisorRewriter — matches Const::DeclareOp used as divisor in
           IE.Divide; replaces the constant with its reciprocal and rewrites
           the Divide as Multiply.
        - FakeQuantizeDivideRewriter — matches FakeQuantize whose output feeds
           a Divide with a constant divisor; absorbs the reciprocal into FQ
           output range and converts the Divide to Multiply.
        - DequantizeDivideRewriter — matches Dequantize whose output feeds a
           Divide with a constant divisor; adjusts the quantization scale and
           converts the Divide to Multiply.
        - NonConstDivisorRewriter — handles Divide with a non-constant divisor
           by inserting IE.PowerOp(divisor, -1) and replacing with Multiply.

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
        AI-generated summary

        Converts 3D or 4D-batch MatMul operations into 1x1 Convolution.
        WHY: MatMul is a software-only operation; Convolution with 1x1 kernels is
        functionally equivalent but executes on the DPU hardware.
        Applies when input1 is 3D or 4D (batch=1) and input2 is 2D.
        Reshapes inputs to 4D, transposes weight matrices as needed, applies
        PermuteCast for layout compatibility, and reshapes output back.
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
// LegalizeConvBackpropData
//

def LegalizeConvBackpropData : PassBase<"legalize-convbackpropdata", "vpux::FunctionPass"> {
    let summary = "Legalize (Group)ConvolutionBackPropData to (Group)TransposedConvolution or 2x2/3x3 Convolution";

    let description = [{
      The pass performs the following conversions:

      1. **(Group)ConvolutionBackPropData → (Group)TransposedConvolution**
         - For general cases, and for GroupConvolutionBackPropData, the pass lowers these operations to their respective TransposedConvolution forms, reusing existing TransposedConvolution lowering. The transformation also handles the required weight reordering:
           - (Group)ConvolutionBackPropData: [(GROUPS,) C_IN, C_OUT, Z, Y, X]
           - (Group)TransposedConvolution:   [(GROUPS,) C_OUT, C_IN, Z, Y, X]

      2. **ConvolutionBackPropData → Multiple 2x2/3x3 Convolutions (with concat and DepthToSpace)**
         - When the `ConvolutionBackPropData` kernel is 4x4 with stride 2 and pad 1, the pass splits the operation into several smaller convolutions (either 2x2 or 3x3), followed by concatenation and a `DepthToSpace` operation.
         - The choice between 2x2 and 3x3 kernels is based on the output height × width (H×W) size, optimizing for memory efficiency and avoiding large DMA writes. This approach improves performance on supported architectures and tensor sizes.
         - The weight order is also changed from `[IC, OC, Y, X]` to `[OC, IC, Y, X]` to match the convention used by other convolution operations in the compiler.
         - Lowering flow for this case:
        ```
        Original operation:
            Input:         Activation: NxICxHxW
                           Weights:    ICxOCxKYxKX
                |
            ConvolutionBackpropData
                |
            Output:        NxOCxHoutxWout

        New subgraph:
            Input:         Activation: NxICxHxW     Weights: ICxOCxKYxKX
                |                                  |
                |                           Split weights into 4 (2x2 or 3x3) kernels
                |                                  |
                +--------------------------+-------+-------+-------+-------+
                |                          |               |       |
            Conv (2x2/3x3, offset0)   Conv (offset1)  Conv (offset2)  Conv (offset3)
                |                          |               |       |
                +--------------------------+---------------+-------+
                |
            Concat (along channel axis)
                |
            DepthToSpace (BLOCKS_FIRST, block size 2)
                |
            Output:        NxOCxHoutxWout (spatially interleaved)
        ```

      The weight order for TransposedConvolution and Convolution matches the convention used by other convolution operations in the compiler, simplifying conversion between transposed and forward convolutions.
    }];

    let constructor = "vpux::IE::createLegalizeConvBackpropDataPass()";

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
        AI-generated summary

        Adjusts convolution weight tensors when inputs come from Concat operations that
        themselves receive sliced data, by extracting only the utilized input channels from
        weights and padding remaining portions with zeros. Handles mixed sliced/non-sliced
        inputs by computing actual channel usage per concat input and reconstructing weights
        via slice, expand, and concat to match the effective input channel count.
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
        AI-generated summary

        Reduces DMA data copy overhead by folding convolution strides into input/kernel
        shapes. Matches convolutions with constant filters where stride can be absorbed by
        reshaping input and adjusting kernel channels via padding. Uses ShapeCast to reshape
        input and Const::DeclareOp to adjust filter shape, producing stride-1 convolutions
        with higher input channel alignment. Decreases expand channel requirements in the
        subsequent ExpandActivation pass, lowering memory bandwidth consumption.
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
        AI-generated summary

        This pass adjusts convolution input shapes by splitting a 1-D spatial dimension
        (e.g. 1x32x256x1) into two factors (e.g. 1x32x64x4), increasing the number of
        DPU workloads that can execute in parallel across clusters.

        Registered patterns:
        - ReshapeConv1DInputWithHalo<ConvolutionOp> — for 1D convolutions ([1,C,1,L] or
           [1,C,L,1]) with K>1, stride=1, dilation=1, no input/output_padding, and an
           activation element type supported by Const::createZerosConst (F16/F32/FP8/
           F4E2M1FN/per-tensor UniformQuantizedType; BF16 and per-axis quant are skipped).
           Folds the long axis into 2D [1, C, numRows, L_row+K-1] with an explicit K-1 halo
           (Concat-of-zero-const + Slice + Concat) so sliding-window semantics are preserved,
           then runs a regular 2D conv (pads=0) and AffineReshape back. Recovers DPU MPE
           utilization that is otherwise lost when H=1 (or W=1). Gated by a MAC threshold
           (see Example below).
        - ReshapeConv1DInputWithHalo<GroupConvolutionOp> — same for 1D grouped/depthwise.
        - ReshapeTransposedConv1DInputWithHalo — same idea for 1D TransposedConvolution with
           kernel % stride == 0, dilation=1 and no pads/output_padding. A strided transposed
           conv scatters each input sample into a K-wide output window, so the halo is mirrored:
           every row reads haloPad = K/stride - 1 extra samples per side and keeps only its own
           valid output range, which stays disjoint across rows (no overlap-add). The input is
           zero-extended when the row count does not divide evenly.
        - ReshapeConvInput<ConvolutionOp> — reshapes Convolution input when one
           spatial dimension is large and the other is 1, splitting the large
           dimension into two factors. Only applies to 1x1 kernels. Favors larger height.
        - ReshapeConvInput<GroupConvolutionOp> — same as above for GroupConvolution.
           Favors larger width.
        - ReshapeSingleConstDWConvInput — reshapes depthwise GroupConvolution input
           with a single constant weight (1x1 kernel) by splitting a 1-D spatial
           dimension into two for better utilization.
        - ReshapeAddInput — reshapes IE::AddOp inputs with 1-D spatial dimensions
           into 2D spatial, improving DPU utilization for element-wise addition.
        - ReshapeExpandDWConvInput — reshapes depthwise GroupConvolution input after
           Expand by adjusting the expanded channels and spatial dimensions.

        Example (ReshapeConvInput):
            Activation: 1x32x256x1     Weights:16x32x1x1
                |                       |
            Reshape: 1x32x64x4          |
                |                       |
                     Conv: 1x16x64x4
                            |
                     Reshape: 1x16x256x1

        Example (ReshapeConv1DInputWithHalo, K=11, Hp=64, SAME pad=5,5):
            Activation: 1x256x1x1600    Weights: 256x256x1x11
                |                        |
            Pad W [5,5] -> 1x256x1x1610  |
                |                        |
            Slice * 64 + Concat H        |
                |                        |
            1x256x64x35                  |
                  \\                    /
                   Conv (pads = 0)
                       |
                   1x256x64x25
                       |
                AffineReshape -> 1x256x1x1600

        Hp is the largest 4-multiple in [mpeAlignment, numDPU * mpeAlignment] that divides
        L_out and keeps L_row = L_out / Hp >= max(K, mpeAlignment) — bigger Hp gives the
        SOH tile-strategy more headroom. A 500 MMACs gate skips layers where Pad/Slice/Concat
        staging would dominate.

        Example (ReshapeTransposedConv1DInputWithHalo, K=8, stride=4, haloPad=1, numRows=17):
            Activation: 1x384x220x1        Weights: 192x384x8x1
                |                           |
            Pad H [1,1] -> 1x384x222x1      |
                |                           |
            Slice * 17 + AffineReshape      AffineReshape -> 192x384x1x8
                |                           |
            Concat H: 1x384x17x14           |
                  \\                       /
                TransposedConv (strides = [1, 4], pads = 0)
                       |
                   1x192x17x60
                       |
                Slice W [4, 56) -> 1x192x17x52   (drop each row's stride*haloPad context)
                       |
                AffineReshape -> 1x192x884x1
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
    let summary = "Adjust maxpool input shape for better hardware utilization and less splitting when handling large kernels";

    let description = [{
        AI-generated summary

        Adjust maxpool input shape for better hardware utilization and less splitting when handling large kernels.

        Registered patterns:
        - ReshapeMaxPoolOutput1x1 — when the MaxPool output is 1x1, reshapes the input from
           [1, C, H, 1] to [1, C, H/Int, Int] (or [1, C, 1, W] to [1, C, Int, W/Int]) so that
           the kernel can operate on a 2D spatial shape instead of a single large dimension.
        - ReshapeMaxPoolInputWithStride — reshapes MaxPool input with strides by expanding
           a 1-D spatial dimension into 2D, adjusting the kernel and stride accordingly.
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
        AI-generated summary

        Converts element-wise Multiply and Add operations with broadcast-compatible
        weight tensors into ScaleShift ops, which can later be lowered to depthwise
        GroupConvolution for DPU execution. Handles weight shapes with spatial
        dimensions by applying permutations (H_EXPAND or W_EXPAND) to reposition dimensions
        into channel position, enabling ScaleShift compatibility. Broadcasts scalar weights
        across channels and handles optional FakeQuantize operations. Skips conversion for
        trivial Reorders with overlapped inputs (HALO regions) that would introduce excessive
        copies.
    }];

    let constructor = "vpux::IE::createConvertToScaleShiftPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableNCEEltwiseMultiply", "enable-nce-eltwise-multiply",
            "bool", "false",
            "Flag to support Multiply to NCEEltwise conversion"
        >
    ];
}

//
// DecomposeLSTMSequence
//

def DecomposeLSTMSequence : PassBase<"decompose-lstm-sequence", "vpux::FunctionPass"> {
    let summary = "Decompose LSTMSequenceOp";

    let description = [{
        AI-generated summary

        Registered patterns (in priority order):
        - DecomposeDynamicLSTMSequenceBidirectionalRewriter (highest priority) —
           decomposes bidirectional LSTMSequence with dynamic shapes into one
           forward and one reverse operator.
        - ExtractWeightsAndBiasesFromLSTMSequenceRewriter — extracts the matrix
           multiplication of inputData and weights, and the addition of biases
           from LSTMSequence to allow them to run on the DPU.
        - DecomposeLSTMSequenceBidirectionalRewriter — decomposes bidirectional
           LSTMSequence (static shapes) into one forward and one reverse operator.
        - UnrollLSTMSequenceToLSTMCellsRewriter (lowest priority) — unrolls
           remaining LSTMSequenceOps into per-timestep LSTMCellOps.
    }];

    let constructor = "vpux::IE::createDecomposeLSTMSequencePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}


//
// DecomposeGRUSequence
//

def DecomposeGRUSequence : PassBase<"decompose-gru-sequence", "vpux::FunctionPass"> {
    let summary = "Decompose GRUSequenceOp";

    let description = [{
        This pass:
        - Extracts the matrix multiplication of inputData and weights, and the
          addition of biases from GRUSequence to allow them to run on the DPU.
        - Convert an GRUSequenceLastPart operator to seqLength GRUCell operators.
          Every GRUCell will be split in 1 MatMul and remaining GruGates operator
          if shouldLinearBeforeReset is true. If shouldLinearBeforeReset is false,
          Cell will be decompose in all basic operation components.
    }];

    let constructor = "vpux::IE::createDecomposeGRUSequencePass()";

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
        AI-generated summary

        Decomposes GRUCellOp into basic operations (MatMul, Multiply, Add, Sigmoid, Tanh,
        Split, Slice) following GRU cell mathematics. Expands the single GRUCell operation
        into three matrix multiplications (xw, hr, rh) with reset and update gate
        computations using Sigmoid activation. Handles optional biases and the
        linear_before_reset mode which changes how reset gate interaction with hidden
        state is computed.
    }];

    let constructor = "vpux::IE::createDecomposeGRUCellPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
//  DecomposeL2Ops
//

def DecomposeL2Ops : PassBase<"decompose-l2-ops", "vpux::FunctionPass"> {
    let summary = "Decompose NormalizeL2 and ReduceL2 ops";

    let description = [{
        AI-generated summary

        Decomposes L2 normalization and reduction operations into simpler element-wise ops.

        - NormalizeL2Op — decomposed into Multiply(input, input) -> ReduceSum -> Sqrt -> Divide(input, sqrt)
           only when axes parameter contains all dimensions of the input tensor.

        - ReduceL2Op — decomposed into Multiply(input, input) -> ReduceSum -> Sqrt.
           Only applied when input total size >= VPU_CHANNEL_ALIGNMENT (16).
           For FP16 output, skips decomposition if the number of reduced elements
           exceeds 64 to avoid accuracy loss from FP16 accumulation overflow.
    }];

    let constructor = "vpux::IE::createDecomposeL2OpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeSoftPlus
//

def DecomposeSoftPlus : PassBase<"decompose-softplus", "vpux::FunctionPass"> {
    let summary = "Decompose SoftPlus into Exp + Add(bias=1) + Log when input is bounded below threshold";

    let description = [{
        The SoftPlus Shave kernel computes:
            SoftPlus(x) = log(1 + exp(x)), if x < threshold (= 11.0)
                        = x,               if x >= threshold
        This pass decomposes IE::SoftPlusOp into IE::ExpOp -> IE::AddOp(const=1.0) -> IE::LogOp
        when the input is statically bounded to (-inf, threshold), making the decomposition
        numerically equivalent to the kernel.

        Currently supported pattern:
            Abs -> Multiply(splat <= 0) -> SoftPlus
        Since Abs >= 0 and the multiplier is <= 0, the product is <= 0 < threshold.

        The resulting ops can be further lowered to DPU-friendly kernels,
        reducing Shave execution time.
    }];

    let constructor = "vpux::IE::createDecomposeSoftPlusPass()";

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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces `IE::BroadcastOp` and `IE::DynamicBroadcastOp` operations
        with `IE::TileOp` and `IE::DynamicTileOp` respectively. After conversion,
        TileOp canonicalization patterns are applied to simplify the results.
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
        AI-generated summary

        Converts Subtract operations to Add with negated second operand, enabling NCE
        execution. Handles four scenarios: tensor-tensor subtraction with optional
        FakeQuantize propagation, constant-tensor subtraction converted to bias in
        GroupConv, tensor-constant subtraction with negated constant, and
        Multiply-with-constant as second input converted to negated Multiply then Add.
        Validates input types (no integer/F32 inputs), output expansion requirements,
        and whether conversion to GroupConv+Add is beneficial versus software execution.
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
        AI-generated summary

        Converts ScaleShift operations to 1x1 DepthWise Convolutions with stride/dilation
        of 1. Extracts scale weights and bias from ScaleShift and reshapes them into
        depthwise filter [channels, 1, 1, 1] format. Creates GroupConvolution with group
        count equal to output channels, transforming channel-wise scaling into NCE-executable
        convolution. Validates through isBeneficialConvertScaleShiftToDW that conversion
        reduces operations and improves DPU utilization.
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
        AI-generated summary

        The pass is a part of `OptimizeActivations` pipeline.

        Registered patterns:
        - FusePostOpsRewriter — matches operations with the EltwiseOp trait (e.g.
           ReLU, LeakyReLU, Sigmoid) and fuses them as post-processing into a
           preceding task that supports post-ops (via LayerWithPostOpInterface).
        - FuseClampRewriter — matches IE::ClampOp and fuses it as a clamp
           post-processing operation into a preceding compatible task.
    }];

    let constructor = "vpux::IE::createFuseActivationOpsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePermuteQuantize
//

def FusePermuteQuantize : PassBase<"fuse-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute combination in 1 common operation";

    let description = [{
        AI-generated summary

        Fuses Reorder (Permute) operations followed by NCE ops (Conv, GroupConv,
        TransposedConv, AvgPool, Add) with quantization into PermuteQuantize ops. Matches
        patterns where F16->quantized type conversion occurs after data reshaping, validating
        that inputs don't have overlapped HALO regions that would require additional copies.
        Converts PermuteQuantize pattern only when non-overlapped (kernel size <= stride) to
        avoid redundant data movement from tiling.
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
        >
    ];
}

//
// LegalizeDilatedConvolution
//

def LegalizeDilatedConvolution : PassBase<"legalize-dilated-conv", "vpux::FunctionPass"> {
    let summary = "Handle dilated convolutions";

    let description = [{
        AI-generated summary

        The pass is a part of `buildHardwareModePipeline` pipeline.

        Expands filters of dilated Convolution and GroupConvolution operations
        by inserting zeros according to the dilation factors. This transforms
        dilated convolutions into equivalent non-dilated convolutions with larger
        kernel sizes, enabling execution on the DPU which does not natively
        support dilation.
    }];

    let constructor = "vpux::IE::createLegalizeDilatedConvolutionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ResolveStridedSlice
//

def ResolveStridedSlice : PassBase<"resolve-strided-slice", "vpux::FunctionPass"> {
    let summary = "Decouple strided slice to slice + reshape";

    let description = [{
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Uses OpenVINO SlicePlan to decompose IE::StridedSlice into simpler operations:
        - Adjusts attributes' rank to match input rank.
        - Replaces StridedSlice with non-zero masks by a simplified StridedSlice (zero masks) + Reshape.
        - Replaces StridedSlice with dense<1> strides by a simple IE::Slice operation.
        - Handles reverse axes (negative strides) by reversing elements along each axis
          via Slice+Concat, then correcting begin/end indices.
        - Merges adjacent unit-sized dimensions for rank reduction when input rank exceeds 4.
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Registered patterns:
        - ParallelStridedSliceToConvolutionConverter — converts multiple parallel
           StridedSlice operations (same input, different channel offsets, stride > 1)
           into a single Convolution with appropriately constructed identity-like weights
           that select the required channels and strides in one DPU operation.
        - StridedSliceOpConverter — converts a single StridedSlice with strides > 1
           into a depthwise Convolution with identity weights and matching strides,
           optionally propagating input FakeQuantize to the weights.
    }];

    let constructor = "vpux::IE::createConvertStridedSlice2ConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapConvertWithSWOp
//

def SwapConvertWithSWOp : PassBase<"swap-convert-with-sw-op", "vpux::FunctionPass"> {
    let summary = "Prepare ConvertOp for DPU fusion";

    let description = [{
        AI-generated summary

        Moves IE.ConvertOp (F16-to-F32) from after reshape-kind operations to directly
        after the preceding NCE or Interpolate operation. Matches ConvertOp with output
        tensors exceeding 36K elements, walks back through Transpose/Reshape/AffineReshape
        chains to find the NCE-supported producer, and inserts a new IE.ConvertOp immediately
        after that producer. The original ConvertOp is erased, enabling subsequent passes to
        fuse the conversion into the NCE operation's output type.
    }];

    let constructor = "vpux::IE::createSwapConvertWithSWOpPass()";

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

        The pass handles IE.ConvertOp with f16 input and f32 output in two ways:
        - fuseWithParentDPUOp — if the Convert has a single-use parent that is a
           supported DPU op (or beneficial Interpolate), the parent output type is
           changed to f32 and the Convert is removed.
        - replaceWithIdentityPool — if the Convert cannot be fused with its parent
           (e.g. no parent, parent has multiple uses), and the platform indicates
           that running Convert on DPU is beneficial, the Convert is replaced by an
           identity AvgPool that performs the f16->f32 type change on the DPU.
           Only applies to 4D tensors with C=1, N=1, and H*W in (512, 8192].
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Registered patterns:
        - SliceConcatRewriter — matches Concat that joins a Slice output with a
           constant, replacing the Concat with an Add to fuse the slice into the
           preceding Convolution.
        - FuseSliceWithConvRewriter — fuses a Slice into its parent Convolution
           by adjusting the convolution output channels and weights to produce
           only the sliced range.
        - SliceOpWithReshapeConverter — promotes a non-4D Slice (rank >= 3, rank != 4)
           that slices only on the innermost dimension into a 4D Slice via Reshape,
           enabling the downstream SliceOpWithLayoutConverter + SliceOpConverter chain
           to convert it to a Convolution.
        - SliceOpWithLayoutConverter — converts a Slice with strided access on a
           non-trivial layout (e.g., NHWC) into a Convolution when the stride is
           on the channel dimension and ShapeCast can provide channel alignment.
        - SliceOpConverter — converts a Slice with stride > 1 into a depthwise
           Convolution when ShapeCast can replace Expand for channel alignment,
           avoiding costly DDR-to-DDR stride copies.
    }];

    let constructor = "vpux::IE::createOptimizeSliceWithStridePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "disableMinHWThreshold", "disable-min-hw-threshold",
            "bool", "false",
            "[Optional] Disable H*W threshold check in isBeneficialToConvertSliceToConv. "
            "When true, the pass converts Slice to Convolution regardless of spatial size."
        >
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Extends 3D input, filter, and output tensors of NCE operations to 4D by
        inserting height = 1.
        [N, C, W] -> [N, C, 1, W]
        strides:    {2} -> strides:    {1, 2}
        pads_begin: {2} -> pads_begin: {0, 2}
        pads_end:   {2} -> pads_end:   {0, 2}
        dilations:  {2} -> dilations:  {1, 2}

        Handled operations:
        - ConvolutionOp, GroupConvolutionOp, TransposedConvolutionOp,
          GroupTransposedConvolutionOp, MaxPoolOp, AvgPoolOp
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces some `GroupConvolution` operations with simpler operations.
        There are three converters (applied in priority order):
        - DepthwiseConvSinglePixelInputToMultiplyConverter (highest priority) — converts
           depthwise GroupConv with single-pixel (1x1) spatial input to IE::Multiply.
        - GroupConvToSingleConvConverter (medium priority) — converts to a single Convolution
           by reconstructing constant weights. Only for GroupConv with constant weights.
        - GroupConvToMultiConvConverter (lowest priority) — general converter that slices
           input and weights per group, creates per-group Convolutions, and concatenates results.
    }];

    let constructor = "vpux::IE::createConvertGroupConvToConvPass()";

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
        AI-generated summary

        The pass is a part of `DefaultHW` pipeline.

        Unrolls 3D convolutions to 2D convolutions so that they can be inferred
        on the DPU, which does not support 3D convolution natively.

        Registered patterns:
        - ConvGeneralRewriter<ConvolutionOp> — unrolls 3D Convolution along the
           depth dimension into multiple 2D Convolutions with appropriate slicing
           and padding, then concatenates results.
        - ConvGeneralRewriter<GroupConvolutionOp> — same for GroupConvolution.
        - TransposedConvGeneralRewriter — unrolls 3D TransposedConvolution into
           multiple 2D TransposedConvolutions with per-depth-slice processing.
        - ConvGeneralAggregation<ConvolutionOp> — aggregates depth-1 3D
           Convolution into equivalent 2D form by removing the unit depth dimension.
        - ConvGeneralAggregation<GroupConvolutionOp> — same for GroupConvolution.
    }];

    let constructor = "vpux::IE::createUnrollConv3dToConv2dPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def UniquifySimilarOps : PassBase<"uniquify-similar-ops", "vpux::FunctionPass"> {
    let summary = "Replace similar (but not identical) operations";

    let description = [{
        AI-generated summary

        The pass is similar to common subexpression elimination in nature,
        however, it handles more comprehensive cases where operations are not
        identical. As such, they cannot be optimized in a general fashion.

        Currently the pass specifically handles IE::MemPermuteOp deduplication.
        When multiple MemPermute operations share the same input, the pass keeps the first
        one and replaces duplicates. Two MemPermute operations are considered duplicates if
        they produce the same type, or if a PermuteCast can bridge the difference in output
        layout/shape.
    }];

    let constructor = "vpux::IE::createUniquifySimilarOpsPass()";

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
        AI-generated summary

        This pass splits inputs of operations with batch > 1 by unrolling along the batch dimension.
        It operates on any operation implementing the IE::UnrollBatchOpInterface, using the
        interface's `doesNeedToUnroll` and `getNumberInputs` methods to determine eligibility.

        For example:
        * `FullyConnected` input with 2x64 geometry will be split by two inputs with 1x64 dimensions.
        * `Convolution` input 3x16x32x64 will be split into three 1x16x32x64 inputs.
        Resulting tensors go to corresponding operations and the outputs are concatenated.

        When skipUnrollBatch is set, Convolution, MaxPool, and AvgPool are excluded.
    }];

    let constructor = "vpux::IE::createUnrollBatchPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "skipUnrollBatch", "skip-unroll-batch",
            "bool", "false",
            "[Optional] Set to true to skip unroll-batch pass for specific operations"
        >
    ];
}

//
// AdjustQKVTransposeForUnrollSDPA
//

def AdjustQKVTransposeForUnrollSDPA : PassBase<"adjust-qkv-transpose-for-unroll-sdpa", "vpux::FunctionPass"> {
    let summary = "Split shared QKV projection before SDPA unrolling";

    let description = [{
        This pass transforms SDPA patterns containing transposes on the Q, K, and V inputs,
        enabling the UnrollSDPA pass to eliminate those transposes.
    }];

    let constructor = "vpux::IE::createAdjustQKVTransposeForUnrollSDPAPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// UnrollSDPAPattern
//

def UnrollSDPAPattern : PassBase<"unroll-sdpa-pattern", "vpux::FunctionPass"> {
    let summary = "Unroll SDPA pattern (MatMul -> Add -> Softmax -> MatMul) and rearrange computation";

    let description = [{
        This pass detects and unrolls the pattern: MatMul -> Add -> Softmax -> MatMul.

        The pass performs the following transformations:
        1. Detects MatMul -> Add -> Softmax -> MatMul patterns
        2. Unroll first Matmul inputs
        3. Creates independent Matmul -> Add -> Softmax -> MatMul chains for each split
        4. Concatenates all results together

        This enables better parallelization and resource utilization on VPU.

        Example:
        Before: MatMul1 -> Add -> Softmax -> MatMul2
        After:
            MatMul1_split1 -> Add1 -> Softmax1 -> MatMul2_split1 \
            MatMul1_split2 -> Add2 -> Softmax2 -> MatMul2_split2 -> Concat
            ...
    }];

    let constructor = "vpux::IE::createUnrollSDPAPatternPass()";

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
        AI-generated summary

        The pass is a part of `InitialTransformations` pipeline.

        Converts MVN6 operations to the simpler MVN1 (MVNOp) representation when
        possible. Supported cases:
        - 2D/3D/4D inputs with normalization along the last axis
        - 4D inputs with normalization across channels (axes [1,2,3]) or spatial
          dimensions (axes [2,3])
        - 4D inputs with single-axis normalization on axis 1 or 2 (uses Transpose
          to move normalized dim to spatial position)
        - 5D inputs with spatial axes [2,3,4] (reshaped to 4D)
        - 3D inputs with axes [1,2] or axis [1] when last dim is 1
        - OUTSIDE_SQRT eps mode accepted only when epsilon <= 1e-3
        - No scale/bias inputs allowed
        Inserts Reshape/DynamicReshape around MVN1 when input rank differs from 4D,
        and Transpose when the normalized axis needs repositioning.
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
        It works differently on some architectures depending on the supported quantization levels
        and performs the following transformations to U16 FakeQuantize (in order):

        1. U16FQConsolidationRewriter
           Restore some U16 FQs that can be removed in step 5. Usually such a restoration is performed by ConvertQuantizeDequantize
           NGraph pass, however, there are cases where one of the operations in chain has multiple users and/or there is data op, e.g:
                                                  -> Result
               FQ(F16) -> Convert(U16) -> Reshape
                                                  -> Convert(F16) -> Mul -> Add
           This rewriter matches:
               FQ -> Convert(optional) -> Reshape/Transpose -> Convert(optional) -> Mul -> Add
           and transforms it into a single FQ.
           It's needed to avoid having separate Quantize and Dequantize ops with conversion into FP16 in-between, which will
           loss precision since FP16 represents only a subset of U16 values.

        2. U16FQConvertToQuantizeRewriter
           Looks for patterns FakeQuantize + Convert(U16) and transforms it into single Quantize layer to avoid converting FQ
           into ScaleShift. Converting FQ into ScaleShift will force the output to be converted to FP16 first and only then it
           will be converted to U16, which is inaccurate since FP16 represents only a subset of U16 values.

        3. LowerFakeQuantizeRewriter
            a. If enableU16ToU8Lowering flag is set to true in the bool options of the pass then enable lowering U16 FakeQuantize
               to U8 using zero-point shift calculation. See ticket E#179583.
            b. The lowering logic will lower a FakeQuantize from U16 to U8 if one customer is either a ConvolutionOp, or a GroupConvolutionOp.
               If it is a MatMulOp, or a FullyConnectedOp then check that both inputs are optionally (ReshapeOp, AffineReshapeOp, TransposeOp),
               and eventually FakeQuantize by searching bottom-up, if so do not lower. Ensure that the splat value has not failed on in_low, in_high,
               out_low, and out_high of FakeQuantizeOp. Ensure that in_low != out_low and in_high != out_high for zero-point calculation.

        4. ReplaceChildU16FakeQuantizeWithReluRewriter (only when enableAdaptiveStripping is set):
           Handles the mirror case of the parent ReLU replacement in step 5 for a parent U16 FQ -> child U16 FQ chain.
           Matches the child FQ as root and replaces it with a ReLU consuming the parent FQ's input when:
           - The child FQ is per tensor with in_low == out_low == 0 (ReLU pattern).
           - The parent FQ is also a U16 FQ but does NOT itself match the ReLU pattern
             (that case is handled by step 5 acting on the parent).
           Must run before RemoveU16FakeQuantizeRewriter (step 5) so that the parent FQ is
           still present and the chain is matchable.

        5. RemoveU16FakeQuantizeRewriter:
           a. If the input to the FakeQuantize is a constant, the quantization is folded directly
              into the constant values and the FakeQuantize op is replaced with the resulting constant.
           b. In case enableAdaptiveStripping is true:
               - If any user of the FQ output is a U16 FQ child and the current FQ is per
                tensor with in_low == out_low == 0 and in_high == out_high and has exactly one use
                (the child FQ), it is replaced with a ReLU activation function. All users of the
                output are scanned for a U16 FQ child, not just the first use-list entry.
              - If no user of the FQ output is a U16 FQ and the FQ is per tensor with
                in_low == out_low == 0 (in_high/out_high are not required to match) and its direct
                predecessor implements LayerWithPostOpInterface (e.g. Add, Multiply) and that
                predecessor does not implement ElemTypeInfoOpInterface (which would indicate an op
                such as MaxPool that manages its own output element type and must not have a ReLU
                fused via this path), it is replaced with a ReLU activation function. Non-U16 FQ
                users (e.g. FQ with 256 levels) do not suppress this rewrite because the FQ is
                semantically a ReLU clamp for all consumers.
              - If the FakeQuantize has in_low != out_low or in_high != out_high, it is replaced with a ScaleShift op.
              - Otherwise the FakeQuantize is completely removed.
           c. In case enableAdaptiveStripping is false and the FakeQuantize is per tensor and the input and output low
              is equal to 0 it is replaced with a ReLU activation function.
           d. Otherwise the FakeQuantize is completely removed.
    }];

    let constructor = "vpux::IE::createHandleU16FakeQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantDialect"
    ];
}

//
// QDQOptimizationAggressive
//

def QDQOptimizationAggressive : PassBase<"qdq-optimization-aggressive", "vpux::ModulePass"> {
    let summary = "Dynamic QDQOptimizationAggressive";

    let description = [{
        Create an MLIR dynamic pipeline to call HandleU16FakeQuantize and its related QDQOptimizationAggressive passes.
        The flag enableQDQOptimizationAggressive is found in InitCompiler and is quieried at runtime when calling the required passes.
    }];

    let constructor = "vpux::IE::createQDQOptimizationAggressivePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantDialect"
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
        AI-generated summary

        The pass is a part of `HardwareMode` pipeline.

        Registered patterns:
        - SplitConvWithPostOpAndFakeQuant — matches FakeQuantize whose input is a
           Convolution that also has a fused post-op consumer. Splits the Convolution
           so the FakeQuantize gets its own dedicated Convolution copy.
        - SplitConvWithOnlyFakeQuantConsumers — matches Convolution with multiple
           FakeQuantize consumers. Duplicates the Convolution for each FakeQuantize
           consumer so each can be independently quantized and fused with bias or
           post-processing.
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
        AI-generated summary

        Converts 2D TransposedConvolution to Upsampling followed by Convolution.
        Requires a convertible op (canConvertTransposedConvToConv) and keeps the
        original dilation attribute on the created IE.ConvolutionOp instead of
        hard-coding [1, 1].

        Upsampling padding is computed from effective kernel size:
            K_eff - 1 = (K - 1) * dilation
        so dilated cases are shape-correct after conversion.

        When output shape is explicitly provided and the converted Conv output
        is larger, the pass inserts Slice to crop back to requested output shape.

        The pass may skip conversion when SEP TransposedConvolution support is
        enabled and preferred for the platform.
    }];

    let constructor = "vpux::IE::createConvertTransposedConv2DToConv2DPass()";

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
        AI-generated summary

        Converts 2D GroupTransposedConvolution to GroupConvolution.
        Validates input/output/filter ranks, requires a valid filter producer,
        and reshapes filter from 5D [G, OC, IC, KY, KX] to 4D by folding groups
        into output channels.

        Generic path:
        - build Upsampling + optional output padding
        - create IE.GroupConvolutionOp with stride [1, 1]
        - preserve original dilation on created GroupConvolutionOp

        Depthwise-dilated fast path (stride=1, no output-padding, no output-shape attr):
        - computes equivalent non-negative padding
        - expands filter by IE.ExpandDilatedOp
        - emits IE.GroupConvolutionOp with dilations [1, 1]
        - falls back to generic path when equivalent padding is negative
    }];

    let constructor = "vpux::IE::createConvertGroupTransposedConvToGroupConvPass()";

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
        AI-generated summary

        Registered patterns:
        - GroupTransposedConvConverter — replaces IE::GroupTransposedConvolution
           by slicing input and weights per group, creating individual
           IE::TransposedConvolution operations per group, and concatenating
           results. Only applicable when SEP (Storage Element Pointer) operations
           are enabled.
        - DepthwiseGroupTransposedConvConverter — converts depthwise
           GroupTransposedConvolution (where groups == input channels) to a single
           IE::TransposedConvolution. Also requires SEP support.
    }];

    let constructor = "vpux::IE::createConvertGroupTransposedConvToTransposedConvPass()";

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
        AI-generated summary

        Splits convolution and pooling operations with strides exceeding 8 into multiple
        tiled operations (tile width = kernel size) concatenated back together. Creates
        output_height x output_width independent ops with stride 1, each processing a
        slice of input matching kernel dimensions with adjusted padding. Reconstructs
        output by concatenating tiled results first along width then along height, handling
        FakeQuantize propagation through temporary FQ cloning.
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
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Registered patterns:
        - UseQuantDequant — splits FakeQuantize with non-constant input into a
           `IE::QuantizeOp -> IE::DequantizeOp` (quant.qcast -> quant.dcast) pair
           using uniform quantized types derived from FQ range parameters.
           In the common case where input data is already quantized correctly,
           the pass adds CastElemType and Dequantize attributes instead of
           inserting a Quantize operation.
        - UseConstDequant — splits FakeQuantize with constant input. Adds
           CastElemType to reinterpret data as the quantized type, then
           IE::DequantizeOp. Applies Quantize only when the input range does
           not match the storage type range.
    }];

    let constructor = "vpux::IE::createSplitFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantDialect"
    ];
}

//
// ConvertToQuantizedOps
//

def ConvertToQuantizedOps : PassBase<"convert-to-quantized-ops", "vpux::FunctionPass"> {
    let summary = "Convert ConvertOp to Quantize/Dequantize operations";

    let description = [{
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        It matches patterns:
        - non-const -> Convert -> ViewLikeOp/TransposeOp -> Convolution/GroupConvolution,
           and replaces ConvertOp with QuantizeCastOp -> DequantizeOp if the non-const input can be propagated to the
           filter of a Convolution/GroupConvolution. We expect that DequantizeOp will be propagated to the
           Convolution/GroupConvolution and then be optimized by mixed precision.

        - [GroupConvolution] -> Round -> Clamp -> Convert,
           or [GroupConvolution] -> Clamp -> Round -> Convert,
           and replaces with Quantize -> QuantizeCast.
    }];

    let constructor = "vpux::IE::createConvertToQuantizedOpsPass()";

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
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Registered patterns:
        - MergeQuantDequant — merges a `Quantize -> Dequantize` pair back into a
           single IE::FakeQuantizeOp, deriving FQ levels and ranges from the
           intermediate quantized type. The pass is used as a fallback to FP16
           computations when quantized types were not consumed by layers.
        - MergeQuantCastDequant — merges a `Quantize -> QuantizeCast -> Dequantize`
           triple into a single IE::FakeQuantizeOp with separate input and output
           ranges derived from the QuantizeCast input/output quantized types.
    }];

    let constructor = "vpux::IE::createMergeFakeQuantPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateAndFuseQuantizeDequantize
//

def PropagateAndFuseQuantizeDequantize : PassBase<"propagate-and-fuse-quantize-dequantize", "vpux::FunctionPass"> {
    let summary = "Propagate Quantize/Dequantize through agnostic operations";

        let description = [{
        AI-generated summary

        The pass is a part of LowPrecision pipeline.

        Registered patterns:
        - PropagateQuantize — propagates QuantizeOp backward through operations
           that implement ElemTypeInfoOpInterface (e.g. Reshape, Transpose), moving
           the quantization closer to producing operations.
        - PropagateDequantize — propagates DequantizeOp forward through operations
           that implement ElemTypeInfoOpInterface, moving dequantization closer to
           consuming operations. Respects SE-ops enablement.
        - FuseDequantizeWithMultiplier — converts
           Quantize -> QuantizeCast(fuse with multiply const) -> Dequantize
           by folding a Multiply constant into the quantization scale via QuantizeCast.
        }];

    let constructor = "vpux::IE::createPropagateAndFuseQuantizeDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantDialect"
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
// AdjustFakeQdqParams
//
def AdjustFakeQdqParams : PassBase<"adjust-fake-qdq-params", "vpux::FunctionPass"> {
    let summary = "Checks FQ layers to see whether input or output quantization params exceeds FP16 range and adjusts the parameters to be in scale.";

    let description = [{
        The pass checks if the FQ params are in FP16 range.
        It updates the parameters of such FQ operations and introduces multiply operators
        that are propagated through the graph.
    }];

    let constructor = "vpux::IE::createAdjustFakeQdqParamsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseQuantizationMultiply
//

def FuseQuantizationMultiply : PassBase<"fuse-fq-and-mul", "vpux::FunctionPass"> {
    let summary = "Fuse Mul into FakeQuantize, Dequantize, or DynamicDequantize";

    let description = [{
    The pass is a part of `InitialTransformations` pipeline.


    FakeQuantize + Multiply

         data  in_L in_H out_L out_H
           |    |    |     |     |
           |    |    |     |     |                data  in_L  in_H  out_L * C  out_H * C
           v    v    v     v     v                  |    |     |        |          |
         +-------------------------+                |    |     |        |          |
         |       FakeQuantize      |                v    v     v        v          v
         +-------------------------+             +------------------------------------+
                      |                =====>    |            FakeQuantize            |
                      v                          +------------------------------------+
                 +----------+                                      |
                 | Multiply | <--- C                               v
                 +----+-----+
                      |
                      v

    Fuse Multiply into FakeQuantize when the fused FakeQuantize is per-tensor or
    per-channel, regardless of whether the input of FakeQuantize is constant or not.

    Dequantize + Multiply

    Constant Weights

         Quantized Weights (Const)
         (scale = S, zp = Z)
                  |
                  v
         +----------------+                       Quantized Weights (Const)
         |   Dequantize   |                        (scale = S * C, zp = Z)
         +----------------+                                  |
                  |                                          v
                  v                  =====>          +----------------+
            +----------+                             |   Dequantize   |
            | Multiply | <--- C                      +----------------+
            +----------+                                     |
                  |                                          v
                  v

    Block Arguments


       Quantized Weights                            Quantized Weights
        (scale = S, zp = Z)                          (scale = S, zp = Z)
                |                                              |
                v                                              v
        +----------------+                           +------------------+
        |   Dequantize   |                           |  QuantizeCast    |
        +----------------+                           | (scale = S*C)    |
                |                                    +------------------+
                v                                              |
          +----------+              =====>                     v
          | Multiply | ← C                            +----------------+
          +----------+                                |   Dequantize   |
                |                                     +----------------+
                v                                              |
                                                               v

    Fuse Multiply into Dequantize by updating the quantization scale:
    - For constant weights: Updates the constant declaration with new quantization
      parameters (scale = original_scale * multiply_factor)
    - For block arguments: Inserts a QuantizeCast operation to
      requantize with the updated scale at runtime

     Supported Quantization Types:
    - UniformQuantizedType (per-tensor): Single scale and zero point
    - UniformPerAxisQuantizedType (per-channel): Per-channel scales along axis 0

    DynamicDequantize + Multiply

      input   scale   zp                       input   scale * C   zp
        |       |     |                          |        |        |
        v       v     v                          v        v        v
      +-------------------+                    +-----------------------+
      | DynamicDequantize |         =====>     |   DynamicDequantize   |
      +-------------------+                    +-----------------------+
                |                                          |
                v                                          v
           +----------+
           | Multiply | <--- C
           +----------+
                |
                v

    Since IE.DynamicDequantize computes `(input - zp) * scale`, a trailing Multiply(C)
    folds directly into its (constant) scale operand as `scale' = scale * C`, for both
    a splat multiplier and a per-channel multiplier that aligns with the scale. The
    weights-import marker attribute is preserved on the rewritten op.

    }];


    let constructor = "vpux::IE::createFuseQuantizationMultiplyPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "fuseFQAndMulWithNonConstInput", "fuse-fq-and-mul-with-non-const-input",
            "bool", "false",
            "Flag to enable or disable the fusion of FQ and Multiply when the FQ has a non-constant input"
        >
    ];
}

//
// EltwiseFakeQuantizeFusion
//

def EltwiseFakeQuantizeFusion : PassBase<"eltwise-fake-quantize-fusion", "vpux::FunctionPass"> {
    let summary = "Fuse Eltwise into FakeQuantize if one of Eltwise's input is a constant scalar (can be also a quantized constant)";

    let description = [{
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Fuses a constant scalar element-wise operation into the input range of
        a FakeQuantize consumer. Handles four element-wise op types with their
        inverse operations:
        - IE::AddOp (inverse: subtraction)
        - IE::SubtractOp (inverse: addition)
        - IE::MultiplyOp (inverse: division)
        - IE::DivideOp (inverse: multiplication)

        The constant scalar value is absorbed into the FakeQuantize input_low and
        input_high parameters using the inverse operation, removing the element-wise op.
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
        AI-generated summary

        Swaps intermediate operations (implementing ElemTypeInfoOpInterface) past
        fusable operations so that bias and activation functions can be fused into
        the main DPU operation.

        Registered patterns:
        - SwapWithBias<AddOp>, SwapWithBias<ScaleShiftOp> — moves reshape-like ops
           before bias Add/ScaleShift so they become adjacent to the convolution.
        - SwapWithActivation<ClampOp> — moves reshape-like ops before Clamp.
        - SwapGeluExpand — swaps Gelu past Expand.
        - SwapTanhSlice, SwapTanhShapeCast — swaps Tanh past Slice/ShapeCast.
        - SwapExpandQuantizeCast — swaps Expand past QuantizeCast.
        - SwapGatherQuantizeCast — swaps Gather past QuantizeCast.
        - SwapDequantMemPermute — swaps Dequantize past MemPermute.
        - SwapBatchedMemPermuteSlice — swaps batched MemPermute past Slice.
        - SwapAffineReshapeFakeQuantize — swaps AffineReshape past FakeQuantize.
        - SwapLeakyReluTranspose — swaps LeakyRelu past Transpose.
        - FindChildActivationAndSwap — interface-based pattern that searches for
            activation children of LayerWithPostOpInterface ops and reorders them.
        }];

    let constructor = "vpux::IE::createSwapOperationsPass()";

    let dependentDialects = [
            "vpux::IE::IEDialect"
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
        "mlir::quant::QuantDialect"
    ];
}

//
// ConvertToPalletizationLUT
//

def ConvertToPalletizationLUT: PassBase<"convert-to-pallet-lut", "vpux::FunctionPass"> {
    let summary = "Convert quantized sub-byte datatypes to a palletized LUT with float/int quantile type";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        This pass detects quantized types in certain conditions and converts them to a float LUT representation by
        fully subtracting the zero point. Depending on the activation type the LUT quantileType chosen can be either fp16 or fp8.
        This pass is not present in NPU37XX pipeline because palletization is not implemented in NPU37XX.
    }];

    let constructor = "vpux::IE::createConvertToPalletizationLUT()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "vpux::type::QuantileDialect"
    ];
}

//
// RemoveQuantDequantSeq
//

def RemoveQuantDequantSeq : PassBase<"remove-quantdequant-seq", "vpux::FunctionPass"> {
    let summary = "Removes quantize->dequantize ops sequence";

    let description = [{
        AI-generated summary

        The optional pass in the `LowPrecision` pipeline.

        Removes Quantize->Dequantize sequences in three scenarios:
        - Direct pattern: Quantize -> Dequantize (simple removal).
        - Chain pattern: Quantize -> [ElemTypeInfoOpInterface chain] -> Dequantize
           (skips Quantize, infers return types through the chain, removes Dequantize).
        - Concat pattern: multiple Quantize -> [ElemTypeInfoOpInterface] -> Concat ->
           [ElemTypeInfoOpInterface] -> Dequantize (skips all Quantize ops, infers types
           through intermediate ops and Concat, removes Dequantize).
        Also erases any remaining fp16->fp16 Dequantize ops created by prior alterations.
    }];

    let constructor = "vpux::IE::createRemoveQuantDequantSeqPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect",
        "mlir::quant::QuantDialect"
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
        "mlir::quant::QuantDialect"
    ];
}

//
// ConvertWeightsToUnsigned
//

def ConvertWeightsToUnsigned : PassBase<"convert-weights-to-unsigned", "vpux::FunctionPass"> {
    let summary = "Shift weight data from a signed range to an unsigned one";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        This pass detects quantized convolution and shifts weights data from a signed range to an unsigned one.
        Uses a platform-specific strategy to determine which conversions to apply:
        - Default: converts I8 weights to U8.
        - Later platforms: additionally converts I4 weights to U4 (symmetric quantization only).

        The weights in the OV .bin file are in Int8.
        The NPU hardware supports both UInt8 and Int8 computation.
        It has support for UInt8 with zero point. For Int8 the zero point is 0.

        It can be advantageous for accuracy to convert the weights to UInt8 with zero point as it enables asymmetric quantization.
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

    let constructor = "vpux::IE::createConvertWeightsToUnsignedPass()";

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
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Converts U4 quantized weights with zero point of 8 to I4 storage type by shifting
        the zero point and storage type range. Handles both per-tensor
        (UniformQuantizedType) and per-axis (UniformQuantizedPerAxisType) quantization
        where all zero points equal 8.

        Registered rewriters:
        - ConstRewriter — rewrites Const::DeclareOp to convert quantized constant content.
        - QuantizeLikeOpRewriter<QuantizeOp> — updates QuantizeOp destination type.
        - QuantizeLikeOpRewriter<QuantizeCastOp> — updates QuantizeCastOp destination type.
        - LayerRewriter — converts any LayerOpInterface result types (skips QuantizeOp/QuantizeCastOp).
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
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Converts Quantize/Dequantize operations to NCE-executable operations in three phases
        using a platform-specific strategy:
        1. AvgPool patterns — converts suitable Quantize/Dequantize to identity AvgPool.
        2. Eltwise patterns — converts per-tensor Quantize/Dequantize to eltwise And
           mixed-precision operations where input2 equals input1 to perform type conversion on DPU.
        3. DW conv patterns — converts per-axis scales with per-tensor zero points
           Quantize/Dequantize to depthwise convolution.
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

        Then after SplitFakeQuant and PropagateAndFuseQuantizeDequantize we have:
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
// FuseConvertWithQDQ
//

def FuseConvertWithQDQ : PassBase<"fuse-convert-with-qdq", "vpux::FunctionPass"> {
    let summary = "Fuse Convert with Quantize into QuantCast operation";

    let description = [{
        AI-generated summary

        Registered rewriters:
        - ConvertQuantizeRewriter — detects pattern Convert(i8 -> FP16) -> Quantize(FP16 -> !quant.uniform<...>)
           and fuses it into a single QuantizeCast(i8 -> !quant.uniform<...>) operation.
           Blocks the fold when Convert input signedness does not match quantized output signedness.
        - DequantizeConvertRewriter — detects pattern Dequantize(!quant.uniform<...> -> FP16) -> Convert(FP16 -> ui8)
           and fuses it into a single QuantizeCast(!quant.uniform<...> -> ui8) operation.
           The fusion is applied only if the dequantize input has identity quantization parameters
           (scale == 1.0 and zeroPoint == 0 for both per-tensor and per-axis quantized types).
    }];

    let constructor = "vpux::IE::createFuseConvertWithQDQPass()";

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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Replaces suitable IE::ReverseOp with a chain of ShapeCast, multiple GroupConvolution
        (depthwise) operations, Concat, and Transpose. Each DW convolution uses a weight matrix
        that picks one element from the reversed position.

        Conditions:
        - Input must be 4D with INDEX reverse mode.
        - Supports reversing one axis or two continuous axes.
        - Kernels must not exceed hardware limits (maxKernelSize and MAX_STRIDE).
        - Input element type must be compatible with GroupConvolutionOp (F16, F32, F64, SI32, or quantized).
        - When axes are not H,W (for 2 axes) or not W (for 1 axis), extra Transpose operations
          are inserted before and after the conversion.
    }];

    let constructor = "vpux::IE::createConvertReverseToDWConvPass()";

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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        After FusePadOps pass, there are Pad Ops that cannot be fused.
        Replace `IE::PadOp` with `IE::ConcatOp` and `Const::DeclareOp`.
        Only `IE::PadMode::CONSTANT` case is supported. Dynamic tensors are skipped.
        Handles per-axis quantized output types by tiling scales and zero points for
        padding constants along the quantization axis.
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
        AI-generated summary

        Optimizes scenarios of IE::StridedSlice and IE::SliceOp without neighboring operations.
        Moves the slice operations upwards through the graph, reducing both compute and memory usage.
        In some cases the slice operation may be safely removed from the graph, if the action of upstreaming it
            only adapts the operations constants.

        Registered rewriters:
        - GenericSliceUpstreaming — moves SliceOp/StridedSliceOp before eltwise-like parent ops
           and FakeQuantize (per-tensor only, or per-axis when slice axis differs from quant axis).
           Only supports 4D shapes and single-use parents with single result.
        - SliceUpstreamWithAffineReshape — handles chains of EltwiseOp -> AffineReshape -> QuantizeCast -> Slice,
           moving the Slice above AffineReshape and EltwiseOp with appropriate shape adjustments.
        Also applies canonicalization patterns for SliceOp and StridedSliceOp.
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces Pooling layers or Convolution layers that have kernels bigger than supported by hardware (11x11),
        with equivalent Pooling (approx equiv in case of prime kernel i.e. 13x13) or Convolution layers.

        Convolution rewriters (applied first):
        - PadLargeKernelRewriter — pads the kernel to a composite size before splitting.
        - ReshapeGlobalConvRewriter — reshapes global convolutions with 1-D spatial extent for better utilization.
        - SliceLargePrimeKernelRewriter — handles prime-sized kernels that cannot be factored.
        - ReshapeLargeConvRewriter — reshapes activation via AffineReshape when IC and one spatial dim equal 1.
        - SliceLargeConvRewriter — slices large kernels into multiple smaller convolutions, then sums results.

        Pooling rewriters (applied second):
        - GeneralAvgPoolRewriter — splits large-kernel AvgPool into smaller AvgPool operations.
        - GeneralMaxPoolRewriter — splits large-kernel MaxPool into smaller MaxPool operations.
        - OverlappedMaxPoolRewriter — handles MaxPool with overlapping padding requirements.
        - SliceLargeKernelAvgPoolRewriter — handles remaining large-kernel AvgPool cases by explicit padding
           and splitting into non-overlapping sub-AvgPools with corrected static_scale.
    }];

    let constructor = "vpux::IE::createHandleLargeKernelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseOpsToMatMul
//

def FuseOpsToMatMul : PassBase<"fuse-ops-to-matmul", "vpux::FunctionPass"> {
    let summary = "Fuse subgraph patterns into IE.MatMul";

    let description = [{
        The pass fuses high-level op subgraphs that are semantically equivalent
        to a matrix multiplication into IE.MatMul, enabling NCE MatMul lowering.

        Before performing the fusions, prioritize optimizing and eliminating Transpose
        operations in the following subgraph..
        ```
             Input1                        Input2
                |                             |
            Transpose                     Transpose
                |                             |
            AffineReshape                AffineReshape
                |                             |
            -----------------------------------------
            |               Multiply                 |
            -----------------------------------------
                               |
                           ReduceSum
                               |
        ```

        Handled patterns:
        1. Dual-axis broadcast Multiply + ReduceSum → IE.MatMul.
           The Multiply has one broadcast-from-1 dim on each input (M and N axes)
           and the ReduceSum contracts over a shared dimension, producing a MatMul
           with K > 1.  The reduce axis may appear at any position in the tensor.
        2. Standalone dual-axis broadcast Multiply (outer product) → IE.MatMul.
           [B,M,1]*[B,1,N] → [B,M,N] with no ReduceSum consumer → MatMul(K=1).
        3. CumSum → IE.MatMul via a triangular constant weight matrix.

        Patterns 1 and 2 are activated only when enable-grouped-matmul is true,
        the resulting MatMul can be lowered to VPU.NCE.MatMul on the current
        target, and the M*N*K product is at least 4096.
        Pattern 3 is activated only when convert-cumsum-to-matmul is true.
    }];

    let constructor = "vpux::IE::createFuseOpsToMatMulPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableGroupedMatMul", "enable-grouped-matmul",
            "bool", "false",
            "Flag to enable or disable grouped MatMul execution"
        >,
        Option<
            "enableConvertCumSumToMatMul", "convert-cumsum-to-matmul",
            "bool", "false",
            "Flag to enable or disable the conversion from CumSum to MatMul"
        >
    ];
}

//
// FuseOneHotSelect
//

def FuseOneHotSelect : PassBase<"fuse-onehot-select", "vpux::FunctionPass"> {
    let summary = "Fold one-hot select over per-branch subgraphs into a single selecting op";

    let description = [{
        Folds the "compute all N branches then one-hot select one" pattern, i.e.

            out = ReduceSum_axis0( onehot(group_id) * Concat_axis0( branch_i ) )

        into a single op that materializes only the selected branch at runtime.

        Currently handled branch kinds:
        * Per-codebook MatMul/FullyConnected select -> Gather of the selected
          weight + a single MatMul.
          Every branch computes `hidden @ W_i^T` (out = hidden @ W_group_id^T); the
          weights are stacked into a single [N, M, K] constant and one Gather by
          `group_id` selects the weight for the fused MatMul.
        * Per-codebook Gather select -> single flat-index Gather.
          Every branch gathers the same `token_id` row of a constant codebook
          E_i[V, D] (out = E_group_id[token_id]); the codebooks are stacked into a
          single [N*V, D] constant and one Gather with the flat index
          `group_id * V + token_id` selects the row.
    }];

    let constructor = "vpux::IE::createFuseOneHotSelectPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapEltwiseAndReduce
//

def SwapEltwiseAndReduce : PassBase<"swap-eltwise-and-reduce", "vpux::FunctionPass"> {
    let summary = "Hoist a reduce operation over a broadcast-compatible elementwise operation";

    let description = [{
        Applies the distributive law to reorder an EltwiseOp+ReduceOp chain into
        ReduceOp+EltwiseOp when one EltwiseOp operand has size==1 on every reduce axis:

            EltwiseOp(A, B) -> ReduceOp(axes, keep_dims)
            =>
            ReduceOp(A, axes, keep_dims=true) -> EltwiseOp(result, B) [-> Reshape]

        The intermediate ReduceOp always uses keep_dims=true to preserve rank, allowing
        B (broadcast on all reduce axes) to be applied correctly to the reduced result.
        When the original keep_dims was false, a Reshape squeezes out the size-1
        reduce axes at the end.

        Supported combinations:
          - Multiply + ReduceSum   (Multiply distributes over sum)
          - Multiply + ReduceMean  (Multiply distributes over mean)
          - Add + ReduceMean       (Add distributes over mean: mean(A+B) = mean(A) + mean(B),
                                   and mean(B) = B when B has size==1 on every reduce axis)

        This eliminates the large intermediate tensor produced by the full EltwiseOp
        and exposes a smaller ReduceOp to subsequent optimization passes such as
        ConvertReduceSumToConv.
    }];

    let constructor = "vpux::IE::createSwapEltwiseAndReducePass()";

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
        Converts ReduceSum operations into equivalent 1x1 Convolution operations using
        all-ones weights. Three rewriters cover different cases:

        - ReduceSumToConvRewriter: 4D input, reduce on channel dim (C).
        - OuterDimReduceSumToConvRewriter: non-4D input, reduce on outermost dim (axis=0).
          Reshapes [A, D1, D2, ...] to [1, A, D1*D2*..., 1] and applies a single Conv
          with IC=A, replacing multi-stage AvgPool decomposition.
        - InnerDimReduceSumToConvRewriter: non-4D input, reduce on innermost dim.

        Each rewriter has its own benefit condition: ReduceSumToConvRewriter applies when
        output W is not NCE-aligned or the op has an adjacent NCE parent/child;
        OuterDimReduceSumToConvRewriter applies when reduceSize exceeds the NCE channel
        alignment or the op has an adjacent NCE parent/child;
        InnerDimReduceSumToConvRewriter applies when the total input size meets a minimum
        threshold for the transformation to be beneficial.
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
        AI-generated summary

        Converts ReduceMean to AvgPool, ReduceMax to MaxPool, ReduceSum to
        AvgPool+Multiply, and ReduceMin to Negate+MaxPool+Negate, enabling NCE/DPU
        execution instead of SHAVE. Inserts Reshape before/after when reduction axes
        are not spatial or keep_dims differs from pooling output rank.
    }];

    let constructor = "vpux::IE::createConvertReduceToPoolingPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableFuseReduceMinMaxToDpu", "enable-fuse-reduce-minmax-to-dpu",
            "bool", "false",
            "Enable fusing ReduceMin/ReduceMax operations into the parent DPU op"
        >
    ];
}

//
// UnrollReduceMinAllAxes
//

def UnrollReduceMinAllAxes : PassBase<"unroll-reducemin-all-axes", "vpux::FunctionPass"> {
    let summary = "unroll reducemin on all axes which cannot be convert to efficient nce pooling ops";

    let description = [{
        AI-generated summary

        Unrolls ReduceMin operations with multiple axes into sequential single-axis
        ReduceMin operations. Detects cases where the output is scalar with total input
        size exceeding profiling thresholds and kernel size constraints, then iteratively
        reduces the largest non-one dimension at each step to avoid suboptimal NCE
        performance.
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
        AI-generated summary

        This pass handles the exclude_pads attribute for AvgPool operations with arbitrary
        pad and stride values. It splits the padded AvgPool into multiple sub-AvgPool regions:
        a center region (no padding), up to four corner regions (top-left, top-right,
        bottom-left, bottom-right), and up to four edge regions (left, right, top, bottom),
        each with adjusted kernel size. The sub-results are combined via IE::ConcatOp.
        Only applies to 4D inputs with FLOOR rounding and non-zero padding.
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
        AI-generated summary

        The pass is a part of `HardwareMode` pipeline.

        Inserts NHWC `Reorder` operations before each input of a `Concat` and an NCHW `Reorder`
        after the `Concat` output when the Concat has exactly two inputs, at least one parent is
        `Transpose` or `AffineReshape`, and the other parent is `FakeQuantize` or
        `AlignedChannelsOpInterface`. This transformation reduces the number of `MemPermute`
        operations in the resulting graph by enabling the Concat to operate in NHWC layout.
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
        AI-generated summary

        The pass is a part of `HardwareMode` pipeline.

        Swaps `Transpose` operation with a per-tensor `Quantize` or `FakeQuantize` operation.
        Applies only when the Quantize/FakeQuantize input is a network input (BlockArgument),
        or in the case of FakeQuantize, when the pattern is
        NetworkInput -> Convert -> FQ -> Transpose -> FQ.
        This transformation reduces the number of `MemPermute` operations in resulting graph
        because the Transpose can later become a PermuteCast.
    }];

    let constructor = "vpux::IE::createSwapTransposeWithFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapConvertWithReshapeKindOps
//

def SwapConvertWithReshapeKindOps : PassBase<"swap-convert-with-reshape-kind-ops", "vpux::FunctionPass"> {
    let summary = "Swaps Reshape kind operations with Convert";

    let description = [{
        AI-generated summary

        The pass is a part of `HardwareMode` pipeline.

        Registered rewriters:
        - OpSwapConverter — swaps Convert with adjacent reshape kind operations
           (AffineReshape, DepthToSpace, Reshape, Squeeze, Transpose, Unsqueeze).
           Network input case: NetworkInput -> Convert -> ReshapeKindOp => NetworkInput -> ReshapeKindOp -> Convert.
           Network output case: Convert -> N reshapeKindOps -> return => N reshapeKindOps -> Convert -> return.
        - PropagateConvertToFuseConverter — when two Convert operations are separated by
           a chain of reshape kind ops or Gather, propagates one Convert through the chain
           to fuse both into a single Convert. Chooses forward or backward propagation
           direction based on which minimizes intermediate data type size.
        - EliminateConvertRoundTripThroughReshapeAndSlice — eliminates a redundant
           Convert(A->B) ... Convert(B->A) round-trip surrounding a reshape kind op that
           feeds one or more Slices. Matched pattern: a leading Convert with a single
           reshape-kind user, whose users are all SliceOps, each of which has a single
           Convert user that converts back to the original type A. The leading and all
           trailing Converts are removed and the intermediate reshape/Slice result types
           are restored to A. NOTE: when A is wider than B (e.g. f32 -> f16 -> f32) the
           intermediate truncation to B is removed, so the trailing A-typed users observe
           higher-precision bits than in the original IR.
    }];

    let constructor = "vpux::IE::createSwapConvertWithReshapeKindOpsPass()";

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
        AI-generated summary

        Eliminates redundant operations from multiple branches by moving common operations
        before Slice/Split points or after elementwise producers.

        Registered rewriters:
        - MoveExpandBeforeSlice — moves Expand before Slice when axes do not conflict.
        - MoveReorderBeforeSlice — moves Reorder before Slice.
        - MoveTransposeBeforeSlice — moves Transpose before Slice when axes do not conflict.
        - MovePermuteCastBeforeSlice — moves PermuteCast before Slice.
        - MoveAffineReshapeBeforeSlice — moves AffineReshape before Slice.
        - MoveMemPermuteBeforeSlice — moves MemPermute before Slice.
        - MoveQuantizeCastBeforeSlice — moves QuantizeCast before Slice.
        - MoveReorderBeforeSplit — moves Reorder before Split when the split axis is in
           a lower memory dim after Reorder.
        - MoveReorderBeforeEltwiseOp — moves identical Reorder operations from eltwise
           outputs back before the eltwise producer.
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
// HandleEltwiseWithSmallHeight
//

def HandleEltwiseWithSmallHeight : PassBase<"handle-eltwise-with-small-height", "vpux::FunctionPass"> {
    let summary = "Optimize AddOp subgraphs with small height dimensions for multi-cluster strategy";

    let description = [{
        AI-generated summary

        This pass identifies AddOp subgraphs with small height (H <= numClusters) dimensions
        and optimizes them for multi-cluster strategy SOH by reshaping H and W dimensions
        (and optionally C) using ShapeCast.

        Conditions:
        - Input must be 4D with static shape.
        - Total operand size must exceed CMX capacity.
        - H dimension must be <= number of clusters.
        - Input channels must be aligned by VPU_CHANNEL_ALIGNMENT (16).

        Two reshaping strategies:
        1. Reshape H*W — finds factors of (H*W) with W >= VPU_SPATIAL_ALIGNMENT.
        2. Reshape C*H*W — when H*W alone cannot meet alignment, reduces C to
           VPU_CHANNEL_ALIGNMENT and redistributes the freed dimension into H*W.
    }];


    let constructor = "vpux::IE::createHandleEltwiseWithSmallHeightPass()";

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
        AI-generated summary

        Propagates AffineReshape operations through downstream consumers to enable
        further optimizations. Registered rewriters:
        -  MoveThroughTranspose — swaps AffineReshape past Transpose.
        -  MoveThroughExpand — swaps AffineReshape past Expand.
        -  MoveThroughConcat — moves multiple AffineReshapes past Concat.
        -  MoveThroughSoftmax — swaps AffineReshape past Softmax.
        -  MoveThroughMVN — swaps AffineReshape past MVN.
        -  MoveThroughEltwiseGeneric<AvgPoolOp> — moves through identity AvgPool.
        -  MoveThroughMultiply — swaps AffineReshape past Multiply.
        -  MoveThroughSlice — swaps AffineReshape past Slice.
        -  MoveTransposeAffineReshapeThroughAdd — moves Transpose+AffineReshape past Add.
        - MoveAffineReshapePermuteCastThroughConcat — moves AffineReshape+PermuteCast past Concat.
        - MoveThroughAdd — swaps AffineReshape past Add.
        - MoveThroughOneInputEltwise — swaps past single-input elementwise ops.
        - MoveThroughConvert — swaps AffineReshape past Convert.
        - ConcatReshapeConcat — folds Concat-Reshape-Concat sequences.
        - MoveThroughQuantizeCast — swaps AffineReshape past QuantizeCast.
        - MoveUpThroughGather — moves AffineReshape upward through Gather.
        Also applies canonicalization patterns for Reshape, AffineReshape, and ShapeCast.
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
        AI-generated summary

        Swaps ShapeCast before its producer operation to reconstruct adjacency broken
        by adjust passes. Supported producers: activations (Abs, Gelu, Swish, HSwish,
        Sigmoid, Tanh), pooling (AvgPool, MaxPool), eltwise (Add, Subtract, Multiply),
        MVN, depthwise GroupConvolution, and SoftMax. Clones the producer with the
        ShapeCast input and inserts a new ShapeCast after it.
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
        AI-generated summary

        Moves Transpose operations past (downstream of) SoftMax, Slice, AvgPool, Multiply,
        RMS, one-input eltwise, and Concat operations by adjusting their axis/offset
        parameters according to the permutation. Also moves Transpose+AffineReshape past
        Add, and Concat past Transpose.
    }];

    let constructor = "vpux::IE::createPropagateTransposePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateDequantThroughConcat
//

def PropagateDequantThroughConcat : PassBase<"propagate-dequant-through-concat", "vpux::FunctionPass"> {
    let summary = "Propagate Dequant operation through Concat";

    let description = [{
        Dequant is not propagated through Concat in PropagateAndFuseQuantizeDequantize pass
        if it has only one quantized input and non-quantized constants.
        Such Concat is usually introduced to add padding (either from Pad or for Convolution).

        This pass applies when the following conditions are met
        1. Concat operation has one and only one quantized input
        2. Quantization is per-tensor
        3. Concat has only one output
            This is needed in case if there was convolution with big kernel. Such convolution is
            splitted into multiple ones and such case doesn't support further dequantize propagation.
            So if we propagate dequantize through such concat, we may end up having many dequantize layers
            and performance regression as a result.
    }];

    let constructor = "vpux::IE::createPropagateDequantThroughConcatPass()";

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
        AI-generated summary

        This pass extracts a function from each `IE::LoopOp` body region.
        For each loop, it creates a private `func::FuncOp` containing the cloned body
        operations and replaces the body with a `func::CallOp` to the outlined function.
        By calling this pass before unrolling, the compiled IR can be greatly reduced
        because multiple iterations share the same function definition.
        When the internal execution condition is a constant with a single use,
        the condition output is omitted from the outlined function signature.
    }];

    let constructor = "vpux::IE::createLoopOutlinerPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ChunkGatedDeltaRuleLoop
//

def ChunkGatedDeltaRuleLoop : PassBase<"chunk-gated-delta-rule-loop", "vpux::FunctionPass"> {
    let summary = "Transform Gated Delta Rule and linear state recurrence loops into chunked computation.";

    let description = [{
        This pass detects `IE::LoopOp` bodies that implement the Gated Delta Rule
        recurrence
        (h_t = decay * h_{t-1} + k ⊗ beta * (v - <h_{t-1}, k>), y_t = <h_t, q>)
        and rewrites them into a chunked form that reduces the sequential trip count
        from N to N/chunk_size while enabling parallel intra-chunk computation.

        The pass also rewrites linear state recurrences of the form
        `h_t = gate_t * h_{t-1} + u_t`, where each `y_t` is accumulated by
        `ScatterUpdate`. This form is transformed by sequentially unrolling
        `chunk_size` iterations in each new loop iteration.

        The transformation is applied only when the loop matches the expected pattern
        and all implementation constraints are satisfied. In particular:
        * the loop trip count must be fixed, with `num_iterations` matching `seqLen`
        * `seqLen` must be divisible by the configured `chunk-size`
        * the execution condition must be constant true (no early termination)
        * loop outputs must use invariant output descriptors only (no concat outputs)
        * the ScatterUpdate indices must derive from the loop iteration argument

        Loops that do not satisfy these requirements are left unchanged.
    }];

    let constructor = "vpux::IE::createChunkGatedDeltaRuleLoopPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<"chunkSize", "chunk-size", "int64_t", "64",
                "Chunk size for the Gated Delta Rule Loop transformation.">
    ];
}

//
// UnrollTensorIterator
//

def UnrollTensorIterator : PassBase<"unroll-tensor-iterator", "vpux::FunctionPass"> {
    let summary = "Unroll tensor iterator op.";

    let description = [{
        AI-generated summary

        Unrolls loop operations by replicating the loop body for each iteration.

        Registered rewriters:
        - TensorIteratorRewriter — unrolls `IE::TensorIteratorOp` by cloning the body region
           for each iteration, handling sliced inputs, invariant inputs, feedback (back-edge)
           inputs, and collecting concat and invariant outputs.
        - LoopRewriter — unrolls `IE::LoopOp` with the same mechanism as TensorIterator,
           additionally handling the current iteration index and execution conditions.
           When execution conditions are not constant, `IE::LoopSelectOp` is introduced
           to select the correct invariant output.
    }];

    let constructor = "vpux::IE::createUnrollTensorIteratorPass()";

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
        AI-generated summary

        Reorders Transpose operations after Concat when all Concat inputs are wrapped in
        identical transposes. Moves the Transpose outside the Concat by updating the concat
        axis and static offset attributes according to the permutation. Creates a single
        Transpose on the concatenated output, reducing the number of transpose operations
        from N to 1.
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
        AI-generated summary

        Converts Power operations with constant integer exponents up to 3 into sequences
        of Multiply operations. Replaces x^2 with x*x, x^3 with x*x*x, and returns the
        identity for exponent=1. Works only with splat constant exponents to enable
        execution on DPU instead of SHAVE.
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
        AI-generated summary

        The pass is a part of `AdjustForVPU` pipeline.

        Converts `IE::UpsamplingOp` into strided `IE::ConcatOp` with zero-filled constants when
        the padding configuration does not allow direct DMA-based upsampling (outputShape = inShape * factor).
        For each spatial dimension with factor > 1 and more than one element, creates a strided concat
        interleaving the original data with zeros. Adds a `IE::SliceOp` when the right padding
        requires trimming, and a `IE::PadOp` for any remaining padding.
        When the padding can be adjusted for DMA, rewrites the UpsamplingOp with corrected pad attributes.
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
// ExpandActivationChannels
//

def ExpandActivationChannels : PassBase<"expand-activation-channels", "vpux::FunctionPass"> {
    let summary = "Align input tensors shape of DPU operation with hardware requirements";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        This pass processes operations that can be compiled as DPU tasks and expands the number
            of channels to a number divisible by 16 if they do not satisfy hardware requirements.
        The input channels may be aligned to 4 instead of 16 for compressed convolutions.
    }];

    let constructor = "vpux::IE::createExpandActivationChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// AlignDimensionsForDPU
//

def AlignDimensionsForDPU : PassBase<"align-dimensions-for-dpu", "vpux::FunctionPass"> {
    let summary = "Align dimensions for DPU operations to meet hardware alignment requirements";

    let description = [{
        This pass expands the width dimension of specific operations to meet DPU hardware alignment requirements.
        The expansion ensures that the width dimension is aligned to the required boundary (typically 16 for FP16).

        Supported operations:
        * IE.PermuteQuantize (NCHW -> NHWC layout transformation)
            - Expands input/output width to be aligned when executing on DPU
            - Applicable when inOrder=NCHW, outOrder=NHWC
            - Adds Expand ops before and Slice ops after the operation to handle alignment
            - To avoid blocking optimization of permuteQuantize, place the pass after buildMemPermuteProcessingPipeline()
              for VPUX37XX+.

        * IE.LogSoftmax
            - Expands the axis dimension to meet alignment requirements
            - Only expands when the axis is on the innermost dimension
            - Uses Expand before and Slice after to maintain correct output shape
    }];

    let constructor = "vpux::IE::createAlignDimensionsForDPUPass()";

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
        AI-generated summary

        `IE.ExpandOp` is used to guarantee the channel alignment requirement for NCE ops. This pass
        reshapes NCE ops to reduce the cost of Expand operations.

        Registered patterns:
        - ExpandEltwiseRewriter<MultiplyOp> — reshapes Multiply inputs to reduce alignment cost.
        - ExpandEltwiseRewriter<SubtractOp> — reshapes Subtract inputs.
        - ExpandEltwiseRewriter<AddOp> — reshapes Add inputs.
        - ExpandGroupConvRewriter — reshapes GroupConvolution inputs.
        - ExpandPermuteQuantizeRewriter — reshapes PermuteQuantize inputs.
        - AdjustPermuteQuantizeRewriter — adjusts PermuteQuantize shape.
        - ExpandPoolingRewriter<AvgPoolOp> — reshapes AvgPool inputs (KY=1, SY=1).
        - ExpandPoolingRewriter<MaxPoolOp> — reshapes MaxPool inputs (KY=1, SY=1).
        - ExpandSingleChannelPoolingRewriter<AvgPoolOp> — handles single-channel AvgPool.
        - ExpandSingleChannelPoolingRewriter<MaxPoolOp> — handles single-channel MaxPool.
        - EltwiseShapeRewriter<AddOp> — reshapes Add for better channel alignment (separate pass).
        Platforms NPU37XX+ all benefit.
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
        AI-generated summary

        The layout does not matter for eltwise ops as long as the input and output layouts are the same.
        Moves the permute ops from the inputs of the eltwise to the output to reduce the number of permute ops.

        Registered rewriters:
        - PermuteEltwiseRewriter<AddOp> — moves MemPermute/PermuteQuantize past Add (2 inputs).
        - PermuteEltwiseRewriter<MultiplyOp> — moves past Multiply (2 inputs).
        - PermuteEltwiseRewriter<SubtractOp> — moves past Subtract (2 inputs).
        - PermuteEltwiseRewriter<GroupConvolutionOp> — moves past eltwise-like GroupConv
           (1x1 kernel, stride 1, no padding, splat constant weights/bias).
        - PermuteEltwiseRewriter<AvgPoolOp> — moves past identity AvgPool (1 input).
        - PermuteEltwiseDiffLayoutRewriter — handles AddOp with inputs from different layouts
           by replacing one branch with a PermuteCast when the other is an NCE op in NHWC.

        The rewriter inserts PermuteCast + ShapeCast before and after the eltwise to maintain
        layout correctness. Traverses through QuantizeCast and ShapeCast ops after the eltwise
        to place the output permute at the optimal position.
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
        AI-generated summary

        Broadcasts mismatched-shape inputs of AddOp to match the output shape.
        Only applies to 4D tensors. Skips broadcasting when the bias input is a constant
        with shape 1xCx1x1, as such cases can be handled more efficiently via ScaleShift
        conversion. Supports both static (IE::createBroadcast) and dynamic
        (IE::createDynamicBroadcast using ShapeOf) broadcasting, but cross-broadcast
        of two dynamic inputs is not supported.
    }];

    let constructor = "vpux::IE::createBroadcastInputForAddPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseInefficientTileForAdd
//

def FuseInefficientTileForAdd : PassBase<"fuse-inefficient-tile-for-add", "vpux::FunctionPass"> {
    let summary = "Fuse inefficient Tile ops into Add op via NUMPY broadcast";

    let description = [{
        Matches the pattern:
            TileOp (broadcast) -> AddOp (NUMPY)
        where the TileOp repeats along the innermost memory dimension of its output.

        Repeating the innermost memory dimension is inefficient for TileDMA because it
        must touch every element. This pass removes the explicit TileOp and replaces the
        corresponding AddOp operand with the un-tiled input, letting AddOp perform the
        broadcast implicitly via NUMPY semantics. The result is handled by the SHAVE
        broadcast-add kernel, which is more efficient for innermost-dimension broadcast.

        Before:
            %tile = IE.Tile(%input) {repeats_values = [1, 1, 1, N]}
                    : tensor<1x8x1x1xf16> -> tensor<1x8x1xNxf16>
            %out  = IE.Add(%tile, %other) {auto_broadcast = NUMPY}
                    : tensor<1x8x1xNxf16>, tensor<1x8x1xNxf16> -> tensor<1x8x1xNxf16>

        After:
            %out  = IE.Add(%input, %other) {auto_broadcast = NUMPY}
                    : tensor<1x8x1x1xf16>, tensor<1x8x1xNxf16> -> tensor<1x8x1xNxf16>
    }];

    let constructor = "vpux::IE::createFuseInefficientTileForAddPass()";

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
        This pass contains two rewrite patterns applied in priority order:

        1. `FuseBroadCastRewriter` (higher priority, always active):
           Matches `Tile (broadcast) -> [optional PermuteCast] -> Multiply` and removes the
           explicit `Tile` by replacing it with the un-tiled input. The `Multiply` then relies
           on NUMPY auto-broadcast to expand the smaller input implicitly. When a `PermuteCast`
           sits between the `Tile` and the `Multiply`, a new small `PermuteCast` on the un-tiled
           input is inserted so the layout contract is preserved.

          2. `TileBroadcastInputRewriter` (lower priority, can be disabled only by setting `broadcast-input-for-multiply=false`):
              Broadcasts mismatched-shape inputs in Multiply operations by materializing an explicit
              `IE.Tile` on the smaller input so it matches the output shape. Performs NCHW-to-NHWC
              casting, tiling, and dimension reordering back to original layout. Only applies when
              broadcasting expands exactly one dimension from 1 -> N and the broadcasted dimension is
              not more inner than the input's innermost non-trivial dimension, and the innermost
              dimension is properly aligned, enabling the Multiply to execute on DPU instead of SHAVE.
    }];

    let constructor = "vpux::IE::createBroadcastInputForMultiplyPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "broadcastInputForMultiply", "broadcast-input-for-multiply",
            "bool", "true",
            "Enable broadcasting input for Multiply ops"
        >
    ];
}

//
// SwapMVNWithTranspose
//

def SwapMVNWithTranspose : PassBase<"swap-mvn-with-transpose", "vpux::FunctionPass"> {
    let summary = "Swaps MVN operation with parent Transpose";

    let description = [{
        AI-generated summary

        The pass is a part of `HardwareMode` pipeline.

        Swaps `IE::MVNOp` with its parent `IE::TransposeOp` when:
        - MVN is not across_channels.
        - Both parent and child Transpose operations swap W and H axes.
        - Parent Transpose has a single use and its input is not a BlockArgument.
        Moves the MVN before the Transpose and places a new Transpose after MVN output.
        This transformation reduces the number of `MemPermute` operations in resulting graph.
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
}

//
// ConvertReorderToPermuteQuantize
//

def ConvertReorderToPermuteQuantize : PassBase<"convert-reorder-to-permute-quantize", "vpux::FunctionPass"> {
    let summary = "Converts IE.Reorder to DPU permute";

    let description = [{
        AI-generated summary

        Converts Reorder operations into PermuteQuantize operations that combine permutation
        and element type conversion in a single DPU operation. Inserts a PermuteCast when needed
        to align input layout, and blocks conversion if the pattern can be better optimized as
        MaxPool. Only applies to float16 input/output reorders.
    }];

    let constructor = "vpux::IE::createConvertReorderToPermuteQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateMemPermuteThroughEltwise
//

def PropagateMemPermuteThroughEltwise : PassBase<"propagate-mem-permute-through-eltwise", "vpux::FunctionPass"> {
    let summary = "Propagates IE.MemPermute backward through elementwise ops (Add, Multiply, Subtract)";

    let description = [{
        Moves a trailing IE.MemPermute backward through elementwise ops (Add, Multiply, Subtract)
        so that consecutive MemPermutes on each input branch become adjacent and can be
        folded by canonicalization. Converts this subgraph:
        ```
            IE.MemPermute -> IE.ShapeCast \
                                           IE.Add -> IE.ShapeCast -> IE.MemPermute
            IE.MemPermute -> IE.ShapeCast /
        ```
        into:
        ```
            IE.MemPermute -> IE.MemPermute -> IE.ShapeCast \
                                                            IE.Add -> IE.ShapeCast
            IE.MemPermute -> IE.MemPermute -> IE.ShapeCast /
        ```
        When two consecutive IE.MemPermute ops are inverses they cancel entirely,
        eliminating the DMA cost. The inverse pass "MovePermutePostEltwise" moves
        identical MemPermutes from inputs to after the output.
    }];

    let constructor = "vpux::IE::createPropagateMemPermuteThroughEltwisePass()";

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
// AdaptODUPermutePass
//

def AdaptODUPermutePass : PassBase<"adapt-odu-permute", "vpux::FunctionPass"> {
    let summary = "Search and adapt the layout for ODU permutation to a better one";

    let description = [{
        number of elements DPU can receive in a single clock cycle from the PPE:

        ---------------------------------------
        | Layout | 8 bits | 16 bits | 32 bits |
        |  NHWC  |   64   |   32    |    16   |
        |  NWHC  |   64   |   32    |    16   |
        |  NWCH  |   16   |   16    |    16   |
        |  NCWH  |   16   |   16    |    16   |
        |  NHCW  |    4   |    4    |     4   |
        |  NCHW  |    4   |    4    |     4   |
        ---------------------------------------

        adapts the layout if posible these subgraph:
        ```
        if H == 1
            Input [NHWC] -> IE.Convolution [NCHW]
        ```
        Into the following subgraph respectively:
        ```
            Input [NHWC] -> IE.Convolution [NCWH] -> IE.PermuteCast  [NCHW]
        ```

    }];

    let constructor = "vpux::IE::createAdaptODUPermutePass()";

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
        AI-generated summary

        This pass handles operations with padding larger than supported on hardware (pad > kernel/2).
        It moves the excess padding from the layer parameter to the input by concatenating
        zero constants. For example, if a convolution's top pad is 5 but HW supports at most 2,
        the pass sets the convolution's pad to 2 and concatenates the input with 3 lines of zero constants.

        Supported operations:
        - ConvGeneralRewriter<ConvolutionOp> — handles IE::ConvolutionOp with large pads.
        - ConvGeneralRewriter<GroupConvolutionOp> — handles IE::GroupConvolutionOp with large pads.
           Accounts for dilation when computing effective pad limits.
        - PoolingGeneralRewriter<MaxPoolOp> — handles IE::MaxPoolOp with large pads.
        - PoolingGeneralRewriter<AvgPoolOp> — handles IE::AvgPoolOp with large pads.
    }];

    let constructor = "vpux::IE::createHandleLargePadsPass()";

}

//
// PropagateMemPermuteBeforeOp
//

def PropagateMemPermuteBeforeOp : PassBase<"propagate-mem-permute-before-op", "vpux::FunctionPass"> {
    let summary = "Propagate IE.MemPermute through concrete op";

    let description = [{
        AI-generated summary

        Propagates permute through concrete op.
        - Converts this subgraph
        ```
            IE.AffineReshape -> IE.MemPermute
        ```
        into
        ```
            [IE.PermuteCast] -> IE.MemPermute -> IE.Reshape -> IE.PermuteCast
        ```

        - Converts this subgraph
        ```
            IE.Upsampling -> IE.MemPermute
        ```
        into
        ```
            IE.MemPermute -> IE.Upsampling -> [IE.PermuteCast]
        ```
        This Upsampling rewrite fires only when the lowered Upsampling DMA goes from
        strided to non-strided after the swap (i.e. a growth axis with factor>1 moves
        from an inner to an outer memory position).

        - Converts this subgraph
        ```
            IE.MVN -> IE.MemPermute
        ```
        into
        ```
            IE.MemPermute -> IE.MVN-> IE.PermuteCast
        ```
        - Converts this subgraph
        ```
            IE.Concat -> IE.Slice -> IE.MemPermute/IE.PermuteQuantize
                      -> IE.Slice -> IE.MemPermute/IE.PermuteQuantize
                      -> ...
        ```
        into
        ```
            IE.MemPermute(input) -> new IE.Concat -> IE.Slice -> [IE.PermuteCast]
                                                  -> IE.Slice -> [IE.PermuteCast]
                                                  -> ...
        ```
        This Concat/Slice closed-form rewrite fires only when it strictly improves DMA
        access (strided Slice DMA becomes non-strided) and does not regress any
        Concat-input DMA from non-strided to strided.

        Canonicalization may fuse IE.MemPermute with another one.
    }];

    let constructor = "vpux::IE::createPropagateMemPermuteBeforeOpPass()";

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
// OptimizeReduceOpsWithMemPermute
//
def OptimizeReduceOpsWithMemPermute : PassBase<"optimize-reduce-ops-with-mem-permute", "vpux::FunctionPass"> {
    let summary = "Optimize IE.ReduceSum / IE.ReduceMean / IE.ReduceMin / IE.ReduceMax with IE.MemPermute";

    let description = [{
        This pass inserts MemPermuteOp before and after Reduce ops if the axis are not on inner most memory dim.
        Then it will be more efficient to execute on SW.Kernel.
    }];

    let constructor = "vpux::IE::createOptimizeReduceOpsWithMemPermutePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableFuseReduceMinMaxToDpu", "enable-fuse-reduce-minmax-to-dpu",
            "bool", "false",
            "Enable fusing ReduceMin/ReduceMax operations into the parent DPU op"
        >
    ];
}

//
// OptimizeIdentityPool
//

def OptimizeIdentityPool : PassBase<"optimize-identity-pools", "vpux::FunctionPass"> {
    let summary = "Optimize identiy pools";

    let description = [{
        AI-generated summary

        Removes identity pooling ops and fuses their post-processing into parent operations.
        Cannot be implemented as a folder/canonicalizer because other passes intentionally
        introduce identity pools.

        Applied in three phases:
        Phase 1:
        1. FuseIdentityQuantizedAvgPool — fuses identity AvgPool with PPE into a parent
           quantized AvgPool permutation, moving post-op and clamp attributes.
        2. FuseIdentityWithQuantizedAdd — fuses identity AvgPool with LeakyRelu post-op
           into a quantized Add (where input1 == input2), moving post-op and clamp.
        Phase 2:
        3. FuseIdentityAvgPoolWithPostOp — fuses identity AvgPool with Relu/LeakyRelu
           post-op or clamp into a parent LayerWithPostOpInterface (e.g., Conv).
        Phase 3:
        4. RemoveIdentityPool<MaxPoolOp> — removes identity MaxPool (kernel=1, stride=1, no pad).
        5. RemoveIdentityPool<AvgPoolOp> — removes identity AvgPool (kernel=1, stride=1, no pad).
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
// ConvertMemPermuteToOpPass
//

def ConvertMemPermuteToOpPass : PassBase<"convert-mem-permute-to-op", "vpux::FunctionPass"> {
    let summary = "Converts an `IE.MemPermute` operation to an `IE.PermuteQuantize` or `IE.MaxPool` operation";

    let description = [{
        Prioritize converting `NCHW-NHWC IE.Reorder`-like `IE.MemPermute` to `IE.PermuteQuantize` as `NCE.Permute` is more performant than `NCE.MaxPool`.

        Then replace remaining `IE.MemPermute` operation with a chain of
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

    let constructor = "vpux::IE::createConvertMemPermuteToOpPass()";

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
        AI-generated summary

        Fuses Sigmoid and Multiply operations into a single IE.Swish. Matches two patterns:
        (1) direct input->Sigmoid->Multiply(input, sigmoid), and (2) quantized
        FakeQuantize->Sigmoid->FakeQuantize->Multiply. Replaces with IE.SwishOp to reduce
        operation count.
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
        AI-generated summary

        For FullyConnected which has first dim = 1, e.g. KV cached Matmul, the performance
        bottleneck is caused by too many small DMA ops instead of compute ops.
        This pass merges unrolled FullyConnected layers to improve performance by using
        more data calculation and fewer DMA ops.

        Registered rewriters:
        - MergeFullyConnectedWithWeightsAsConstant — merges parallel FC branches that share
           the same input and have constant weights (via FQ -> Reshape -> Transpose or direct
           constant). Concatenates weights and input slices into grouped FC operations.
        - MergeFullyConnectedForDQPatternWithConvert — merges FC branches with
           dequantize-quantize patterns using Convert ops.
        - MergeFullyConnectedForDQPatternWithDequantize — merges FC branches with
           dequantize patterns.
    }];

    let constructor = "vpux::IE::createMergeFullyConnectedPass()";

    let options = [
        Option<"mergeUnrolledMatmulForLargeOC", "merge-unrolled-matmul-for-large-oc", "bool", /*default=*/"false",
               "Enable merging FC branches with dequantize pattern">
    ];

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
                                                                             IE.Add ([N, W], [N, W])
        ```
        1. Match `IE.FakeQuantize -> IE.Concat -> IE.AffineReshape -> IE.Transpose -> IE.FullyConnected` pattern.
        2. Reshape all `IE.FakeQuantize` operations from 1xHxW to HxW.
        3. Split `Input` by the number of groups (G).
        4. Build a `IE.MatMul` for each group using splits from step 3 and reshape results from step 2.
        5. Accumulate `IE.MatMul` results with `IE.Add` operations.

        Motivation: this transformation eliminates `IE.Concat` from the subgraph.
        Such concat is problematic because its axis (d0) is different from the quantization axis (d2).
        `PropagateAndFuseQuantizeDequantize` pass cannot handle the case when the axes differ.
        Therefore, a dequantize cannot be fused into a convolution.
        Without the `IE.Concat`, the fusion is possible.
    }];

    let constructor = "vpux::IE::createUnrollFullyConnectedPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MoveMultiplyDividePostOp
//

def MoveMultiplyDividePostOp : PassBase<"move-multiply-divide-post-op", "vpux::FunctionPass"> {
    let summary = "Move `IE.Multiply` and `IE.Divide` post op";

    let description = [{
        AI-generated summary

        This pass registers three rewriter patterns:

        - MoveMultiplyDividePostLayerGeneric<IE::MultiplyOp> and
           MoveMultiplyDividePostLayerGeneric<IE::DivideOp>:
        Move IE.Multiply / IE.Divide before IE.MatMul or IE.FullyConnected when the
        matmul/FC output tensor is smaller than the multiply/divide input tensor.
        ```
                                        (1x32x1024x80)      (1x1x1x1)
                                            \               /
                      (1x32x1x80)       IE.Multiply/IE.Divide (1x32x1024x80)
                            \                    /
                            IE.Matmul(1x32x1x1024)
        ```
        To:
        ```
                        (1x32x1x80)    1x32x1024x80)
                            \            /
                            IE.Matmul(1x32x1x1024)     (1x1x1x1)
                                    \                   /
                          IE.Multiply/IE.Divide (1x32x1x1024)
        ```
        If matmul output size is smaller than input, after swap multiply and divide with matmul, we will
        reduce the computation of multiply and divide. For above case, 80 times saving.

        - MoveMultiplyDividePostConcat:
        When a Concat feeds an IE.Multiply with a broadcastable constant, the Multiply is
        pushed before the Concat so it operates on each smaller concat input individually.
        This enables later fusion of the Multiply into preceding operations.
    }];

    let constructor = "vpux::IE::createMoveMultiplyDividePostOpPass()";

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
        AI-generated summary

        - OptimizeGroupConvConcat pattern
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

        - OptimizeConvConcat pattern
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

        - OptimizeSliceMultiplyConcat
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
        AI-generated summary

        - InputsTo2D: Unsqueeze 1d inputs to 2d.
        - TransposeInputs (requires `enable-grouped-matmul`): Insert `IE.Transpose` if
        transpose_a is set. Transform
        ```
            IE.MatMul(16x32, 16x64) {transpose_a = true}
        ```
        into
        ```
            IE.Transpose(16x32 -> 32x16) -> IE.MatMul(16x32, 16x64) {transpose_a = false}
        ```
        Also inserts `IE.Transpose` if transpose_b is not set.
        ```
            IE.MatMul(16x32, 32x64) {transpose_b = false}
        ```
        into
        ```
            IE.Transpose(32x64 -> 64x32) -> IE.MatMul(16x32, 64x32) {transpose_b = true}
        ```
        - CollapseBatch (requires `enable-grouped-matmul`): Collapse batch dimension into
        rows when the second input has 2d shape.
        ```
            IE.MatMul(1x4x32x64, 64x32) {transpose_b = true}
        ```
        becomes
        ```
            IE.Reshape(1x4x32x64 -> 128x64) -> IE.FullyConnected(128x64, 64x32)
        ```
        - To4D (requires `enable-grouped-matmul`): Fold batch dimensions to 4D form.
        2x3x4x5x6 becomes 1x24x5x6.
        - SoftMaxTo4D: Reshapes IE.SoftMax to 4D when preceded by a reshaped IE.MatMul,
        to maintain SDP (MatMul-SoftMax-MatMul) fusion.
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

def MergeParallelFullyConnected : PassBase<"merge-parallel-fully-connected", "vpux::FunctionPass"> {
    let summary = "Merge parallel 'IE.FullyConnectedOp'";

    let description = [{
        Merges N parallel IE.FullyConnectedOp operations that share the same input tensor
        into a single IE.FullyConnectedOp with concatenated weights, reducing kernel scheduling
        overhead and improving DPU tile utilization on NPU.

        Two merge strategies are supported:

        Strategy 1 — FQ/DynDQ weight chain (original, for GPTQ LLM models):
        Weights pass through FakeQuantize (or DynamicDequantize) + AffineReshape + Transpose.
        The pass merges constants along the output-channel axis and recovers per-FC outputs
        via IE.Slice ops:
        ```
        cst_set1(2x3x2,...)    cst_set2(2x3x3,...)
              \  ...  /               \  ...  /
          IE.FakeQuantize(2x3x2)  IE.FakeQuantize(2x3x3)
                  |                       |
          IE.AffineReshape(6x2)   IE.AffineReshape(6x3)
                  |                       |
          IE.Transpose(2x6)       IE.Transpose(3x6)
                   \           Input(1x6)  /
                    \          /       \  /
             IE.FullyConnected(1x2)  IE.FullyConnected(1x3)
        ```
        Becomes:
        ```
        cst_set_concat(2x3x5,...)
               \  ...  /
           IE.FakeQuantize(2x3x5)
                   |
           IE.AffineReshape(6x5)
                   |
           IE.Transpose(5x6)    Input(1x6)
                    \           /
                  IE.FullyConnected(1x5)
                  /               \
          IE.Slice(1x2)         IE.Slice(1x3)
        ```

        Strategy 2 — Direct constant weights + Concat elimination:
        Weights are direct IE.Const::DeclareOp values (no FQ/DynDQ preprocessing).
        All FC outputs (through an optional IE.AffineReshapeOp) must feed the same
        IE.ConcatOp. The pass merges weights along axis 0 and replaces the downstream
        Concat with a single IE.ReshapeOp — no Slice ops are emitted:
        ```
        cst_w1(N1xK)   cst_w2(N2xK)
             |    Input(BxK)   |
              \   /       \   /
          FC(BxN1)       FC(BxN2)
              |               |
        AffineReshape   AffineReshape    (optional)
              \               /
           IE.Concat(axis=a) → out_shape
        ```
        Becomes:
        ```
        Input(BxK)    Concat(cst_w1, cst_w2, axis=0)
                  \        merged_weight((N1+N2)xK)
                   \      /
                  FC(Bx(N1+N2))
                        |
                IE.Reshape(out_shape)       (Concat eliminated)
        ```
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

def ShrinkMatmulGroups : PassBase<"shrink-matmul-groups", "vpux::FunctionPass"> {
    let summary = "Shrink Matmul groups number";

    let description = [{
        Shrinks the group count of a grouped MatMul when the RHS tensor is obtained
        through a BroadcastOp, reducing memory traffic by eliminating the Broadcast
        and allowing the resulting smaller MatMul to fit in CMX.

        Three patterns are handled:

        Case 1 - Broadcast -> AffineReshape -> MatMul:
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
        becomes:
            LHS             RHS
        1x24x1x64       1x8x1x1024x64
            |               |
        Reshape         Reshape
            |               |
        1x8x3x64        1x8x1024x64
            \               /
                 MatMul

        Case 2 - Broadcast -> AffineReshape -> Transpose(4D) -> MatMul:
        same as Case 1 but with an outer Transpose (NCWH) after AffineReshape.

        Case 3 - Broadcast -> Transpose(5D swap last 2 dims) -> AffineReshape -> MatMul:
        handles KV-cache attention MatMuls in decode models where a 5D last-2-dims
        Transpose sits between the Broadcast and the AffineReshape:
                            RHS
                        1x2x1xAxB
                            |
                        Broadcast
                            |
                        1x2x7xAxB
                            |
                        Transpose (d0,d1,d2,d4,d3)
                            |
                        1x2x7xBxA
                            |
                        AffineReshape
            LHS             |
        1x14x1xB        1x14xBxA
            \               /
                 MatMul
        becomes a 2-group MatMul with the Broadcast eliminated.
    }];

    let constructor = "vpux::IE::createShrinkMatmulGroupsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagatePermuteCast
//

def PropagatePermuteCast : PassBase<"propagate-permute-cast", "vpux::FunctionPass"> {
    let summary = "Propagates permute cast through various ops to enable better fusion opportunities";

    let description = [{
        This pass propagates PermuteCast up in the network through various ops such as Dequantize,
        Slice, Concat, etc. to simplify IR patterns and make further optimizations easier to apply.
    }];

    let constructor = "vpux::IE::createPropagatePermuteCastPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertConstantDynamicDequantizeToDequantize
//

def ConvertConstantDynamicDequantizeToDequantize : PassBase<"convert-const-dynamic-dequantize-to-dequantize", "vpux::FunctionPass"> {
    let summary = "Embed constant scale/ZP of DynamicDequantize into the quantized type";

    let description = [{
        This is a bridge-adaptation pass.

        When a DynamicDequantize op has compile-time-constant scale (and optional
        zero-point) operands, embed those constants directly into a
        Quantized Type type via QuantizeCast, then replace the DynamicDequantize
        with a static Dequantize.

        Input subgraph:
        ```
        %scale = Const  tensor<...>      (constant)
        %zp    = Const  tensor<...>        (optional constant)
        %input          tensor<...>
        %out = IE.DynamicDequantize(%input, %scale [, %zp])
                        tensor<...>
        ```
        Output subgraph:
        ```
        %cast = IE.QuantizeCast(%input)
                        tensor<...>
        %out  = IE.Dequantize(%cast)  tensor<...>
        ```
        This allows downstream passes to continue using the static
        IE.DequantizeOp path while higher-level passes migrate to
        IE.DynamicDequantizeOp.
    }];

    let constructor = "vpux::IE::createConvertConstantDynamicDequantizeToDequantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDQRawDataTypeToQuantized
//

def ConvertDQRawDataTypeToQuantized : PassBase<"convert-dq-raw-data-type-to-quantized", "vpux::FunctionPass"> {
    let summary = "Normalize raw storage types in DynamicDequantize inputs to QuantizedType";

    let description = [{
        For each IE.DynamicDequantize op whose input carries a raw (non-quantized) element type
        (e.g. si4, ui4, QuantileType, f8E4M3FN), insert an IE.QuantizeCast that attaches an
        identity quant.uniform type (scale=1.0, zp=0) to the input tensor.

        This normalizes the IR invariant expected by downstream passes: every
        IE.DynamicDequantize input must have a QuantizedType element type.

        Input subgraph:
        ```
        %input          tensor<...xi4>     (raw storage type)
        %scale          tensor<...>
        %out = IE.DynamicDequantize(%input, %scale)
                        tensor<...>
        ```
        Output subgraph:
        ```
        %cast = IE.QuantizeCast(%input)
                        tensor<...x!quant.uniform<i4:f16, 1.0:0>>
        %out  = IE.DynamicDequantize(%cast, %scale)
                        tensor<...>
        ```
    }];

    let constructor = "vpux::IE::createConvertDQRawDataTypeToQuantizedPass()";

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
// ConvertDynamicDequantizeToFakeQuantize
//

def ConvertDynamicDequantizeToFakeQuantize : PassBase<"convert-dynamic-dequantize-to-fake-quantize", "vpux::FunctionPass"> {
    let summary = "Reconstruct 'IE.FakeQuantize' from a weights-import 'IE.DynamicDequantize'";

    let description = [{
        Bridge pass. Reconstructs the exact IE.FakeQuantize that the FakeQuantize weights import would have
        produced from the IE.DynamicDequantize emitted by the DynamicDequantize weights import. This shields the downstream
        FakeQuantize-centric pipeline while the weights import switches to DynamicDequantize.
        DynamicDequantize forms that are not the weights-import shape (non-constant scale, Gather/KV-cache, etc.)
        are left untouched.
    }];

    let constructor = "vpux::IE::createConvertDynamicDequantizeToFakeQuantizePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertDequantizeToDynamicDequantize
//

def ConvertDequantizeToDynamicDequantize : PassBase<"convert-dequantize-to-dynamic-dequantize", "vpux::FunctionPass"> {
    let summary = "Replace 'IE.Dequantize' with 'IE.DynamicDequantize' extracting scale/ZP into explicit constant operands";

    let description = [{
        Replaces each IE.Dequantize whose input has a quant.uniform element type with an
        IE.DynamicDequantize whose scale and zero-point are explicit constant tensor operands
        derived from the quantized element type.

        Input subgraph:
        ```
        %input : tensor<... x !quant.uniform<i8:f32, scale:zp>>
        %out = IE.Dequantize(%input)
        ```

        Output subgraph:
        ```
        %scale = Const : tensor<... x f32>
        %zp    = Const : tensor<... x i8>
        %out = IE.DynamicDequantize(%input, %scale, %zp)
        ```
    }];

    let constructor = "vpux::IE::createConvertDequantizeToDynamicDequantizePass()";

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
        AI-generated summary

        Converts IR generated by the reifyResultShapes lowering machinery into hardware-compatible
        operations. Replaces tensor.dim ops with IE::ShapeOf followed by IE::Slice to extract
        specific dimensions, converts arith.divsi into IE::DivideOp, and converts arith.addi
        into IE::AddOp respectively.
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
        This pass contains two rewrite patterns:

        - ConvertToGather: When SliceOp branches share the same source and are finally concatenated,
          convert these SliceOp into single Gather operation.
          For NPU4000+, Gather operation will be mapped to GatherDMA.
          With this conversion, DMA workloads can be reduced and benefit performance.

        - MergeParallelGatherSameIndices: When N GatherOps read from N different constant of equal shape,
          share the same indices input, and all feed a single ConcatOp,
          replace them with one GatherOp over a concatenated table.
          Per-table index offsets are added via an AddOp, and the ConcatOp is replaced by a ReshapeOp.
    }];

    let constructor = "vpux::IE::createConvertParallelSlicesToGatherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertGatherElementsToGather
//

def ConvertGatherElementsToGather : PassBase<"convert-gather-elements-to-gather", "vpux::FunctionPass"> {
    let summary = "Convert GatherElements op to Gather op";

    let description = [{
        The pass is a part of `AdjustForVPU` pipeline.

        This pass replaces suitable `GatherElements` operations with `Gather` operations
        using two strategies:

        1. GatherElementsFlatConverter (higher priority):
           Applicable when data and indices have the same shape and all dimensions after
           the gather axis are 1, reshapes to 2D, splits into chunks of at most 32768
           indices each, adds row offsets, performs flat Gather(axis=0) per chunk, then
           concatenates and reshapes back.

        2. GatherElementsOpConverter:
           Handles the case where GatherElements has a Tile producer on the gather axis,
           converting to Squeeze + Gather.
    }];

    let constructor = "vpux::IE::createConvertGatherElementsToGatherPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapOperationsWithGatherAndSlice
//

def SwapOperationsWithGatherAndSlice : PassBase<"swap-operations-with-gather-and-slice", "vpux::FunctionPass"> {
    let summary = "Reorder operations around GatherOp and SliceOp";

    let description = [{
        AI-generated summary

        Moves two-input eltwise operations (Multiply, Subtract, DynamicDequantize) and type
        conversion operations (Convert) before GatherOp when Gather reduces tensor size.
        Executing the eltwise on the smaller pre-gathered tensor decreases memory footprint
        and computational cost.
    }];

    let constructor = "vpux::IE::createSwapOperationsWithGatherAndSlicePass()";

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
// InputQuantizationRestoration
//

def InputQuantizationRestoration : PassBase<"input-quantization-restoration", "vpux::FunctionPass"> {
    let summary = "Support ONNX style quantized network input";

    let description = [{
        The pass is a part of `PostImportPipeline` pipeline.
        And is executed just before PropagateFQ pass.

        The purpose of this pass is to keep input quantized by introducing a proper FakeQuantize operation
        in place of the operations that are not present in the original ONNX model.
        In cases where ONNX models have quantized inputs, the first layer may only include a Dequantize
        operation without a corresponding Quantize operation.

        Example:
        ```
            Input (u8) -> Convert (f32) -> Multiply (f32) or
            Input (u8) -> Convert (f32) -> Subtract (f32) or
            Input (u8) -> Convert (f32) -> Subtract (f32) -> Multiply (f32)
        ```
        To:
        ```
            Input (u8) -> Convert (f32) -> FakeQuantize (f32)
        ```
    }];

    let constructor = "vpux::IE::createInputQuantizationRestorationPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateAndCleanUpFQ
//

def PropagateAndCleanUpFQ : PassBase<"propagate-and-clean-up-fq", "vpux::FunctionPass"> {
    let summary = "Propagates FQ up and down through FQ-agnostic ops and then remove FQ operations with equal input and output low/high values.";

    let description = [{
        The pass is a part of `PostImportPipeline` pipeline.
        And was moved here from the nGraph transformations.

        The pass includes 2 types of transformations:

        1. Propagates FQ up and down through FQ-agnostic ops.
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

        2. Removes FQ operations where:
            - input low == output low && input high == output high
            - producer and all consumers are "view-like"(not "computational") operations e.g. TransposeOp, SplitOp, ReshapeOp, etc.

        This is a temporary solution that should be replaced with PropagateAndFuseQuantizeDequantize: #-152958
    }];

    let constructor = "vpux::IE::createPropagateAndCleanUpFQPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeMatMulThroughSlice
//

def DecomposeMatMulThroughSlice : PassBase<"decompose-matmul-through-slice", "vpux::FunctionPass"> {
    let summary = "Decompose a fused MatMul + VariadicSplit into separate per-slice MatMuls";
    let description = [{
        This pass is part of the `PostImportPipeline` pipeline and must run
        before `ConvertVariadicSplitToStridedSlice`.

        Looking for the following pattern:
            MatMul(act, W_qkv)
            -> Add(bias)
            -> Reshape / AffineReshape
            -> VariadicSplit(equal split lengths)

        Replaces it with N separate MatMul+Add+Reshape sequences, each
        operating on a column-slice of W_qkv and the corresponding bias slice.
        The VariadicSplit is eliminated entirely. Each per-output MatMul is
        smaller and independent, avoiding DDR->DDR copies that the original
        strided VariadicSplit requires.

    }];

    let constructor = "vpux::IE::createDecomposeMatMulThroughSlicePass()";

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

//
// ApplyDynamicBoundaryCorrection
//

def ApplyDynamicBoundaryCorrection : PassBase<"apply-dynamic-boundary-correction", "vpux::FunctionPass"> {
    let summary = "Clear garbage from pad area for static tensors, which has been coverted to static tensors by DynamicExpandOp";

    let description = [{
        The "apply-dynamic-boundary-correction" pass addresses the issue of dynamic garbage in tensors when dynamic DPU operations are unsupported.
        Operations computed using an upper bound process the entire tensor, including garbage, which can cause accuracy issues
        in grouped operations like convolution or pooling. In case kernel size > 1 by dynamic dimension
        This pass inserts a sequence of operations IE.DynamicDataMask -> IE.Multiply before each originally dynamic operation,
        processed as static, to remove dynamic garbage.

        The mask can be generated in two different modes:
            - In terms of elements (MultiplyOp required to apply mask):
                For fp16: `0x3c00`
                For int8: `0x01`
                For int32: `0x00000001`
            - In terms of bit mask (BitwiseAndOp required to apply mask):
                For fp16: `0xFFFF`
                For int8: `0xFF`
                For int32: `0xFFFFFFFF`
        Currently the first option is used for all cases. The second option could be used in case of integer tensors
        but there is the room for future optimizations

        Example Before:
            %1 = IE.Add(%0, %cst_0) : tensor<1x3x?x?xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x?x?xf16>
            %2 = IE.Convolution(%1, %cst) : tensor<1x3x?x?xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x?x?xf16>
        Example After:
            %1 = IE.Add(%0, %cst_0) : tensor<1x3x?x?xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x?x?xf16>
            %mask = IE.DynamicDataMask(%1) : tensor<1x3x?x?xf16> -> tensor<1x3x?x?xf16>
            %cleaned = IE.Multiply(%1, %mask) : tensor<1x3x?x?xf16>, tensor<1x3x?x?xf16> -> tensor<1x3x?x?xf16>
            %2 = IE.Convolution(%cleaned, %cst) : tensor<1x3x?x?xf16>, tensor<3x3x3x3xf16> -> tensor<1x3x?x?xf16>
    }];

    let constructor = "vpux::IE::createApplyDynamicBoundaryCorrectionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FixDynamicOpsLocations
//

def FixDynamicOpsLocations : PassBase<"fix-dynamic-ops-locations", "vpux::FunctionPass"> {
    let summary = "Resolve duplicate operation locations in dynamic ops by appending unique counters";

    let description = [{
        AI-generated summary

        Detects operations with duplicate source locations (created by IR
        unrolling/cloning) and appends counter suffixes ("duplicated_1", "duplicated_2")
        to make them unique. Skips operations with unknown locations.
    }];

    let constructor = "vpux::IE::createFixDynamicOpsLocationsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ReassociateMultiply
//

def ReassociateMultiply : PassBase<"reassociate-multiply", "vpux::FunctionPass"> {
    let summary = "Reassociate multiply ops to reduce FLOPS";

    let description = [{

        1x32x1024x1024      1x32x1x1                            1x32x1x1        1x1x1x1
                \          /                                        \        /
                  Multiply       1x1x1x1   =>   1x32x1024x1024       Multiply
                      \         /                           \       /
                        Multiply                             Multiply
    }];

    let constructor = "vpux::IE::createReassociateMultiplyPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConsolidateNF4WeightsPattern
//

def ConsolidateNF4WeightsPattern : PassBase<"consolidate-nf4-weights-pattern", "vpux::FunctionPass"> {
    let summary = "Convert nf4 weights patten with GatherOp to QuantizeCast. Note that only uint4 type weights are supported.";

    let description = [{

                           Const(uint4 weights)
                                  |                         QuantizeCast
                    Const(LUT)  Convert               =>         |
                         \      /                          Convert (optional)
                          Gather
                            |
                    Convert (optional)

    }];

    let constructor = "vpux::IE::createConsolidateNF4WeightsPatternPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DumpStatisticsOfIeOps
//

def DumpStatisticsOfIeOps : PassBase<"dump-statistics-of-ie-ops", "vpux::ModulePass"> {
    let summary = "Reports operations found in IE dialect IR";

    let description = [{
        This pass analyzes the IE dialect operations and reports statistics on
        various operations found.

        Note that the main user of this pass at the moment is weights
        separation, in particular, a pass is used to analyze the init schedule.
    }];

    let constructor = "vpux::IE::createDumpStatisticsOfIeOpsPass()";
}

//
// ConvertBlockCacheAttentionToFlashSDPA
//

def ConvertBlockCacheAttentionToFlashSDPA : PassBase<"convert-block-cache-attention-to-flash-sdpa", "vpux::FunctionPass"> {
    let summary = "Converts IE.Attention with block-based KV-cache concat inputs into a cascade of IE.FlashSDPA ops";

    let description = [{
        Matches IE::AttentionOp whose K and V inputs are formed by a Concat of
        paged/blocked KV-cache blocks (optionally wrapped by a GQA head-fold
        Reshape: 4D Concat [1, kvH, S, E] -> [kvH, 1, S, E]).  "Block-based"
        refers to KV-cache blocking, not VPU-level computation tiling.

        Rewrites the op into a cascade of IE::FlashSDPAOp with online softmax,
        one per merged block group.  Adjacent blocks are greedily merged until
        the cumulative sequence length is a multiple of 16.  When a GQA Reshape
        wrapper is detected, Q is un-folded to batch=1 so FlashSDPA handles
        head-grouping natively.

        Preconditions: no bias/padSizeS/sink operand; >= 3 Concat inputs;
        total source sequence length >= 2048 (prefill) or >= 8192 (decode)
        and divisible by 16.
    }];

    let constructor = "vpux::IE::createConvertBlockCacheAttentionToFlashSDPAPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSDPAToFlashSDPA
//

def ConvertSDPAToFlashSDPA : PassBase<"convert-sdpa-to-flash-sdpa", "vpux::FunctionPass"> {
    let summary = "Converts IE.SDPA to IE.FlashSDPA";

    let description = [{
        AI-generated summary

        Transforms IE.SDPAOp into IE.FlashSDPAOp, which supports tiled execution of
        attention to reduce peak memory usage. Pre-multiplies the query by a scale factor
        (1/sqrt(head_dim)), and initializes running-max, running-sum, and output accumulator
        tensors needed by the FlashAttention tiling algorithm.
    }];

    let constructor = "vpux::IE::createConvertSDPAToFlashSDPAPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertToMixPrecision
//

def ConvertToMixedPrecision: PassBase<"convert-to-mixed-precision", "vpux::FunctionPass"> {
    let summary = "Convert DPU task without fake quantize behind to mixed-precision operation";

    let description = [{
        AI-generated summary

        The pass is a part of `LowPrecision` pipeline.

        Converts DPU tasks to mixed-precision operations by removing unnecessary
        Dequantize/Quantize operations around NCE tasks. Uses a platform-specific strategy
        that registers the following rewriters:

        - FloatOutConvRewriter: Removes Dequantize from IE.Convolution inputs when mixed
          precision is supported.
        - FloatOutGroupConvRewriter: Same for IE.GroupConvolution.
        - FloatOutTransposedConvRewriter: Same for IE.TransposedConvolution.
        - FloatOutMatMulRewriter: Same for IE.MatMul.
        - FloatOutAvgPoolRewriter: Removes Dequantize from IE.AvgPool input (per-tensor
          quantization only).
        - FloatOutAddRewriter: Removes Dequantize from both IE.Add inputs.
        - QuantizeWithNCERewriter: Fuses a trailing IE.Quantize into its NCE producer by
          changing the producer output type to quantized, eliminating the separate quantize
          operation.

        The `enable-float-in-quant-weights-mixed-mode` option controls whether mixed mode
        for NCE tasks with float input and quantized weights is enabled.
    }];

    let constructor = "vpux::IE::createConvertToMixedPrecision()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableFloatInQuantWeightsMixedMode", "enable-float-in-quant-weights-mixed-mode",
            "bool", "true",
            "Enable mixed mode for NCE tasks with float input and quantized weights"
        >
    ];
}

//
// SwapMaxPoolWithActivation
//

def SwapMaxPoolWithActivation : PassBase<"swap-maxpool-with-act", "vpux::FunctionPass"> {
    let summary = "Swaps the MaxPool and activation";

    let description = [{
        AI-generated summary

        This pass is needed for VPUX37XX only since HW MaxPool does not support post-op operations.
        Operations are swapped only if there is an operation before MaxPool that supports post-ops.

        Currently handles only IE.ReLU activation. When ReLU follows a MaxPool whose
        producer supports post-ops, the pass swaps the two operations so ReLU can be
        fused as a post-op into the preceding NCE task.
    }];

    let constructor = "vpux::IE::createSwapMaxPoolWithActivation()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateReorderToNCE
//

def PropagateReorderToNCE : PassBase<"propagate-reorder-to-nce", "vpux::FunctionPass"> {
    let summary = "Propagate reorder back to NCE task through act shave layers";

    let description = [{
        Converts these subgraphs:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> Act shave layer -> IE.Reorder [NCHW]
        ```
        Into the following subgraphs respectively:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> IE.Reorder [NCHW] -> Act shave layer [NCHW]
        ```
    }];

    let constructor = "vpux::IE::createPropagateReorderToNCEPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertFFTToConv
//

def ConvertFFTToConv : PassBase<"convert-fft-to-conv", "vpux::FunctionPass"> {
    let summary = "Replace FFT operations (I/R/ DFT) with a subgraph of smaller operations";

    let description = [{
        AI-generated summary

        Decomposes DFT, IDFT, RDFT, and IRDFT operations into reshape, transpose, and
        matrix multiplication subgraphs. Generates twiddle factor weight matrices from
        cos/sin coefficients so that the FFT computation reduces to MatMul, which can
        run on the DPU.
    }];

    let constructor = "vpux::IE::createConvertFFTToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeSTFT
//

def DecomposeSTFT : PassBase<"decompose-stft", "vpux::FunctionPass"> {
    let summary = "Decompose STFT operation into windowing, frame extraction, and RDFT operations";

    let description = [{
        This pass decomposes Short-Time Fourier Transform (STFT) operations into a sequence of
        primitive IE operations according to the OpenVINO STFT-15 specification.

        The decomposition follows this pipeline:
        1. Frame Extraction: Use IE.Loop with slice_input_descs to extract overlapping frames
           from the input signal with specified frame_size and frame_step
        2. Windowing: Apply IE.Multiply to element-wise multiply each frame with the window function
        3. Zero Padding: Use IE.Pad if FFT length > frame length (optional)
        4. RDFT: Apply IE.RDFT to each windowed frame to get frequency domain representation
        5. Output Formatting: Use IE.Transpose/IE.Reshape to match transpose_frames attribute

        Input signal shapes supported:
        - 1D: [signal_size] -> Output: [frames, fft_results, 2] or [fft_results, frames, 2]
        - 2D: [batch, signal_size] -> Output: [batch, frames, fft_results, 2] or [batch, fft_results, frames, 2]

        The pass leverages existing optimized IE operations:
        - IE.Loop for efficient frame extraction with slice descriptors
        - IE.RDFT for the core FFT computation (reuses existing implementation)
        - IE.Multiply for windowing function application
        - IE.Pad for zero-padding when needed
        - IE.Transpose for output layout control

        This approach provides better optimization opportunities than a monolithic STFT kernel
        and enables hardware-specific optimizations for each component operation.
    }];

    let constructor = "vpux::IE::createDecomposeSTFTPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// DecomposeISTFT
//

def DecomposeISTFT : PassBase<"decompose-istft", "vpux::FunctionPass"> {
    let summary = "Decompose ISTFT operation into windowing, frame extraction, and RDFT operations";

    let description = [{
        This pass decomposes Inverse Short-Time Fourier Transform (ISTFT) operations into a sequence of
        primitive IE operations to reconstruct time-domain signals from frequency-domain representations.

        The decomposition follows this pipeline:
        1. Frame Extraction: Use IE.Slice operations to extract individual frequency frames from the input
        2. Frame Reshaping: Use IE.Reshape to prepare each frame for IRDFT processing
        3. IRDFT: Apply IE.IRDFT to each frequency frame to convert back to time domain
        4. Windowing: Apply IE.Multiply to element-wise multiply each time frame with the window function (if provided)
        5. Overlap-Add Reconstruction: Use IE.Pad and IE.Add operations to reconstruct the final signal:
       - IE.Pad each frame to the full output signal length with appropriate zero-padding
       - IE.Add all padded frames using overlap-add method to handle overlapping regions

        Input frequency domain shapes supported:
        - 3D: [frames, fft_bins, 2] -> Output: [signal_length]
        - 4D: [batch, frames, fft_bins, 2] -> Output: [batch, signal_length]

        The pass leverages existing optimized IE operations:
        - IE.Slice for efficient frame extraction from frequency domain input
        - IE.Reshape for tensor shape adjustments between processing stages
        - IE.IRDFT for the core inverse FFT computation
        - IE.Multiply for optional windowing function application
        - IE.Pad for zero-padding frames to full signal length before overlap-add
        - IE.Add for overlap-add reconstruction of the continuous time domain signal
        - Const.DeclareOp for creating zero-initialized output buffer

        This decomposition improves performance over a single monolithic ISTFT operation.

    }];

    let constructor = "vpux::IE::createDecomposeISTFTPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ProcessAsymmetricZeroPointsForMatmul
//

def ProcessAsymmetricZeroPointsForMatmul : PassBase<"process-asymmetric-zero-points-for-matmul", "vpux::FunctionPass"> {
    let summary = "Decompose matmuls to change u8 zero point to 128";

    let description = [{
        Decompose matmuls ( which are mapped to convolution at this stage) and convolutions with 1x1 kernel size.
        This decomposition changes zero point of u8 weights to 128 to enable mixed precision execution. Mixed precision
        required i8 weights with ZP=0 ,so we can fix zero point of u8 to 128 here to enable i8 conversion.

        Original Formula for Matmul with u8 quantized weights
            A: Float matrix of shape (I, J) , input matrix
            B_q: Integer quantized matrix (uint8) derived from B, shape (J, K), weights matrix
            S: Float scale vector for B, shape (1, K) (one scale per column k of B)
            ZP: Integer zero-point vector for B, shape (1, K) (one ZP per column k of B)
            C: Resulting float output matrix, shape (I, K)

            B[j, k] = (B_q[j, k] - ZP[k]) * S[k]  ( Dequantization of B_q)
            C[i, k] = sum_{j=0}^{J-1} (A[i, j] * B[j, k])
            When B is subtituted
            C[i, k] = sum_{j=0}^{J-1} (A[i, j] * [(B_q[j, k] - ZP[k]) * S[k]])

            C[i, k] = S[k] * (sum_{j=0}^{J-1} A[i, j] * B_q[j, k]) - S[k] * ZP[k] * (sum_{j=0}^{J-1} A[i, j])
            So from above we can see that zp related calculation can be separated and does not depend on inside iteration

            Originally a matmul with zp 128 has this formula
            C[i, k] = S[k] * (sum_{j=0}^{J-1} A[i, j] * (B_q[j, k] - 128)
            if zp not = 128
            C[i, k] = S[k] * (sum_{j=0}^{J-1} A[i, j] * (B_q[j, k] - zp[k])
            to convert to zp 128 we add fix (+ zp[k]  -128 ) to inside , we also subtract it so results stays same
            Added value does not depend on B so or internal matmul multiplication so can be pulled out :
            C[i, k] = S[k] * (sum_{j=0}^{J-1} A[i, j] * (B_q[j, k] - zp[k] + zp[k]  -128]) - S[k] * (zp[k] -128) sum_{j=0}^{J-1} A[i, j]
            Negate subtract to use add Op we end up with final formula
            C[i, k] = Subtract( [Matmul(A[i, k],B_128[j, k])] , [S[k] * (- zp[k]+128)  * ReduceSum(A,dim=1)]);

    }];

    let constructor = "vpux::IE::createProcessAsymmetricZeroPointsForMatmulPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "matmulMixedPrecisionDecompositionRatio", "matmul-mixed-precision-decomposition-ratio",
            "double", "250",
            "Determines when to enable Matmul Mixed Precision Decomposition"
             "Ratio = (MatMul input size)/(Sum of Inputs of newly added ops by decomposition)"
        >
    ];
}

//
// ConvertWeightsToI8Pass
//

def ConvertWeightsToI8 : PassBase<"convert-weights-to-i8", "vpux::FunctionPass"> {
    let summary = "Converts weights to i8 for mixed precision convolution and matmul";

    let description = [{
        Converts weights to i8 for mixed precision convolution and matmul.
        This pass converts u8 weights with zero point 128 to i8 weights with zero point 0.
        For mixed precison support i8 weights must have zero point of 0.
        Matmuls are converted to Convolutions before this pass so this pass only deals
        with Convolutions.
        Reason for conversion is to enable mixed precision convolution with F16 activation
        and i8 weights to reduce memory/dma usage.
        Simply convert data types and reduces values by 128.
        Pass converts below pattern :
        Tensor:F16 -> Convolution <- Dequantise(F16) <- ConstDeclare(u8: ZeroPoint:128) to :
        Tensor:F16 -> Convolution <- Dequantise(F16) <- ConstDeclare(i8: ZeroPoint:0)
        Which is transformed by next pass ( ConvertToMixedPrecision ) to :
        Tensor:F16 -> Convolution <- ConstDeclare(i8: ZeroPoint:0)
        As a result resulting constDeclare wont be affected by future passes ConvertToU8/I4 etc
    }];

    let constructor = "vpux::IE::createConvertWeightsToI8Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FusePermuteQuantizeExpand
//

def FusePermuteQuantizeExpand : PassBase<"fuse-permute-quantize-expand", "vpux::FunctionPass"> {
    let summary = "Converts Quantize-MemPermute-Expand combination in 1 common operation";

    let description = [{
        AI-generated summary

        Fuses multi-operation patterns into single composite operations. Combines
        Expand-MemPermute-QuantizeCast patterns (NCHW to NHWC with channel expansion
        in C dimension only), and fuses Expand-MemPermute-QuantizeCast-Expand patterns
        into PermuteQuantizeOp followed by QuantizeCastOp, reducing total operation count.
    }];

    let constructor = "vpux::IE::createFusePermuteQuantizeExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseOutstandingDequant
//

def FuseOutstandingDequant: PassBase<"fuse-outstanding-dequant", "vpux::FunctionPass"> {
    let summary = "Fuse outstanding dequantize after NCE task";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass walks through quant-agnostic ops and removes dequantize after
        NCE task with quantized output. If Dequantize implicitly holds a ReLUx clamp,
        that is a Clamp range of [0, x], then explicitly create this Clamp after the NCEOp.
        If it does not exist then fuse Dequantize without creating clamp. Before creating clamp
        check that attribute of NCEOp contains clamp already, and merge the two clamp values together.
        This pass always creates explicit clamps rather than fusing as clamp attribute.

        Allowed operations to walk through are defined by isAllowedElemTypeOp.
        These are ops that do not affect data values.
        They are "View-like" ops, where Clamp won't be affected if it's moved through these ops.

        If no ReLUx implicitly found in Dequantize, then Convert :
        `u8_Convolution_u8 -> u8_AllowedElemTypeOp_u8 -> ... -> u8_Dequantize_f16 -> f16_SWKernel_f16`
        To :
        `u8_Convolution_f16 -> f16_AllowedElemTypeOp_f16 -> ... -> f16_SWKernel_f16`

        If ReLUx implicitly found in Dequantize, then Convert :
        `u8_Convolution_u8 -> u8_AllowedElemTypeOp_u8 -> ... -> u8_Dequantize_f16 -> f16_SWKernel_f16`
        To :
        `u8_Convolution_f16 -> f16_Clamp_f16 -> f16_AllowedElemTypeOp_f16 -> ... -> f16_SWKernel_f16`
    }];

    let constructor = "vpux::IE::createFuseOutstandingDequant()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ConvertSubGRUSequenceToConv
//

def ConvertSubGRUSequenceToConv : PassBase<"convert-sub-gru-sequence-to-conv", "vpux::FunctionPass"> {
    let summary = "Convert sub GRUSequence to convolution";

    let description = [{
        GRUSequence can be split into two parts, GRUSequenceFirstPart and GRUSequenceLastPart.
        GRUSequenceFirstPart is similar to matrix multiply, and GRUSequenceLastPart will use output
        of GRUSequenceFirstPart and other inputs of original GRUSequence to get outputs.

        This pass will convert GRUSequenceFirstPart to HW convolution to utilize hardware feature of NPU
        for performance improvement.
    }];

    let constructor = "vpux::IE::createConvertSubGRUSequenceToConvPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseScale
//

def FuseScale : PassBase<"fuse-scale", "vpux::FunctionPass"> {
    let summary = "Fuses suitable `IE.Multiply` with const operand to `IE.Convolution` or `IE.AvgPool`";

    let description = [{
        AI-generated summary

        Fuses `IE.Multiply` with a constant splat operand into preceding or following
        `IE.Convolution` or `IE.AvgPool` operations. Fusing multiplication by a constant
        makes the multiplication effectively no-op on several HW architectures.

        Apply the pass before ConvertToScaleShift and after ConvertDivideToMultiply to
        increase the applicability.

        The pass registers the following rewriter patterns:

        - InsertMultiplyBeforeConcat: When `IE.Multiply(Concat, Const)` is found and every
           concat branch has a preceding `IE.Convolution`, the Multiply is pushed inside
           each concat branch so it can be fused later.
        - FuseStaticScaleToPpeBackward<IE::ConvolutionOp / IE::AvgPoolOp>: Fuses Multiply
           into the `static_scale` attribute of a preceding Convolution or AvgPool.
        - FuseStaticScaleToWeights<IE::ConvolutionOp>: Fuses Multiply into the following
           Convolution weights (rescales constant weights or FakeQuantize output ranges).
        - FuseStaticScaleToPpeForward<IE::AvgPoolOp>: Fuses Multiply into the `static_scale`
           attribute of a following AvgPool.
        - FuseStaticScaleToPpeForwardWithScaleCheck<IE::ConvolutionOp>: Same as forward
           fusion into Convolution but only when scale >= 1.0 to avoid accuracy degradation.
        - FuseRoPEQueryScaleToSlicedConvs: Fuses a scalar `IE.Multiply` scale that fans out to
           several per-head `IE.Convolution` operations through `IE.Slice` and view ops
           (e.g. AffineReshape) into each convolution's `static_scale`. Applies only when the
           scaled value feeds convolution input #0, the convolution has no bias and no
           post-op/clamp, and consumes FP16 activations.
        Also we could fuse channel wise scales to convolution if it could support new weights table format
        Apply the pass before ConvertToScaleShift and after ConvertDivideToMultiply to increase
        the applicability.
    }];

    let constructor = "vpux::IE::createFuseScalePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseReordersPass
//

def FuseReordersPass : PassBase<"fuse-reorders", "vpux::FunctionPass"> {
    let summary = "Fuses reorder to previous NCE task as ODU permutation";

    let description = [{
        Converts these subgraphs:
        ```
            Input [NHWC] -> IE.Convolution [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.GroupConvolution [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.MaxPool [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.AvgPool [NHWC] -> IE.Reorder [NCHW]
            Input [NHWC] -> IE.Add [NHWC] -> IE.Reorder [NCHW]
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

    let constructor = "vpux::IE::createFuseReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// PropagateExpand
//

def PropagateExpand : PassBase<"propagate-expand", "vpux::FunctionPass"> {
    let summary = "Propagate Expand operation in order to fuse it with other layers";

    let description = [{
        AI-generated summary

        Propagate Expand through various operations so that Expand can be fused with
        preceding or following layers. The pass registers four rewriter patterns:

        - EltwiseShapeCastRewriter: Propagates Expand through quantized eltwise Add
           operations with ShapeCast, enabling PermuteQuantize fusion with Expand in
           the FusePermuteQuantizeExpand pass.
        - SliceShapeCastRewriter: Propagates Expand through Slice + ShapeCast chains.
        - DepthToSpaceSliceRewriter: Propagates Expand before DepthToSpace when D2S uses
           padded channels descriptor (block_size=2, channels <= 16).
        - SpaceToDepthSliceRewriter: Propagates Expand before SpaceToDepth so that the
           following Convolution can execute with expanded input and padded filter on DPU.
    }];

    let constructor = "vpux::IE::createPropagateExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// InsertIdentityPoolBeforeOp
//

def InsertIdentityPoolBeforeOp : PassBase<"insert-identity-pool-before-op", "vpux::FunctionPass"> {
    let summary = "Insert Maxpool/AvgPool op before Activation ops and MemPermute ops";

    let description = [{
        AI-generated summary

        Inserts an identity AvgPool before LeakyRelu, Clamp, or ReLU activations so
        these can later be fused as post-ops into the NCE AvgPool.
        The AvgPool is only inserted when the activation does not already have a suitable
        NCE producer with an available PPE slot, and when the resulting AvgPool is supported
        by NCEAveragePoolOp.
    }];

    let constructor = "vpux::IE::createInsertIdentityPoolBeforeOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// LoggingWeightsQuantFusedIntoTaskPass
//

def LoggingWeightsQuantFusedIntoTask: PassBase<"logging-weights-quant-fused-into-task", "vpux::FunctionPass"> {
    let summary = "Flag that weights quant is fused into task";

    let description = [{
        The pass is a part of the `LowPrecision` pipeline and is used for debugging purposes only.

        This pass logs diagnostic information about weight quantization fusion into tasks. It walks through
        weight constants and block arguments for operations like Convolution, MatMul, and GroupConvolution,
        and logs whether the weights have quantized element types (int8/int4), indicating that weight
        quantization has been fused into the task.

        The pass generates trace-level logs for two scenarios:
        - "WAC" (Weights As Constant): When weights are provided as quantized constants
        - "WAI" (Weights As Input): When weights are block arguments with quantized types

        This pass does not modify the IR and is purely for debugging analysis. It can be enabled
        via NPU_COMPILATION_MODE_PARAMS with:
        "NPU_COMPILATION_MODE_PARAMS": "enable-log-dynamic-quant=true"

        Note: This option is separate from the public NPU_COMPILER_DYNAMIC_QUANTIZATION configuration and
        is only available through NPU_COMPILATION_MODE_PARAMS for advanced debugging.
    }];

    let constructor = "vpux::IE::createLoggingWeightsQuantFusedIntoTaskPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// ReduceNumTilesForSmallModelsPass
//

def ReduceNumTilesForSmallModelsPass: PassBase<"reduce-num-tiles-for-small-models", "vpux::ModulePass"> {
    let summary = "Reduce number of clusters for small models";

    let description = [{
        There are some cases when running a model on all clusters can be slower than on one cluster.

        Based on observations, it happens on transformer models when the following conditions are met:
            1) The model's inference latency is small, e.g. few hundreds of microseconds.
            2) The shapes for most of the layers are small, e.g. ~10,000 elements.
            3) From these small layers there are a lot of Add & Multiply that are not fused into Convolution,
                so they end up being a separate DPU operation.

        Running such a model on all clusters introduces overhead, that is bigger than the improvement mostly because:
            1) Shave initialization takes longer, as we need to initialize more shaves.
            2) There are a lot of tasks (both compute and DMA), which creates pressure on scheduler.
            3) A lot of small DMA transactions create pressure on DMA unit.

        The pass makes a choice based on two main heuristics comparing them with experimental numbers:
            1) Mean and standard deviation of tensor sizes.
            2) Ratio of MatMul/Multiply/SoftMax operations (that are frequently met in transformer models).
    }];

    let constructor = "vpux::IE::createReduceNumTilesForSmallModelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];

    let options = [
        Option<
            "enableSDPAContributions", "enable-sdpa-contributions",
            "bool", "false",
            "Count SDPAOp as equivalent Multiply + 2xMatMul + SoftMax when computing attention-model ratios"
        >
    ];
}

//
// OptimizeSliceExpand
//

def OptimizeSliceExpand : PassBase<"optimize-slice-expand", "vpux::FunctionPass"> {
    let summary = "Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand";

    let description = [{
        AI-generated summary

        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimizes Slice-Expand patterns and Slice-ImplicitOps-Expand patterns to avoid
        extra DMAs. Registers the following rewriter patterns:

        - OptimizeSliceExpand: Direct Slice -> Expand elimination.
        - OptimizeExpandSlice: Expand -> Slice elimination.
        - OptimizeSliceImplicitExpand<QuantizeCastOp/HSwishOp/SwishOp/GeluOp/ClampOp>:
          Optimizes Slice -> implicit activation -> Expand chains.
        - OptimizeSliceSoftmaxExpand: Slice -> SoftMax -> Expand optimization.
        - OptimizeSliceLayoutCastExpand: Slice -> LayoutCast -> Expand.
        - OptimizeSlicePermuteCastExpand: Slice -> PermuteCast -> Expand.
        - OptimizeSliceShapeCastExpand<HSwishOp/SwishOp/GeluOp/SigmoidOp>:
          Slice -> ShapeCast -> activation -> Expand.
        - OptimizeSliceConcatExpandWithViewLikeOps: Slice -> Concat -> Expand with view-like ops.
        - OptimizeSliceConcatExpand: Slice -> Concat -> Expand.
        - OptimizeSlicePReluExpand: Slice -> PReLU -> Expand.
        - OptimizeSliceEltwiseExpand<MultiplyOp/AddOp/SubtractOp/DivideOp>:
          Slice -> eltwise -> Expand for two-input operations.
        - SliceAfterAddForLayoutCastExpandAddRewriter: LayoutCast -> Expand -> Add chains.
        - OptimizeSliceOpsExpand: General Slice -> op1 -> op2 -> ... -> opN -> Expand chains.
    }];

    let constructor = "vpux::IE::createOptimizeSliceExpandPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// FuseDepthToSpaceExpandChannels
//

def FuseD2SExpandChannels : PassBase<"fuse-depth-to-space-expand-channels", "vpux::FunctionPass"> {
    let summary = "Fuse depth to space and expand channels";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimize patterns where DepthToSpace is followed by Expand, in the case where DepthToSpace has only one user and the Expand operation expands only the channels.
    }];

    let constructor = "vpux::IE::createFuseD2SExpandChannelsPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// OptimizeActivationsPipelineRewriterExecutor
//

def OptimizeActivationsPipelineRewriterExecutor : PassBase<"run-optimize-activations-rewriters", "vpux::FunctionPass"> {
    let summary = "Build optimize activations pipeline rewriter collection";

    let description = [{
        AI-generated summary

        Executes a dynamically-loaded rewriter collection that optimizes activation functions
        and related operations. Registers rewriters for swapping operation order for efficiency,
        inserting identity pools, swapping maxpool with activations, and fusing consecutive
        activation operations. Supports selective execution via the rewriter name option.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >
    ];

    let constructor = "vpux::IE::createOptimizeActivationsPipelineRewriterExecutorPass()";
}

//
// AdjustForVPUPipelineRewriterExecutor
//

def AdjustForVPUPipelineRewriterExecutor : PassBase<"run-adjust-for-vpu-rewriters", "vpux::FunctionPass"> {
    let summary = "Build adjust for VPU pipeline rewriter collection";

    let description = [{
        AI-generated summary

        Applies a collection of rewriters that adapt IR for VPU-specific constraints. Registers
        rewriters for merging tile operations with slices, converting large convolutions into
        multiple convolutions with addition, merging weight-shared convolutions, handling
        per-axis fake quantization with concat, and fusing padding operations. Supports selective
        execution via the rewriter name option.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >
    ];

    let constructor = "vpux::IE::createAdjustForVPUPipelineRewriterExecutorPass()";
}

//
// LegalizeEpsilonUsage
//

def LegalizeEpsilonUsage : PassBase<"legalize-epsilon-usage", "vpux::FunctionPass"> {
    let summary = "Legalize Epsilon Usage To Prevent Divide-By-Zero Conditions";

    let description = [{
        AI-generated summary

        Low-precision normalization subgraphs and layers often rely on small positive epsilon values to
        prevent divide-by-zero. If epsilon is too small it can cease to function properly and lead to NaNs
        in the output when the input tensor contains exclusively or mostly zeros.

        This pass legalizes epsilon values in IE.NormalizeL2 and IE.RMS operations with f16
        element types. When the epsilon attribute is smaller than the minimum safe epsilon for
        float16 mixed-precision arithmetic, the pass replaces it with the minimum safe value.
    }];

    let constructor = "vpux::IE::createLegalizeEpsilonUsagePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MemPermuteProcessingPipelineRewriterExecutor
//

def MemPermuteProcessingPipelineRewriterExecutor : PassBase<"run-mem-permute-processing-rewriters", "vpux::FunctionPass"> {
    let summary = "Build mem permute processing pipeline rewriter collection";

    let description = [{
        AI-generated summary

        Executes rewriters specialized for memory permutation optimization. Swaps memory permute
        operations with expands, propagates memory permutation before operations, optimizes
        concat-convolution combinations, and fuses memory permutation operations. Supports
        selective execution via the rewriter name option.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >
    ];

    let constructor = "vpux::IE::createMemPermuteProcessingPipelineRewriterExecutorPass()";
}

//
// InitialLowPrecisionTransformationsPipelineRewriterExecutor
//

def InitialLowPrecisionTransformationsPipelineRewriterExecutor : PassBase<"run-initial-low-precision-transformations-rewriters", "vpux::FunctionPass"> {
    let summary = "Build initial low precision transformations pipeline rewriter collection";

    let description = [{
        This pass runs the set of rewriters from DecomposeMultiZPQuantization, WeightsDequantizeToDynamicDequantize,
        ConsolidateWeightsDequantization and ConsolidateActivationFP8Quantization from the InitialLowPrecisionTransformations pipeline.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >
    ];

    let constructor = "vpux::IE::createInitialLowPrecisionTransformationsPipelineRewriterExecutorPass()";
}

// BatchOpProcessingPipelineRewriterExecutor
//

def BatchOpProcessingPipelineRewriterExecutor : PassBase<"run-batch-op-processing-rewriters", "vpux::FunctionPass"> {
    let summary = "Build batch op processing pipeline rewriter collection";
    let description = [{
        AI-generated summary

        Registers and executes rewriters for batch dimension handling. Reshapes MatMul inputs
        to 2D form when beneficial and propagates eltwise operations through batch-dimension
        concat operations. Supports selective execution via the rewriter name option and
        optional grouped MatMul execution toggle.
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >,
        Option<
            "enableGroupedMatMul", "enable-grouped-matmul",
            "bool", "false",
            "Flag to enable or disable grouped MatMul execution"
        >
    ];

    let constructor = "vpux::IE::createBatchOpProcessingPipelineRewriterExecutorPass()";
}

//
// ForbidFourBitOutputs
//

def ForbidFourBitOutputs : PassBase<"forbid-four-bit-outputs", "vpux::ModulePass"> {
    let summary = "Stop compilation if network output has 4 bits";

    let description = [{
        The compiler does not have support for 4-bit network outputs.
        If such outputs are detected, the pass fails, which will cause the compilation to stop.
    }];

    let constructor = "vpux::IE::createForbidFourBitOutputsPass()";
}

//
// OptimizeQuantizationForAttention
//

def OptimizeQuantizationForAttention : PassBase<"optimize-quantization-for-attention", "vpux::FunctionPass"> {
    let summary = "Move ScatterUpdate to operate on quantized (si8) KV cache data";

    let description = [{
        This pass optimizes the quantization/dequantization pattern around ScatterUpdate operations in KV-cache inference.
        It moves the scatter from f32 to si8 domain by:
            1. Removing the dequant chain (Convert si8→f32 + Multiply scale) before the scatter
            2. Quantizing the update token (Multiply + Round + Clamp + Convert) before inserting
            3. Performing the ScatterUpdate directly in si8
            4. Re-dequantizing the output for downstream SDPA consumers
        This reduces memory bandwidth and storage for KV-cache by keeping the cache tensor in int8 throughout the scatter operation.
    }];

    let constructor = "vpux::IE::createOptimizeQuantizationForAttentionPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
