//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES
#define VPUX_COMPILER_DIALECT_IE_ARCH_37XX_PASSES

include "mlir/Pass/PassBase.td"

//
// InsertIdentityPoolBeforeOp
//

def InsertIdentityPoolBeforeOp : PassBase<"insert-identity-pool-before-op", "vpux::FunctionPass"> {
    let summary = "Insert Maxpool/AvgPool op before Activation ops and MemPermute ops";

    let description = [{
        The pass inserts MaxPool/AvgPool op before Activation ops and MemPermute ops.
        MaxPool/AvgPool will be lowered to a NCE op and the activation op will be fused into it.
        Supported activations: LeakyRelu, Clamp.
    }];

    let constructor = "vpux::IE::arch37xx::createInsertIdentityPoolBeforeOpPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// MapBilinearInterpolateOnDPUPass
//

def MapBilinearInterpolateOnDPUPass : PassBase<"map-bilinear-interpolate-on-dpu", "vpux::FunctionPass"> {
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

    let constructor = "vpux::IE::arch37xx::createMapBilinearInterpolateOnDPUPass()";

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
// OptimizeSliceExpand
//

def OptimizeSliceExpand : PassBase<"optimize-slice-expand", "vpux::FunctionPass"> {
    let summary = "Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand";

    let description = [{
        The pass is a part of `buildHardwareModePipeline` pipeline.

        Optimize patterns Slice->Expand and Slice->Implicit operations ->Expand in order to avoid extra DMAs
    }];

    let constructor = "vpux::IE::arch37xx::createOptimizeSliceExpandPass()";

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
        Propagate Expand through Eltwise Add in case layers before might be fused with Expand
        in following cases:
        1. PermuteQuntize might be fused with Expand in FusePermuteQuantizeExpand pass
        2. DepthToSpace is used with padded channels' descriptor
        3. SpaceToDepth might be executed with expanded input on dpu with following convolution with padded filter
    }];

    let constructor = "vpux::IE::arch37xx::createPropagateExpandPass()";

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
        Converts Quantize-MemPermute-Expand combination in 1 common operation.
    }];

    let constructor = "vpux::IE::arch37xx::createFusePermuteQuantizeExpandPass()";

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

        This pass processes operations, which can be compile as a DPU tasks and
            expands channels number to number divisible by 16 in case they doesn't satisfy hardware requirements.
        The input channels could be aligned to 4 instead of 16 for compressed convolutions.
    }];

    let constructor = "vpux::IE::arch37xx::createExpandActivationChannelsPass()";

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
// ConvertFFTToConv
//

def ConvertFFTToConv : PassBase<"convert-fft-to-conv", "vpux::FunctionPass"> {
    let summary = "Replace FFT operations (I/R/ DFT) with a subgraph of smaller operations";

    let description = [{
        Decomposes `FFT` operations (DFT, IDFT, RDFT, IRDFT) into smaller `convolution` friendly operations.
    }];

    let constructor = "vpux::IE::arch37xx::createConvertFFTToConvPass()";

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

    let constructor = "vpux::IE::arch37xx::createConvertSubGRUSequenceToConvPass()";

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
        The pass is a part of `LowPrecision` pipeline.
        Converts DPU task to mixed-precision operation where there is no quantize operation for the output of a DPU task
    }];

    let constructor = "vpux::IE::arch37xx::createConvertToMixedPrecision()";

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
// OptimizeNetworkInputConvert
//

def OptimizeNetworkInputConvert: PassBase<"optimize-network-input-convert", "vpux::FunctionPass"> {
    let summary = "Fuses outstanding Convert operations into input of quantized consumers";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.
        To avoid extra FP->Quant Convert operations at network start, fuses these Convert ops into quantized consumers
        resulting mixed precision operations with FP input and Quant output.
        As a side effect, also the original quantized weights are dequantized.
        There is later logic which attempts to enable mixed precision of FP input activations and Quant weights.
    }];

    let constructor = "vpux::IE::arch37xx::createOptimizeNetworkInputConvertPass()";

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

    let constructor = "vpux::IE::arch37xx::createPropagateReorderToNCEPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// SwapMaxPoolWithActivation
//

def SwapMaxPoolWithActivation : PassBase<"swap-maxpool-with-act", "vpux::FunctionPass"> {
    let summary = "Swaps the MaxPool and activation";

    let description = [{
        This pass is needed for NPU37XX only since HW MaxPool does not support post-op operations.
        Operations are swapped only if there is an operation before MaxPool that supports post-ops.
    }];

    let constructor = "vpux::IE::arch37xx::createSwapMaxPoolWithActivation()";

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

    let constructor = "vpux::IE::arch37xx::createFuseReordersPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def FuseStaticScale : PassBase<"fuse-static-scale", "vpux::FunctionPass"> {
    let summary = "Fuses suitable `IE.Multiply` with const operand to `IE.Convolution` or `IE.AvgPool`";

    let description = [{
        This transformation takes the following subgraph:
        ```
            IE.Convolution / IE.AvgPool     Const.Declare
                        \                       /
                                IE.Multiply
        ```
        and fuses the `IE.Multiply` into `IE.Convolution` or `IE.AvgPool` (see `static_scale`
        attribute and weights table generation for details).

        Fusing multiplication by a constant to a Conv or Pooling makes the
        multiplication itself effectively no-op on several HW architectures.

        Aplly the pass before ConvertToScaleShift and after ConvertDivideToMultiply to increase
        the applicability.
    }];

    let constructor = "vpux::IE::arch37xx::createFuseStaticScalePass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
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

    let constructor = "vpux::IE::arch37xx::createConvertWeightsToI8Pass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

def ProcessAsymmetricZeroPointsForConvolution : PassBase<"process-asymmetric-zero-points-for-convolution", "vpux::FunctionPass"> {
    let summary = "Changes zero point of convolution weights to 128 via decomposition";

    let description = [{
        Changes zero point of convolution weights to 128 via decomposition.
        This is done to support mixed precision convolution which must have f16 input and i8 weights with zero
        point of 0, only u8 weights with zero point of 128 are converted to i8 ( by ConvertWeightsToI8Pass) so
        this pass fixes zero points of u8 weights to 128.
        Pass first look for this pattern for convolution filter:
        FakeQuantize -> Optional(Transpose) -> Optional(AffineReshape)  -> Convolution
        Full pattern with Transpose and AffineReshape captures matmuls which are converted to Convolutions.
        FakeQuantize -> Convolution pattern captures ops which were originally convolution in the input graph.
        Additional conditions are checked for the pattern to be valid.
        Quantized type must be unsigned 8 bit number with zero point different than 128.
        Converts : input -> Convolution <- FakeQuantize <- constDeclare
        To :
        P1 =  input -> Convolution <- (ModifiedFakeQuantize ZeroPoint = 128 ) <- constDeclare
        P2 =  input -> Convolution <- constDeclare ( same size as original filter, values are (128-ZeroPoint) * scale)
            P1 -> Add <- P2
        Current implementation causes a regression as it increases #DPU task 2x, and DMA size is not any
        less than original flow. There are plans to reduce DMA size E#126278.
    }];

    let constructor = "vpux::IE::arch37xx::createProcessAsymmetricZeroPointsForConvolutionPass()";

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
        NCE task with quantized output.

        Converts :
        `u8_Convolution_u8 -> u8_ElemTypeInfoOp_u8 -> ... -> u8_Dequantize_f16 -> f16_SWKernel_f16`
        To :
        `u8_Convolution_f16 -> f16_ElemTypeInfoOp_f16 -> ... -> f16_SWKernel_f16`
    }];

    let constructor = "vpux::IE::arch37xx::createFuseOutstandingDequant()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

//
// WeightsQuantFusedIntoTaskPass
//

def WeightsQuantFusedIntoTask: PassBase<"weights-quant-fused-into-task", "vpux::FunctionPass"> {
    let summary = "Flag that weights quant is fused into task";

    let description = [{
        The pass is a part of `LowPrecision` pipeline.

        Pass walks through weights constants and flag if it's fused into task
    }];

    let constructor = "vpux::IE::arch37xx::createWeightsQuantFusedIntoTaskPass()";

    let dependentDialects = [
        "vpux::IE::IEDialect"
    ];
}

#endif
