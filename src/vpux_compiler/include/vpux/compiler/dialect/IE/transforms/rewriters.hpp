//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux {
namespace IE {

//
// InitialLowPrecisionTransformationsPipeline
//

/*
Due to the limitation that HW doesn't have support for per-channel zero-point,
this pass applies the solution described below to decompose the original quantization pattern
with per-channel zero-point into the proposed pattern.
Only group-wise quantization with quant params constants is supported by this pass for now.

       Weights        Zero-points    weights                  Parameter   Zero-points
            |          |                  |                   /   |            |
       Convert        Convert        Convert       Scales    / Reshape     Convert Scales
             \        /                    \        /       /     |           |    /
              Subtract  Scales    =>        Multiply       /  ReduceSum    Multiply
                 |     /                           \      /        \      /
Parameter      Multiply                             MatMul          MatMul
         \      /                                         \        /
          MatMul                                           Subtract
*/
void registerDecomposeMultiZPQuantizationRewriters(RewriterRegistry& registry,
                                                   ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                   Logger log = Logger::global());

/*
Replaces:
Constant -> [Convert (to fp)] -> Multiply (scale) with
Constant -> DynamicDequantize(weights_q, scale).
*/
void registerWeightsDequantizeToDynamicDequantizeRewriters(RewriterRegistry& registry,
                                                           ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                           Logger log = Logger::global());

/*
Replaces:
  INT4 weights as params -> Convert (to fp16) -> Subtract (w params) -> Multiply (w params) -> Conv with
  INT4 weights as params -> QuantCast (dummy quant zp=0 scale=1) -> DynamicDequantize (w scale and zp params) -> Conv
 */
void registerConsolidateWeightsDequantizationRewriters(RewriterRegistry& registry,
                                                       ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                       Logger log = Logger::global());
/*
These rewriters target FP8 quantized activation with 4-bit weights.
Tehy identify a MatMul / FullyConnected operation with FP8 input and INT4 weights, where both the weights and the
quantization parameters are inputs. This pattern then decomposes FakeConvert to GroupConv->Quantize->Dequantize->Divide,
then move divide post MatMul/FC.
*/
void registerConsolidateActivationFP8QuantizationRewriters(RewriterRegistry& registry,
                                                           ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                           Logger log = Logger::global());

//
// Rewriters used in more than one pipeline
//

void registerPropagateMemPermuteBeforeOpRewriters(RewriterRegistry& registry,
                                                  ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                  Logger log);
void registerSwapOperationsRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                     size_t index, bool seOpsEnabled, Logger log);
void registerInsertIdentityPoolBeforeOpRewriters(RewriterRegistry& registry,
                                                 ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                                 Logger log);
void registerFuseActivationOpsRewriters(RewriterRegistry& registry, Logger log = Logger::global());

//
// AdjustForVPU Pipeline
//

/*
This rewriter creates `FakeQuantize` operation, which combines per-channel quantization from `Concat` inputs,
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
*/
void registerPerAxisFQConcatRewriters(RewriterRegistry& registry, Logger log = Logger::global());

/*
This rewriter converts ShuffleChannels to Reshape->Transpose->Reshape.
*/
void registerConvertShuffleChannelsRewriters(RewriterRegistry& registry, Logger log = Logger::global());

/*
                               Input(2x1x10x80)
                                       |
                               IE.Tile(2x2x10x80)
                                       |
                               IE.Reshape(1x4x10x80)
                                       |
            -----------------------------------------------------------
            |                   |                   |                 |
       IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)
To:
                               Input(2x1x10x80)
                                       |
            -----------------------------------------------------------
            |                   |                   |                 |
       IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80) IE.Slice(1x1x10x80)
*/
void registerMergeTileWithSliceRewriters(RewriterRegistry& registry, Logger log = Logger::global());

/*
Converts large `Convolution` Op into multiple smaller `Convolution` Op followed by an `Add` Op.

The purpose is to optimize the `Convolution` by splitting it into smaller pieces along the input channels.
There are two performance benefits:
1. Reduces the overlapped input data, thereby decreasing the DMA size
2. Reduces the number of tiles, preventing excessive tiling and improving workload efficiency
*/
void registerConvertLargeConvToMultiConvWithAddRewriters(RewriterRegistry& registry, Logger log = Logger::global());

/*
The optimization is used for LLM GQA. Assume queries(input) is 28, value or key(filter)
is 4, that means each 7 queries share the same value or key(filter). Originally we have
28 conv in total, after optimization, only have 4 convs. Reduce Op number will benefit to
runtime idle, and also for some case convolution input H increase from 1 to 7 also benefit
to HW efficient.
*/
void registerMergeWeightsSharedConvRewriters(RewriterRegistry& registry, Logger log = Logger::global());

/*
PadOp with CONSTANT model, pad value is 0 and the padding is needed in H and W dimensions only.
Merge [Pad] -> [Conv] into [Conv].
Merge [Pad] -> [GroupConv] into [GroupConv].
Merge [Pad] -> [MaxPool] into [MaxPool].
*/
void registerFusePadOpsRewriters(RewriterRegistry& registry, Logger log = Logger::global());

//
// BuildBatchOpProcessing Pipeline
//

/*
This pass converts `MatMul` inputs to 2d.
For example, `MatMul` input with 4x1x64 geometry will be split to four inputs with 1x64 dimensions.
Resulting inputs with filters go to `MatMul` operations and the outputs are concatenated.
*/
void registerMatMulInputsTo2dRewriters(RewriterRegistry& registry, Logger log,
                                       ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index,
                                       bool enableGroupedMatMul);

/*
Folds inverse Reshape pairs in GQA SDPA chains for batch-unrolled MatMul sequences.
For unrolled chains (Concat -> Reshape -> [Add] -> SoftMax -> Reshape -> Slices),
removes the Reshape round-trip by tiling the mask into the shrunk domain, keeping
SoftMax in the more efficient head-shrunk layout.
*/
void registerEliminateReshapeRoundtripInSDPARewriters(RewriterRegistry& registry, Logger log,
                                                      ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index);

/*
Move ops after concat to place after each batch unrolled matmul.
Currently only softmax is enabled.
*/
void registerPropagateOpThroughBatchConcatRewriters(RewriterRegistry& registry, Logger log,
                                                    ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index);

//
// BuildMemPermuteProcessing Pipeline
//

/*
The pass is a part of `MemPermute processing` pipeline.
This pass swap Reorder-like `MemPermute` and `Expand` operation order for optimization.
For subgraph MemPermute -> Expand, it will be converted as Expand -> MemPermute,
which will be further optimized in later pass with single DMA op.
*/
void registerSwapMemPermuteAndExpandRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                              size_t index, Logger log);
/*
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

The Convolution has weight[2C, C, H+1, 1] with Pad[0, 0, 0, 0] and Strides[1, 1]
The weights values are filled as follows:
                0                 1          ...        C-1
OC0:      [1, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
OC1:      [0, 0,..., 0, 1], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
OC2:      [0, 0,..., 0, 0], [1, 0,..., 0, 0], ...,  [0, 0, ..., 0, 0]
OC3:      [0, 0,..., 0, 0], [0, 0,..., 0, 1], ...,  [0, 0, ..., 0, 0]
...
OC(2C-2): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [1, 0, ..., 0, 0]
OC(2C-1): [0, 0,..., 0, 0], [0, 0,..., 0, 0], ...,  [0, 0, ..., 0, 1]
*/
void registerOptimizeConcatWithConvRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                             size_t index, Logger log);
/*
Adjusts Convolution input shape and kernel shape to get a better performance
*/
void registerAdjustConvolutionShapeRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                             size_t index, Logger log);
/*
Converts inefficient Concat operations on the innermost dimension to more efficient sequences.

For Concat operations, this pass applies transformation as below:

Original pattern:
```
    input1 [N,H,W,1] -> Concat(axis=3) -> [N,H,W,numInputs]
    input2 [N,H,W,1]
    ...
```

Optimized sequence:
1. Converting all inputs to NCHW layout using PermuteCast, so the innermost dimension becomes the channel.
2. Concatenating along the channel axis (axis=1), which is more efficient for memory access and hardware execution.
3. Applying MemPermute to move the concatenated channel back to the innermost position, restoring the original
layout.
*/
void registerOptimizeInnermostConcatRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                              size_t index, Logger log);
/*
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
*/
void registerFuseMemPermuteRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                     size_t index, Logger log);

//
// Canonicalizers
//

void registerReshapeOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index);
void registerMemPermuteOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                   size_t index);
void registerConcatOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels, size_t index);
void registerAffineReshapeOpRewriters(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                      size_t index);
void registerConvertOpRewriters(RewriterRegistry& registry);

//
// OptimizeActiavtions Pipeline
//

void registerSwapMaxpoolWithActivation(RewriterRegistry& registry, ArrayRef<mlir::PatternBenefit> benefitLevels,
                                       size_t index, Logger log);

}  // namespace IE
}  // namespace vpux
