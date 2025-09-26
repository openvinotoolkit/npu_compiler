//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-slice-expand %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @OptimizeSliceExpand
module @OptimizeSliceExpand {

func.func @main(%arg0: tensor<1x80x28x28xf16>) -> tensor<1x80x28x27xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 1] [1, 70, 28, 27] : tensor<1x80x28x28xf16> to tensor<1x70x28x27xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x27xf16> -> tensor<1x80x28x27xf16>
    return %1 : tensor<1x80x28x27xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      tensor<1x80x28x28xf16> to tensor<1x80x28x27xf16>
    // CHECK:       return [[VAR0]] : tensor<1x80x28x27xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCastExpand
module @OptimizeSliceQuantizeCastExpand {

func.func @main(%arg0: tensor<1x80x28x28x!qElemType>) -> tensor<1x80x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %2 : tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.QuantizeCast(%arg0)
    // CHECK-SAME:      tensor<1x80x28x28x!qElemType> -> tensor<1x80x28x28x!qElemType1>
    // CHECK:       return [[VAR0]] : tensor<1x80x28x28x!qElemType1>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCastTwoBranchesExpand
module @OptimizeSliceQuantizeCastTwoBranchesExpand {

func.func @main(%arg0: tensor<1x80x28x28x!qElemType>) -> (tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType, {order = #NHWC}>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType1>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %1, %3 : tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:  [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
    // CHECK-SAME:  {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.QuantizeCast(%arg0)
    // CHECK-SAME:      tensor<1x80x28x28x!qElemType> -> tensor<1x80x28x28x!qElemType1>
    // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>
}

}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCast4ChannelExpand
module @OptimizeSliceQuantizeCast4ChannelExpand {

func.func @main(%arg0: tensor<1x16x28x28x!qElemType>) -> tensor<1x4x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 28, 28] : tensor<1x16x28x28x!qElemType> to tensor<1x1x28x28x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x1x28x28x!qElemType> -> tensor<1x1x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x1x28x28x!qElemType1> -> tensor<1x4x28x28x!qElemType1>
    return %2 : tensor<1x4x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
    // CHECK-SAME:      tensor<1x16x28x28x!qElemType> to tensor<1x4x28x28x!qElemType>
    // CHECK:       [[VAR1:%.+]] = IE.QuantizeCast([[VAR0]])
    // CHECK-SAME:      tensor<1x4x28x28x!qElemType> -> tensor<1x4x28x28x!qElemType1>
    // CHECK:       return [[VAR1]] : tensor<1x4x28x28x!qElemType1>
}

}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @DoNotOptimizeSliceQuantizeCastExpandUserCanUseAutopad
module @DoNotOptimizeSliceQuantizeCastExpandUserCanUseAutopad {
config.PipelineOptions @Options {
   config.Option @VPU.AutoPaddingIDU : true
}
// CHECK: ([[INPUT:%.+]]: tensor<1x16x64x64x!qElemType>)
func.func @main(%arg0: tensor<1x16x64x64x!qElemType>) -> tensor<1x16x64x64x!qElemType1> {
    %filter = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %slice = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x16x64x64x!qElemType> to tensor<1x3x64x64x!qElemType>
    %cast = IE.QuantizeCast(%slice) {dstElemType = !qElemType1} : tensor<1x3x64x64x!qElemType> -> tensor<1x3x64x64x!qElemType1>
    %expand = IE.Expand(%cast) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64x!qElemType1> -> tensor<1x16x64x64x!qElemType1>
    %conv = IE.Convolution(%expand, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64x!qElemType1>, tensor<16x16x1x1xf16> -> tensor<1x16x64x64x!qElemType1>
    return %conv : tensor<1x16x64x64x!qElemType1>

    // CHECK: [[FILTER:%.+]] = const.Declare tensor<16x16x1x1xf16>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 64, 64]
    // CHECK: [[CAST:%.+]] = IE.QuantizeCast([[SLICE]]) {dstElemType = !qElemType1}
    // CHECK: [[EXPAND:%.+]] = IE.Expand([[CAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[FILTER]])
    // CHECK: return [[CONV]]
}

}

// -----

// CHECK-LABEL: @OptimizeSliceConcatExpand
module @OptimizeSliceConcatExpand {

func.func @main(%arg0: tensor<1x80x4x4xf16>, %arg1: tensor<1x80x4x24xf16>) -> tensor<1x80x4x28xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x80x4x4xf16> to tensor<1x70x4x4xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x70x4x4xf16>, tensor<1x70x4x24xf16> -> tensor<1x70x4x28xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x28xf16> -> tensor<1x80x4x28xf16>
   return %3 : tensor<1x80x4x28xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Concat(%arg0, %arg1)
   // CHECK-SAME:      tensor<1x80x4x4xf16>, tensor<1x80x4x24xf16> -> tensor<1x80x4x28xf16>
   // CHECK:       return [[VAR0]] : tensor<1x80x4x28xf16>

}
}

// -----

// CHECK-LABEL: @NotOptimizeSliceConcatExpand
module @NotOptimizeSliceConcatExpand {

func.func @main(%arg0: tensor<1x80x4x4xf16>, %arg1: tensor<1x70x4x24xf16>) -> tensor<1x80x4x28xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x80x4x4xf16> to tensor<1x70x4x4xf16>
   %2 = IE.Concat(%0, %arg1) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x70x4x4xf16>, tensor<1x70x4x24xf16> -> tensor<1x70x4x28xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x28xf16> -> tensor<1x80x4x28xf16>
   return %3 : tensor<1x80x4x28xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x70x4x4xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x4x4xf16>, tensor<1x70x4x24xf16> -> tensor<1x70x4x28xf16>
   // CHECK:       IE.Expand
   // CHECK-SAME:      tensor<1x70x4x28xf16> -> tensor<1x80x4x28xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatExpand
module @NoOptimizeSliceConcatExpand {

func.func @main(%arg0: tensor<1x80x4x24xf16>, %arg1: tensor<1x80x4x24xf16>) -> tensor<1x144x4x24xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x70x4x24xf16>, tensor<1x70x4x24xf16> -> tensor<1x140x4x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x140x4x24xf16> -> tensor<1x144x4x24xf16>
   return %3 : tensor<1x144x4x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x4x24xf16>, tensor<1x70x4x24xf16> -> tensor<1x140x4x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x140x4x24xf16> -> tensor<1x144x4x24xf16>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceTwoConcatsExpand
module @OptimizeSliceTwoConcatsExpand {

func.func @main(%arg0: tensor<1x16x128x200xf16, {order = #NHWC}>) -> tensor<1x16x130x202xf16, {order = #NHWC}> {
   %cst_0 = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>]
   %cst_1 = const.Declare tensor<1x1x128x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x128x1xf16>, [#const.Reorder<#NHWC>]

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   %1 = IE.Concat(%cst_1, %0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]} : tensor<1x1x128x1xf16, {order = #NHWC}>, tensor<1x1x128x200xf16, {order = #NHWC}>, tensor<1x1x128x1xf16, {order = #NHWC}> -> tensor<1x1x128x202xf16, {order = #NHWC}>

   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]} : tensor<1x1x1x202xf16, {order = #NHWC}>, tensor<1x1x128x202xf16, {order = #NHWC}>, tensor<1x1x1x202xf16, {order = #NHWC}> -> tensor<1x1x130x202xf16, {order = #NHWC}>

   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   return %3 : tensor<1x16x130x202xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x16x128x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x128x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x16x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], %arg0, [[CST_0]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]}
   // CHECK-SAME:      : tensor<1x16x128x1xf16, {order = #NHWC}>, tensor<1x16x128x200xf16, {order = #NHWC}>, tensor<1x16x128x1xf16, {order = #NHWC}> -> tensor<1x16x128x202xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CONCAT_0]], [[CST_1]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]}
   // CHECK-SAME:      : tensor<1x16x1x202xf16, {order = #NHWC}>, tensor<1x16x128x202xf16, {order = #NHWC}>, tensor<1x16x1x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   // CHECK:       return [[CONCAT_1]] : tensor<1x16x130x202xf16, {order = #NHWC}>

}
}

// -----

// CHECK-LABEL: @OptimizeSliceTwoConcatsExpandForSliceAxisNotInLastMemDim
module @OptimizeSliceTwoConcatsExpandForSliceAxisNotInLastMemDim {
// CHECK-LABEL: @main
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x128x200xf16>
func.func @main(%arg0: tensor<1x16x128x200xf16>) -> tensor<1x16x130x202xf16> {
   %cst_0 = const.Declare tensor<1x1x1x202xf16> = dense<0.000000e+00> : tensor<1x1x1x202xf16>
   %cst_1 = const.Declare tensor<1x1x128x1xf16> = dense<0.000000e+00> : tensor<1x1x128x1xf16>

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200] : tensor<1x16x128x200xf16> to tensor<1x1x128x200xf16>
   %1 = IE.Concat(%cst_1, %0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]} : tensor<1x1x128x1xf16>, tensor<1x1x128x200xf16>, tensor<1x1x128x1xf16> -> tensor<1x1x128x202xf16>

   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]} : tensor<1x1x1x202xf16>, tensor<1x1x128x202xf16>, tensor<1x1x1x202xf16> -> tensor<1x1x130x202xf16>

   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16> -> tensor<1x16x130x202xf16>
   return %3 : tensor<1x16x130x202xf16>

   // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x16x128x1xf16> = dense<0.000000e+00> : tensor<1x1x128x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x16x1x202xf16> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], [[INPUT]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]}
   // CHECK-SAME:      : tensor<1x16x128x1xf16>, tensor<1x16x128x200xf16>, tensor<1x16x128x1xf16> -> tensor<1x16x128x202xf16>
   // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CONCAT_0]], [[CST_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]}
   // CHECK-SAME:      : tensor<1x16x1x202xf16>, tensor<1x16x128x202xf16>, tensor<1x16x1x202xf16> -> tensor<1x16x130x202xf16>
   // CHECK:       return [[CONCAT_1]] : tensor<1x16x130x202xf16>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceTwoConcatsExpandWithConstBroadcastAttribute
module @OptimizeSliceTwoConcatsExpandWithConstBroadcastAttribute {

func.func @main(%arg0: tensor<1x16x128x200xf16, {order = #NHWC}>) -> tensor<1x16x130x202xf16, {order = #NHWC}> {
   %cst_0 = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<3 : i64, 202 : i64>, #const.Reorder<#NHWC>]
   %cst_1 = const.Declare tensor<1x1x128x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<2 : i64, 128 : i64>, #const.Reorder<#NHWC>]

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   %1 = IE.Concat(%cst_1, %0, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]} : tensor<1x1x128x1xf16, {order = #NHWC}>, tensor<1x1x128x200xf16, {order = #NHWC}>, tensor<1x1x128x1xf16, {order = #NHWC}> -> tensor<1x1x128x202xf16, {order = #NHWC}>

   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]} : tensor<1x1x1x202xf16, {order = #NHWC}>, tensor<1x1x128x202xf16, {order = #NHWC}>, tensor<1x1x1x202xf16, {order = #NHWC}> -> tensor<1x1x130x202xf16, {order = #NHWC}>

   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   return %3 : tensor<1x16x130x202xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x16x128x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<2 : i64, 128 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x16x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<3 : i64, 202 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], %arg0, [[CST_0]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]}
   // CHECK-SAME:      : tensor<1x16x128x1xf16, {order = #NHWC}>, tensor<1x16x128x200xf16, {order = #NHWC}>, tensor<1x16x128x1xf16, {order = #NHWC}> -> tensor<1x16x128x202xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST_1]], [[CONCAT_0]], [[CST_1]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]}
   // CHECK-SAME:      : tensor<1x16x1x202xf16, {order = #NHWC}>, tensor<1x16x128x202xf16, {order = #NHWC}>, tensor<1x16x1x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   // CHECK:       return [[CONCAT_1]] : tensor<1x16x130x202xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceTwoConcatsExpand
module @NotOptimizeSliceTwoConcatsExpand {

func.func @main(%arg0: tensor<1x16x128x200xf16, {order = #NHWC}>) -> tensor<1x16x130x202xf16, {order = #NHWC}> {

   %cst_0 = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>]

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   %4 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 1] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x1xf16, {order = #NHWC}>

   %1 = IE.Concat(%4, %0, %4) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]} : tensor<1x1x128x1xf16, {order = #NHWC}>, tensor<1x1x128x200xf16, {order = #NHWC}>, tensor<1x1x128x1xf16, {order = #NHWC}> -> tensor<1x1x128x202xf16, {order = #NHWC}>

   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]} : tensor<1x1x1x202xf16, {order = #NHWC}>, tensor<1x1x128x202xf16, {order = #NHWC}>, tensor<1x1x1x202xf16, {order = #NHWC}> -> tensor<1x1x130x202xf16, {order = #NHWC}>

   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   return %3 : tensor<1x16x130x202xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>]
   // CHECK:       [[SLICE_0:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200]
   // CHECK-SAME:      : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   // CHECK:       [[SLICE_1:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 1]
   // CHECK-SAME:      : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x1xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[SLICE_1]], [[SLICE_0]], [[SLICE_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]}
   // CHECK-SAME:      : tensor<1x1x128x1xf16, {order = #NHWC}>, tensor<1x1x128x200xf16, {order = #NHWC}>, tensor<1x1x128x1xf16, {order = #NHWC}> -> tensor<1x1x128x202xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST]], [[CONCAT_0]], [[CST]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]}
   // CHECK-SAME:      : tensor<1x1x1x202xf16, {order = #NHWC}>, tensor<1x1x128x202xf16, {order = #NHWC}>, tensor<1x1x1x202xf16, {order = #NHWC}> -> tensor<1x1x130x202xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   // CHECK:       return [[EXPAND]] : tensor<1x16x130x202xf16, {order = #NHWC}>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatAxisHExpand
module @NoOptimizeSliceConcatAxisHExpand {

func.func @main(%arg0: tensor<1x70x20x24xf16>, %arg1: tensor<1x70x20x24xf16>) -> tensor<1x80x20x24xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x20x24xf16> -> tensor<1x80x20x24xf16>
   return %3 : tensor<1x80x20x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x70x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x70x20x24xf16> -> tensor<1x80x20x24xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceConcatAxisHExpand2
module @NoOptimizeSliceConcatAxisHExpand2 {

func.func @main(%arg0: tensor<1x80x20x24xf16>, %arg1: tensor<1x80x20x24xf16>) -> tensor<1x70x30x24xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 10, 24] : tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 10, 0]} : tensor<1x70x20x24xf16> -> tensor<1x70x30x24xf16>
   return %3 : tensor<1x70x30x24xf16>

   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Slice
   // CHECK-SAME:      tensor<1x80x20x24xf16> to tensor<1x70x10x24xf16>
   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x70x10x24xf16>, tensor<1x70x10x24xf16> -> tensor<1x70x20x24xf16>
   // CHECK-NEXT:  IE.Expand
   // CHECK-SAME:      tensor<1x70x20x24xf16> -> tensor<1x70x30x24xf16>

}
}

// -----

// CHECK-LABEL: @NoOptimizeSliceExpand
module @NoOptimizeSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x3x4xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1 : tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x80x3x4xf16>

}
}

// -----

// CHECK-LABEL: @NotOptimizeSliceExpandDueToOffset
module @NotOptimizeSliceExpandDueToOffset {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x20x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 60, 0, 0] [1, 10, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   return %1 : tensor<1x20x4x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x20x4x4xf16>
}
}

// -----

// CHECK-LABEL: @DeleteSliceExpand
module @DeleteSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK-NOT:   IE.Slice
   // CHECK:       [[VAR0:%.+]] = IE.Expand(%arg0)
   // CHECK-SAME:      tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[VAR0]] : tensor<1x80x4x4xf16>

}
}

// -----

// CHECK-LABEL: @NoSliceExpand
module @NoSliceExpand {

func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK:       [[SLICE:%.*]] = IE.Slice %arg0 [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   // CHECK:       [[EXPAND:%.*]] = IE.Expand([[SLICE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[EXPAND]] : tensor<1x80x4x4xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeTwoBranchesSliceExpand
module @OptimizeTwoBranchesSliceExpand {

func.func @main(%arg0: tensor<1x80x4x4xf16>) -> (tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>) {


   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16>
   %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1, %2 : tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   // CHECK:       [[VAR2:%.+]] = IE.Slice %arg0
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x80x3x4xf16>
   // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>
}
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePattern
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePattern(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x3x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   return %1 : tensor<1x3x32x32xf16>

   // CHECK-NOT:    IE.Expand
   // CHECK-NOT:    IE.Slice
   // CHECK:        return [[INPUT]] : tensor<1x3x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternUnsupportedOffset
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePatternUnsupportedOffset(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x3x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   return %1 : tensor<1x3x32x32xf16>

   // Nothing should be changed
   // The input data of the Expand and the output data of the Slice are different because of the offset
   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[EXPAND]] [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   // CHECK:        [[SLICE]] : tensor<1x3x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternToExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @OptimizeExpandSlicePatternToExpand(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x4x32x32xf16>
   // CHECK-NOT:    IE.Slice
   // CHECK:        [[EXPAND]] : tensor<1x4x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternToSlice
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x5x32x32xf16>
func.func @OptimizeExpandSlicePatternToSlice(%arg0: tensor<1x5x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK-NOT:    IE.Expand
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x5x32x32xf16> to tensor<1x4x32x32xf16>
   // CHECK:        [[SLICE]] : tensor<1x4x32x32xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSliceWithIterationTimeLargerThan10
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x12x32x32xf16>
func.func @OptimizeExpandSliceWithIterationTimeLargerThan10(%arg0: tensor<1x12x32x32xf16>) -> tensor<1x12x32x32xf16> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x32x32xf16> -> tensor<1x16x32x32xf16>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %2 = IE.Slice %0 [0, 3, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %3 = IE.Slice %0 [0, 6, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %4 = IE.Slice %0 [0, 9, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %5 = IE.Slice %0 [0, 1, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %6 = IE.Slice %0 [0, 4, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %7 = IE.Slice %0 [0, 7, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %8 = IE.Slice %0 [0, 10, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %9 = IE.Slice %0 [0, 2, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %10 = IE.Slice %0 [0, 5, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %11 = IE.Slice %0 [0, 8, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %12 = IE.Slice %0 [0, 11, 0, 0] [1, 1, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x1x32x32xf16>
   %13 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) {per_axis = #IE.Concat<axis = 1 : i64>} :
            tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>,
            tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>,
            tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16>, tensor<1x1x32x32xf16> -> tensor<1x12x32x32xf16>

   return %13 : tensor<1x12x32x32xf16>

   // CHECK-NOT:    IE.Expand
   // CHECK:        [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 11, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 8, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 5, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE3:%.+]] = IE.Slice [[INPUT]] [0, 2, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE4:%.+]] = IE.Slice [[INPUT]] [0, 10, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE5:%.+]] = IE.Slice [[INPUT]] [0, 7, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE6:%.+]] = IE.Slice [[INPUT]] [0, 4, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE7:%.+]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE8:%.+]] = IE.Slice [[INPUT]] [0, 9, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE9:%.+]] = IE.Slice [[INPUT]] [0, 6, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE10:%.+]] = IE.Slice [[INPUT]] [0, 3, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[SLICE11:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 32, 32]
   // CHECK:        [[CONCAT:%.+]] = IE.Concat([[SLICE11]], [[SLICE10]], [[SLICE9]], [[SLICE8]], [[SLICE7]], [[SLICE6]],
   // CHECK:                                    [[SLICE5]], [[SLICE4]], [[SLICE3]], [[SLICE2]], [[SLICE1]], [[SLICE0]])

   // CHECK:        [[CONCAT]] : tensor<1x12x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceHSwishExpand
module @OptimizeSliceHSwishExpand {

func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.HSwish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.HSwish(%arg0)
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSwishExpand
module @OptimizeSliceSwishExpand {

func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.Swish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.Swish(%arg0)
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceGeluExpand
module @OptimizeSliceGeluExpand {

func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.Gelu(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.Gelu(%arg0)
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSigmoidShapeCastExpand
module @OptimizeSliceSigmoidShapeCastExpand {

func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid(%arg0)
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [16, 1, 1, 1]}
   // CHECK-SAME:     inputs([[SIGMOID]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[SHAPECAST]] : tensor<16x1x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceSigmoidShapeCastExpandDueToExpandOnAnotherAxis
module @NotOptimizeSliceSigmoidShapeCastExpandDueToExpandOnAnotherAxis {

func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<8x16x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<8x16x1x1xf16, {order = #NHWC}>
   return %3 : tensor<8x16x1x1xf16, {order = #NHWC}>

   // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[SLICE]])
   // CHECK-SAME:      tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [8, 1, 1, 1]}
   // CHECK-SAME:     inputs([[SIGMOID]] : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}>
   // CHECK-SAME:                   -> tensor<8x16x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[EXPAND]] : tensor<8x16x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceSigmoidShapeCastExpandDueToShapeCast2Dims
module @NotOptimizeSliceSigmoidShapeCastExpandDueToShapeCast2Dims {

func.func @main(%arg0: tensor<1x16x3x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 3, 1] : tensor<1x16x3x1xf16, {order = #NHWC}> to tensor<1x8x3x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x3x1xf16, {order = #NHWC}> -> tensor<1x8x3x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [1, 24, 1, 1]} inputs(%1 : tensor<1x8x3x1xf16, {order = #NHWC}>) -> tensor<1x24x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>
   return %3 : tensor<1x32x1x1xf16, {order = #NHWC}>

   // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 3, 1] : tensor<1x16x3x1xf16, {order = #NHWC}> to tensor<1x8x3x1xf16, {order = #NHWC}>
   // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[SLICE]])
   // CHECK-SAME:      tensor<1x8x3x1xf16, {order = #NHWC}> -> tensor<1x8x3x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 24, 1, 1]}
   // CHECK-SAME:     inputs([[SIGMOID]] : tensor<1x8x3x1xf16, {order = #NHWC}>) -> tensor<1x24x1x1xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SHAPECAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x1x1xf16, {order = #NHWC}>
   // CHECK-SAME:                   -> tensor<1x32x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[EXPAND]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSwishShapeCastExpand
module @OptimizeSliceSwishShapeCastExpand {

func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Swish(%0) {beta_value = 1.0} : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[SWISH:%.+]] = IE.Swish(%arg0)
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [16, 1, 1, 1]}
   // CHECK-SAME:     inputs([[SWISH]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[SHAPECAST]] : tensor<16x1x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceHSwishShapeCastExpand
module @OptimizeSliceHSwishShapeCastExpand {

func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.HSwish(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[HSWISH:%.+]] = IE.HSwish(%arg0)
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [16, 1, 1, 1]}
   // CHECK-SAME:     inputs([[HSWISH]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[SHAPECAST]] : tensor<16x1x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceGeluShapeCastExpand
module @OptimizeSliceGeluShapeCastExpand {

func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Gelu(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[GELU:%.+]] = IE.Gelu(%arg0)
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [16, 1, 1, 1]}
   // CHECK-SAME:     inputs([[GELU]] : tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[SHAPECAST]] : tensor<16x1x1x1xf16, {order = #NHWC}>
}
}

// -----

// CHECK-LABEL: @OptimizeExpandOverSameDimWithSingleSlice
module @OptimizeExpandOverSameDimWithSingleSlice {

func.func @main(%arg0: tensor<1x96x180x320xf16>, %arg1: tensor<1x96x180x320xf16>) -> tensor<1x192x180x320xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 84, 180, 320] : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   %1 = IE.Concat(%arg1, %0) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   return %2 : tensor<1x192x180x320xf16>

   // CHECK:       IE.Concat
   // CHECK-SAME:      tensor<1x96x180x320xf16>, tensor<1x96x180x320xf16> -> tensor<1x192x180x320xf16>

}
}

// -----

// CHECK-LABEL: @NotOptimizeExpandOverSameDimWithSingleSlice
module @NotOptimizeExpandOverSameDimWithSingleSlice  {

func.func @main(%arg0: tensor<1x96x180x320xf16>, %arg1: tensor<1x96x180x320xf16>) -> tensor<1x192x180x320xf16> {

   %0 = IE.Slice %arg0 [0, 12, 0, 0] [1, 84, 180, 320] : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   %1 = IE.Concat(%arg1, %0) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   return %2 : tensor<1x192x180x320xf16>

   // CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 12, 0, 0] [1, 84, 180, 320]
   // CHECK-SAME:      : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   // CHECK:       [[CONCAT:%.+]] = IE.Concat(%arg1, [[SLICE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 96, 0, 0]]}
   // CHECK-SAME:      : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
   // CHECK-SAME:      : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   // CHECK:       return [[EXPAND]] : tensor<1x192x180x320xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceExpandDueToAddInput
module @NotOptimizeSliceExpandDueToAddInput {

func.func @main(%arg0: tensor<1x12x64x64xf16, {order = #NHWC}>, %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>) -> tensor<1x16x64x64xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>

   return %3 : tensor<1x16x64x64xf16, {order = #NHWC}>

   // CHECK:       [[SLICE0:%.+]]  = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND0]], [[EXPAND1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]] : tensor<1x16x64x64xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceExpandIfCannotShapeCastForAdd
module @FuseSliceExpandIfCannotShapeCastForAdd {

func.func @main(%arg0: tensor<1x12x11x11xf16, {order = #NHWC}>, %arg1: tensor<1x3x11x11xf16, {order = #NHWC}>) -> tensor<1x16x11x11xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 11, 11] : tensor<1x12x11x11xf16, {order = #NHWC}> to tensor<1x3x11x11xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x11x11xf16, {order = #NHWC}>, tensor<1x16x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>

   return %3 : tensor<1x16x11x11xf16, {order = #NHWC}>

   // CHECK:       [[EXPAND0:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND0]], [[EXPAND1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x11x11xf16, {order = #NHWC}>, tensor<1x16x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]] : tensor<1x16x11x11xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceExpandForEltwisIfExpandHasMultiUsers
module @OptimizeSliceExpandForEltwisIfExpandHasMultiUsers {

func.func @main(%arg0: tensor<1x12x64x64xf16, {order = #NHWC}>, %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>) -> (tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x1x64x64xf16, {order = #NHWC}>)    {
   %filter = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %4 = IE.Convolution(%1, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>

   return %3, %4 : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x1x64x64xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]
   // CHECK:       [[EXPAND0:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND0]], [[EXPAND1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]], [[CONV]] : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x1x64x64xf16, {order = #NHWC}>

}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.013957817414227655:161>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeNumeberSliceOpsTest
module @LargeNumeberSliceOpsTest {

func.func @main(%arg0: tensor<1x25x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x100x56x56x!qElemType, {order = #NHWC}> {
   %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 7, 0, 0]} : tensor<1x25x56x56x!qElemType, {order = #NHWC}> -> tensor<1x32x56x56x!qElemType, {order = #NHWC}>
   %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %2 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %3 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %4 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %5 = IE.Slice %0 [0, 1, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %6 = IE.Slice %0 [0, 2, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %7 = IE.Slice %0 [0, 3, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %8 = IE.Slice %0 [0, 4, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %9 = IE.Slice %0 [0, 5, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %10 = IE.Slice %0 [0, 6, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %11 = IE.Slice %0 [0, 7, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %12 = IE.Slice %0 [0, 8, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %13 = IE.Slice %0 [0, 9, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %14 = IE.Slice %0 [0, 10, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %15 = IE.Slice %0 [0, 11, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %16 = IE.Slice %0 [0, 12, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %17 = IE.Slice %0 [0, 13, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %18 = IE.Slice %0 [0, 14, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %19 = IE.Slice %0 [0, 15, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %20 = IE.Slice %0 [0, 16, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %21 = IE.Slice %0 [0, 17, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %22 = IE.Slice %0 [0, 18, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %23 = IE.Slice %0 [0, 19, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %24 = IE.Slice %0 [0, 20, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %25 = IE.Slice %0 [0, 21, 0, 0] [1, 4, 56, 56] : tensor<1x32x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   %26 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}> -> tensor<1x100x56x56x!qElemType, {order = #NHWC}>

   return %26 : tensor<1x100x56x56x!qElemType, {order = #NHWC}>

   // CHECK:       [[SLICE0:%.+]]  = IE.Slice %arg0 [0, 21, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE1:%.+]]  = IE.Slice %arg0 [0, 20, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE2:%.+]]  = IE.Slice %arg0 [0, 19, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE3:%.+]]  = IE.Slice %arg0 [0, 18, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE4:%.+]]  = IE.Slice %arg0 [0, 17, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE5:%.+]]  = IE.Slice %arg0 [0, 16, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE6:%.+]]  = IE.Slice %arg0 [0, 15, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE7:%.+]]  = IE.Slice %arg0 [0, 14, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE8:%.+]]  = IE.Slice %arg0 [0, 13, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE9:%.+]]  = IE.Slice %arg0 [0, 12, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE10:%.+]]  = IE.Slice %arg0 [0, 11, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE11:%.+]]  = IE.Slice %arg0 [0, 10, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE12:%.+]]  = IE.Slice %arg0 [0, 9, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE13:%.+]]  = IE.Slice %arg0 [0, 8, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE14:%.+]]  = IE.Slice %arg0 [0, 7, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE15:%.+]]  = IE.Slice %arg0 [0, 6, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE16:%.+]]  = IE.Slice %arg0 [0, 5, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE17:%.+]]  = IE.Slice %arg0 [0, 4, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE18:%.+]]  = IE.Slice %arg0 [0, 3, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE19:%.+]]  = IE.Slice %arg0 [0, 2, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE20:%.+]]  = IE.Slice %arg0 [0, 1, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE21:%.+]]  = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE22:%.+]]  = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE23:%.+]]  = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE24:%.+]]  = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>

   // CHECK:       [[CONCAT:%.+]] = IE.Concat([[SLICE24]], [[SLICE23]], [[SLICE22]], [[SLICE21]], [[SLICE20]], [[SLICE19]], [[SLICE18]], [[SLICE17]], [[SLICE16]], [[SLICE15]], [[SLICE14]], [[SLICE13]], [[SLICE12]], [[SLICE11]], [[SLICE10]], [[SLICE9]], [[SLICE8]], [[SLICE7]], [[SLICE6]], [[SLICE5]], [[SLICE4]], [[SLICE3]], [[SLICE2]], [[SLICE1]], [[SLICE0]])
   // CHECK-SAME:      : tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}>, tensor<1x4x56x56x!qElemType, {order = #NHWC}> -> tensor<1x100x56x56x!qElemType, {order = #NHWC}>

   // CHECK:       return [[CONCAT]] : tensor<1x100x56x56x!qElemType, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSlicePReluExpandWithConstInput
module @OptimizeSlicePReluExpandWithConstInput {
// CHECK-LABEL: @main
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x482x642xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x482x642xf16, {order = #NHWC}>) -> tensor<1x16x482x642xf16, {order = #NHWC}> {
   %cst = const.Declare tensor<1x5x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>]
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 5, 482, 642] : tensor<1x16x482x642xf16, {order = #NHWC}> to tensor<1x5x482x642xf16, {order = #NHWC}>
   %1 = IE.PRelu(%0, %cst) : tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x1x1xf16, {order = #NHWC}> -> tensor<1x5x482x642xf16, {order = #NHWC}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>

   return %2 : tensor<1x16x482x642xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
   // CHECK:       [[PRELU:%.+]] = IE.PRelu([[INPUT]], [[CST]])
   // CHECK-SAME:      tensor<1x16x482x642xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>
   // CHECK:       return [[PRELU]] : tensor<1x16x482x642xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSlicePReluExpand
module @OptimizeSlicePReluExpand {
// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT_0:%.+]]: tensor<1x16x482x642xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x16x482x642xf16, {order = #NHWC}>, %arg1: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x482x642xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 5, 482, 642] : tensor<1x16x482x642xf16, {order = #NHWC}> to tensor<1x5x482x642xf16, {order = #NHWC}>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 5, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x5x1x1xf16, {order = #NHWC}>
   %2 = IE.PRelu(%0, %1) : tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x1x1xf16, {order = #NHWC}> -> tensor<1x5x482x642xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>

   return %3 : tensor<1x16x482x642xf16, {order = #NHWC}>

   // CHECK:       [[PRELU:%.+]] = IE.PRelu([[INPUT_0]], [[INPUT_1]])
   // CHECK-SAME:      tensor<1x16x482x642xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>
   // CHECK:       return [[PRELU]] : tensor<1x16x482x642xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSlicePReluExpand
module @NotOptimizeSlicePReluExpand {
// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT_0:%.+]]: tensor<1x16x482x642xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x5x482x642xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x16x482x642xf16, {order = #NHWC}>, %arg1: tensor<1x5x482x642xf16, {order = #NHWC}>) -> tensor<1x16x482x642xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 5, 482, 642] : tensor<1x16x482x642xf16, {order = #NHWC}> to tensor<1x5x482x642xf16, {order = #NHWC}>
   %2 = IE.PRelu(%0, %arg1) : tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x5x482x642xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>

   return %3 : tensor<1x16x482x642xf16, {order = #NHWC}>

   // CHECK:       [[SLICE:%.+]] = IE.Slice [[INPUT_0]]
   // CHECK-SAME:      tensor<1x16x482x642xf16, {order = #NHWC}> to tensor<1x5x482x642xf16, {order = #NHWC}>
   // CHECK:       [[PRELU:%.+]] = IE.PRelu([[SLICE]], [[INPUT_1]])
   // CHECK-SAME:      tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x5x482x642xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[PRELU]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x482x642xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>
   // CHECK:       return [[EXPAND]] : tensor<1x16x482x642xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSlicePReluTwoConcatsExpand
module @OptimizeSlicePReluTwoConcatsExpand {
// CHECK-LABEL: @main
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x482x642xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x482x642xf16, {order = #NHWC}>) -> tensor<1x16x484x644xf16, {order = #NHWC}> {
   %cst = const.Declare tensor<1x5x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>]
   %cst_0 = const.Declare tensor<1x5x482x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x5x482x1xf16>, [#const.Reorder<#NHWC>]
   %cst_1 = const.Declare tensor<1x5x1x644xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x5x1x644xf16>, [#const.Reorder<#NHWC>]

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 5, 482, 642] : tensor<1x16x482x642xf16, {order = #NHWC}> to tensor<1x5x482x642xf16, {order = #NHWC}>
   %1 = IE.PRelu(%0, %cst) : tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x1x1xf16, {order = #NHWC}> -> tensor<1x5x482x642xf16, {order = #NHWC}>
   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 643]]} : tensor<1x5x482x1xf16, {order = #NHWC}>, tensor<1x5x482x642xf16, {order = #NHWC}>, tensor<1x5x482x1xf16, {order = #NHWC}> -> tensor<1x5x482x644xf16, {order = #NHWC}>
   %3 = IE.Concat(%cst_1, %2, %cst_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 483, 0]]} : tensor<1x5x1x644xf16, {order = #NHWC}>, tensor<1x5x482x644xf16, {order = #NHWC}>, tensor<1x5x1x644xf16, {order = #NHWC}> -> tensor<1x5x484x644xf16, {order = #NHWC}>
   %4 = IE.Expand(%3) {pads_begin = [0, 0, 0, 0], pads_end = [0, 11, 0, 0]} : tensor<1x5x484x644xf16, {order = #NHWC}> -> tensor<1x16x484x644xf16, {order = #NHWC}>

   return %4 : tensor<1x16x484x644xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare
   // CHECK-SAME:      tensor<1x16x1x644xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x5x1x644xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
   // CHECK-DAG:   [[CST_0:%.+]] = const.Declare
   // CHECK-SAME:      tensor<1x16x482x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x5x482x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
   // CHECK-DAG:   [[CST_1:%.+]] = const.Declare
   // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
   // CHECK:       [[PRELU:%.+]] = IE.PRelu([[INPUT]], [[CST_1]])
   // CHECK-SAME:      tensor<1x16x482x642xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x482x642xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], [[PRELU]], [[CST_0]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 643]]}
   // CHECK-SAME:      : tensor<1x16x482x1xf16, {order = #NHWC}>, tensor<1x16x482x642xf16, {order = #NHWC}>, tensor<1x16x482x1xf16, {order = #NHWC}> -> tensor<1x16x482x644xf16, {order = #NHWC}>
   // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CST]], [[CONCAT_0]], [[CST]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 483, 0]]}
   // CHECK-SAME:      : tensor<1x16x1x644xf16, {order = #NHWC}>, tensor<1x16x482x644xf16, {order = #NHWC}>, tensor<1x16x1x644xf16, {order = #NHWC}> -> tensor<1x16x484x644xf16, {order = #NHWC}>
   // CHECK:       return [[CONCAT_1]] : tensor<1x16x484x644xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceMultiplyExpand
module @OptimizeSliceMultiplyExpand {
// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT_0:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x96x128x128xf16, {order = #NHWC}>, %arg1: tensor<1x96x128x128xf16, {order = #NHWC}>) -> tensor<1x96x128x128xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x91x128x128xf16, {order = #NHWC}>, tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x91x128x128xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>

   return %3 : tensor<1x96x128x128xf16, {order = #NHWC}>

   // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT_0]], [[INPUT_1]])
   // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<1x96x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
   // CHECK:       return [[MULTIPLY]] : tensor<1x96x128x128xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceAddExpand
module @OptimizeSliceAddExpand {
// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT_0:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x96x128x128xf16, {order = #NHWC}>, %arg1: tensor<1x96x128x128xf16, {order = #NHWC}>) -> tensor<1x96x128x128xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x91x128x128xf16, {order = #NHWC}>, tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x91x128x128xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>

   return %3 : tensor<1x96x128x128xf16, {order = #NHWC}>

   // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]])
   // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<1x96x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]] : tensor<1x96x128x128xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSubtractExpand
module @OptimizeSliceSubtractExpand {
// CHECK-LABEL: @main
// CHECK-SAME: ([[INPUT_0:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>, [[INPUT_1:%.+]]: tensor<1x96x128x128xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x96x128x128xf16, {order = #NHWC}>, %arg1: tensor<1x96x128x128xf16, {order = #NHWC}>) -> tensor<1x96x128x128xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 91, 128, 128] : tensor<1x96x128x128xf16, {order = #NHWC}> to tensor<1x91x128x128xf16, {order = #NHWC}>
   %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x91x128x128xf16, {order = #NHWC}>, tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x91x128x128xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 5, 0, 0]} : tensor<1x91x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>

   return %3 : tensor<1x96x128x128xf16, {order = #NHWC}>

   // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]])
   // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<1x96x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
   // CHECK:       return [[SUBTRACT]] : tensor<1x96x128x128xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceClampExpand
module @OptimizeSliceClampExpand {
// CHECK-LABEL: @main
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
   %1 = IE.Clamp(%0) {min = 1.0, max = 3.0} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   return %2 : tensor<1x16x1x1xf16, {order = #NHWC}>

   // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]])
   // CHECK-SAME:      {max = 3.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
   // CHECK:       return [[CLAMP]]  : tensor<1x16x1x1xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSoftmaxExpand
module @OptimizeSliceSoftmaxExpand {
// CHECK-LABEL:       @fuseSliceSoftmaxExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @fuseSliceSoftmaxExpand(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:       [[OUTPUT:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64, padSize = 3 : i64} :
  // CHECK-SAME:  tensor<1x80x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  // CHECK:       return [[OUTPUT]]
}
}

// -----

// CHECK-LABEL: @fuseSliceSoftmaxExpand_NCHW
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x192x225x240xf16>)
func.func @fuseSliceSoftmaxExpand_NCHW(%arg0: tensor<1x192x225x240xf16>) -> tensor<1x192x225x240xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 192, 225, 225] : tensor<1x192x225x240xf16> to tensor<1x192x225x225xf16>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x192x225x225xf16> -> tensor<1x192x225x225xf16>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]} : tensor<1x192x225x225xf16> -> tensor<1x192x225x240xf16>

    return %2 : tensor<1x192x225x240xf16>

    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64, padSize = 15 : i64} : tensor<1x192x225x240xf16> -> tensor<1x192x225x240xf16>
    // CHECK: return [[SOFTMAX]] : tensor<1x192x225x240xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpand
module @notOptimizeSliceSoftmaxExpand {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x81x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpand(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x81x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x81x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x81x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 1 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpandAxisNotC
module @notOptimizeSliceSoftmaxExpandAxisNotC {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpandAxisNotC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpandAxisNotC(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 2 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 2 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpandWithOffsetOfSliceNotAllZero
module @notOptimizeSliceSoftmaxExpandWithOffsetOfSliceNotAllZero {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpandWithOffsetOfSliceNotAllZero
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpandWithOffsetOfSliceNotAllZero(%arg0: tensor<1x80x8x16xf16, {order = #NHWC}>) -> tensor<1x80x8x16xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 3, 0, 0] [1, 77, 8, 16] : tensor<1x80x8x16xf16, {order = #NHWC}> to tensor<1x77x8x16xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x77x8x16xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x77x8x16xf16, {order = #NHWC}> -> tensor<1x80x8x16xf16, {order = #NHWC}>
  return %3 : tensor<1x80x8x16xf16, {order = #NHWC}>

  // CHECK:    [[CUT_INPUT:%.+]] = IE.Slice [[INPUT]]
  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[CUT_INPUT]]) {axisInd = 1 : i64}
  // CHECK:    [[OUTPUT:%.+]] = IE.Expand([[OUT_SOFTMAX]])
  // CHECK:    return [[OUTPUT]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.956:128, 0.785:128, 0.567:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>
!qElemType2 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.567:128, 0.567:128, 0.567:128}>

// CHECK-LABEL: @OptimizeSliceConcatExpandForQuantizedType
module @OptimizeSliceConcatExpandForQuantizedType {
// CHECK-LABEL:       @OptimizeSliceConcatExpandForQuantizedType
// CHECK-SAME:        [[INPUT1:%arg0]]: tensor<1x6x32x56x!qElemType>,
// CHECK-SAME:        [[INPUT2:%arg1]]: tensor<1x6x32x56x!qElemType>) -> tensor<2x6x32x56x!qElemType>
func.func @OptimizeSliceConcatExpandForQuantizedType(%arg0: tensor<1x6x32x56x!qElemType>, %arg1: tensor<1x6x32x56x!qElemType>) -> tensor<2x6x32x56x!qElemType> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 32, 56] : tensor<1x6x32x56x!qElemType> to tensor<1x3x32x56x!qElemType1>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 32, 56] : tensor<1x6x32x56x!qElemType> to tensor<1x3x32x56x!qElemType1>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x3x32x56x!qElemType1>, tensor<1x3x32x56x!qElemType1> -> tensor<2x3x32x56x!qElemType1>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<2x3x32x56x!qElemType1> -> tensor<2x6x32x56x!qElemType2>
   %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType} : tensor<2x6x32x56x!qElemType2> -> tensor<2x6x32x56x!qElemType>
   return %4 : tensor<2x6x32x56x!qElemType>

   // CHECK:       [[VAR0:%.+]] = IE.Concat([[INPUT1]], [[INPUT2]])
   // CHECK-SAME:      tensor<1x6x32x56x!qElemType>, tensor<1x6x32x56x!qElemType> -> tensor<2x6x32x56x!qElemType>
   // CHECK:       return [[VAR0]] : tensor<2x6x32x56x!qElemType>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeExpandSlicePattern
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @OptimizeExpandSlicePattern(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x3x32x32xf16>
   %1 = IE.LayoutCast(%0) {dst_order = #NHWC} : tensor<1x3x32x32xf16> -> tensor<1x3x32x32xf16, {order = #NHWC}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   return %2 : tensor<1x16x32x32xf16, {order = #NHWC}>

   // CHECK-NOT:    IE.Expand
   // CHECK-NOT:    IE.Slice
   // CHECK:        [[LAYOUTCAST:%.+]] = IE.LayoutCast([[INPUT]])
   // CHECK:        return [[LAYOUTCAST]] : tensor<1x16x32x32xf16, {order = #NHWC}>
}
