//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --optimize-slice-expand %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @OptimizeSliceExpand
module @OptimizeSliceExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x80x28x28xf16>) -> tensor<1x80x28x27xf16>
func.func @main(%arg0: tensor<1x80x28x28xf16>) -> tensor<1x80x28x27xf16> {
    %0 = IE.Slice %arg0 [0, 0, 0, 1] [1, 70, 28, 27] : tensor<1x80x28x28xf16> to tensor<1x70x28x27xf16>
    %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x27xf16> -> tensor<1x80x28x27xf16>
    return %1 : tensor<1x80x28x27xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:      tensor<1x80x28x28xf16> to tensor<1x80x28x27xf16>
    // CHECK:       return [[VAR0]] : tensor<1x80x28x27xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCastExpand
module @OptimizeSliceQuantizeCastExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x80x28x28x!qElemType>) -> tensor<1x80x28x28x!qElemType1>
func.func @main(%arg0: tensor<1x80x28x28x!qElemType>) -> tensor<1x80x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %2 : tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.QuantizeCast([[ARG_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x80x28x28x!qElemType>) -> (tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>)
func.func @main(%arg0: tensor<1x80x28x28x!qElemType>) -> (tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType, {order = #NHWC}>
    %2 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType1>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x28x28x!qElemType1> -> tensor<1x80x28x28x!qElemType1>
    return %1, %3 : tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
    // CHECK-SAME:  [0, 0, 0, 0] [1, 70, 28, 28] : tensor<1x80x28x28x!qElemType> to tensor<1x70x28x28x!qElemType>
    // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
    // CHECK-SAME:  {dstOrder = #NHWC} : tensor<1x70x28x28x!qElemType> -> tensor<1x70x28x28x!qElemType, {order = #NHWC}>
    // CHECK:       [[VAR2:%.+]] = IE.QuantizeCast([[ARG_0]])
    // CHECK-SAME:      tensor<1x80x28x28x!qElemType> -> tensor<1x80x28x28x!qElemType1>
    // CHECK:       return [[VAR1]], [[VAR2]] : tensor<1x70x28x28x!qElemType, {order = #NHWC}>, tensor<1x80x28x28x!qElemType1>
}

}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1445073146446075E-5>
!qElemType1 = !quant.uniform<u8:f16, 1.5722536573223038E-5>

// CHECK-LABEL: @OptimizeSliceQuantizeCast4ChannelExpand
module @OptimizeSliceQuantizeCast4ChannelExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x28x28x!qElemType>) -> tensor<1x4x28x28x!qElemType1>
func.func @main(%arg0: tensor<1x16x28x28x!qElemType>) -> tensor<1x4x28x28x!qElemType1> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 28, 28] : tensor<1x16x28x28x!qElemType> to tensor<1x1x28x28x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x1x28x28x!qElemType> -> tensor<1x1x28x28x!qElemType1>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 3, 0, 0]} : tensor<1x1x28x28x!qElemType1> -> tensor<1x4x28x28x!qElemType1>
    return %2 : tensor<1x4x28x28x!qElemType1>

    // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
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
   config.Option @config.AutoPaddingIDU : true
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x80x4x4xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x80x4x24xf16>
func.func @main(%arg0: tensor<1x80x4x4xf16>, %arg1: tensor<1x80x4x24xf16>) -> tensor<1x80x4x28xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 4, 4] : tensor<1x80x4x4xf16> to tensor<1x70x4x4xf16>
   %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 70, 4, 24] : tensor<1x80x4x24xf16> to tensor<1x70x4x24xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x70x4x4xf16>, tensor<1x70x4x24xf16> -> tensor<1x70x4x28xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x4x28xf16> -> tensor<1x80x4x28xf16>
   return %3 : tensor<1x80x4x28xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x128x200xf16, {order = #NHWC}>
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
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], [[ARG_0]], [[CST_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x128x200xf16, {order = #NHWC}>
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
   // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], [[ARG_0]], [[CST_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x128x200xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x128x200xf16, {order = #NHWC}>) -> tensor<1x16x130x202xf16, {order = #NHWC}> {

   %cst_0 = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>]

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 200] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   %4 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 1] : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x1xf16, {order = #NHWC}>

   %1 = IE.Concat(%4, %0, %4) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 201]]} : tensor<1x1x128x1xf16, {order = #NHWC}>, tensor<1x1x128x200xf16, {order = #NHWC}>, tensor<1x1x128x1xf16, {order = #NHWC}> -> tensor<1x1x128x202xf16, {order = #NHWC}>

   %2 = IE.Concat(%cst_0, %1, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 129, 0]]} : tensor<1x1x1x202xf16, {order = #NHWC}>, tensor<1x1x128x202xf16, {order = #NHWC}>, tensor<1x1x1x202xf16, {order = #NHWC}> -> tensor<1x1x130x202xf16, {order = #NHWC}>

   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x130x202xf16, {order = #NHWC}> -> tensor<1x16x130x202xf16, {order = #NHWC}>
   return %3 : tensor<1x16x130x202xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x202xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x1x202xf16>, [#const.Reorder<#NHWC>]
   // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 128, 200]
   // CHECK-SAME:      : tensor<1x16x128x200xf16, {order = #NHWC}> to tensor<1x1x128x200xf16, {order = #NHWC}>
   // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 128, 1]
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x70x4x4xf16>
func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x3x4xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1 : tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x80x3x4xf16>

}
}

// -----

// CHECK-LABEL: @NotOptimizeSliceExpandDueToOffset
module @NotOptimizeSliceExpandDueToOffset {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x70x4x4xf16>
func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x20x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 60, 0, 0] [1, 10, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   return %1 : tensor<1x20x4x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
   // CHECK-SAME:      tensor<1x70x4x4xf16> to tensor<1x10x4x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Expand([[VAR0]])
   // CHECK-SAME:      tensor<1x10x4x4xf16> -> tensor<1x20x4x4xf16>
   // CHECK:       return [[VAR1]] : tensor<1x20x4x4xf16>
}
}

// -----

// CHECK-LABEL: @DeleteSliceExpand
module @DeleteSliceExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x70x4x4xf16>
func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK-NOT:   IE.Slice
   // CHECK:       [[VAR0:%.+]] = IE.Expand([[ARG_0]])
   // CHECK-SAME:      tensor<1x70x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[VAR0]] : tensor<1x80x4x4xf16>

}
}

// -----

// CHECK-LABEL: @NoSliceExpand
module @NoSliceExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x70x4x4xf16>
func.func @main(%arg0: tensor<1x70x4x4xf16>) -> tensor<1x80x4x4xf16> {

   %0 = IE.Slice %arg0 [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   return %1 : tensor<1x80x4x4xf16>

   // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 10, 0, 0] [1, 60, 4, 4] : tensor<1x70x4x4xf16> to tensor<1x60x4x4xf16>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[SLICE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 20, 0, 0]} : tensor<1x60x4x4xf16> -> tensor<1x80x4x4xf16>
   // CHECK:       return [[EXPAND]] : tensor<1x80x4x4xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeTwoBranchesSliceExpand
module @OptimizeTwoBranchesSliceExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x80x4x4xf16>
func.func @main(%arg0: tensor<1x80x4x4xf16>) -> (tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>) {


   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 70, 3, 4] : tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16>
   %1 = IE.Reorder(%0) {dstOrder = #NHWC} : tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   %2 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x70x3x4xf16> -> tensor<1x80x3x4xf16>
   return %1, %2 : tensor<1x70x3x4xf16, {order = #NHWC}>, tensor<1x80x3x4xf16>

   // CHECK:       [[VAR0:%.+]] = IE.Slice [[ARG_0]]
   // CHECK-SAME:      tensor<1x80x4x4xf16> to tensor<1x70x3x4xf16>
   // CHECK:       [[VAR1:%.+]] = IE.Reorder([[VAR0]])
   // CHECK-SAME:      tensor<1x70x3x4xf16> -> tensor<1x70x3x4xf16, {order = #NHWC}>
   // CHECK:       [[VAR2:%.+]] = IE.Slice [[ARG_0]]
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x257x257xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.HSwish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.HSwish([[ARG_0]])
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSwishExpand
module @OptimizeSliceSwishExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x257x257xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.Swish(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.Swish([[ARG_0]])
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceGeluExpand
module @OptimizeSliceGeluExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x257x257xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x257x257xf16, {order = #NHWC}>) -> tensor<1x16x257x257xf16, {order = #NHWC}> {
   %3 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 257, 257] : tensor<1x16x257x257xf16, {order = #NHWC}> to tensor<1x8x257x257xf16, {order = #NHWC}>
   %4 = IE.Gelu(%3) : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x8x257x257xf16, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x8x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   return %5 : tensor<1x16x257x257xf16, {order = #NHWC}>

   // CHECK:       [[VAR0:%.+]] = IE.Gelu([[ARG_0]])
   // CHECK-SAME:      tensor<1x16x257x257xf16, {order = #NHWC}> -> tensor<1x16x257x257xf16, {order = #NHWC}>
   // CHECK:       return [[VAR0]] : tensor<1x16x257x257xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceSigmoidShapeCastExpand
module @OptimizeSliceSigmoidShapeCastExpand {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[ARG_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<8x16x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<8x16x1x1xf16, {order = #NHWC}>
   return %3 : tensor<8x16x1x1xf16, {order = #NHWC}>

   // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x3x1xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x16x3x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 3, 1] : tensor<1x16x3x1xf16, {order = #NHWC}> to tensor<1x8x3x1xf16, {order = #NHWC}>
   %1 = IE.Sigmoid(%0) : tensor<1x8x3x1xf16, {order = #NHWC}> -> tensor<1x8x3x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [1, 24, 1, 1]} inputs(%1 : tensor<1x8x3x1xf16, {order = #NHWC}>) -> tensor<1x24x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf16, {order = #NHWC}>
   return %3 : tensor<1x32x1x1xf16, {order = #NHWC}>

   // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 8, 3, 1] : tensor<1x16x3x1xf16, {order = #NHWC}> to tensor<1x8x3x1xf16, {order = #NHWC}>
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Swish(%0) {beta_value = 1.0} : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[SWISH:%.+]] = IE.Swish([[ARG_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.HSwish(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[HSWISH:%.+]] = IE.HSwish([[ARG_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @main(%arg0 : tensor<1x16x1x1xf16, {order = #NHWC}>)->tensor<16x1x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x8x1x1xf16, {order = #NHWC}>
   %1 = IE.Gelu(%0) : tensor<1x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x1x1xf16, {order = #NHWC}>
   %2 = IE.ShapeCast {shape = [8, 1, 1, 1]} inputs(%1 : tensor<1x8x1x1xf16, {order = #NHWC}>) -> tensor<8x1x1x1xf16, {order = #NHWC}>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<8x1x1x1xf16, {order = #NHWC}> -> tensor<16x1x1x1xf16, {order = #NHWC}>
   return %3 : tensor<16x1x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:       [[GELU:%.+]] = IE.Gelu([[ARG_0]])
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x96x180x320xf16>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x96x180x320xf16>
func.func @main(%arg0: tensor<1x96x180x320xf16>, %arg1: tensor<1x96x180x320xf16>) -> tensor<1x192x180x320xf16> {

   %0 = IE.Slice %arg0 [0, 12, 0, 0] [1, 84, 180, 320] : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   %1 = IE.Concat(%arg1, %0) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   return %2 : tensor<1x192x180x320xf16>

   // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 12, 0, 0] [1, 84, 180, 320]
   // CHECK-SAME:      : tensor<1x96x180x320xf16> to tensor<1x84x180x320xf16>
   // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_1]], [[SLICE]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 96, 0, 0]]}
   // CHECK-SAME:      : tensor<1x96x180x320xf16>, tensor<1x84x180x320xf16> -> tensor<1x180x180x320xf16>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
   // CHECK-SAME:      : tensor<1x180x180x320xf16> -> tensor<1x192x180x320xf16>
   // CHECK:       return [[EXPAND]] : tensor<1x192x180x320xf16>
}
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeRedundantSliceExpandWithEltwiseUser
module @OptimizeRedundantSliceExpandWithEltwiseUser {

// CHECK: func.func @main([[INPUT1:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
func.func @main(%input1: tensor<1x32x64x64xf16, {order = #NHWC}>, %input2: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
   %slice = IE.Slice %input1 [0, 0, 0, 0] [1, 24, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x24x64x64xf16, {order = #NHWC}>
   %expand = IE.Expand(%slice) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x24x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
   %eltwise = IE.Add(%expand, %input2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
   return %eltwise : tensor<1x32x64x64xf16, {order = #NHWC}>

   // CHECK-NOT:  IE.Slice
   // CHECK-NOT:  IE.Expand
   // CHECK:      [[ADD:%.+]] = IE.Add([[INPUT1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>
   // CHECK:      return [[ADD]]

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceExpandDueToAddInput
module @NotOptimizeSliceExpandDueToAddInput {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x12x64x64xf16, {order = #NHWC}>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x3x64x64xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x12x64x64xf16, {order = #NHWC}>, %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>) -> tensor<1x16x64x64xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>

   return %3 : tensor<1x16x64x64xf16, {order = #NHWC}>

   // CHECK:       [[SLICE0:%.+]]  = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND0:%.+]] = IE.Expand([[SLICE0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand([[ARG_1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND0]], [[EXPAND1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]] : tensor<1x16x64x64xf16, {order = #NHWC}>

}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceExpandIfCannotShapeCastForAdd
module @FuseSliceExpandIfCannotShapeCastForAdd {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x12x11x11xf16, {order = #NHWC}>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x3x11x11xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x12x11x11xf16, {order = #NHWC}>, %arg1: tensor<1x3x11x11xf16, {order = #NHWC}>) -> tensor<1x16x11x11xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 11, 11] : tensor<1x12x11x11xf16, {order = #NHWC}> to tensor<1x3x11x11xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x11x11xf16, {order = #NHWC}>, tensor<1x16x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>

   return %3 : tensor<1x16x11x11xf16, {order = #NHWC}>

   // CHECK:       [[EXPAND0:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand([[ARG_1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND0]], [[EXPAND1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x11x11xf16, {order = #NHWC}>, tensor<1x16x11x11xf16, {order = #NHWC}> -> tensor<1x16x11x11xf16, {order = #NHWC}>
   // CHECK:       return [[ADD]] : tensor<1x16x11x11xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceExpandForEltwisIfExpandHasMultiUsers
module @OptimizeSliceExpandForEltwisIfExpandHasMultiUsers {

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x12x64x64xf16, {order = #NHWC}>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: tensor<1x3x64x64xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x12x64x64xf16, {order = #NHWC}>, %arg1: tensor<1x3x64x64xf16, {order = #NHWC}>) -> (tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x1x64x64xf16, {order = #NHWC}>)    {
   %filter = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 64, 64] : tensor<1x12x64x64xf16, {order = #NHWC}> to tensor<1x3x64x64xf16, {order = #NHWC}>
   %1 = IE.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %2 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   %4 = IE.Convolution(%1, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x1x64x64xf16, {order = #NHWC}>

   return %3, %4 : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x1x64x64xf16, {order = #NHWC}>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]
   // CHECK:       [[EXPAND0:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND1:%.+]] = IE.Expand([[ARG_1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64xf16, {order = #NHWC}>
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

// CHECK-LABEL: func.func @main
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x25x56x56x!qElemType, {order = #NHWC}>
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

   // CHECK:       [[SLICE0:%.+]]  = IE.Slice [[ARG_0]] [0, 21, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE1:%.+]]  = IE.Slice [[ARG_0]] [0, 20, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE2:%.+]]  = IE.Slice [[ARG_0]] [0, 19, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE3:%.+]]  = IE.Slice [[ARG_0]] [0, 18, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE4:%.+]]  = IE.Slice [[ARG_0]] [0, 17, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE5:%.+]]  = IE.Slice [[ARG_0]] [0, 16, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE6:%.+]]  = IE.Slice [[ARG_0]] [0, 15, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE7:%.+]]  = IE.Slice [[ARG_0]] [0, 14, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE8:%.+]]  = IE.Slice [[ARG_0]] [0, 13, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE9:%.+]]  = IE.Slice [[ARG_0]] [0, 12, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE10:%.+]]  = IE.Slice [[ARG_0]] [0, 11, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE11:%.+]]  = IE.Slice [[ARG_0]] [0, 10, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE12:%.+]]  = IE.Slice [[ARG_0]] [0, 9, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE13:%.+]]  = IE.Slice [[ARG_0]] [0, 8, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE14:%.+]]  = IE.Slice [[ARG_0]] [0, 7, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE15:%.+]]  = IE.Slice [[ARG_0]] [0, 6, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE16:%.+]]  = IE.Slice [[ARG_0]] [0, 5, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE17:%.+]]  = IE.Slice [[ARG_0]] [0, 4, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE18:%.+]]  = IE.Slice [[ARG_0]] [0, 3, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE19:%.+]]  = IE.Slice [[ARG_0]] [0, 2, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE20:%.+]]  = IE.Slice [[ARG_0]] [0, 1, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE21:%.+]]  = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE22:%.+]]  = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE23:%.+]]  = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>
   // CHECK:       [[SLICE24:%.+]]  = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 56, 56] : tensor<1x25x56x56x!qElemType, {order = #NHWC}> to tensor<1x4x56x56x!qElemType, {order = #NHWC}>

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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @notOptimizeSliceSoftmaxExpandDifferentAxis
module @notOptimizeSliceSoftmaxExpandDifferentAxis {
// CHECK-LABEL:       @notFuseSliceSoftmaxExpandDifferentAxis
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x592x580x1xf16, {order = #NHWC}>) -> tensor<1x592x580x1xf16, {order = #NHWC}>
func.func @notFuseSliceSoftmaxExpandDifferentAxis(%arg0: tensor<1x592x580x1xf16, {order = #NHWC}>) -> tensor<1x592x580x1xf16, {order = #NHWC}> {
  %1 = IE.Slice %arg0 [0, 0, 0, 0] [1, 592, 577, 1] : tensor<1x592x580x1xf16, {order = #NHWC}> to tensor<1x592x577x1xf16, {order = #NHWC}>
  %2 = IE.SoftMax(%1) {axisInd = 1 : i64, padSize = 15 : i64} : tensor<1x592x577x1xf16, {order = #NHWC}> -> tensor<1x592x577x1xf16, {order = #NHWC}>
  %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x592x577x1xf16, {order = #NHWC}> -> tensor<1x592x580x1xf16, {order = #NHWC}>
  return %3 : tensor<1x592x580x1xf16, {order = #NHWC}>

  // CHECK:    [[OUT_SOFTMAX:%.+]] = IE.SoftMax([[INPUT]]) {axisInd = 1 : i64, padSize = 15 : i64}
  // CHECK:    return [[OUT_SOFTMAX]]
}
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.956:128, 0.785:128, 0.567:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128}>
!qElemType2 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.567:128, 0.567:128, 0.567:128}>

// CHECK-LABEL: @OptimizeSliceConcatExpandForQuantizedType
module @OptimizeSliceConcatExpandForQuantizedType {
// CHECK-LABEL:       @OptimizeSliceConcatExpandForQuantizedType
// CHECK-SAME:        [[INPUT1:%[^:]+]]: tensor<1x6x32x56x!qElemType>,
// CHECK-SAME:        [[INPUT2:%[^:]+]]: tensor<1x6x32x56x!qElemType>) -> tensor<2x6x32x56x!qElemType>
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHWN = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithElimination
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x4xf16, {order = #NHWC}>
func.func @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithElimination(%arg0: tensor<1x1504x20x4xf16, {order = #NHWC}>) -> tensor<20x4x1504x1xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 20, 4] : tensor<1x1504x20x4xf16, {order = #NHWC}> to tensor<1x1500x20x4xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #CHWN} : tensor<1x1500x20x4xf16, {order = #NHWC}> -> tensor<20x4x1500x1xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<20x4x1500x1xf16> -> tensor<20x4x1504x1xf16>

   return %2 : tensor<20x4x1504x1xf16>

   // CHECK-NOT:    IE.Slice
   // CHECK-NOT:    IE.Expand
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x1504x20x4xf16, {order = #NHWC}> -> tensor<20x4x1504x1xf16>

   // CHECK:        return [[PERMUTECAST]] : tensor<20x4x1504x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithSlice
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x16xf16, {order = #NHWC}>
func.func @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithSlice(%arg0: tensor<1x1504x20x16xf16, {order = #NHWC}>) -> tensor<1x16x20x1502xf16, {order = #NHCW}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 20, 16] : tensor<1x1504x20x16xf16, {order = #NHWC}> to tensor<1x1500x20x16xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1500x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1500xf16, {order = #NHCW}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x16x20x1500xf16, {order = #NHCW}> -> tensor<1x16x20x1502xf16, {order = #NHCW}>

   return %2 : tensor<1x16x20x1502xf16, {order = #NHCW}>

   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1502, 20, 16] : tensor<1x1504x20x16xf16, {order = #NHWC}> to tensor<1x1502x20x16xf16, {order = #NHWC}>
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1502x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1502xf16, {order = #NHCW}>

   // CHECK:        return [[PERMUTECAST]] : tensor<1x16x20x1502xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithExpand
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x16xf16, {order = #NHWC}>
func.func @OptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneWithExpand(%arg0: tensor<1x1504x20x16xf16, {order = #NHWC}>) -> tensor<1x16x20x1505xf16, {order = #NHCW}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 20, 16] : tensor<1x1504x20x16xf16, {order = #NHWC}> to tensor<1x1500x20x16xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1500x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1500xf16, {order = #NHCW}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 5]} : tensor<1x16x20x1500xf16, {order = #NHCW}> -> tensor<1x16x20x1505xf16, {order = #NHCW}>

   return %2 : tensor<1x16x20x1505xf16, {order = #NHCW}>

   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x1504x20x16xf16, {order = #NHWC}> -> tensor<1x1505x20x16xf16, {order = #NHWC}>
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[EXPAND]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1505x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1505xf16, {order = #NHCW}>

   // CHECK:        return [[PERMUTECAST]] : tensor<1x16x20x1505xf16, {order = #NHCW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHWN = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NotOptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneIncompatibleMemAxis
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x4xf16, {order = #NHWC}>
func.func @NotOptimizeSlicePermuteCastExpandPatternSliceSizeGreaterThanOneIncompatibleMemAxis(%arg0: tensor<1x1504x20x4xf16, {order = #NHWC}>) -> tensor<20x5x1500x1xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 20, 4] : tensor<1x1504x20x4xf16, {order = #NHWC}> to tensor<1x1500x20x4xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #CHWN} : tensor<1x1500x20x4xf16, {order = #NHWC}> -> tensor<20x4x1500x1xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<20x4x1500x1xf16> -> tensor<20x5x1500x1xf16>

   return %2 : tensor<20x5x1500x1xf16>

   // CHECK:    IE.Slice
   // CHECK:    IE.PermuteCast
   // CHECK:    IE.Expand
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @OptimizeSlicePermuteCastExpandPatternSliceSizeIsOneNCHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x4xf16, {order = #NHWC}>
func.func @OptimizeSlicePermuteCastExpandPatternSliceSizeIsOneNCHW(%arg0: tensor<1x1504x20x4xf16, {order = #NHWC}>) -> tensor<1x20x4x1504xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 20, 4] : tensor<1x1504x20x4xf16, {order = #NHWC}> to tensor<1x1x20x4xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x20x4xf16, {order = #NHWC}> -> tensor<1x20x4x1xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 1503]} : tensor<1x20x4x1xf16> -> tensor<1x20x4x1504xf16>

   return %2 : tensor<1x20x4x1504xf16>

   // CHECK-NOT:    IE.Slice
   // CHECK-NOT:    IE.Expand
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1504x20x4xf16, {order = #NHWC}> -> tensor<1x20x4x1504xf16>

   // CHECK:        return [[PERMUTECAST]] : tensor<1x20x4x1504xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @OptimizeSlicePermuteCastExpandPatternSameDstOrder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x5x1xf16, {order = #NHWC}>
func.func @OptimizeSlicePermuteCastExpandPatternSameDstOrder(%arg0: tensor<1x512x5x1xf16, {order = #NHWC}>) -> tensor<5x512x1x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 500, 5, 1] : tensor<1x512x5x1xf16, {order = #NHWC}> to tensor<1x500x5x1xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NHWC, mem_perm = #CNHW} : tensor<1x500x5x1xf16, {order = #NHWC}> -> tensor<5x500x1x1xf16, {order = #NHWC}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<5x500x1x1xf16, {order = #NHWC}> -> tensor<5x512x1x1xf16, {order = #NHWC}>
   return %2 : tensor<5x512x1x1xf16, {order = #NHWC}>

   // CHECK-NOT:    IE.Slice
   // CHECK-NOT:    IE.Expand
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #map} : tensor<1x512x5x1xf16, {order = #NHWC}> -> tensor<5x512x1x1xf16, {order = #NHWC}>

   // CHECK:        return [[PERMUTECAST]] : tensor<5x512x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotOptimizeSlicePermuteCastExpandPatternSliceSizeIsOneNotNCHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x4xf16, {order = #NHWC}>
func.func @NotOptimizeSlicePermuteCastExpandPatternSliceSizeIsOneNotNCHW(%arg0: tensor<1x1504x20x4xf16, {order = #NHWC}>) -> tensor<1x20x1504x4xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 20, 4] : tensor<1x1504x20x4xf16, {order = #NHWC}> to tensor<1x1x20x4xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x1x20x4xf16, {order = #NHWC}> -> tensor<1x20x1x4xf16>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1503, 0]} : tensor<1x20x1x4xf16> -> tensor<1x20x1504x4xf16>

   return %2 : tensor<1x20x1504x4xf16>

   // CHECK:    IE.Slice
   // CHECK:    IE.PermuteCast
   // CHECK:    IE.Expand
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038406767097173954>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeExpandSliceWithIterationTimeLargerThan60
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x256x3x!qElemType, {order = #NHWC}>
func.func @OptimizeExpandSliceWithIterationTimeLargerThan60(%arg0: tensor<1x256x256x3x!qElemType, {order = #NHWC}>) -> tensor<1x256x336x3x!qElemType, {order = #NHWC}> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]} : tensor<1x256x256x3x!qElemType, {order = #NHWC}> -> tensor<1x256x256x16x!qElemType, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 256, 256, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x256x3x!qElemType, {order = #NHWC}>
    %2 = IE.Slice %0 [0, 0, 40, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %0 [0, 0, 39, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %4 = IE.Slice %0 [0, 0, 38, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %5 = IE.Slice %0 [0, 0, 37, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %6 = IE.Slice %0 [0, 0, 36, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %7 = IE.Slice %0 [0, 0, 35, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %8 = IE.Slice %0 [0, 0, 34, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %9 = IE.Slice %0 [0, 0, 33, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %10 = IE.Slice %0 [0, 0, 32, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %11 = IE.Slice %0 [0, 0, 31, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %12 = IE.Slice %0 [0, 0, 30, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %13 = IE.Slice %0 [0, 0, 29, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %14 = IE.Slice %0 [0, 0, 28, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %15 = IE.Slice %0 [0, 0, 27, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %16 = IE.Slice %0 [0, 0, 26, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %17 = IE.Slice %0 [0, 0, 25, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %18 = IE.Slice %0 [0, 0, 24, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %19 = IE.Slice %0 [0, 0, 23, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %20 = IE.Slice %0 [0, 0, 22, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %21 = IE.Slice %0 [0, 0, 21, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %22 = IE.Slice %0 [0, 0, 20, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %23 = IE.Slice %0 [0, 0, 19, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %24 = IE.Slice %0 [0, 0, 18, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %25 = IE.Slice %0 [0, 0, 17, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %26 = IE.Slice %0 [0, 0, 16, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %27 = IE.Slice %0 [0, 0, 15, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %28 = IE.Slice %0 [0, 0, 14, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %29 = IE.Slice %0 [0, 0, 13, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %30 = IE.Slice %0 [0, 0, 12, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %31 = IE.Slice %0 [0, 0, 11, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %32 = IE.Slice %0 [0, 0, 10, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %33 = IE.Slice %0 [0, 0, 9, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %34 = IE.Slice %0 [0, 0, 8, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %35 = IE.Slice %0 [0, 0, 7, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %36 = IE.Slice %0 [0, 0, 6, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %37 = IE.Slice %0 [0, 0, 5, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %38 = IE.Slice %0 [0, 0, 4, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %39 = IE.Slice %0 [0, 0, 3, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %40 = IE.Slice %0 [0, 0, 2, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %41 = IE.Slice %0 [0, 0, 1, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %42 = IE.Slice %0 [0, 0, 254, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %43 = IE.Slice %0 [0, 0, 253, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %44 = IE.Slice %0 [0, 0, 252, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %45 = IE.Slice %0 [0, 0, 251, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %46 = IE.Slice %0 [0, 0, 250, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %47 = IE.Slice %0 [0, 0, 249, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %48 = IE.Slice %0 [0, 0, 248, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %49 = IE.Slice %0 [0, 0, 247, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %50 = IE.Slice %0 [0, 0, 246, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %51 = IE.Slice %0 [0, 0, 245, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %52 = IE.Slice %0 [0, 0, 244, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %53 = IE.Slice %0 [0, 0, 243, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %54 = IE.Slice %0 [0, 0, 242, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %55 = IE.Slice %0 [0, 0, 241, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %56 = IE.Slice %0 [0, 0, 240, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %57 = IE.Slice %0 [0, 0, 239, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %58 = IE.Slice %0 [0, 0, 238, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %59 = IE.Slice %0 [0, 0, 237, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %60 = IE.Slice %0 [0, 0, 236, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %61 = IE.Slice %0 [0, 0, 235, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %62 = IE.Slice %0 [0, 0, 234, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %63 = IE.Slice %0 [0, 0, 233, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %64 = IE.Slice %0 [0, 0, 232, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %65 = IE.Slice %0 [0, 0, 231, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %66 = IE.Slice %0 [0, 0, 230, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %67 = IE.Slice %0 [0, 0, 229, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %68 = IE.Slice %0 [0, 0, 228, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %69 = IE.Slice %0 [0, 0, 227, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %70 = IE.Slice %0 [0, 0, 226, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %71 = IE.Slice %0 [0, 0, 225, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %72 = IE.Slice %0 [0, 0, 224, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %73 = IE.Slice %0 [0, 0, 223, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %74 = IE.Slice %0 [0, 0, 222, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %75 = IE.Slice %0 [0, 0, 221, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %76 = IE.Slice %0 [0, 0, 220, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %77 = IE.Slice %0 [0, 0, 219, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %78 = IE.Slice %0 [0, 0, 218, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %79 = IE.Slice %0 [0, 0, 217, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %80 = IE.Slice %0 [0, 0, 216, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %81 = IE.Slice %0 [0, 0, 215, 0] [1, 256, 1, 3] : tensor<1x256x256x16x!qElemType, {order = #NHWC}> to tensor<1x256x1x3x!qElemType, {order = #NHWC}>
    %82 = IE.Concat(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %1, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0], [0, 0, 6, 0], [0, 0, 7, 0], [0, 0, 8, 0], [0, 0, 9, 0], [0, 0, 10, 0], [0, 0, 11, 0], [0, 0, 12, 0], [0, 0, 13, 0], [0, 0, 14, 0], [0, 0, 15, 0], [0, 0, 16, 0], [0, 0, 17, 0], [0, 0, 18, 0], [0, 0, 19, 0], [0, 0, 20, 0], [0, 0, 21, 0], [0, 0, 22, 0], [0, 0, 23, 0], [0, 0, 24, 0], [0, 0, 25, 0], [0, 0, 26, 0], [0, 0, 27, 0], [0, 0, 28, 0], [0, 0, 29, 0], [0, 0, 30, 0], [0, 0, 31, 0], [0, 0, 32, 0], [0, 0, 33, 0], [0, 0, 34, 0], [0, 0, 35, 0], [0, 0, 36, 0], [0, 0, 37, 0], [0, 0, 38, 0], [0, 0, 39, 0], [0, 0, 40, 0], [0, 0, 296, 0], [0, 0, 297, 0], [0, 0, 298, 0], [0, 0, 299, 0], [0, 0, 300, 0], [0, 0, 301, 0], [0, 0, 302, 0], [0, 0, 303, 0], [0, 0, 304, 0], [0, 0, 305, 0], [0, 0, 306, 0], [0, 0, 307, 0], [0, 0, 308, 0], [0, 0, 309, 0], [0, 0, 310, 0], [0, 0, 311, 0], [0, 0, 312, 0], [0, 0, 313, 0], [0, 0, 314, 0], [0, 0, 315, 0], [0, 0, 316, 0], [0, 0, 317, 0], [0, 0, 318, 0], [0, 0, 319, 0], [0, 0, 320, 0], [0, 0, 321, 0], [0, 0, 322, 0], [0, 0, 323, 0], [0, 0, 324, 0], [0, 0, 325, 0], [0, 0, 326, 0], [0, 0, 327, 0], [0, 0, 328, 0], [0, 0, 329, 0], [0, 0, 330, 0], [0, 0, 331, 0], [0, 0, 332, 0], [0, 0, 333, 0], [0, 0, 334, 0], [0, 0, 335, 0]]} : tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x256x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x256x1x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x256x336x3x!quant.uniform<u8:f16, 0.0038406767097173954>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %82 : tensor<1x256x336x3x!qElemType, {order = #NHWC}>

   // CHECK-NOT:    IE.Expand
   // CHECK-DAG:    [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 215, 0] [1, 256, 1, 3]
   // CHECK-DAG:    [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 216, 0] [1, 256, 1, 3]
   // CHECK-DAG:    [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 217, 0] [1, 256, 1, 3]
   // CHECK-DAG:    [[SLICE79:%.+]] = IE.Slice [[INPUT]] [0, 0, 40, 0] [1, 256, 1, 3]
   // CHECK:        [[CONCAT:%.+]] = IE.Concat
   // CHECK:        return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceLayoutCastExpandAdd
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x12x77x80xf16>
func.func @OptimizeSliceLayoutCastExpandAdd(%arg0: tensor<1x12x77x80xf16>) -> tensor<1x12x77x77xf16> {
   %cst = const.Declare tensor<1x16x77x77xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x77x77xf16>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 12 : i64>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 4, 0, 0]>]
   %slice = IE.Slice %arg0 [0, 0, 0, 0] [1, 12, 77, 77] : tensor<1x12x77x80xf16> to tensor<1x12x77x77xf16>
   %lc = IE.LayoutCast(%slice) {dst_order = #NHWC} : tensor<1x12x77x77xf16> -> tensor<1x12x77x77xf16, {order = #NHWC}>
   %expand = IE.Expand(%lc) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16, {order = #NHWC}>
   %add = IE.Add(%expand, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, input_padding = [0, 4, 0, 0], output_padding = [0, 4, 0, 0]} : tensor<1x16x77x77xf16, {order = #NHWC}>, tensor<1x16x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16, {order = #NHWC}>
   %sliceOut = IE.Slice %add [0, 0, 0, 0] [1, 12, 77, 77] : tensor<1x16x77x77xf16, {order = #NHWC}> to tensor<1x12x77x77xf16, {order = #NHWC}>
   %lcOut = IE.LayoutCast(%sliceOut) {dst_order = #NCHW} : tensor<1x12x77x77xf16, {order = #NHWC}> -> tensor<1x12x77x77xf16>

   return %lcOut : tensor<1x12x77x77xf16>

   // CHECK-DAG:   [[CONST:%.+]] = const.Declare tensor<1x16x77x80xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x77x77xf16>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 12 : i64>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>, #const.LayoutCast<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 4, 0, 0]>]
   // CHECK:       [[LC:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x12x77x80xf16> -> tensor<1x12x77x80xf16, {order = #NHWC}>
   // CHECK:       [[EXPAND:%.+]] = IE.Expand([[LC]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]} : tensor<1x12x77x80xf16, {order = #NHWC}> -> tensor<1x16x77x80xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND]], [[CONST]])
   // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 12, 77, 80] : tensor<1x16x77x80xf16, {order = #NHWC}> to tensor<1x12x77x80xf16, {order = #NHWC}>
   // CHECK:       [[LC_OUT:%.+]] = IE.LayoutCast([[SLICE_0]]) {dst_order = #NCHW} : tensor<1x12x77x80xf16, {order = #NHWC}> -> tensor<1x12x77x80xf16>
   // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[LC_OUT]] [0, 0, 0, 0] [1, 12, 77, 77] : tensor<1x12x77x80xf16> to tensor<1x12x77x77xf16>
   // CHECK:       return [[SLICE_1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceLayoutCastAddLayoutCast
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x16x77x80xf16>
func.func @OptimizeSliceLayoutCastAddLayoutCast(%arg0: tensor<1x16x77x80xf16>) -> tensor<1x16x77x80xf16> {
   %cst = const.Declare tensor<1x16x77x77xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x77x77xf16>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 16 : i64>, #const.LayoutCast<#NHWC>]
   %cst_0 = const.Declare tensor<1x16x77x77xf16> = dense<1.0> : tensor<1x16x77x77xf16>
   %cst_1 = const.Declare tensor<1x16x77x77xf16> = dense<2.0> : tensor<1x16x77x77xf16>
   %cst_2 = const.Declare tensor<1x16x77x77xf16> = dense<3.0> : tensor<1x16x77x77xf16>

   %slice = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 77, 77] : tensor<1x16x77x80xf16> to tensor<1x16x77x77xf16>
   %lc = IE.LayoutCast(%slice) {dst_order = #NHWC} : tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16, {order = #NHWC}>
   %add = IE.Add(%lc, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x77xf16, {order = #NHWC}>, tensor<1x16x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16, {order = #NHWC}>
   %lcOut = IE.LayoutCast(%add) {dst_order = #NCHW} : tensor<1x16x77x77xf16, {order = #NHWC}> -> tensor<1x16x77x77xf16>
   %sub = IE.Subtract(%lcOut, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x77xf16>, tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16>
   %mul = IE.Multiply(%sub, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x77xf16>, tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16>
   %div = IE.Divide(%mul, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x77xf16>, tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16>
   %softmax = IE.SoftMax(%div) {axisInd = 3 : i64} : tensor<1x16x77x77xf16> -> tensor<1x16x77x77xf16>
   %expand = IE.Expand(%softmax) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 3]} : tensor<1x16x77x77xf16> -> tensor<1x16x77x80xf16>

   return %expand : tensor<1x16x77x80xf16>

   // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x77x80xf16> = dense<3.000000e+00> : tensor<1x16x77x77xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>]
   // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x16x77x80xf16> = dense<2.000000e+00> : tensor<1x16x77x77xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>]
   // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x16x77x80xf16> = dense<1.000000e+00> : tensor<1x16x77x77xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>]
   // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x16x77x80xf16, {order = #NHWC}> = dense<1.000000e+00>
   // CHECK-SAME:       tensor<1x1x77x77xf16>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 16 : i64>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>, #const.LayoutCast<#NHWC>]

   // CHECK:       [[LC_IN:%.+]] = IE.LayoutCast([[INPUT]]) {dst_order = #NHWC} : tensor<1x16x77x80xf16> -> tensor<1x16x77x80xf16, {order = #NHWC}>
   // CHECK:       [[ADD:%.+]] = IE.Add([[LC_IN]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x80xf16, {order = #NHWC}>, tensor<1x16x77x80xf16, {order = #NHWC}> -> tensor<1x16x77x80xf16, {order = #NHWC}>
   // CHECK:       [[LC_OUT:%.+]] = IE.LayoutCast([[ADD]]) {dst_order = #NCHW} : tensor<1x16x77x80xf16, {order = #NHWC}> -> tensor<1x16x77x80xf16>
   // CHECK:       [[SUB:%.+]] = IE.Subtract([[LC_OUT]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x80xf16>, tensor<1x16x77x80xf16> -> tensor<1x16x77x80xf16>
   // CHECK:       [[MUL:%.+]] = IE.Multiply([[SUB]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x80xf16>, tensor<1x16x77x80xf16> -> tensor<1x16x77x80xf16>
   // CHECK:       [[DIV:%.+]] = IE.Divide([[MUL]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x77x80xf16>, tensor<1x16x77x80xf16> -> tensor<1x16x77x80xf16>
   // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[DIV]]) {axisInd = 3 : i64, padSize = 3 : i64} : tensor<1x16x77x80xf16> -> tensor<1x16x77x80xf16>
   // CHECK:       return [[SOFTMAX]] : tensor<1x16x77x80xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceConcatExpandThroughViewLikeOps
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>)
func.func @OptimizeSliceConcatExpandThroughViewLikeOps(%arg0: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg1: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg2: tensor<1x64x580x1xf16, {order = #NHWC}>) -> tensor<1x64x580x3xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %4 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16> -> tensor<1x577x3x64xf16>
   %7 = IE.PermuteCast(%6) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x577x3x64xf16> -> tensor<1x64x577x3xf16, {order = #NHWC}>
   %8 = IE.Expand(%7) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x64x577x3xf16, {order = #NHWC}> -> tensor<1x64x580x3xf16, {order = #NHWC}>

   return %8 : tensor<1x64x580x3xf16, {order = #NHWC}>

   // CHECK-NOT:           IE.Slice
   // CHECK-NOT:           IE.Expand
   // CHECK:               [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x64x580x1xf16, {order = #NHWC}>, tensor<1x64x580x1xf16, {order = #NHWC}>, tensor<1x64x580x1xf16, {order = #NHWC}> -> tensor<1x64x580x3xf16, {order = #NHWC}>
   // CHECK:               return [[CONCAT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#CWNH = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL: @OptimizeSliceConcatExpandThroughViewLikeOpsInsertShapeCast
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>)
func.func @OptimizeSliceConcatExpandThroughViewLikeOpsInsertShapeCast(%arg0: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg1: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg2: tensor<1x64x580x1xf16, {order = #NHWC}>) -> tensor<1x192x580x1xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #CWNH} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<577x64x1x1xf16>
   %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 577, 1, 64]} : tensor<577x64x1x1xf16> -> tensor<1x577x1x64xf16>
   %3 = IE.PermuteCast(%2) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x577x1x64xf16> -> tensor<1x1x577x64xf16, {order = #NHCW}>

   %4 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #CWNH} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<577x64x1x1xf16>
   %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 577, 1, 64]} : tensor<577x64x1x1xf16> -> tensor<1x577x1x64xf16>
   %7 = IE.PermuteCast(%6) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x577x1x64xf16> -> tensor<1x1x577x64xf16, {order = #NHCW}>

   %8 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %9 = IE.PermuteCast(%8) {dst_order = #NCHW, mem_perm = #CWNH} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<577x64x1x1xf16>
   %10 = IE.AffineReshape(%9) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 577, 1, 64]} : tensor<577x64x1x1xf16> -> tensor<1x577x1x64xf16>
   %11 = IE.PermuteCast(%10) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x577x1x64xf16> -> tensor<1x1x577x64xf16, {order = #NHCW}>

   %12 = IE.Concat(%3, %7, %11) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]} : tensor<1x1x577x64xf16, {order = #NHCW}>, tensor<1x1x577x64xf16, {order = #NHCW}>, tensor<1x1x577x64xf16, {order = #NHCW}> -> tensor<1x3x577x64xf16, {order = #NHCW}>
   %13 = IE.PermuteCast(%12) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x577x64xf16, {order = #NHCW}> -> tensor<1x577x3x64xf16>
   %14 = IE.AffineReshape(%13) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 577, 192, 1]} : tensor<1x577x3x64xf16> -> tensor<1x577x192x1xf16>
   %15 = IE.PermuteCast(%14) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x577x192x1xf16> -> tensor<1x192x577x1xf16, {order = #NHWC}>
   %16 = IE.Expand(%15) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x192x577x1xf16, {order = #NHWC}> -> tensor<1x192x580x1xf16, {order = #NHWC}>

   return %16 : tensor<1x192x580x1xf16, {order = #NHWC}>

   // CHECK-NOT:           IE.Slice
   // CHECK-NOT:           IE.Expand
   // CHECK:               [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x64x580x1xf16, {order = #NHWC}>, tensor<1x64x580x1xf16, {order = #NHWC}>, tensor<1x64x580x1xf16, {order = #NHWC}> -> tensor<1x64x580x3xf16, {order = #NHWC}>
   // CHECK:               [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 192, 580, 1]} inputs([[CONCAT]] : tensor<1x64x580x3xf16, {order = #NHWC}>) -> tensor<1x192x580x1xf16, {order = #NHWC}>
   // CHECK:               return [[SHAPECAST]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.01>
!qElemType1 = !quant.uniform<u8:f16, 0.02>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceConcatExpandThroughViewLikeOpsInsertQuantCast
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>)
func.func @OptimizeSliceConcatExpandThroughViewLikeOpsInsertQuantCast(%arg0: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, %arg2: tensor<1x64x580x1x!qElemType, {order = #NHWC}>) -> tensor<1x64x580x3x!qElemType1, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %4 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x577x1x64x!qElemType>, tensor<1x577x1x64x!qElemType>, tensor<1x577x1x64x!qElemType> -> tensor<1x577x3x64x!qElemType>
   %7 = IE.PermuteCast(%6) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x577x3x64x!qElemType> -> tensor<1x64x577x3x!qElemType, {order = #NHWC}>
   %8 = IE.QuantizeCast(%7) {dstElemType = !qElemType1} : tensor<1x64x577x3x!qElemType, {order = #NHWC}> -> tensor<1x64x577x3x!qElemType1, {order = #NHWC}>
   %9 = IE.Expand(%8) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x64x577x3x!qElemType1, {order = #NHWC}> -> tensor<1x64x580x3x!qElemType1, {order = #NHWC}>

   return %9 : tensor<1x64x580x3x!qElemType1, {order = #NHWC}>

   // CHECK-NOT:           IE.Slice
   // CHECK-NOT:           IE.Expand
   // CHECK:               [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x64x580x1x!qElemType, {order = #NHWC}>, tensor<1x64x580x1x!qElemType, {order = #NHWC}>, tensor<1x64x580x1x!qElemType, {order = #NHWC}> -> tensor<1x64x580x3x!qElemType, {order = #NHWC}>
   // CHECK:               [[QUANTCAST:%.+]] = IE.QuantizeCast([[CONCAT]]) {dstElemType = !qElemType1} : tensor<1x64x580x3x!qElemType, {order = #NHWC}> -> tensor<1x64x580x3x!qElemType1, {order = #NHWC}>
   // CHECK:               return [[QUANTCAST]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.01>
!qElemType1 = !quant.uniform<u8:f16, 0.02>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceConcatExpandThroughViewLikeOpsInsertShapeCastQuantCast
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x64x580x1x!qElemType, {order = #NHWC}>)
func.func @OptimizeSliceConcatExpandThroughViewLikeOpsInsertShapeCastQuantCast(%arg0: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x580x1x!qElemType, {order = #NHWC}>, %arg2: tensor<1x64x580x1x!qElemType, {order = #NHWC}>) -> tensor<1x1x580x192x!qElemType1, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %4 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1x!qElemType, {order = #NHWC}> to tensor<1x64x577x1x!qElemType, {order = #NHWC}>
   %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1x!qElemType, {order = #NHWC}> -> tensor<1x577x1x64x!qElemType>

   %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x577x1x64x!qElemType>, tensor<1x577x1x64x!qElemType>, tensor<1x577x1x64x!qElemType> -> tensor<1x577x3x64x!qElemType>
   %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 577, 192, 1]} : tensor<1x577x3x64x!qElemType> -> tensor<1x577x192x1x!qElemType>
   %8 = IE.PermuteCast(%7) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x577x192x1x!qElemType> -> tensor<1x1x577x192x!qElemType, {order = #NHWC}>
   %9 = IE.QuantizeCast(%8) {dstElemType = !qElemType1} : tensor<1x1x577x192x!qElemType, {order = #NHWC}> -> tensor<1x1x577x192x!qElemType1, {order = #NHWC}>
   %10 = IE.Expand(%9) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x1x577x192x!qElemType1, {order = #NHWC}> -> tensor<1x1x580x192x!qElemType1, {order = #NHWC}>

   return %10 : tensor<1x1x580x192x!qElemType1, {order = #NHWC}>

   // CHECK-NOT:           IE.Slice
   // CHECK-NOT:           IE.Expand
   // CHECK:               [[CONCAT:%.+]] = IE.Concat([[INPUT0]], [[INPUT1]], [[INPUT2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x64x580x1x!qElemType, {order = #NHWC}>, tensor<1x64x580x1x!qElemType, {order = #NHWC}>, tensor<1x64x580x1x!qElemType, {order = #NHWC}> -> tensor<1x64x580x3x!qElemType, {order = #NHWC}>
   // CHECK:               [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 1, 580, 192]} inputs([[CONCAT]] : tensor<1x64x580x3x!qElemType, {order = #NHWC}>) -> tensor<1x1x580x192x!qElemType, {order = #NHWC}>
   // CHECK:               [[QUANTCAST:%.+]] = IE.QuantizeCast([[SHAPECAST]]) {dstElemType = !qElemType1} : tensor<1x1x580x192x!qElemType, {order = #NHWC}> -> tensor<1x1x580x192x!qElemType1, {order = #NHWC}>
   // CHECK:               return [[QUANTCAST]]
}

// -----

// CHECK-LABEL: @NotOptimizeSliceConcatExpandThroughViewLikeOpsIncompatibleSliceConcat
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<580x64x1x1xf16>, [[INPUT1:%.+]]: tensor<580x64x1x1xf16>, [[INPUT2:%.+]]: tensor<580x64x1x1xf16>)
func.func @NotOptimizeSliceConcatExpandThroughViewLikeOpsIncompatibleSliceConcat(%arg0: tensor<580x64x1x1xf16>, %arg1: tensor<580x64x1x1xf16>, %arg2: tensor<580x64x1x1xf16>) -> tensor<580x32x2x3xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>

   %2 = IE.Slice %arg1 [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>

   %4 = IE.Slice %arg2 [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>

   %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<577x32x2x1xf16>, tensor<577x32x2x1xf16>, tensor<577x32x2x1xf16> -> tensor<577x32x2x3xf16>
   %7 = IE.Expand(%6) {pads_begin = [0, 0, 0, 0], pads_end = [3, 0, 0, 0]} : tensor<577x32x2x3xf16> -> tensor<580x32x2x3xf16>

   return %7 : tensor<580x32x2x3xf16>

   // CHECK:              [[SLICE0:%.+]] = IE.Slice [[INPUT0]] [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   // CHECK:              [[RESHAPE0:%.+]] = IE.AffineReshape([[SLICE0]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>
   // CHECK:              [[SLICE1:%.+]] = IE.Slice [[INPUT1]] [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   // CHECK:              [[RESHAPE1:%.+]] = IE.AffineReshape([[SLICE1]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>
   // CHECK:              [[SLICE2:%.+]] = IE.Slice [[INPUT2]] [0, 0, 0, 0] [577, 64, 1, 1] : tensor<580x64x1x1xf16> to tensor<577x64x1x1xf16>
   // CHECK:              [[RESHAPE2:%.+]] = IE.AffineReshape([[SLICE2]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [577, 32, 2, 1]} : tensor<577x64x1x1xf16> -> tensor<577x32x2x1xf16>
   // CHECK:              [[CONCAT:%.+]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]], [[RESHAPE2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<577x32x2x1xf16>, tensor<577x32x2x1xf16>, tensor<577x32x2x1xf16> -> tensor<577x32x2x3xf16>
   // CHECK:               [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [3, 0, 0, 0]} : tensor<577x32x2x3xf16> -> tensor<580x32x2x3xf16>
   // CHECK:               return [[EXPAND]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceConcatExpandThroughViewLikeOpsIncompatibleConcatExpand
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>, [[INPUT2:%.+]]: tensor<1x64x580x1xf16, {order = #NHWC}>)
func.func @NotOptimizeSliceConcatExpandThroughViewLikeOpsIncompatibleConcatExpand(%arg0: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg1: tensor<1x64x580x1xf16, {order = #NHWC}>, %arg2: tensor<1x64x580x1xf16, {order = #NHWC}>) -> tensor<1x1x1734x64xf16, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %2 = IE.Slice %arg1 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %4 = IE.Slice %arg2 [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   %5 = IE.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>

   %6 = IE.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16> -> tensor<1x577x3x64xf16>
   %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1731, 64, 1]} : tensor<1x577x3x64xf16> -> tensor<1x1731x64x1xf16>
   %8 = IE.PermuteCast(%7) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1731x64x1xf16> -> tensor<1x1x1731x64xf16, {order = #NHWC}>
   %9 = IE.Expand(%8) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x1x1731x64xf16, {order = #NHWC}> -> tensor<1x1x1734x64xf16, {order = #NHWC}>

   return %9 : tensor<1x1x1734x64xf16, {order = #NHWC}>

   // CHECK:               [[SLICE0:%.+]] = IE.Slice [[INPUT0]] [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   // CHECK:               [[PERMUTECAST0:%.+]] = IE.PermuteCast([[SLICE0]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>
   // CHECK:               [[SLICE1:%.+]] = IE.Slice [[INPUT1]] [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   // CHECK:               [[PERMUTECAST1:%.+]] = IE.PermuteCast([[SLICE1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>
   // CHECK:               [[SLICE2:%.+]] = IE.Slice [[INPUT2]] [0, 0, 0, 0] [1, 64, 577, 1] : tensor<1x64x580x1xf16, {order = #NHWC}> to tensor<1x64x577x1xf16, {order = #NHWC}>
   // CHECK:               [[PERMUTECAST2:%.+]] = IE.PermuteCast([[SLICE2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x577x1xf16, {order = #NHWC}> -> tensor<1x577x1x64xf16>
   // CHECK:               [[CONCAT:%.+]] = IE.Concat([[PERMUTECAST0]], [[PERMUTECAST1]], [[PERMUTECAST2]])
   // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]} : tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16>, tensor<1x577x1x64xf16> -> tensor<1x577x3x64xf16>
   // CHECK:               [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1731, 64, 1]} : tensor<1x577x3x64xf16> -> tensor<1x1731x64x1xf16>
   // CHECK:               [[PERMUTECAST3:%.+]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1731x64x1xf16> -> tensor<1x1x1731x64xf16, {order = #NHWC}>
   // CHECK:               [[EXPAND:%.+]] = IE.Expand([[PERMUTECAST3]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 3, 0]} : tensor<1x1x1731x64xf16, {order = #NHWC}> -> tensor<1x1x1734x64xf16, {order = #NHWC}>
   // CHECK:                return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @OptimizeSliceViewLikeChainExpand_PermuteCastAffineReshape
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x3x64xf16, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_PermuteCastAffineReshape(%arg0: tensor<1x1504x3x64xf16, {order = #NHWC}>) -> tensor<1x192x1504x1xf16> {
   // Slice on logical dim C (mem dim 3 in NHWC): 1504 -> 1500
   // PermuteCast identity mem_perm reinterprets layout
   // AffineReshape merges dims 1,2 and splits dim 3
   // Expand pads dim 2: 1500 -> 1504
   // Optimization: enlarge Slice by 4 on C-axis, propagate through chain, eliminate Expand
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 3, 64] : tensor<1x1504x3x64xf16, {order = #NHWC}> to tensor<1x1500x3x64xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>

   return %3 : tensor<1x192x1504x1xf16>

   // Enlarged Slice becomes identity (size = source) and is canonicalized away
   // CHECK-NOT:    IE.Slice
   // CHECK-NOT:    IE.Expand
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1504x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1504xf16>
   // CHECK:        [[RESHAPE:%.+]] = IE.ShapeCast {shape = [1, 192, 1504, 1]} inputs([[PERMUTECAST]] : tensor<1x3x64x1504xf16>) -> tensor<1x192x1504x1xf16>
   // CHECK:        return [[RESHAPE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @OptimizeSliceViewLikeChainExpand_NotEnoughRoom
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1502x3x64xf16, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_NotEnoughRoom(%arg0: tensor<1x1502x3x64xf16, {order = #NHWC}>) -> tensor<1x192x1504x1xf16> {
   // Source has 1502 on C-axis, slice takes 1500, but enlarging by 4 would need 1504 > 1502
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 3, 64] : tensor<1x1502x3x64xf16, {order = #NHWC}> to tensor<1x1500x3x64xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>

   return %3 : tensor<1x192x1504x1xf16>

   // Pattern should NOT match: source (1502) cannot cover the full extent (need 1504)
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1500, 3, 64]
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   // CHECK:        [[RESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[RESHAPE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>
   // CHECK:        return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @OptimizeSliceViewLikeChainExpand_NotFullSourceExtent
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1508x3x64xf16, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_NotFullSourceExtent(%arg0: tensor<1x1508x3x64xf16, {order = #NHWC}>) -> tensor<1x192x1504x1xf16> {
   // Source has 1508 on C-axis, slice takes 1500, enlarging by 4 gives 1504 which fits
   // but 1504 != 1508 so enlarged Slice does NOT cover the full source extent
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 3, 64] : tensor<1x1508x3x64xf16, {order = #NHWC}> to tensor<1x1500x3x64xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>

   return %3 : tensor<1x192x1504x1xf16>

   // Pattern should NOT match: enlarged Slice (1504) does not cover full source (1508)
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1500, 3, 64]
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   // CHECK:        [[RESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[RESHAPE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>
   // CHECK:        return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @OptimizeSliceViewLikeChainExpand_NonZeroOffset
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x3x64xf16, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_NonZeroOffset(%arg0: tensor<1x1504x3x64xf16, {order = #NHWC}>) -> tensor<1x192x1504x1xf16> {
   // Slice offset is non-zero (4), so enlarged Slice [4:1504] does not start at 0
   %0 = IE.Slice %arg0 [0, 4, 0, 0] [1, 1500, 3, 64] : tensor<1x1504x3x64xf16, {order = #NHWC}> to tensor<1x1500x3x64xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>

   return %3 : tensor<1x192x1504x1xf16>

   // Pattern should NOT match: slice offset is non-zero, enlarged Slice is still a subview
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 4, 0, 0] [1, 1500, 3, 64]
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1500x3x64xf16, {order = #NHWC}> -> tensor<1x3x64x1500xf16>
   // CHECK:        [[RESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]])
   // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 192, 1500, 1]} : tensor<1x3x64x1500xf16> -> tensor<1x192x1500x1xf16>
   // CHECK:        [[EXPAND:%.+]] = IE.Expand([[RESHAPE]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 4, 0]} : tensor<1x192x1500x1xf16> -> tensor<1x192x1504x1xf16>
   // CHECK:        return [[EXPAND]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @OptimizeSliceViewLikeChainExpand_SingleViewLikeNotMatched
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1504x20x16xf16, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_SingleViewLikeNotMatched(%arg0: tensor<1x1504x20x16xf16, {order = #NHWC}>) -> tensor<1x16x20x1502xf16, {order = #NHCW}> {
   // Only one view-like op (PermuteCast): handled by OptimizeSlicePermuteCastExpand, not this pattern
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1500, 20, 16] : tensor<1x1504x20x16xf16, {order = #NHWC}> to tensor<1x1500x20x16xf16, {order = #NHWC}>
   %1 = IE.PermuteCast(%0) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1500x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1500xf16, {order = #NHCW}>
   %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x16x20x1500xf16, {order = #NHCW}> -> tensor<1x16x20x1502xf16, {order = #NHCW}>

   return %2 : tensor<1x16x20x1502xf16, {order = #NHCW}>

   // This triggers OptimizeSlicePermuteCastExpand (single PermuteCast), not the chain pattern
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1502, 20, 16] : tensor<1x1504x20x16xf16, {order = #NHWC}> to tensor<1x1502x20x16xf16, {order = #NHWC}>
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x1502x20x16xf16, {order = #NHWC}> -> tensor<1x16x20x1502xf16, {order = #NHCW}>
   // CHECK:        return [[PERMUTECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#HNWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.032154855983597894>
!qElemType1 = !quant.uniform<f8E4M3FN:f16, 0.012785908366952623>

// CHECK: [[QELEMTYPE0:!.+]] = !quant.uniform<f8E4M3FN:f16, 0.032154855983597894>
// CHECK: [[QELEMTYPE1:!.+]] = !quant.uniform<f8E4M3FN:f16, 0.012785908366952623>

// CHECK: @OptimizeSliceViewLikeChainExpand_QuantizeCast
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
func.func @OptimizeSliceViewLikeChainExpand_QuantizeCast(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>) -> tensor<1x48x4096x1x!qElemType1, {order = #NHWC}> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4x!qElemType, {order = #NHWC}> to tensor<1x40x1024x4x!qElemType, {order = #NHWC}>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4x!qElemType, {order = #NHWC}> -> tensor<1x40x4096x1x!qElemType, {order = #NHWC}>
   %2 = IE.PermuteCast(%1) {dst_order = #HNWC, mem_perm = #NCHW} : tensor<1x40x4096x1x!qElemType, {order = #NHWC}> -> tensor<4096x40x1x1x!qElemType, {order = #HNWC}>
   %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType1} : tensor<4096x40x1x1x!qElemType, {order = #HNWC}> -> tensor<4096x40x1x1x!qElemType1, {order = #HNWC}>
   %4 = IE.PermuteCast(%3) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<4096x40x1x1x!qElemType1, {order = #HNWC}> -> tensor<1x40x4096x1x!qElemType1, {order = #NHWC}>
   %5 = IE.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<1x40x4096x1x!qElemType1, {order = #NHWC}> -> tensor<1x48x4096x1x!qElemType1, {order = #NHWC}>

   return %5 : tensor<1x48x4096x1x!qElemType1, {order = #NHWC}>

   // CHECK-NOT:    IE.Slice
   // CHECK-NOT:    IE.Expand
   // CHECK: [[PERMUTECAST0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x48x1024x4x[[QELEMTYPE0]], {order = #NHWC}> -> tensor<1x1024x4x48x[[QELEMTYPE0]]>
   // CHECK: [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 4096, 1, 48]} inputs([[PERMUTECAST0]] : tensor<1x1024x4x48x[[QELEMTYPE0]]>) -> tensor<1x4096x1x48x[[QELEMTYPE0]]>
   // CHECK: [[PERMUTECAST1:%.+]] = IE.PermuteCast([[SHAPECAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4096x1x48x[[QELEMTYPE0]]> -> tensor<1x48x4096x1x[[QELEMTYPE0]], {order = #NHWC}>
   // CHECK: [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[PERMUTECAST1]]) {dstElemType = [[QELEMTYPE1]]} : tensor<1x48x4096x1x[[QELEMTYPE0]], {order = #NHWC}> -> tensor<1x48x4096x1x[[QELEMTYPE1]], {order = #NHWC}>

   // CHECK: return [[QUANTIZECAST]]
}
