//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>
// CHECK-LABEL: @SingleQuantParams
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SingleQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.078431372549019607:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @DifferentQuantParams
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @DifferentQuantParams(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u16:f32, 1.000000e+00>
// CHECK-LABEL: @SplitFakeQuantU16
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @SplitFakeQuantU16(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<65535.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<65535.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.039215686274509803>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @OneDifferentQuantParam
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @OneDifferentQuantParam(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.011764705882352941:85>

// CHECK-LABEL: @BroadcastQuantParam
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x2x30x30xf32>)
func.func @BroadcastQuantParam(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.011764705882352941:85>

// CHECK-LABEL: @BroadcastQuantParamDiffRanks
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x2x30x30xf32>)
func.func @BroadcastQuantParamDiffRanks(%arg0: tensor<1x2x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %input_low = const.Declare tensor<1xf32> = dense<-1.0> : tensor<1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[2.0]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x2x30x30xf32> ->
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x2x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x2x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 0.078431372549019607:128>

// CHECK-LABEL: @UseDequantize
func.func @UseDequantize() -> tensor<1x3x30x30xf32> {
    %input = const.Declare tensor<1x3x30x30xf32> =
        dense<10> : tensor<1x3x30x30xui8>, [#const.CastElemType<f32>]

    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-10.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<10.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x3x30x30x!qElemType> =
    // CHECK-SAME:      dense<10> : tensor<1x3x30x30xui8>, [#const.CastElemType<f32>, #const.CastElemType<!qElemType>]

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>
    // CHECK-SAME:      -> tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK-LABEL: @UseRescale
func.func @UseRescale() -> tensor<1x2x30x30xf32> {
    %input = const.Declare tensor<1x2x30x30xf32> = dense<1.0> : tensor<1x2x30x30xf32>
    %input_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-2.0]],[[-1.0]]]]> : tensor<1x2x1x1xf32>
    %input_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[2.0]],[[1.0]]]]> : tensor<1x2x1x1xf32>
    %output_low = const.Declare tensor<1x2x1x1xf32> = dense<[[[[-1.0]],[[-0.5]]]]> : tensor<1x2x1x1xf32>
    %output_high = const.Declare tensor<1x2x1x1xf32> = dense<[[[[1.0]],[[0.5]]]]> : tensor<1x2x1x1xf32>

    %0 = IE.FakeQuantize(%input, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x2x30x30xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x30x30xf32>

    return %0 : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL0:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>

    // CHECK-DAG:       [[VAL1:%.*]] = const.Declare tensor<1x2x30x30xf32> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x30x30xf32>, [#const.Rescale<2.000000e+00 : f64>]

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8<0:1>:f32, 1.000000e+00>

// CHECK-LABEL: @Level2Quantization
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Level2Quantization(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 2 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar0AsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Scalar0AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>

    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Dequantize([[VAL0]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 7.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @Scalar7AsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @Scalar7AsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<7.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f32, 4.000000e+00:2>
// CHECK: !qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NegativeScalarAsInputRange
// CHECK-SAME:       ([[INPUT:%.+]]: tensor<1x3x30x30xf32>)
func.func @NegativeScalarAsInputRange(%arg0: tensor<1x3x30x30xf32>) -> tensor<1x3x30x30xf32> {
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<-4.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x3x30x30xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x30x30xf32>

    return %0 : tensor<1x3x30x30xf32>


    // CHECK:       [[VAL0:%.*]] = IE.Quantize([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType}
    // CHECK-SAME:      tensor<1x3x30x30xf32> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]])
    // CHECK-SAME:      {dstElemType = !qElemType1}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType> ->
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1>

    // CHECK:       [[VAL2:%.*]] = IE.Dequantize([[VAL1]])
    // CHECK-SAME:      {dstElemType = f32}
    // CHECK-SAME:      tensor<1x3x30x30x!qElemType1> ->
    // CHECK-SAME:      tensor<1x3x30x30xf32>

    // CHECK:       return [[VAL2]]
}

// -----

// CHECK-LABEL: @PerChannelQuant
func.func @PerChannelQuant(%arg0: tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32> {
    %input_low = const.Declare tensor<1x16x1x1xf32> =
        dense<[[[[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]],
        [[0.000000e+00]], [[-0.297849566]], [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]],
        [[0.000000e+00]], [[-0.381789744]], [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %input_high = const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]], [[23.3847122]],
        [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]], [[12.3310165]],
        [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]], [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>
    %output_low = const.Declare tensor<1x16x1x1xf32> = dense<[[[[0.000000e+00]], [[0.000000e+00]],
        [[0.000000e+00]], [[0.000000e+00]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.297849566]],
        [[-0.382785916]], [[-0.399385154]], [[-3.750000e-01]], [[0.000000e+00]], [[-0.381789744]],
        [[0.000000e+00]], [[-3.750000e-01]], [[-3.750000e-01]], [[-0.389199734]]]]> : tensor<1x16x1x1xf32>
    %output_high =  const.Declare tensor<1x16x1x1xf32> = dense<[[[[2.88798332]], [[17.5819988]],
        [[23.3847122]], [[6.1077733]], [[1.875000e+01]], [[3.81057024]], [[0.192161009]], [[13.5615578]],
        [[12.3310165]], [[3.609375]], [[2.681180e+00]], [[3.0952239]], [[3.04886699]], [[1.556250e+01]],
        [[2.70967746]], [[8.63315773]]]]> : tensor<1x16x1x1xf32>

    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} :
        tensor<1x16x112x112xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x112x112xf32>

    return %fq : tensor<1x16x112x112xf32>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK-LABEL: @PerChannelQuantInput
func.func @PerChannelQuantInput(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299xf16> {
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<2.63867188> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.65820313> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %input_high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[2.551250e+02]], [[2.670000e+02]], [[2.780000e+02]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>]
    %input_low =  const.Declare tensor<1x3x1x1xf16> = dense<[[[[-49.28125]], [[-35.65625]], [[-31.828125]]]]> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>]

    %1 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x299x299xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x299x299xf16>

    return %1 : tensor<1x3x299x299xf16>

    // CHECK:       IE.FakeQuantize
    // CHECK-NOT:       IE.Quantize
    // CHECK-NOT:       IE.Dequantize
    // CHECK-NOT:       IE.QuantizeCast
}

// -----

// CHECK: !qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>
// CHECK-LABEL: @ConstantsSplitFakeQuant
func.func @ConstantsSplitFakeQuant(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x16x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.Quantize<!qElemType>, #const.CastElemType<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i4:f16, 1.3385416666666667>
// CHECK-LABEL: @I4ConstantsSplitFakeQuant
func.func @I4ConstantsSplitFakeQuant(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<16x1x1x1xf16> = dense<-1.007810e+01> : tensor<16x1x1x1xf16>
    %cst_3 = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+01> : tensor<16x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<16x16x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>
    return %1 : tensor<1x16x16x16xf16>
    // CHECK-NOT:       IE.FakeQuantize
    // CHECK:       [[Q_CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType> = dense<0.000000e+00> : tensor<16x16x1x1xf16>,
    // CHECK-SAME:      [#const.CastElemType<!qElemType>]
    // CHECK:       [[DQ_CST:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16}
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[ARG0:%.*]], [[DQ_CST]])
    // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i8:f16, 0.39215686274509803:127>
// CHECK:       !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:127>
// CHECK-LABEL: @BringConstIn8BitRepresentableFormat
func.func @BringConstIn8BitRepresentableFormat() -> tensor<1x3x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+02> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x3x1x1xf16> = dense<[[[-6.90000e+01]], [[-2.500000e+02]], [[-2.300000e+02]]]> : tensor<3x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 3, 1, 1]>]
    %0 = IE.FakeQuantize(%cst_2, %cst_1, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
    return %0 : tensor<1x3x1x1xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[Q_CST:%.*]] = const.Declare tensor<1x3x1x1x!qElemType> =
    // CHECK-SAME{LITERAL}:      dense<[[[-6.900000e+01]], [[-2.500000e+02]], [[-2.300000e+02]]]> : tensor<3x1x1xf32>
    // CHECK-SAME:      [#const.CastElemType<f16>, #const.Reshape<[1, 3, 1, 1]>, #const.Quantize<!qElemType1>, #const.CastElemType<!qElemType>]
    // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>

    //CHECK: return [[DEQUANTIZE]] : tensor<1x3x1x1xf16>
}

// -----

// CHECK:       !qElemType = !quant.uniform<i8:f16, 0.0028435202205882352:127>
// CHECK:       !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:127>
// CHECK-LABEL: @BringConstIn8BitRepresentableFormatWithBroadcast
func.func @BringConstIn8BitRepresentableFormatWithBroadcast() -> tensor<1x12x1x1xf16> {
    %cst_0 = const.Declare tensor<1x12x1x1xf16> = dense<2.300000e+02> : tensor<1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>, #const.Rescale<-1.000000e+00 : f64>, #const.Broadcast<1 : i64, 12 : i64>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-7.252680e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%cst_0, %cst_3, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x12x1x1xf16>
    return %0 : tensor<1x12x1x1xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[Q_CST:%.*]] = const.Declare tensor<1x12x1x1x!qElemType> =
    // CHECK-SAME:      dense<2.300000e+02> : tensor<1xf32>
    // CHECK-SAME:      [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>, #const.Rescale<-1.000000e+00 : f64>, #const.Broadcast<1 : i64, 12 : i64>, #const.Quantize<!qElemType1>, #const.CastElemType<!qElemType>]
    // CHECK: [[DEQUANTIZE:%.*]] = IE.Dequantize([[Q_CST]]) {dstElemType = f16} : tensor<1x12x1x1x!qElemType> -> tensor<1x12x1x1xf16>
    //CHECK: return [[DEQUANTIZE]] : tensor<1x12x1x1xf16>
}


// -----

// CHECK-LABEL: @NonConstNoSplitFakeQuantForMultiZP
func.func @NonConstNoSplitFakeQuantForMultiZP(%arg0: tensor<1x3x16x16xf16>, %arg1: tensor<3x3x1x1xf16>) -> tensor<1x3x16x16xf16> {
    %cst_low = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]]]> : tensor<3x1x1x1xf16>
    %cst_high = const.Declare tensor<3x1x1x1xf16> = dense<[[[[2.0]]], [[[1.2]]], [[[1.5]]]]> : tensor<3x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg1, %cst_low, %cst_high, %cst_low, %cst_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<3x3x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare
    // CHECK:           [[FQ:%.*]] = IE.FakeQuantize
    // CHECK:           [[CONV:%.*]] = IE.Convolution
    // CHECK:       return [[CONV]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.48553921568627451:127>
// CHECK: !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:127>

// CHECK-LABEL: @GarbageFq
// CHECK-SAME: -> tensor<1x3x1x1xf16>
func.func @GarbageFq() -> tensor<1x3x1x1xf16> {
    %data = const.Declare tensor<1x3x1x1xf16> = dense<[[[8]], [[249]], [[222]]]> : tensor<3x1x1xui8>,
    [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>,
     #const.Rescale<-1.000000e+00 : f64>]

    %in_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
    %out_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.238130e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>

    // Note: our compiler is able to generate FQ operations with invalid ranges
    // (where FQ maps [-255; 0] to [-123; 0]). however, this pass currently
    // deals with it "specially".

    %fq_with_bad_input_range = IE.FakeQuantize(%data, %in_low, %high, %out_low, %high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
        : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x3x1x1xf16>

    return %fq_with_bad_input_range : tensor<1x3x1x1xf16>

    // CHECK: [[NEW_CONST:%.+]] = const.Declare tensor<1x3x1x1x!qElemType>
    // CHECK-SAME{LITERAL}: dense<[[[8]], [[249]], [[222]]]>
    // CHECK-SAME: [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f16>, #const.Rescale<-1.000000e+00 : f64>, #const.Quantize<!qElemType1>, #const.CastElemType<!qElemType>]

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[NEW_CONST]]) {dstElemType = f16}
    // CHECK: return [[DQ]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002222"
    }
  }
#-}

!quantFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0}>

// CHECK: !qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}:0.02211761474609375>

// CHECK-LABEL: @NF4ConstantsSplitFakeQuant
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @NF4ConstantsSplitFakeQuant(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-8.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<7.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.359010e-02> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.064510e-02> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<!quantFloatType>, #const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = !quantFloatType} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    return %1 : tensor<1x1x29x29xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2xui4> isSplat, [#const.ConvertElemType<ui8>, #const.CastElemType<!QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-8.000000e-01,-0.69999999999999996,-6.000000e-01,-5.000000e-01,-4.000000e-01,-3.000000e-01,0.000000e+00,1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01,6.000000e-01,0.69999999999999996,1.000000e+00}>>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DQ_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    // CHECK: return [[CONV]]
}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16, 0.002948760986328125>

// CHECK-LABEL: @I2ConstantsSplitFakeQuant
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @I2ConstantsSplitFakeQuant(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-2.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-5.897520e-03> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    return %1 : tensor<1x1x29x29xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DQ_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    // CHECK: return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001B"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 0.002948760986328125:2>

// CHECK-LABEL: @U2ConstantsSplitFakeQuant
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @U2ConstantsSplitFakeQuant(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-5.897520e-03> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    return %1 : tensor<1x1x29x29xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DQ_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    // CHECK: return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001B"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<i2:f16, 0.002948760986328125>

// CHECK-LABEL: @I2TernaryWeightsConstantsSplitFakeQuant
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @I2TernaryWeightsConstantsSplitFakeQuant(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-2.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-5.897520e-03> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<2.948760e-03> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    return %1 : tensor<1x1x29x29xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2xsi2>, [#const.ConvertElemType<si8>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DQ_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    // CHECK: return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000013"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<u2:f16, 0.002948760986328125:1>

// CHECK-LABEL: @U2TernaryWeightsConstantsSplitFakeQuant
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x28x28xf16>
func.func @U2TernaryWeightsConstantsSplitFakeQuant(%arg0: tensor<1x1x28x28xf16>) -> tensor<1x1x29x29xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-2.948760e-03> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<5.897520e-03> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x2x2xf16> = dense_resource<blob> : tensor<1x1x2x2xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 4 : i64} : tensor<1x1x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    return %1 : tensor<1x1x29x29xf16>

    // CHECK-NOT:       IE.FakeQuantize
    // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x2x2x!qElemType> = dense_resource<blob> : tensor<1x1x2x2xui2>, [#const.ConvertElemType<ui8>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_CST:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16}
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DQ_CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1x28x28xf16>, tensor<1x1x2x2xf16> -> tensor<1x1x29x29xf16>
    // CHECK: return [[CONV]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000012"
    }
  }
#-}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16:0, {-0.0010833141850490197,6.7138671875E-4,-6.371591605392157E-4}>

// CHECK-LABEL: @NegativeScalesSplitFakeQuant
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3xf16>
func.func @NegativeScalesSplitFakeQuant(%arg0: tensor<1x3xf16>) -> tensor<1x3xf16> {
    %cst = const.Declare tensor<1x1xf16> = dense<-1.280000e+02> : tensor<1x1xf16>
    %cst_0 = const.Declare tensor<1x1xf16> = dense<1.270000e+02> : tensor<1x1xf16>
    %cst_1 = const.Declare tensor<3x1xf16> = dense<[[1.386720e-01], [-8.593750e-02], [8.154300e-02]]> : tensor<3x1xf16>
    %cst_2 = const.Declare tensor<3x1xf16> = dense<[[-1.375730e-01], [8.526610e-02], [-8.093260e-02]]> : tensor<3x1xf16>
    %cst_3 = const.Declare tensor<3x3xf16> = dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<3x1xf16>, tensor<3x1xf16> -> tensor<3x3xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
    return %1 : tensor<1x3xf16>

    // CHECK: [[WT:%.+]] = const.Declare tensor<3x3x!qElemType>
    // CHECK-SAME{LITERAL}: dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_WT:%.+]] = IE.Dequantize([[WT]]) {dstElemType = f16} : tensor<3x3x!qElemType> -> tensor<3x3xf16>
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[DQ_WT]]) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
    // CHECK: return [[FC]] : tensor<1x3xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {-2.22213e-18,2.77587e-17,-1.61586e-16}>

// CHECK-LABEL: @VerySmallScalesSplitFakeQuant
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3xf16>
func.func @VerySmallScalesSplitFakeQuant(%arg0: tensor<1x3xf16>) -> tensor<1x3xf16> {
    %cst = const.Declare tensor<1x1xf16> = dense<-1.280000e+02> : tensor<1x1xf16>
    %cst_0 = const.Declare tensor<1x1xf16> = dense<1.270000e+02> : tensor<1x1xf16>
    %cst_1 = const.Declare tensor<3x1xf16> = dense<[[2.84432e-16], [-3.55311e-15], [2.0683e-14]]> : tensor<3x1xf16>
    %cst_2 = const.Declare tensor<3x1xf16> = dense<[[-2.8221e-16], [3.52535e-15], [-2.05215e-14]]> : tensor<3x1xf16>
    %cst_3 = const.Declare tensor<3x3xf16> = dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%cst_3, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<3x3xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<3x1xf16>, tensor<3x1xf16> -> tensor<3x3xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
    return %1 : tensor<1x3xf16>

    // CHECK: [[WT:%.+]] = const.Declare tensor<3x3x!qElemType>
    // CHECK-SAME{LITERAL}: dense<[[64, 63, 112], [-8, 62, -8], [8, 63, 16]]> : tensor<3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ_WT:%.+]] = IE.Dequantize([[WT]]) {dstElemType = f16} : tensor<3x3x!qElemType> -> tensor<3x3xf16>
    // CHECK: [[FC:%.+]] = IE.FullyConnected([[INPUT]], [[DQ_WT]]) : tensor<1x3xf16>, tensor<3x3xf16> -> tensor<1x3xf16>
    // CHECK: return [[FC]] : tensor<1x3xf16>
}

// -----

// CHECK-LABEL: @NotSplitFQWithNonStaticLowHighValues
func.func @NotSplitFQWithNonStaticLowHighValues(%arg0: tensor<1x256x2048x1xf16>, %arg1: tensor<1x1x1xf16>, %arg2: tensor<1x1x1xf16>, %arg3: tensor<1x1x1xf16>, %arg4: tensor<1x1x1xf16>) -> tensor<1x256x2048x1xf16> {
    %input_low = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %input_high = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %output_low = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    %output_high = IE.AffineReshape(%arg4) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x256x2048x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x256x2048x1xf16>

    return %0 : tensor<1x256x2048x1xf16>

    // CHECK:           [[IN_LOW:%.+]] = IE.AffineReshape
    // CHECK:           [[IN_HIGH:%.+]] = IE.AffineReshape
    // CHECK:           [[OUT_LOW:%.+]] = IE.AffineReshape
    // CHECK:           [[OUT_HIGH:%.+]] = IE.AffineReshape
    // CHECK:           [[FQ:%.+]] = IE.FakeQuantize

    // CHECK:           return [[FQ]]
}
