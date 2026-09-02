//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @Conv2dLeakyReluWithQuantize
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @Conv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x3x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<!QuantileType.quantile<ui4:ui8, {0.0,16.0,32.0,48.0,64.0,80.0,96.0,112.0,128.0,144.0,160.0,176.0,192.0,208.0,224.0,240.0}>:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK: !qElemType = !quant.uniform<!QuantileType.quantile<ui4:ui8, {0.000000e+00,1.600000e+01,3.200000e+01,4.800000e+01,6.400000e+01,8.000000e+01,9.600000e+01,1.120000e+02,1.280000e+02,1.440000e+02,1.600000e+02,1.760000e+02,1.920000e+02,2.080000e+02,2.240000e+02,2.400000e+02}>:f16, 1.1534313725490195:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionConvQuantile
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>)
func.func @MixedPrecisionConvQuantile(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  //CHECK: [[CONV:%.+]] = IE.Convolution([[QUANT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType1>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<i8:f16:0, {0.1085171568627451,0.0043868719362745098,0.0011484183517156863,0.0015251608455882353,-0.023115808823529413,0.0092486213235294118,-0.0024605545343137254,-5.5218864889705881E-4,0.022280943627450981,-0.0096047794117647065,-0.025742953431372548,0.01208639705882353,4.7164617800245099E-4,-0.022916666666666665,0.01014859068627451,-0.020687806372549019,1.1920928955078125E-7,-9.0451708026960788E-4,-0.0028301164215686274,0.020657169117647058,0.029725796568627449,0.0014466528799019608,0.12061887254901961,6.4505782781862744E-4,0.0023399203431372548,0.003393075980392157,0.0095818014705882359,0.013534007352941177,-0.010497089460784313,0.011251531862745098,0.025314031862745098,0.02688419117647059}>

// CHECK-LABEL: @AvoidMixedPrecisionForConvWithPostOpReluAndNegativeScales
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x448x448xf32>)
func.func @AvoidMixedPrecisionForConvWithPostOpReluAndNegativeScales(%arg0: tensor<1x3x448x448xf32>) -> tensor<1x32x224x224xf32> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<1.0> : tensor<1x32x1x1xf16>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<32x3x3x3x!qElemType> = dense<1> : tensor<32x3x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
    %0 = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<32x3x3x3x!qElemType> -> tensor<32x3x3x3xf16>
    %1 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x448x448xf32> -> tensor<1x3x448x448xf16>
    %2 = IE.Convolution(%1, %0, %cst) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        post_op = #IE.Relu<>,
        strides = [2, 2]
    } : tensor<1x3x448x448xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x224x224xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x32x224x224xf16> -> tensor<1x32x224x224xf32>
    return %3 : tensor<1x32x224x224xf32>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>, [#const.CastElemType<f16>]
    //CHECK: [[CST_0:%.+]] = const.Declare tensor<32x3x3x3x!qElemType> = dense<1> : tensor<32x3x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<32x3x3x3x!qElemType> -> tensor<32x3x3x3xf16>
    //CHECK: [[VAL1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x448x448xf32> -> tensor<1x3x448x448xf16>
    //CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], post_op = #IE.Relu<>, strides = [2, 2]} : tensor<1x3x448x448xf16>, tensor<32x3x3x3xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x224x224xf16>
    //CHECK: [[VAL3:%.+]] = IE.Convert([[VAL2]]) {dstElemType = f32} : tensor<1x32x224x224xf16> -> tensor<1x32x224x224xf32>
    //CHECK: return [[VAL3]] : tensor<1x32x224x224xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @DoNotConvertConvWithLeakyRelu
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @DoNotConvertConvWithLeakyRelu(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>, #const.CastElemType<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    return %4 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[DEQUANT_WEIGHTS:%.+]] = IE.Dequantize([[WEIGHTS]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[DEQUANT_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @DoNotConvertConvWithLeakyReluConsumer
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @DoNotConvertConvWithLeakyReluConsumer(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>, #const.CastElemType<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %5 = IE.LeakyRelu(%4) {negative_slope = 2.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %5 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[DEQUANT_WEIGHTS:%.+]] = IE.Dequantize([[WEIGHTS]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[DEQUANT_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    // CHECK: [[LEAKY:%.+]] = IE.LeakyRelu([[CONV]]) {negative_slope = 2.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[LEAKY]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.28679281496534159>

// CHECK-LABEL: @FuseQuantizeIntoMultiply
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x16x4x4xf16>
func.func @FuseQuantizeIntoMultiply(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    return %1 : tensor<1x16x4x4x!qElemType>

    // CHECK-NOT: IE.Quantize
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
    // CHECK-SAME:     : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    // CHECK: return [[MULTIPLY]]
}

// -----

!qElemType = !quant.uniform<i8:f16:1, {0.10000000000000001,0.20000000000000001,0.30000000000000004}>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyPerAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x3x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x3x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyPerAxis(%arg0: tensor<1x3x4x4xf16>, %arg1: tensor<1x3x4x4xf16>) -> tensor<1x3x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x3x4x4xf16>, tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x4x4xf16> -> tensor<1x3x4x4x!qElemType>
    return %1 : tensor<1x3x4x4x!qElemType>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]])
    // CHECK-SAME:     : tensor<1x3x4x4xf16>, tensor<1x3x4x4xf16> -> tensor<1x3x4x4xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK: return [[QUANTIZE]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.28679281496534159>
!qElemType1 = !quant.uniform<u8:f16, 0.081176862529679844:128>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyMultipleConsumers
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x16x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyMultipleConsumers(%arg0: tensor<1x16x4x4xf16>, %arg1: tensor<1x16x4x4xf16>) -> (tensor<1x16x4x4x!qElemType>, tensor<1x16x4x4x!qElemType1>) {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>}
        : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType>
    %2 = IE.Quantize(%0) {dstElemType = !qElemType1} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4x!qElemType1>
    return %1, %2 : tensor<1x16x4x4x!qElemType>, tensor<1x16x4x4x!qElemType1>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]])
    // CHECK-SAME:     : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
    // CHECK: [[Q1:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK: [[Q2:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType1}
    // CHECK: return [[Q1]], [[Q2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0013184806879828958:128>

// CHECK-LABEL: @DoNotFuseQuantizeIntoMultiplyBroadcastInputs
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x1x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<1x4x4x4xf16>
func.func @DoNotFuseQuantizeIntoMultiplyBroadcastInputs(%arg0: tensor<1x1x4x4xf16>,
                                                        %arg1: tensor<1x4x4x4xf16>) -> tensor<1x4x4x4x!qElemType> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1x1x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x4x4x4xf16> -> tensor<1x4x4x4x!qElemType>
    return %1 : tensor<1x4x4x4x!qElemType>

    // CHECK-NOT: IE.Multiply{{.*}}-> tensor<1x4x4x4x!qElemType>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG_0]], [[ARG_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:     : tensor<1x1x4x4xf16>, tensor<1x4x4x4xf16> -> tensor<1x4x4x4xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[MULTIPLY]]) {dstElemType = !qElemType}
    // CHECK-SAME:     : tensor<1x4x4x4xf16> -> tensor<1x4x4x4x!qElemType>
    // CHECK: return [[QUANTIZE]]
}
