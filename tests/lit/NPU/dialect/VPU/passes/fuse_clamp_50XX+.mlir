//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-clamp --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp32to128
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClamp32to128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <LPRELU>,
            clamp_low = 0.0 : f64,
            clamp_high = 255.0 : f64,
            prelu_alpha = [1.250000e-01],
            adder = 0.0 : f64>
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 3.200000e+01 : f64
    } : tensor<1x256x56x56x!qElemType, {order = #NHWC}> -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %1 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.250000e-01],
    // CHECK-SAME:          adder = 0.000000e+00 : f64>
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to120
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @FloatClamp0to120(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <LPRELU>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 3.4028234663852886E+38 : f64,
            prelu_alpha = [1.250000e-01],
            adder = 0.0 : f64>
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.200000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.250000e-01],
    // CHECK-SAME:          adder = 0.000000e+00 : f64>
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to128
// CHECK-SAME:  ([[INPUT_0:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%.+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @FloatClamp0to128(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <LPRELU>,
            clamp_low = -3.4028234663852886E+38 : f64,
            clamp_high = 22400.0 : f64,
            prelu_alpha = [1.250000e-01],
            adder = 0.0 : f64>
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x256x56x56xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Clamp
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.250000e-01]
    // CHECK-SAME:          adder = 0.000000e+00 : f64>
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 13.571496821384804:117>

// CHECK-LABEL: @ConvWithMultipleConsumers
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>)
func.func @ConvWithMultipleConsumers(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> (tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>) {
    %weights = const.Declare tensor<9216x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
            clamp_low = 0.0 : f64,
            clamp_high = 255.0 : f64,
            prelu_alpha = [1.000000e+00],
            adder = 0.0 : f64>,
        rawFilterShape = [9216, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<9216x32x3x3x!qElemType, {order = #NHWC}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 4608, 16, 16] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> to tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>
    %2 = VPU.Clamp(%0) {
        max = 1.280000e+02 : f64,
        min = 0.000000e+00 : f64
    } : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %1, %2 : tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:  [[WEIGHTS:%.+]] = const.Declare tensor<9216x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK:      [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME: {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <NOOP>,
    // CHECK-SAME:      clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:      clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:      prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      adder = 0.000000e+00 : f64>
    // CHECK-SAME: rawFilterShape = [9216, 32, 3, 3],
    // CHECK-SAME: strides = [1, 1]}
    // CHECK-SAME: -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:      [[SLICE:%.+]] = VPU.Slice [[CONV_0]] [0, 0, 0, 0] [1, 4608, 16, 16] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> to tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>
    // CHECK:      [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME: {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:  ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <NOOP>,
    // CHECK-SAME:      clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:      clamp_high = 9.4315315167232363 : f64,
    // CHECK-SAME:      prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      adder = 0.000000e+00 : f64>
    // CHECK-SAME:  rawFilterShape = [9216, 32, 3, 3],
    // CHECK-SAME:  strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:      return [[SLICE]], [[CONV_1]] : tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>, tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
}
