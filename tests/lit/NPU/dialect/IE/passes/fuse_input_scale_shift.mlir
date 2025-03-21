//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" -mlir-print-elementsattrs-with-hex-if-larger=-1 --fuse-input-scale-shift --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: func.func @SplatScales([[ARG0:%.+]]: tensor<1x3x224x224xf32>)
func.func @SplatScales(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x8x112x112xf32> {
    %scales = const.Declare tensor<1x3x1x1xf32> = dense<0.0174255371> : tensor<1x3x1x1xf32>
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Multiply(%arg0, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK-DAG:               [[WEIGHTS_OUT_HIHG:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.0483667552]]], [[[0.00282828277]]], [[[0.0404081456]]], [[[0.00188446045]]], [[[0.00188446045]]], [[[0.016109433]]], [[[0.00977523718]]], [[[0.0193486288]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS_OUT_LOW:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[-0.00156653463]]], [[[-9.1604299E-5]]], [[[-0.00130876584]]], [[[-6.10351563E-5]]], [[[-6.10351563E-5]]], [[[-5.217630e-04]]], [[[-3.16606864E-4]]], [[[-6.2667625E-4]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[2.550000e+02, 2.550000e+02, 2.550000e+02]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.362888545]], [[-0.144634902]], [[0.441007137]], [[-0.0820473805]], [[-0.0959633961]], [[0.637237727]], [[0.349784434]], [[0.209193677]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.45700073> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.98651123> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[WEIGHTS_OUT_LOW]], [[WEIGHTS_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
    // CHECK:                   [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    // CHECK:                   return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

// CHECK: func.func @DifferenScales([[ARG0:%.+]]: tensor<1x3x224x224xf32>)
func.func @DifferenScales(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x8x112x112xf32> {
    %scales = const.Declare tensor<1x3x1x1xf32> = dense<[[[[0.0174255371]], [[0.0175018311]], [[0.0170593262]]]]> : tensor<1x3x1x1xf32>
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Multiply(%arg0, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK-DAG:               [[WEIGHTS_OUT_HIHG:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.0488494299]]], [[[0.00285650766]]], [[[0.0408113971]]], [[[0.00188446045]]], [[[0.00188446045]]], [[[0.0162701979]]], [[[0.00987278856]]], [[[0.0195417162]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS_OUT_LOW:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[-0.00158216781]]], [[[-9.25184649E-5]]], [[[-0.0013218266]]], [[[-6.10351563E-5]]], [[[-6.10351563E-5]]], [[[-5.26969961E-4]]], [[[-3.19766434E-4]]], [[[-6.32930081E-4]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[2.540000e+02, 2.540000e+02, 2.540000e+02]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.348501623]], [[-0.145476192]], [[0.428987533]], [[-0.0821912512]], [[-0.0961072668]], [[0.632445871]], [[0.34687674]], [[0.203438342]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.44337463> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.97549438> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[WEIGHTS_OUT_LOW]], [[WEIGHTS_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
    // CHECK:                   [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    // CHECK:                   return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

// CHECK: func.func @NoScales([[ARG0:%.+]]: tensor<1x3x224x224xf32>)
func.func @NoScales(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x8x112x112xf32> {
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Add(%arg0, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.FakeQuantize(%1, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.Convolution(%2, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %4 = IE.Add(%3, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %4 : tensor<1x8x112x112xf32>


    // CHECK-DAG:               [[WEIGHTS_OUT_HIHG:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.0483667552]]], [[[0.00282828277]]], [[[0.0404081456]]], [[[0.00188446045]]], [[[0.00188446045]]], [[[0.016109433]]], [[[0.00977523718]]], [[[0.0193486288]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS_OUT_LOW:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[-0.00156653463]]], [[[-9.1604299E-5]]], [[[-0.00130876584]]], [[[-6.10351563E-5]]], [[[-6.10351563E-5]]], [[[-5.217630e-04]]], [[[-3.16606864E-4]]], [[[-6.2667625E-4]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[2.550000e+02, 2.550000e+02, 2.550000e+02]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.380503565]], [[-0.143604845]], [[0.455723643]], [[-0.0818712338]], [[-0.0957872495]], [[0.643104672]], [[0.35334453]], [[0.216240391]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.530000e+02> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[WEIGHTS_OUT_LOW]], [[WEIGHTS_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
    // CHECK:                   [[ADD:%.+]] = IE.Add([[CONV]], [[BIAS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    // CHECK:                   return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @TransposeInput([[ARG0:%.+]]: tensor<1x224x224x3xf32>)
func.func @TransposeInput(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x8x112x112xf32> {
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x224x224x3xf32> -> tensor<1x3x224x224xf32>

    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NWCH}
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]])
    // CHECK:    [[ADD:%.+]] = IE.Add([[CONV]], {{[^:]+}})
    // CHECK:    return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @TransposeConvertInput([[ARG0:%.+]]: tensor<1x224x224x3xf16>)
func.func @TransposeConvertInput(%arg0: tensor<1x224x224x3xf16>) -> tensor<1x8x112x112xf32> {
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x224x224x3xf16> -> tensor<1x224x224x3xf32>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x224x224x3xf32> -> tensor<1x3x224x224xf32>

    %3 = IE.Add(%2, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.FakeQuantize(%3, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %5 = IE.Convolution(%4, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %6 = IE.Add(%5, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %6 : tensor<1x8x112x112xf32>

    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32}
    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NWCH}
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]])
    // CHECK:    [[ADD:%.+]] = IE.Add([[CONV]], {{[^:]+}})
    // CHECK:    return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @UnsupportedInput([[ARG0:%.+]]: tensor<1x3x224x224xf32>)
func.func @UnsupportedInput(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x8x112x112xf32> {
    %scales = const.Declare tensor<1x3x1x1xf32> = dense<0.0174255371> : tensor<1x3x1x1xf32>
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weightsInLow = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %weightsInHigh = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %weightsOutLow = const.Declare tensor<8x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]], [[[-0.00619094493]]], [[[-0.206200793]]], [[[-0.125123039]]], [[[-0.247662395]]]]> : tensor<8x1x1x1xf32>
    %weightsOutHigh = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]], [[[0.00614257809]]], [[[0.204589844]]], [[[0.124145515]]], [[[0.245727539]]]]> : tensor<8x1x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xf32> = dense<10> : tensor<8x3x3x3xsi8>, [#const.CastElemType<f32>]
    %weightsFQ = IE.FakeQuantize(%weights, %weightsInLow, %weightsInHigh, %weightsOutLow, %weightsOutHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<8x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<8x1x1x1xf32>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x3x224x224xf32> -> tensor<1x3x224x224xf32>

    %2 = IE.Multiply(%1, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.Add(%2, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.FakeQuantize(%3, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %5 = IE.Convolution(%4, %weightsFQ) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %6 = IE.Add(%5, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %6 : tensor<1x8x112x112xf32>

    // CHECK:    [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize
    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:    [[SCALE:%.+]] = IE.Multiply([[SOFTMAX]], {{[^:]+}})
    // CHECK:    [[SHIFT:%.+]] = IE.Add([[SCALE]], {{[^:]+}})
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[SHIFT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_FQ]])
    // CHECK:    [[ADD:%.+]] = IE.Add([[CONV]], {{[^:]+}})
    // CHECK:    return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @FQWeightsWithMultiUsers([[ARG0:%.+]]: tensor<1x3x180x320xf16>)
func.func @FQWeightsWithMultiUsers(%arg0: tensor<1x3x180x320xf16>) -> (tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_4 = const.Declare tensor<32x3x3x3xf32> = dense<10> : tensor<32x3x3x3xsi8>, [#const.CastElemType<f32>]
    %cst_5 = const.Declare tensor<32x1x1x1xf32> = dense<1.0> : tensor<32x1x1x1xf32>
    %cst_6 = const.Declare tensor<32x1x1x1xf32> = dense<1.0> : tensor<32x1x1x1xf32>
    %cst_7 = const.Declare tensor<1x32x1x1xf32> = dense<1.0> : tensor<1x32x1x1xf32>

    %cst_248 = const.Declare tensor<12x3x5x5xf32> = dense<10> : tensor<12x3x5x5xsi8>, [#const.CastElemType<f32>]
    %cst_249 = const.Declare tensor<12x1x1x1xf32> = dense<[[[[0.602677524]]], [[[0.5713588]]], [[[0.544255078]]], [[[0.484283864]]], [[[0.496469975]]], [[[0.567459047]]], [[[0.551877379]]], [[[0.521942496]]], [[[0.550553083]]], [[[0.519279301]]], [[[0.48845908]]], [[[0.549172223]]]]> : tensor<12x1x1x1xf32>
    %cst_250 = const.Declare tensor<12x1x1x1xf32> = dense<[[[[-6.074230e-01]]], [[[-0.575857699]]], [[[-0.548540592]]], [[[-0.488097131]]], [[[-0.500379086]]], [[[-0.57192713]]], [[[-0.556222856]]], [[[-0.526052177]]], [[[-0.554888189]]], [[[-0.52336812]]], [[[-0.492305219]]], [[[-0.55349642]]]]> : tensor<12x1x1x1xf32>
    %cst_251 = const.Declare tensor<1x12x1x1xf32> = dense<1.0> : tensor<1x12x1x1xf32>
    %cst_262 = const.Declare tensor<1x1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>
    %cst_263 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>

    %25 = IE.FakeQuantize(%cst_4, %cst_262, %cst_263, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<32x3x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<32x1x1x1xf32>, tensor<32x1x1x1xf32> -> tensor<32x3x3x3xf32>
    %26 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x3x180x320xf16> -> tensor<1x3x180x320xf32>
    %27 = IE.Add(%26, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x180x320xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x180x320xf32>
    %28 = IE.FakeQuantize(%27, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x180x320xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x180x320xf32>
    %29 = IE.Convolution(%28, %25) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x180x320xf32>, tensor<32x3x3x3xf32> -> tensor<1x32x180x320xf32>
    %30 = IE.Add(%29, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x180x320xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x180x320xf32>

    %1 = IE.FakeQuantize(%cst_248, %cst_262, %cst_263, %cst_250, %cst_249) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<12x3x5x5xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<12x1x1x1xf32>, tensor<12x1x1x1xf32> -> tensor<12x3x5x5xf32>
    %146 = IE.Convolution(%28, %1) {dilations = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x180x320xf32>, tensor<12x3x5x5xf32> -> tensor<1x12x180x320xf32>
    %147 = IE.Add(%146, %cst_251) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x180x320xf32>, tensor<1x12x1x1xf32> -> tensor<1x12x180x320xf32>

    return %30, %147 : tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>

    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32}
    // CHECK:    [[IN_FQ_1:%.+]] = IE.FakeQuantize([[CONVERT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[IN_FQ_2:%.+]] = IE.FakeQuantize
    // CHECK:    [[IN_FQ_3:%.+]] = IE.FakeQuantize([[CONVERT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[IN_FQ_4:%.+]] = IE.FakeQuantize

    // CHECK:    [[CONV_1:%.+]] = IE.Convolution([[IN_FQ_3]], [[IN_FQ_4]])
    // CHECK:    [[ADD_1:%.+]] = IE.Add([[CONV_1]], {{[^:]+}})

    // CHECK:    [[CONV_2:%.+]] = IE.Convolution([[IN_FQ_1]], [[IN_FQ_2]])
    // CHECK:    [[ADD_2:%.+]] = IE.Add([[CONV_2]], {{[^:]+}})

    // CHECK:    return [[ADD_1]], [[ADD_2]] : tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>
}
