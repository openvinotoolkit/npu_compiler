//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" -mlir-print-elementsattrs-with-hex-if-larger=-1 --fuse-input-scale-shift --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// The weights branch is imported as a marked IE.DynamicDequantize. The pass fuses the input scale/shift into the activation FakeQuantize and re-encodes
// the rescaled weights as raw unsigned storage feeding a new marked IE.DynamicDequantize.

// CHECK: func.func @SplatScales([[ARG0:%.+]]: tensor<1x3x224x224xf32>)
func.func @SplatScales(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x8x112x112xf32> {
    %scales = const.Declare tensor<1x3x1x1xf32> = dense<0.0174255371> : tensor<1x3x1x1xf32>
    %shifts = const.Declare tensor<1x3x1x1xf32> = dense<[[[[-1.8046875]], [[-2.03515625]], [[-2.109375]]]]> : tensor<1x3x1x1xf32>
    %actLow = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %actHigh = const.Declare tensor<1x1x1x1xf32> = dense<2.50789928> : tensor<1x1x1x1xf32>
    %biases = const.Declare tensor<1x8x1x1xf32> = dense<[[[[0.358398438]], [[-0.144897461]], [[0.437255859]], [[-0.0820922852]], [[-9.600830e-02]], [[0.635742188]], [[0.348876953]], [[0.207397461]]]]> : tensor<1x8x1x1xf32>
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667525]]], [[[0.000282828277]]], [[[0.00404081447]]], [[[0.0000483667573]]], [[[0.0000483667573]]], [[[0.0016109433]]], [[[0.000977523718]]], [[[0.00193486293]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Multiply(%arg0, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK-DAG:               [[WEIGHTS_ZP:%.+]] = const.Declare tensor<8x1x1x1xui8> = dense<8> : tensor<8x1x1x1xui8>
    // CHECK-DAG:               [[WEIGHTS_SCALE:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[1.95816814E-4]]], [[[1.14505374E-5]]], [[[1.63595731E-4]]], [[[7.62939453E-6]]], [[[7.62939453E-6]]], [[[6.52203744E-5]]], [[[3.9575858E-5]]], [[[7.83345312E-5]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xui8> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[255, 255, 255]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.362888545]], [[-0.144634902]], [[0.441007137]], [[-0.0820473805]], [[-0.0959633961]], [[0.637237727]], [[0.349784434]], [[0.209193677]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.45700073> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.98651123> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[WEIGHTS_SCALE]], [[WEIGHTS_ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
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
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667525]]], [[[0.000282828277]]], [[[0.00404081447]]], [[[0.0000483667573]]], [[[0.0000483667573]]], [[[0.0016109433]]], [[[0.000977523718]]], [[[0.00193486293]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Multiply(%arg0, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK-DAG:               [[WEIGHTS_ZP:%.+]] = const.Declare tensor<8x1x1x1xui8> = dense<8> : tensor<8x1x1x1xui8>
    // CHECK-DAG:               [[WEIGHTS_SCALE:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[1.97770962E-4]]], [[[1.15648081E-5]]], [[[1.65228324E-4]]], [[[7.62939453E-6]]], [[[7.62939453E-6]]], [[[6.58712379E-5]]], [[[3.99708042E-5]]], [[[7.91162674E-5]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xui8> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[254, 254, 254]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.348501623]], [[-0.145476192]], [[0.428987533]], [[-0.0821912512]], [[-0.0961072668]], [[0.632445871]], [[0.34687674]], [[0.203438342]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.44337463> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.97549438> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[WEIGHTS_SCALE]], [[WEIGHTS_ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
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
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667525]]], [[[0.000282828277]]], [[[0.00404081447]]], [[[0.0000483667573]]], [[[0.0000483667573]]], [[[0.0016109433]]], [[[0.000977523718]]], [[[0.00193486293]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>
    %1 = IE.Add(%arg0, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %2 = IE.FakeQuantize(%1, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.Convolution(%2, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %4 = IE.Add(%3, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %4 : tensor<1x8x112x112xf32>

    // CHECK-DAG:               [[WEIGHTS_ZP:%.+]] = const.Declare tensor<8x1x1x1xui8> = dense<8> : tensor<8x1x1x1xui8>
    // CHECK-DAG:               [[WEIGHTS_SCALE:%.+]] = const.Declare tensor<8x1x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[1.95816814E-4]]], [[[1.14505374E-5]]], [[[1.63595731E-4]]], [[[7.62939453E-6]]], [[[7.62939453E-6]]], [[[6.52203744E-5]]], [[[3.9575858E-5]]], [[[7.83345312E-5]]]]> : tensor<8x1x1x1xf32>
    // CHECK-DAG:               [[WEIGHTS:%.+]] = const.Declare tensor<8x3x3x3xui8> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[255, 255, 255]
    // CHECK-DAG:               [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf32> =
    // CHECK-SAME{LITERAL}:                     dense<[[[[0.380503565]], [[-0.143604845]], [[0.455723643]], [[-0.0818712338]], [[-0.0957872495]], [[0.643104672]], [[0.35334453]], [[0.216240391]]]]> : tensor<1x8x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_IN_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:               [[ACT_OUT_HIHG:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.530000e+02> : tensor<1x1x1x1xf32>
    // CHECK:                   [[IN_FQ:%.+]] = IE.FakeQuantize([[ARG0]], [[ACT_IN_LOW]], [[ACT_IN_HIHG]], [[ACT_OUT_LOW]], [[ACT_OUT_HIHG]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK:                   [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[WEIGHTS_SCALE]], [[WEIGHTS_ZP]]) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:                   [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]}
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
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667525]]], [[[0.000282828277]]], [[[0.00404081447]]], [[[0.0000483667573]]], [[[0.0000483667573]]], [[[0.0016109433]]], [[[0.000977523718]]], [[[0.00193486293]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x224x224x3xf32> -> tensor<1x3x224x224xf32>

    %2 = IE.Add(%1, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.FakeQuantize(%2, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.Convolution(%3, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %5 = IE.Add(%4, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %5 : tensor<1x8x112x112xf32>

    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]]) {order_value = #NWCH}
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]])
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
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667525]]], [[[0.000282828277]]], [[[0.00404081447]]], [[[0.0000483667573]]], [[[0.0000483667573]]], [[[0.0016109433]]], [[[0.000977523718]]], [[[0.00193486293]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x224x224x3xf16> -> tensor<1x224x224x3xf32>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<1x224x224x3xf32> -> tensor<1x3x224x224xf32>

    %3 = IE.Add(%2, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.FakeQuantize(%3, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %5 = IE.Convolution(%4, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %6 = IE.Add(%5, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %6 : tensor<1x8x112x112xf32>

    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32}
    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NWCH}
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]])
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
    %weights = const.Declare tensor<8x3x3x3xsi8> = dense<10> : tensor<8x3x3x3xsi8>
    // Per-channel weight scales (mirror the develop FakeQuantize per-channel out ranges).
    %weightsScale = const.Declare tensor<8x1x1x1xf32> = dense<[[[[0.00483667570196]]], [[[0.000282828261569]]], [[[0.00404081578431]]], [[[4.83667569412e-05]]], [[[4.83667569412e-05]]], [[[0.00161094367451]]], [[[0.000977523741176]]], [[[0.00193486248627]]]]> : tensor<8x1x1x1xf32>
    %weightsDD = IE.DynamicDequantize(%weights, %weightsScale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<8x3x3x3xsi8>, tensor<8x1x1x1xf32> -> tensor<8x3x3x3xf32>

    %1 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x3x224x224xf32> -> tensor<1x3x224x224xf32>

    %2 = IE.Multiply(%1, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %3 = IE.Add(%2, %shifts) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x224x224xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x224x224xf32>
    %4 = IE.FakeQuantize(%3, %actLow, %actHigh, %actLow, %actHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x224x224xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x224x224xf32>
    %5 = IE.Convolution(%4, %weightsDD) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x3x224x224xf32>, tensor<8x3x3x3xf32> -> tensor<1x8x112x112xf32>
    %6 = IE.Add(%5, %biases) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x112x112xf32>, tensor<1x8x1x1xf32> -> tensor<1x8x112x112xf32>
    return %6 : tensor<1x8x112x112xf32>

    // The input scale/shift is not fused (preceded by an unsupported SoftMax) so the weights stay a plain DynamicDequantize.
    // CHECK:    [[WEIGHTS_DD:%.+]] = IE.DynamicDequantize
    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK:    [[SCALE:%.+]] = IE.Multiply([[SOFTMAX]], {{[^:]+}})
    // CHECK:    [[SHIFT:%.+]] = IE.Add([[SCALE]], {{[^:]+}})
    // CHECK:    [[IN_FQ:%.+]] = IE.FakeQuantize([[SHIFT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[CONV:%.+]] = IE.Convolution([[IN_FQ]], [[WEIGHTS_DD]])
    // CHECK:    [[ADD:%.+]] = IE.Add([[CONV]], {{[^:]+}})
    // CHECK:    return [[ADD]] : tensor<1x8x112x112xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @DDQWeightsWithMultiUsers([[ARG0:%.+]]: tensor<1x3x180x320xf16>)
func.func @DDQWeightsWithMultiUsers(%arg0: tensor<1x3x180x320xf16>) -> (tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-2.52764654> : tensor<1x1x1x1xf32>
    %cst_7 = const.Declare tensor<1x32x1x1xf32> = dense<1.0> : tensor<1x32x1x1xf32>
    %cst_251 = const.Declare tensor<1x12x1x1xf32> = dense<1.0> : tensor<1x12x1x1xf32>

    %w0 = const.Declare tensor<32x3x3x3xsi8> = dense<10> : tensor<32x3x3x3xsi8>
    %w0Scale = const.Declare tensor<32x1x1x1xf32> = dense<0.00393700786> : tensor<32x1x1x1xf32>
    %ddq0 = IE.DynamicDequantize(%w0, %w0Scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<32x3x3x3xsi8>, tensor<32x1x1x1xf32> -> tensor<32x3x3x3xf32>

    %w1 = const.Declare tensor<12x3x5x5xsi8> = dense<10> : tensor<12x3x5x5xsi8>
    // Per-channel weight scales (mirror the develop FakeQuantize per-channel out ranges).
    %w1Scale = const.Declare tensor<12x1x1x1xf32> = dense<[[[[0.00474549225098]]], [[[0.00449888823137]]], [[[0.00428547321569]]], [[[0.00381325880392]]], [[[0.00390921200392]]], [[[0.00446818108627]]], [[[0.00434549111765]]], [[[0.00410978303137]]], [[[0.00433506381176]]], [[[0.00408881341569]]], [[[0.00384613450588]]], [[[0.00432419075686]]]]> : tensor<12x1x1x1xf32>
    %ddq1 = IE.DynamicDequantize(%w1, %w1Scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} : tensor<12x3x5x5xsi8>, tensor<12x1x1x1xf32> -> tensor<12x3x5x5xf32>

    %26 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x3x180x320xf16> -> tensor<1x3x180x320xf32>
    %27 = IE.Add(%26, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x180x320xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x180x320xf32>
    %28 = IE.FakeQuantize(%27, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x180x320xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x180x320xf32>
    %29 = IE.Convolution(%28, %ddq0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x180x320xf32>, tensor<32x3x3x3xf32> -> tensor<1x32x180x320xf32>
    %30 = IE.Add(%29, %cst_7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x180x320xf32>, tensor<1x32x1x1xf32> -> tensor<1x32x180x320xf32>

    %146 = IE.Convolution(%28, %ddq1) {dilations = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x180x320xf32>, tensor<12x3x5x5xf32> -> tensor<1x12x180x320xf32>
    %147 = IE.Add(%146, %cst_251) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x180x320xf32>, tensor<1x12x1x1xf32> -> tensor<1x12x180x320xf32>

    return %30, %147 : tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>

    // Each conv consumer of the shared activation FakeQuantize gets its own fused input FakeQuantize and
    // re-encoded weights DynamicDequantize.
    // CHECK:    [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32}
    // CHECK:    [[IN_FQ_1:%.+]] = IE.FakeQuantize([[CONVERT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_DD_1:%.+]] = IE.DynamicDequantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {dstElemType = f32, vpux.weights_import_dyn_dequant}
    // CHECK:    [[IN_FQ_2:%.+]] = IE.FakeQuantize([[CONVERT]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}})
    // CHECK:    [[WEIGHTS_DD_2:%.+]] = IE.DynamicDequantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {dstElemType = f32, vpux.weights_import_dyn_dequant}

    // CHECK:    [[CONV_1:%.+]] = IE.Convolution([[IN_FQ_2]], [[WEIGHTS_DD_2]])
    // CHECK:    [[ADD_1:%.+]] = IE.Add([[CONV_1]], {{[^:]+}})

    // CHECK:    [[CONV_2:%.+]] = IE.Convolution([[IN_FQ_1]], [[WEIGHTS_DD_1]])
    // CHECK:    [[ADD_2:%.+]] = IE.Add([[CONV_2]], {{[^:]+}})

    // CHECK:    return [[ADD_1]], [[ADD_2]] : tensor<1x32x180x320xf32>, tensor<1x12x180x320xf32>
}
