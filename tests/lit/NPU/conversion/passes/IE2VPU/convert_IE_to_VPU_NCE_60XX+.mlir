//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true enable-dcim=true power-optimized=false" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ClampLowInF16PReLU
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @ClampLowInF16PReLU(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = -2.312500e+00 : f64>,
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[SCALE_TABLE:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:    [[BIAS_TABLE:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [64, 16, 3, 3] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:            clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:            scale = 1.000000e+00 : f64, prelu_alpha = [-2.312500e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:            strides = [2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK: return [[CONV]] : tensor<1x64x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 0.078737745098039214>

// CHECK-LABEL: @BiasFuncForI8Weights
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @BiasFuncForI8Weights(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.0787377431>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<12.7003889>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 0.078737745098039214 : f64, prelu_alpha = [1.000000e-01], bias = 12.700389105058367 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @ConvWithStaticScale(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        // Note: 5.24156e-40 is '374050' when bit-cast to int
        static_scale = 5.24156e-40 : f32
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<5.241560e-40>
    // CHECK:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 5.2415569058069783E-40 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 2.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE4channelsIDUAutopad
module @ConvToNCE4channelsIDUAutopad {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>

        %0 = IE.Convolution(%arg0, %weights, %bias) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1],
                post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
                input_padding = [0, 12, 0, 0],
                output_padding = [0, 12, 0, 0]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
                -> tensor<1x16x16x16xf16, {order = #NHWC}>

        return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
        // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>

        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      input_padding = [0, 12, 0, 0],
        // CHECK-SAME:      output_padding = [0, 12, 0, 0],
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
        // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
        // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
        // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 2.000000e+00 : f64, adder = 0.000000e+00 : f64>
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEWithBiasBroadCast
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCEWithBiasBroadCast(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 2.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithReluRewriter
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvWithReluRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.Relu<>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [-0.000000e+00], bias = 2.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithSprLUTTanh
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvWithSprLUTTanh(%input: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>

    %output = IE.Convolution(%input, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.Tanh<>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %output : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <TANH>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 2.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense_resource<__elided__>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, -0.078737745098039214>
// CHECK: !qElemType = !quant.uniform<i8:f16, -0.078737745098039214>

// CHECK-LABEL: @QuantConvWithNegativeScales
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @QuantConvWithNegativeScales(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<-0.0787377431>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<-12.7003889>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = -0.078737745098039214 : f64, prelu_alpha = [1.000000e-01], bias = -12.700389105058367 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 0.078737745098039214>

// CHECK-LABEL: @BiasFuncForI8Weights
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @BiasFuncForI8Weights(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.0787377431>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<12.7003889>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 0.078737745098039214 : f64, prelu_alpha = [1.000000e-01], bias = 12.700389105058367 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:       [[BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE]], [[BIAS]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 2.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

!qElemTypeInOut = !quant.uniform<ui8:f16, 0.007>
!qElemTypeWeights = !quant.uniform<ui8:f16, 2.5>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEConvWithMPEEngineDCIMui8
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NHWC}>)
func.func @ConvToNCEConvWithMPEEngineDCIMui8(%arg0: tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>) -> tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}> {
    %weights = const.Declare tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemTypeWeights>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>, tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}>, tensor<1x512x1x1xf16>
            -> tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>

    return %0 : tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<2.500000e+00> : tensor<512x1x1x1xf32>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<57.1428566> : tensor<512x1x1x1xf32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 0 : i64>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 2.500000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 57.142857142857139 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x120x16x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>, tensor<512x1x1x1xf32>, tensor<512x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x512x120x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x512x120x16x!qElemType, {order = #NHWC}>
}

// -----

!qElemTypeInOut = !quant.uniform<ui8:f16, 0.007>
!qElemTypeWeights = !quant.uniform<ui8:f16, 2.5>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEConvWithMPEEngineSCLui8
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x120x16x!qElemType, {order = #NHWC}>)
func.func @ConvToNCEConvWithMPEEngineSCLui8(%arg0: tensor<1x32x120x16x!qElemTypeInOut, {order = #NHWC}>) -> tensor<1x48x120x16x!qElemTypeInOut, {order = #NHWC}> {
    %weights = const.Declare tensor<48x32x1x1x!qElemTypeWeights, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x32x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemTypeWeights>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x48x1x1xf16> = dense<1.000000e+00> : tensor<1x48x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x32x120x16x!qElemTypeInOut, {order = #NHWC}>, tensor<48x32x1x1x!qElemTypeWeights, {order = #NHWC}>, tensor<1x48x1x1xf16>
            -> tensor<1x48x120x16x!qElemTypeInOut, {order = #NHWC}>

    return %0 : tensor<1x48x120x16x!qElemTypeInOut, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<48x32x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<2.500000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<48x1x1x1xf32> = dense<57.1428566>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 0 : i64, activation_zp = 0 : i64>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 2.500000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 57.142857142857139 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x32x120x16x!qElemType, {order = #NHWC}>, tensor<48x32x1x1x!qElemType1, {order = #NHWC}>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x48x120x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x48x120x16x!qElemType, {order = #NHWC}>
}

// -----

!qElemTypeInOut = !quant.uniform<si8:f16, 0.007>
!qElemTypeWeights = !quant.uniform<si8:f16, 2.5>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEConvWithMPEEngineDCIMsi8
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x120x16x!qElemType, {order = #NHWC}>)
func.func @ConvToNCEConvWithMPEEngineDCIMsi8(%arg0: tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>) -> tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}> {
    %weights = const.Declare tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemTypeWeights>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>, tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}>, tensor<1x512x1x1xf16>
            -> tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>

    return %0 : tensor<1x512x120x16x!qElemTypeInOut, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<2.500000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 0 : i64>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -1.280000e+02 : f64,
    // CHECK-SAME:          clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:          scale = 2.500000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 57.142857142857139 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x512x120x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x512x120x16x!qElemType, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>

// CHECK: func.func @LowerMatMulToNCE([[INPUT:%.+]]: tensor<1x128x64x32xf16>)
func.func @LowerMatMulToNCE(%input : tensor<1x128x64x32xf16>) -> tensor<1x128x64x64xf16> {
    %matmul = IE.MatMul(%input, %input) { transpose_b } : tensor<1x128x64x32xf16>, tensor<1x128x64x32xf16> -> tensor<1x128x64x64xf16>
    return %matmul : tensor<1x128x64x64xf16>

    // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_0]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x1x32x64x1xf16, {order = #GNHWC}>

    // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x64x32x1x1xf16, {order = #GNHWC}>

    // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<1.000000e+00>
    // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [128, 64, 32, 1, 1] {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64
    // CHECK-SAME:      clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } -> tensor<128x1x64x64x1xf16, {order = #GNHWC}>


    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[MATMUL]])
    // CHECK-SAME:      tensor<128x1x64x64x1xf16, {order = #GNHWC}> -> tensor<128x64x64x1x1xf16>

    // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[MEMPERM]])
    // CHECK-SAME:      tensor<128x64x64x1x1xf16> -> tensor<1x128x64x64xf16>


    // CHECK:       return [[AFFINE_RESHAPE_2]] : tensor<1x128x64x64xf16>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<u8:f16, 0.021319432352103439:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0033220431383918312>
!qElemType2 = !quant.uniform<u8:f16, 0.029685012966978782:128>

// CHECK: func.func @LowerQuantMatMulToNCE()
func.func @LowerQuantMatMulToNCE() -> tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>> {
    %INP1 = const.Declare tensor<1x12x576x144x!quant.uniform<u8:f16, 0.0033220431383918312>> = dense<1.0> : tensor<1x12x576x144xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0033220431383918312>> ]
    %INP2 = const.Declare tensor<1x12x32x144x!quant.uniform<u8:f16, 0.029685012966978782:128>> = dense<1.0> : tensor<1x12x32x144xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.029685012966978782:128>> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x12x576x144x!quant.uniform<u8:f16, 0.0033220431383918312>>,
     tensor<1x12x32x144x!quant.uniform<u8:f16, 0.029685012966978782:128>> -> tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>>
    return %MATMUL : tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<12x1x144x576x1x!qElemType1, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<12x32x144x1x1x!qElemType2, {order = #GNHWC}>

    // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<1x32x1x1x1xf32> = dense<0.00462558726>
    // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<1x32x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [12, 32, 144, 1, 1] {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -1.280000e+02 : f64
    // CHECK-SAME:      clamp_high = 1.270000e+02 : f64
    // CHECK-SAME:      scale = 0.0046255872112980888 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } -> tensor<12x1x32x576x1x!qElemType, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0045223289302417213>

// CHECK: func.func @LowerMixedPrecisionMatMulToNCE
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x1x64xf16>)
func.func @LowerMixedPrecisionMatMulToNCE(%arg0: tensor<1x8x1x64xf16>) -> tensor<1x8x1x128xf16> {
    %cst = const.Declare tensor<1x8x128x64x!qElemType> = dense<1.000000e+00> : tensor<1x8x128x64xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %quantize = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType>
    %matmul = IE.MatMul(%quantize, %cst) {transpose_b} : tensor<1x8x1x64x!qElemType>, tensor<1x8x128x64x!qElemType> -> tensor<1x8x1x128xf16>
    return %matmul : tensor<1x8x1x128xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:     -> tensor<8x1x64x1x1x!qElemType, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<8x128x64x1x1x!qElemType, {order = #GNHWC}>

    // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<1x128x1x1x1xf32> = dense<2.04514581E-5>
    // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<1x128x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [8, 128, 64, 1, 1] {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:      scale = 2.0451458953301231E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:     -> tensor<8x1x128x1x1xf16, {order = #GNHWC}>
}

// -----

!qElemTypeInOut   = !quant.uniform<ui8:f16, 0.007>
!qElemTypeWeights = !quant.uniform<ui8:f16, 2.5>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotMapToDCIM
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x512x120x8200x!qElemType, {order = #NHWC}>)
func.func @DoNotMapToDCIM( %arg0: tensor<1x512x120x8200x!qElemTypeInOut, {order = #NHWC}>) -> tensor<1x512x120x8200x!qElemTypeInOut, {order = #NHWC}> {
    //  In DCIM mode tensor_size_x + pad_count_right must be <= 8192!
    %weights = const.Declare
        tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<512x512x1x1xf16>,
            [#const.CastElemType<ui8>, #const.CastElemType<!qElemTypeWeights>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x512x1x1xf16>
        = dense<1.000000e+00> : tensor<1x512x1x1xf16>
    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations  = [1, 1],
            pads_begin = [0, 0],
            pads_end   = [0, 0],
            strides    = [1, 1]
        } : tensor<1x512x120x8200x!qElemTypeInOut, {order = #NHWC}>, tensor<512x512x1x1x!qElemTypeWeights, {order = #NHWC}>, tensor<1x512x1x1xf16>
        -> tensor<1x512x120x8200x!qElemTypeInOut, {order = #NHWC}>

    return %0 : tensor<1x512x120x8200x!qElemTypeInOut, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x512x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<2.500000e+00> : tensor<512x1x1x1xf32>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<57.1428566> : tensor<512x1x1x1xf32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 0 : i64, activation_zp = 0 : i64>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 2.500000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 57.142857142857139 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x120x8200x!qElemType, {order = #NHWC}>, tensor<512x512x1x1x!qElemType1, {order = #NHWC}>, tensor<512x1x1x1xf32>, tensor<512x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x512x120x8200x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x512x120x8200x!qElemType, {order = #NHWC}>


}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<u8:f16, 0.021319432352103439:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0033220431383918312>
!qElemType2 = !quant.uniform<u8:f16, 0.029685012966978782:128>

// CHECK: func.func @LowerMatMultToDCIM()
func.func @LowerMatMultToDCIM() -> tensor<1x128x64x64x!qElemType> {
    %INP1 = const.Declare tensor<1x128x64x32x!qElemType1> = dense<1.0> : tensor<1x128x64x32xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
    %INP2 = const.Declare tensor<1x128x64x32x!qElemType2> = dense<1.0> : tensor<1x128x64x32xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x128x64x32x!qElemType1>,
     tensor<1x128x64x32x!qElemType2> -> tensor<1x128x64x64x!qElemType>
    return %MATMUL : tensor<1x128x64x64x!qElemType>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<128x1x32x64x1x!qElemType1, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<128x64x32x1x1x!qElemType2, {order = #GNHWC}>

    // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<0.00462558726>
    // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [128, 64, 32, 1, 1] {
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 128 : i64, activation_zp = 0 : i64>,
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -1.280000e+02 : f64
    // CHECK-SAME:      clamp_high = 1.270000e+02 : f64
    // CHECK-SAME:      scale = 0.0046255872112980888 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } -> tensor<128x1x64x64x1x!qElemType, {order = #GNHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<i16:f16, 5.000000e-01:2888>
!qElemType1 = !quant.uniform<i8:f16, 1.0000152590218967>

// CHECK: func.func @LowerI16MatMulToDCIM()
func.func @LowerI16MatMulToDCIM() -> tensor<1x128x64x64x!qElemType> {
    %INP1 = const.Declare tensor<1x128x64x32x!qElemType> = dense<1.0> : tensor<1x128x64x32xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType> ]
    %INP2 = const.Declare tensor<1x128x64x32x!qElemType1> = dense<1.0> : tensor<1x128x64x32xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType1> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x128x64x32x!qElemType>,
     tensor<1x128x64x32x!qElemType1> -> tensor<1x128x64x64x!qElemType>
    return %MATMUL : tensor<1x128x64x64x!qElemType>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<128x1x32x64x1x!qElemType, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<128x64x32x1x1x!qElemType1, {order = #GNHWC}>

    // CHECK:       [[SCALE_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<1.00001526>
    // CHECK:       [[BIAS_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [128, 64, 32, 1, 1] {
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 2888 : i64>,
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -3.565600e+04 : f64
    // CHECK-SAME:      clamp_high = 2.987900e+04 : f64
    // CHECK-SAME:      scale = 1.0000152590218967 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 2.888000e+03 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<128x1x64x64x1x!qElemType, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:1, {0.0190698242187499996,0.0181949869791666674,0.0194848632812500003,0.0188582356770833329,
                                       0.0290698242187499996,0.0281949869791666674,0.0294848632812500003,0.0288582356770833329,
                                       0.0390698242187499996,0.0381949869791666674,0.0394848632812500003,0.0388582356770833329,
                                       0.0490698242187499996,0.0481949869791666674,0.0494848632812500003,0.0488582356770833329}>

// CHECK-LABEL: @AvgPoolToNCEAsQuantization
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @AvgPoolToNCEAsQuantization(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    return %avg_pool : tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:                 [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<[
    // CHECK-SAME{LITERAL}:       [[[52.4388695]]], [[[54.9601936]]], [[[51.321888]]], [[[53.0272293]]],
    // CHECK-SAME{LITERAL}:       [[[34.3999329]]], [[[35.4672966]]], [[[33.9157066]]], [[[34.652153]]],
    // CHECK-SAME{LITERAL}:       [[[25.5952015]]], [[[26.1814461]]], [[[25.3261604]]], [[[25.7345695]]],
    // CHECK-SAME{LITERAL}:       [[[20.3791237]]], [[[20.7490463]]], [[[2.020820e+01]]], [[[20.4673786]]]]> : tensor<16x1x1x1xf32>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {kernel_size = [1, 1]
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -1.280000e+02 : f64,
    // CHECK-SAME:                   clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, strides = [1, 1]}

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<ui4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvWithStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x32x1x1xf16>)
func.func @ConvWithStaticScale(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
                               %arg2: tensor<1x32x1x1xf16>)
        -> tensor<1x32x16x16xf16, {order = #NHWC}> {
            
    %0 = IE.LayoutCast(%arg2) {dst_order = #NHWC} : tensor<1x32x1x1xf16> -> tensor<1x32x1x1xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x1xf32, {order = #NHWC}>
        
    %2 = IE.LayoutCast(%1) {dst_order = #NCHW} : tensor<1x32x1x1xf32, {order = #NHWC}> -> tensor<1x32x1x1xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [32, 1, 1, 1]} : tensor<1x32x1x1xf32> -> tensor<32x1x1x1xf32>
    %4 = IE.Convolution(%arg0, %arg1, %3) {
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        static_scale = 0.247436523 : f32
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1x!qElemType, {order = #NHWC}>, tensor<32x1x1x1xf32>
        -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x32x16x16xf16, {order = #NHWC}>
    
    // CHECK:   [[LC_IN_1:%.+]] = IE.LayoutCast([[ARG2]]) {dst_order = #NHWC} : tensor<1x32x1x1xf16> -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.247436523>
    // CHECK-DAG:   [[BIAS_TABLE_1:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[LC_IN_1]], [[SCALE_TABLE]], [[BIAS_TABLE_1]])

    // CHECK:   [[LC_OUT:%.+]] = IE.LayoutCast([[MAXPOOL]]) {dst_order = #NCHW}
    // CHECK:   [[AR_OUT:%.+]] = IE.AffineReshape([[LC_OUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [32, 1, 1, 1]}

    // CHECK-DAG:   [[BIAS_TABLE_2:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[AR_OUT]], [[BIAS_TABLE_2]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<ui4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvWithStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x96x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<96x96x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x32x3x1xf16>)
func.func @ConvWithStaticScale(%arg0: tensor<1x96x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<96x96x1x1x!qElemType, {order = #NHWC}>,
                               %arg2: tensor<1x32x3x1xf16>)
        -> tensor<1x96x16x16xf16, {order = #NHWC}> {
            
    %0 = IE.LayoutCast(%arg2) {dst_order = #NHWC} : tensor<1x32x3x1xf16> -> tensor<1x32x3x1xf16, {order = #NHWC}>
    %1 = IE.MaxPool(%0) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]} : tensor<1x32x3x1xf16, {order = #NHWC}> -> tensor<1x32x3x1xf32, {order = #NHWC}>
        
    %2 = IE.LayoutCast(%1) {dst_order = #NCHW} : tensor<1x32x3x1xf32, {order = #NHWC}> -> tensor<1x32x3x1xf32>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [96, 1, 1, 1]} : tensor<1x32x3x1xf32> -> tensor<96x1x1x1xf32>
    %4 = IE.Convolution(%arg0, %arg1, %3) {
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        static_scale = 0.247436523 : f32
    } : tensor<1x96x16x16xf16, {order = #NHWC}>, tensor<96x96x1x1x!qElemType, {order = #NHWC}>, tensor<96x1x1x1xf32>
        -> tensor<1x96x16x16xf16, {order = #NHWC}>

    return %4 : tensor<1x96x16x16xf16, {order = #NHWC}>

    
    // CHECK:   [[LC_IN_1:%.+]] = IE.LayoutCast([[ARG2]]) {dst_order = #NHWC} : tensor<1x32x3x1xf16> -> tensor<1x32x3x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[SCALE_TABLE:%.+]] = const.Declare tensor<1x32x3x1xf16, {order = #NHWC}> = dense<0.247436523> : tensor<96x1x1x1xf32>, [#const.Reshape<[1, 32, 3, 1]>, #const.CastElemType<f16>, #const.LayoutCast<#NHWC>]

    // CHECK:   [[MULTIPLY:%.+]] = VPU.NCE.Eltwise([[LC_IN_1]], [[SCALE_TABLE]])

    // CHECK:   [[LC_OUT:%.+]] = IE.LayoutCast([[MULTIPLY]]) {dst_order = #NCHW}
    // CHECK:   [[AR_OUT:%.+]] = IE.AffineReshape([[LC_OUT]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [96, 1, 1, 1]}

    // CHECK-DAG:   [[BIAS_TABLE:%.+]] = const.Declare tensor<96x1x1x1xf32> = dense<0.000000e+00> : tensor<96x1x1x1xf32>

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[AR_OUT]], [[BIAS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<ui4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvWithStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x32x1x1xf32>)
func.func @ConvWithStaticScale(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
                               %arg2: tensor<1x32x1x1xf32>)
        -> tensor<1x32x16x16xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1, %arg2) {
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 0>,
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        static_scale = 0.247436523 : f32
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1x!qElemType, {order = #NHWC}>, tensor<1x32x1x1xf32>
        -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK:   [[SCALE_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.247436523>
    // CHECK:   [[BIAS_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00>
    // CHECK:   [[MULTIPLY:%.+]] = IE.Multiply([[ARG2]], [[SCALE_TABLE]])

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[MULTIPLY]], [[BIAS_TABLE]]) rawFilterShape [32, 32, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<ui4:f16, 1.000000e+00>

// CHECK-LABEL: @ConvWithOutStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<1x32x1x1xf32>)
func.func @ConvWithOutStaticScale(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<32x32x1x1x!qElemType, {order = #NHWC}>,
                               %arg2: tensor<1x32x1x1xf32>)
        -> tensor<1x32x16x16xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1, , %arg2) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<32x32x1x1x!qElemType, {order = #NHWC}>, tensor<1x32x1x1xf32>
        -> tensor<1x32x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x32x16x16xf16, {order = #NHWC}>

    // CHECK:   [[BIAS_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[ARG2]], [[BIAS_TABLE]]) rawFilterShape [32, 32, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:1, {0.0190698242187499996,0.0181949869791666674,0.0194848632812500003,0.0188582356770833329,
                                       0.0290698242187499996,0.0281949869791666674,0.0294848632812500003,0.0288582356770833329,
                                       0.0390698242187499996,0.0381949869791666674,0.0394848632812500003,0.0388582356770833329,
                                       0.0490698242187499996,0.0481949869791666674,0.0494848632812500003,0.0488582356770833329}>

// CHECK-LABEL: @AvgPoolToNCEAsDequantization
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6x!qElemType, {order = #NHWC}>)
func.func @AvgPoolToNCEAsDequantization(%arg0: tensor<1x16x6x6x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %avg_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6x!qElemType, {order = #NHWC}> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %avg_pool : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:                 [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<[
    // CHECK-SAME{LITERAL}:       [[[0.0190698244]]], [[[0.0181949865]]], [[[0.0194848627]]], [[[0.0188582353]]],
    // CHECK-SAME{LITERAL}:       [[[0.0290698241]]], [[[0.0281949863]]], [[[0.0294848625]]], [[[0.0288582351]]],
    // CHECK-SAME{LITERAL}:       [[[0.0390698239]]], [[[0.0381949879]]], [[[0.0394848622]]], [[[0.0388582349]]],
    // CHECK-SAME{LITERAL}:       [[[0.0490698256]]], [[[0.0481949858]]], [[[0.0494848639]]], [[[0.0488582365]]]]> : tensor<16x1x1x1xf32>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {kernel_size = [1, 1]
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, strides = [1, 1]}

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @AvgPoolToNCE(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x4x4xf16, {order = #NHWC}>
    
    // CHECK:             [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [3, 3]
    // CHECK-SAME:           pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:           ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:              clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:              clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:              prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK:           return [[OUT]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @AveragePoolToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x14x14xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [2, 2]
         } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [2, 2],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x14x14xf16, {order = #NHWC}>
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCEWithScaleTable
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>
// CHECK-SAME:     [[SCALE_TABLE:%.+]]: tensor<16x1x1x1xf32>)
func.func @MaxPoolToNCEWithScaleTable(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf32>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %max_pool = IE.MaxPool(%arg0, %arg1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<16x1x1x1xf32> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %max_pool : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {kernel_size = [1, 1]
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:                   resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            clamp = {min = 0.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolFP32ToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolFP32ToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf32, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf32, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf32, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCEWithScaleTable
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>
// CHECK-SAME:     [[SCALE_TABLE:%.+]]: tensor<16x1x1x1xf32>)
func.func @AvgPoolToNCEWithScaleTable(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>, %arg1: tensor<16x1x1x1xf32>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %avg_pool = IE.AvgPool(%arg0, %arg1) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<16x1x1x1xf32> -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %avg_pool : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {kernel_size = [1, 1]
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00],
    // CHECK-SAME:                   adder = 0.000000e+00 : f64>, strides = [1, 1]}

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithDynamicScaleInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG2:%.+]]: tensor<16x1x1x1xf32>)
func.func @ConvWithDynamicScaleInput(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
                                     %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>,
                                     %arg2: tensor<16x1x1x1xf32>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %arg1, %bias, %arg2) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            static_scale = 1.000000e+00 : f32
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>,
            tensor<1x16x1x1xf16>, tensor<16x1x1x1xf32>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[ARG2]], [[BIAS]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:1, {0.0190698242187499996,0.0181949869791666674,0.0194848632812500003,0.0188582356770833329,
                                       0.0290698242187499996,0.0281949869791666674,0.0294848632812500003,0.0288582356770833329,
                                       0.0390698242187499996,0.0381949869791666674,0.0394848632812500003,0.0388582356770833329,
                                       0.0490698242187499996,0.0481949869791666674,0.0494848632812500003,0.0488582356770833329}>

// CHECK-LABEL: @EltwiseAddToNCEAsQuantization
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @EltwiseAddToNCEAsQuantization(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x6x6x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } :
        tensor<1x16x6x6xf16, {order = #NHWC}>, tensor<1x16x6x6xf16, {order = #NHWC}>
        -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:                 [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<[
    // CHECK-SAME{LITERAL}:       [[[52.4388695]]], [[[54.9601936]]], [[[51.321888]]], [[[53.0272293]]],
    // CHECK-SAME{LITERAL}:       [[[34.3999329]]], [[[35.4672966]]], [[[33.9157066]]], [[[34.652153]]],
    // CHECK-SAME{LITERAL}:       [[[25.5952015]]], [[[26.1814461]]], [[[25.3261604]]], [[[25.7345695]]],
    // CHECK-SAME{LITERAL}:       [[[20.3791237]]], [[[20.7490463]]], [[[2.020820e+01]]], [[[20.4673786]]]]> : tensor<16x1x1x1xf32>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {
    // CHECK-SAME:                op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -1.280000e+02 : f64,
    // CHECK-SAME:                   clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME:                -> tensor<1x16x6x6x!qElemType, {order = #NHWC}>

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16:1, {0.0095349121093750000,0.0090974934895833337,0.0097424316406250002,0.0094291178385416665,
                                       0.0145349121093750000,0.0140974934895833337,0.0147424316406250002,0.0144291178385416665,
                                       0.0195349121093750000,0.0190974934895833337,0.0197424316406250002,0.0194291178385416665,
                                       0.0245349121093750000,0.0240974934895833337,0.0247424316406250002,0.0244291178385416665}>

// CHECK-LABEL: @EltwiseAddToNCEAsDequantization
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6x!qElemType, {order = #NHWC}>)
func.func @EltwiseAddToNCEAsDequantization(%arg0: tensor<1x16x6x6x!qElemType, {order = #NHWC}>) -> tensor<1x16x6x6xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } :
        tensor<1x16x6x6x!qElemType, {order = #NHWC}>, tensor<1x16x6x6x!qElemType, {order = #NHWC}>
        -> tensor<1x16x6x6xf16, {order = #NHWC}>

    return %0 : tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:                 [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<[
    // CHECK-SAME{LITERAL}:       [[[0.00953491218]]], [[[0.00909749325]]], [[[0.00974243134]]], [[[0.00942911766]]],
    // CHECK-SAME{LITERAL}:       [[[0.0145349121]]], [[[0.0140974931]]], [[[0.0147424312]]], [[[0.0144291176]]],
    // CHECK-SAME{LITERAL}:       [[[0.019534912]]], [[[0.019097494]]], [[[0.0197424311]]], [[[0.0194291174]]],
    // CHECK-SAME{LITERAL}:       [[[0.0245349128]]], [[[0.0240974929]]], [[[0.0247424319]]], [[[0.0244291183]]]]> : tensor<16x1x1x1xf32>

    // CHECK:                 [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

    // CHECK:                 [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]], [[SCALE_TABLE]], [[BIAS_TABLE]]) {
    // CHECK-SAME:                op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:                   clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:                   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME:                -> tensor<1x16x6x6xf16, {order = #NHWC}>

    // CHECK:                   return [[OUT]] : tensor<1x16x6x6xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// IE.Convolution with a non-constant (function-argument) bias: isNewWeightTableFormat is already
// true for IE.Convolution starting on NPU60XX, so the conversion converts the bias to f32,
// reshapes it to [OC, 1, 1, 1], and passes it as the weight-table bias operand instead of the
// compile-time zero-constant used in the static-bias case (see @BiasFuncForI8Weights above).
//
// CHECK-LABEL: @ConvToNCEWithNonConstantBias
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[BIAS:%.+]]: tensor<1x16x1x1xf16>)
func.func @ConvToNCEWithNonConstantBias(
        %arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %weights, %arg1) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // Non-constant bias is converted to f32 via a DPU identity MaxPool (createConvertPoolingForScaleTable
    // -- faster than a SW Convert), reshaped to [OC, 1, 1, 1], and passed directly as the
    // weight-table bias operand rather than the compile-time zero-constant generated in the
    // no-bias case. Plain Convolution does not need a DataPointerTable (unlike DepthConvolution).
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[BIAS_NHWC:%.+]] = IE.LayoutCast([[BIAS]]) {dst_order = #NHWC}
    // CHECK:       [[BIAS_POOL:%.+]] = VPU.NCE.MaxPool([[BIAS_NHWC]]) {kernel_size = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_CONV:%.+]] = IE.LayoutCast([[BIAS_POOL]]) {dst_order = #NCHW}
    // CHECK:       [[BIAS_RESHAPE:%.+]] = IE.Reshape([[BIAS_CONV]])
    // CHECK-SAME:      {shape_value = [16, 1, 1, 1]}
    // CHECK-SAME:      : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE]], [[BIAS_RESHAPE]]) rawFilterShape [16, 16, 1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[OUT]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// The f16-to-f32 bias MaxPool is created during this pass, after layout adjustment, so it must be
// NHWC to lower to NCE; an NCHW MaxPool would fall back to a SW MaxPool that cannot change precision.
// CHECK-LABEL: @DepthConvToNCEWithLargeF16RuntimeBias
// CHECK-SAME:     ([[INPUT:%[^:]+]]: tensor<1x5120x4x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[BIAS:%[^:]+]]: tensor<1x5120x1x1xf16>)
func.func @DepthConvToNCEWithLargeF16RuntimeBias(
        %arg0: tensor<1x5120x4x4xf16, {order = #NHWC}>,
        %arg1: tensor<1x5120x1x1xf16>) -> tensor<1x5120x4x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x1x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.GroupConvolution(%arg0, %weights, %arg1) {
            dilations = [1, 1],
            groups = 5120 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x5120x4x4xf16, {order = #NHWC}>, tensor<5120x1x1x1xf16, {order = #NHWC}>, tensor<1x5120x1x1xf16>
            -> tensor<1x5120x4x4xf16, {order = #NHWC}>
    return %0 : tensor<1x5120x4x4xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[BIAS_IN:%.+]] = IE.Reshape([[BIAS]]) {shape_value = [1, 256, 4, 5]}
    // CHECK:       [[BIAS_NHWC:%.+]] = IE.LayoutCast([[BIAS_IN]]) {dst_order = #NHWC}
    // CHECK-SAME:      -> tensor<1x256x4x5xf16, {order = #NHWC}>
    // CHECK:       [[BIAS_POOL:%.+]] = VPU.NCE.MaxPool([[BIAS_NHWC]])
    // CHECK-SAME:      -> tensor<1x256x4x5xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_NCHW:%.+]] = IE.LayoutCast([[BIAS_POOL]]) {dst_order = #NCHW}
    // CHECK:       [[BIAS_RESHAPE:%.+]] = IE.Reshape([[BIAS_NCHW]]) {shape_value = [5120, 1, 1, 1]}
    // CHECK-SAME:      -> tensor<5120x1x1x1xf32>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, [[BIAS_RESHAPE]])
    // CHECK:       return [[OUT]] : tensor<1x5120x4x4xf16, {order = #NHWC}>
}

// -----

// An f32 runtime bias is already in the weight-table precision. Conversion must reshape it
// directly, without creating a redundant f16-to-f32 MaxPool conversion.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DepthConvToNCEWithF32RuntimeBias
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[BIAS:%.+]]: tensor<1x16x1x1xf32>)
func.func @DepthConvToNCEWithF32RuntimeBias(
        %arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.GroupConvolution(%arg0, %weights, %arg1) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf32>
            -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.MaxPool
    // CHECK:       [[BIAS_RESHAPE:%.+]] = IE.Reshape([[BIAS]])
    // CHECK-SAME:      {shape_value = [16, 1, 1, 1]}
    // CHECK-SAME:      : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[WEIGHTS:%.+]], {{.+}}, [[SCALE]], [[BIAS_RESHAPE]])
    // CHECK:       return [[OUT]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

// A MaxPool-produced f32 tensor is already in weight-table bias precision. DepthConvToNCE must
// reshape that value directly rather than insert another MaxPool conversion.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DepthConvToNCEWithMaxPoolRuntimeBias
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[BIAS_INPUT:%.+]]: tensor<1x16x3x4xf16, {order = #NHWC}>)
func.func @DepthConvToNCEWithMaxPoolRuntimeBias(
        %arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x3x4xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %pool = IE.MaxPool(%arg1) {
            kernel_size = [3, 4],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x3x4xf16, {order = #NHWC}> -> tensor<1x16x1x1xf32, {order = #NHWC}>
    %0 = IE.GroupConvolution(%arg0, %weights, %pool) {
            dilations = [1, 1],
            groups = 16 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>,
            tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[POOL:%.+]] = VPU.NCE.MaxPool([[BIAS_INPUT]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = IE.Reshape([[POOL]])
    // CHECK-SAME:      {shape_value = [16, 1, 1, 1]}
    // CHECK-SAME:      : tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<16x1x1x1xf32>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], {{.+}}, {{.+}}, {{.+}}, [[BIAS_RESHAPE]])
    // CHECK:       return [[OUT]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}
