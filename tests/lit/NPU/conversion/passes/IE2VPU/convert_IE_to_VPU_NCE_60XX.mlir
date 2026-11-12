//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true enable-dcim=true power-optimized=false" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemTypeInOut = !quant.uniform<si8:f16, 0.007>
!qElemTypeWeights = !quant.uniform<si8:f16, 2.5>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEConvWithMPEEngineSCLsi8
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x120x16x!qElemType, {order = #NHWC}>)
func.func @ConvToNCEConvWithMPEEngineSCLsi8(%arg0: tensor<1x16x120x16x!qElemTypeInOut, {order = #NHWC}>) -> tensor<1x32x120x16x!qElemTypeInOut, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x1x1x!qElemTypeWeights, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemTypeWeights>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x120x16x!qElemTypeInOut, {order = #NHWC}>, tensor<32x16x1x1x!qElemTypeWeights, {order = #NHWC}>, tensor<1x32x1x1xf16>
            -> tensor<1x32x120x16x!qElemTypeInOut, {order = #NHWC}>

    return %0 : tensor<1x32x120x16x!qElemTypeInOut, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<2.500000e+00>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<32x1x1x1xf32> = dense<57.1428566>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 0 : i64, activation_zp = 0 : i64>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -1.280000e+02 : f64,
    // CHECK-SAME:          clamp_high = 1.270000e+02 : f64,
    // CHECK-SAME:          scale = 2.500000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 57.142857142857139 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x120x16x!qElemType, {order = #NHWC}>, tensor<32x16x1x1x!qElemType1, {order = #NHWC}>, tensor<32x1x1x1xf32>, tensor<32x1x1x1xf32>
    // CHECK-SAME:      -> tensor<1x32x120x16x!qElemType, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x32x120x16x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I4WeightsConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @I4WeightsConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x32x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
    // CHECK-SAME:      tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 16]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 16]>]
    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.33854163>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [16, 16, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.3385416666666667 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

// Check that an IE.Convolution whose filter uses !quant.quantile weights
// (produced by a QuantizeCast → AffineReshape → QuantizeCast → AffineReshape →
// QuantizeCast → PermuteCast → Concat chain) is lowered to VPU.NCE.Convolution
// with the correct weight-table constants and NCE weight-tensor layout
// (Reshape → Expand → LayoutCast).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Quantile quant type: lookup-table index encoding, per-output-channel scales.
!qElemType0 = !quant.uniform<i4:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>
!qElemType1 = !quant.uniform<!quant.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>

// CHECK-DAG: [[QTYPE0:!.+]] = !quant.uniform<i4:f16:0,
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<!quant.quantile<ui4:si8,

// CHECK: @NCEConvolutionWithQuantileTypeFilter
// CHECK-SAME: ([[ACT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS:%.+]]: tensor<16x4xsi4>)
func.func @NCEConvolutionWithQuantileTypeFilter(
        %arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<16x4xsi4>
) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    // Zero-padding constant for the filter (input-channel dimension: 4 → 16).
    %cst_zero = const.Declare tensor<16x12x1x1x!qElemType1, {order = #NHWC}> =
        dense<0> : tensor<16x12x1x1xui8, {order = #NHWC}>

    // Filter producer chain: si4 weights → QuantileType per-channel quant tensor.
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i4:f16, 1.000000e+00>}
        : tensor<16x4xsi4> -> tensor<16x4x!quant.uniform<i4:f16, 1.000000e+00>>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 1, 4]}
        : tensor<16x4x!quant.uniform<i4:f16, 1.000000e+00>>
        -> tensor<1x16x1x4x!quant.uniform<i4:f16, 1.000000e+00>>
    %2 = IE.QuantizeCast(%1) {dstElemType = !quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>}
        : tensor<1x16x1x4x!quant.uniform<i4:f16, 1.000000e+00>>
        -> tensor<1x16x1x4x!quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 4, 1, 1]}
        : tensor<1x16x1x4x!quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>>
        -> tensor<16x4x1x1x!qElemType0>
    %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType1}
        : tensor<16x4x1x1x!qElemType0> -> tensor<16x4x1x1x!qElemType1>
    %5 = IE.PermuteCast(%4) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<16x4x1x1x!qElemType1> -> tensor<16x4x1x1x!qElemType1, {order = #NHWC}>
    %6 = IE.Concat(%5, %cst_zero) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]}
        : tensor<16x4x1x1x!qElemType1, {order = #NHWC}>,
          tensor<16x12x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<16x16x1x1x!qElemType1, {order = #NHWC}>

    %7 = IE.Convolution(%arg0, %6) {
        dilations = [1, 1], input_padding = [0, 12, 0, 0], output_padding = [0, 0, 0, 0],
        pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x1x1xf16, {order = #NHWC}>,
        tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
        -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %7 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // Filter producer chain is preserved unchanged up to and including the Concat.
    // CHECK-DAG:   [[ZERO:%.+]] = const.Declare tensor<16x12x1x1x[[QTYPE1]], {order = #NHWC}> = dense<0>

    // CHECK:       [[Q_CAST0:%.+]] = IE.QuantizeCast([[WEIGHTS]])
    // CHECK:       [[AFF_RESHAPE0:%.+]] = IE.AffineReshape([[Q_CAST0]])
    // CHECK:       [[Q_CAST1:%.+]] = IE.QuantizeCast([[AFF_RESHAPE0]])
    // CHECK:       [[AFF_RESHAPE1:%.+]] = IE.AffineReshape([[Q_CAST1]])
    // CHECK:       [[Q_CAST2:%.+]] = IE.QuantizeCast([[AFF_RESHAPE1]])
    // CHECK:       [[FILTER_NHWC:%.+]] = IE.PermuteCast([[Q_CAST2]])
    // CHECK-SAME:      -> tensor<16x4x1x1x[[QTYPE1]], {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[FILTER_NHWC]], [[ZERO]])
    // CHECK-SAME{{LITERAL}}:      static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]
    // CHECK-SAME:      -> tensor<16x16x1x1x[[QTYPE1]], {order = #NHWC}>

    // The 16x16 filter is reshaped and expanded into NCE weight layout [OC, 1, 1, IC_aligned]
    // then LayoutCast to NHWC before being passed to VPU.NCE.Convolution.
    // CHECK:       [[RESHAPE:%.+]] = VPU.Reshape([[CONCAT]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 16]
    // CHECK-SAME:      -> tensor<16x1x1x16x[[QTYPE1]]>
    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[RESHAPE]])
    // CHECK-SAME:      pads_end = [0, 0, 0, 16]
    // CHECK-SAME:      -> tensor<16x1x1x32x[[QTYPE1]]>
    // CHECK:       [[LAYOUT:%.+]] = VPU.LayoutCast([[EXPAND]])
    // CHECK-SAME:      -> tensor<16x1x1x32x[[QTYPE1]], {order = #NHWC}>

    // CHECK-DAG:   [[SCALES:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<{{.+}}> : tensor<16x1x1x1xf32>
    // CHECK-DAG:   [[BIASES:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ACT]], [[LAYOUT]], [[SCALES]], [[BIASES]])
    // CHECK-SAME:      input_padding = [0, 12, 0, 0]
    // CHECK-SAME:      weight_zp = 0
    // CHECK-SAME:      output_padding = [0, 0, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!quantileType = !quant.uniform<!quant.quantile<ui4:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>

// CHECK: [[QQ:!.+]] = !quant.uniform<!quant.quantile<ui4:f16,

// CHECK: @ConvWeightsAsInputsNotBlockArgUnaICNeedsWorkloadAlignment
// CHECK-SAME:     ([[ACT:%.+]]: tensor<1x16x128x8xf16, {order = #NHWC}>, [[WRAW:%.+]]: tensor<4x16x1x1x[[QQ]], {order = #NHWC}>)
func.func @ConvWeightsAsInputsNotBlockArgUnaICNeedsWorkloadAlignment(
    %act: tensor<1x16x128x8xf16, {order = #NHWC}>,%wraw: tensor<4x16x1x1x!quantileType, {order = #NHWC}>)
        -> tensor<1x16x128x8xf16, {order = #NHWC}> {
  %0 = IE.Expand(%wraw) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]}
    : tensor<4x16x1x1x!quantileType, {order = #NHWC}> -> tensor<16x16x1x1x!quantileType, {order = #NHWC}>

  %2 = IE.Convolution(%act, %0) {
    dilations = [1, 1], input_padding = [0, 12, 0, 0], output_padding = [0, 12, 0, 0],
    pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
  } : tensor<1x16x128x8xf16, {order = #NHWC}>, tensor<16x16x1x1x!quantileType, {order = #NHWC}>
    -> tensor<1x16x128x8xf16, {order = #NHWC}>

  return %2 : tensor<1x16x128x8xf16, {order = #NHWC}>

  // CHECK:       [[EXPAND_OC:%.+]] = IE.Expand([[WRAW]])
  // CHECK-SAME:    : tensor<4x16x1x1x[[QQ]], {order = #NHWC}> -> tensor<16x16x1x1x[[QQ]], {order = #NHWC}>
  // CHECK:       [[RESHAPE:%.+]] = VPU.Reshape([[EXPAND_OC]])
  // CHECK-SAME:    : tensor<16x16x1x1x[[QQ]], {order = #NHWC}> -> tensor<16x1x1x16x[[QQ]]>

  // CHECK:       [[EXPAND_WORKLOAD:%.+]] = VPU.Expand([[RESHAPE]])
  // CHECK-SAME:    : tensor<16x1x1x16x[[QQ]]> -> tensor<16x1x1x32x[[QQ]]>

  // CHECK:       [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[EXPAND_WORKLOAD]])
  // CHECK-SAME:    : tensor<16x1x1x32x[[QQ]]> -> tensor<16x1x1x32x[[QQ]], {order = #NHWC}>

  // CHECK:       VPU.NCE.Convolution([[ACT]], [[LAYOUT_CAST]]
  // CHECK-SAME:      input_padding = [0, 12, 0, 0], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 0 : i64>,
  // CHECK-SAME:      output_padding = [0, 12, 0, 0]
  // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:    : tensor<1x16x128x8xf16, {order = #NHWC}>, tensor<16x1x1x32x!qElemType, {order = #NHWC}>
  // CHECK-SAME:    -> tensor<1x16x128x8xf16, {order = #NHWC}>
}
