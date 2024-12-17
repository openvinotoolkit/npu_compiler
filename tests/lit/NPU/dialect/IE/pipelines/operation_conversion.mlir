//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --operation-conversion %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


// CHECK-LABEL: @OperationConversionAllOpsSubset
func.func @OperationConversionAllOpsSubset(%arg0: tensor<1x16x8x12xf16>) -> tensor<1x12x8x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x8x12xf16> -> tensor<1x1x8x12xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 12], strides = [1, 1]} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  %cst_exponent = const.Declare tensor<1xf16> = dense<2.0> : tensor<1xf16>
  %sqd = IE.SquaredDiff(%1, %cst_exponent) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x8x1xf16>, tensor<1xf16> -> tensor<1x12x8x1xf16>
  return %sqd : tensor<1x12x8x1xf16>


  //CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NCHW>]
  //CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
  //CHECK:     [[VAL0:%.+]] = IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x8x12xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x8x12xf16>
  //CHECK:     [[VAL1:%.+]] = IE.Transpose([[VAL0]]) {order_value = #NWHC} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  //CHECK:     [[VAL2:%.+]] = IE.SquaredDiff([[VAL1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x8x1xf16>, tensor<1xf16> -> tensor<1x12x8x1xf16>

}

// -----

// CHECK: @ConvertFCToConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16xf16>)
func.func @ConvertFCToConv(%arg0: tensor<1x16xf16>) -> tensor<1x64xf16> {
    %weights = const.Declare tensor<64x16xf16> = dense<1.0> : tensor<64x16xf16>
    %bias = const.Declare tensor<1x64xf16> = dense<1.0> : tensor<1x64xf16>
    %0 = IE.FullyConnected(%arg0, %weights, %bias) : tensor<1x16xf16>, tensor<64x16xf16>, tensor<1x64xf16> -> tensor<1x64xf16>

    return %0 : tensor<1x64xf16>

    // CHECK-NOT:   IE.FullyConnected

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16> = dense<1.000000e+00> : tensor<64x16xf16>, [#const.Reshape<[64, 16, 1, 1]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64xf16>, [#const.Reshape<[1, 64, 1, 1]>]

    // CHECK:       [[VAL0:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME:      shape_value = [1, 16, 1, 1]
    // CHECK-SAME:    : tensor<1x16xf16> -> tensor<1x16x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[VAL0]], [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       [[VAL3:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME:      shape_value = [1, 64]
    // CHECK-SAME:    : tensor<1x64x1x1xf16> -> tensor<1x64xf16>
    // CHECK:       return [[VAL3]]
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:       @MatMul4dInputsTo2d
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<1x2x1x512xf16>) -> tensor<1x2x1x40xf16>
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf16>) -> tensor<1x2x1x40xf16> {
  %cst = const.Declare tensor<40x512xf16> = dense<1.000000e+00> : tensor<1x2x512x40xf32>,
      [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#CN>, #const.CastElemType<f16>]
  %cst_0 = const.Declare tensor<40x512xf16> = dense<1.000000e+00> : tensor<1x2x512x40xf32>,
      [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>, #const.Transpose<#CN>, #const.CastElemType<f16>]

  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf16> to tensor<1x1x1x512xf16>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf16> -> tensor<1x512xf16>

  %2 = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf16> to tensor<1x1x1x512xf16>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf16> -> tensor<1x512xf16>

  %4 = IE.FullyConnected(%1, %cst_0) : tensor<1x512xf16>, tensor<40x512xf16> -> tensor<1x40xf16>
  %5 = IE.FullyConnected(%3, %cst) : tensor<1x512xf16>, tensor<40x512xf16> -> tensor<1x40xf16>

  %6 = IE.AffineReshape(%4) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf16> -> tensor<1x1x1x40xf16>
  %7 = IE.AffineReshape(%5) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40xf16> -> tensor<1x1x1x40xf16>
  %8 = IE.Concat(%6, %7) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x2x1x40xf16>
  return %8 : tensor<1x2x1x40xf16>

  // CHECK-DAG:      [[CST_0:%.+]] = const.Declare tensor<40x512x1x1xf16> = dense<1.000000e+00>
  // CHECK-DAG-SAME:          : tensor<1x2x512x40xf32>, [#const.SubView<[0, 1, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>,
  // CHECK-DAG-SAME:                                     #const.Transpose<#CN>, #const.Reshape<[40, 512, 1, 1]>, #const.CastElemType<f16>]

  // CHECK-DAG:      [[CST_1:%.+]] = const.Declare tensor<40x512x1x1xf16> = dense<1.000000e+00>
  // CHECK-DAG-SAME:          : tensor<1x2x512x40xf32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 512, 40]>, #const.Reshape<[512, 40]>,
  // CHECK-DAG-SAME:                                     #const.Transpose<#CN>, #const.Reshape<[40, 512, 1, 1]>, #const.CastElemType<f16>]

  // CHECK:          [[SLICE_1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf16> to tensor<1x1x1x512xf16>
  // CHECK:          [[SLICE_2:%.+]] = IE.Slice [[ARG0]] [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf16> to tensor<1x1x1x512xf16>

  // CHECK:          [[IN_1:%.+]] = IE.AffineReshape([[SLICE_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 512, 1, 1]} : tensor<1x1x1x512xf16> -> tensor<1x512x1x1xf16>
  // CHECK:          [[CONV_1:%.+]] = IE.Convolution([[IN_1]], [[CST_1]])
  // CHECK-SAME:              {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:          : tensor<1x512x1x1xf16>, tensor<40x512x1x1xf16> -> tensor<1x40x1x1xf16>

  // CHECK:          [[IN_2:%.+]] = IE.AffineReshape([[SLICE_2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 512, 1, 1]} : tensor<1x1x1x512xf16> -> tensor<1x512x1x1xf16>
  // CHECK:          [[CONV_2:%.+]] = IE.Convolution([[IN_2]], [[CST_0]])
  // CHECK-SAME:              {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:          : tensor<1x512x1x1xf16>, tensor<40x512x1x1xf16> -> tensor<1x40x1x1xf16>

  // CHECK:          [[OUT_1_4D:%.+]] = IE.AffineReshape([[CONV_1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40x1x1xf16> -> tensor<1x1x1x40xf16>
  // CHECK:          [[OUT_2_4D:%.+]] = IE.AffineReshape([[CONV_2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 40]} : tensor<1x40x1x1xf16> -> tensor<1x1x1x40xf16>

  // CHECK:          [[CONCAT:%.+]] = IE.Concat([[OUT_1_4D]], [[OUT_2_4D]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x2x1x40xf16>

  // CHECK return [[OUT]] : tensor<1x2x1x40xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:       @MatMulWithGroupQuant
// CHECK-SAME:   ([[ARG0:%.+]]: tensor<16x3072xf16>)
func.func @MatMulWithGroupQuant(%arg0: tensor<16x3072xf16>) -> tensor<16x4096xf16> {
    %WEIGHTS = const.Declare tensor<3x1024x4096xf16> = dense<1.0> : tensor<3x1024x4096xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<1x1024x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1024x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1024, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<1x1024x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1024x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1024, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS_2:%.+]] = const.Declare tensor<1x1024x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1024x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1024, 4096]>, #const.CastElemType<f16>]

    %IN_LOW = const.Declare tensor<1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[IN_LOW:%.+]] = const.Declare tensor<1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1xf32>
    %IN_HIGH = const.Declare tensor<1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1xf32>

    %OUT_LOW = const.Declare tensor<3x1x4096xf16> = dense<-1.0> : tensor<3x1x4096xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_LOW_0:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_LOW_1:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_LOW_2:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]

    %OUT_HIGH = const.Declare tensor<3x1x4096xf16> = dense<1.0> : tensor<3x1x4096xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_HIGH_0:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_HIGH_1:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[OUT_HIGH_2:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<1.000000e+00> :
    // CHECK-DAG-SAME:  tensor<3x1x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1, 4096]>, #const.CastElemType<f16>]

    %FQ = IE.FakeQuantize(%WEIGHTS, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 16 : i64
    } : tensor<3x1024x4096xf16>,
        tensor<1x1x1xf16>,
        tensor<1x1x1xf16>,
        tensor<3x1x4096xf16>,
        tensor<3x1x4096xf16>
            -> tensor<3x1024x4096xf16>

    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[WEIGHTS_0]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_0]], [[OUT_HIGH_0]])
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[WEIGHTS_1]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_1]], [[OUT_HIGH_1]])
    // CHECK: [[FQ_2:%.+]] = IE.FakeQuantize([[WEIGHTS_2]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW_2]], [[OUT_HIGH_2]])

    %RESHAPE = IE.AffineReshape(%FQ) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 4096]
    } : tensor<3x1024x4096xf16> -> tensor<3072x4096xf16>
    // CHECK:   [[RESHAPE_FQ_0:%.+]] = IE.AffineReshape([[FQ_0]])
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  : tensor<1x1024x4096xf16> -> tensor<1024x4096xf16>

    // CHECK:   [[RESHAPE_FQ_1:%.+]] = IE.AffineReshape([[FQ_1]])
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  : tensor<1x1024x4096xf16> -> tensor<1024x4096xf16>

    // CHECK:   [[RESHAPE_FQ_2:%.+]] = IE.AffineReshape([[FQ_2]])
    // CHECK-SAME:      shape_value = [1024, 4096]
    // CHECK-SAME:  : tensor<1x1024x4096xf16> -> tensor<1024x4096xf16>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE) {
        order_value = #CN
    } : tensor<3072x4096xf16> -> tensor<4096x3072xf16>

    %GEMM = IE.FullyConnected(%arg0, %TRANSPOSE_RHS) : tensor<16x3072xf16>, tensor<4096x3072xf16> -> tensor<16x4096xf16>
    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[ARG0]] [0, 0] [16, 1024] : tensor<16x3072xf16> to tensor<16x1024xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[ARG0]] [0, 1024] [16, 1024] : tensor<16x3072xf16> to tensor<16x1024xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[ARG0]] [0, 2048] [16, 1024] : tensor<16x3072xf16> to tensor<16x1024xf16>

    // CHECK:   [[TRANSPOSE_FQ_0:%.+]] = IE.Transpose([[RESHAPE_FQ_0]])
    // CHECK-SAME:  tensor<1024x4096xf16> -> tensor<4096x1024xf16>

    // CHECK:   [[SLICE_0_4D:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME:    shape_value = [16, 1024, 1, 1]
    // CHECK-SAME:  : tensor<16x1024xf16> -> tensor<16x1024x1x1xf16>
    // CHECK:   [[FQ_0_4D:%.+]] = IE.AffineReshape([[TRANSPOSE_FQ_0]])
    // CHECK-SAME:    shape_value = [4096, 1024, 1, 1]
    // CHECK-SAME:  : tensor<4096x1024xf16> -> tensor<4096x1024x1x1xf16>
    // CHECK:   [[GEMM_0:%.+]] = IE.Convolution([[SLICE_0_4D]], [[FQ_0_4D]])
    // CHECK-SAME:  tensor<16x1024x1x1xf16>, tensor<4096x1024x1x1xf16> -> tensor<16x4096x1x1xf16>
    // CHECK:   [[GEMM_0_2D:%.+]] = IE.AffineReshape([[GEMM_0]])
    // CHECK-SAME:    shape_value = [16, 4096]
    // CHECK-SAME:  : tensor<16x4096x1x1xf16> -> tensor<16x4096xf16>

    // CHECK:   [[TRANSPOSE_FQ_1:%.+]] = IE.Transpose([[RESHAPE_FQ_1]])
    // CHECK-SAME:  tensor<1024x4096xf16> -> tensor<4096x1024xf16>

    // CHECK:   [[SLICE_1_4D:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME:     shape_value = [16, 1024, 1, 1]
    // CHECK-SAME:  : tensor<16x1024xf16> -> tensor<16x1024x1x1xf16>
    // CHECK:   [[FQ_1_4D:%.+]] = IE.AffineReshape([[TRANSPOSE_FQ_1]])
    // CHECK-SAME:     shape_value = [4096, 1024, 1, 1]
    // CHECK-SAME:  : tensor<4096x1024xf16> -> tensor<4096x1024x1x1xf16>
    // CHECK:   [[GEMM_1:%.+]] = IE.Convolution([[SLICE_1_4D]], [[FQ_1_4D]])
    // CHECK-SAME:  tensor<16x1024x1x1xf16>, tensor<4096x1024x1x1xf16> -> tensor<16x4096x1x1xf16>
    // CHECK:   [[GEMM_1_2D:%.+]] = IE.AffineReshape([[GEMM_1]])
    // CHECK-SAME:     shape_value = [16, 4096]
    // CHECK-SAME:  : tensor<16x4096x1x1xf16> -> tensor<16x4096xf16>

    // CHECK:   [[TRANSPOSE_FQ_2:%.+]] = IE.Transpose([[RESHAPE_FQ_2]])
    // CHECK-SAME:  tensor<1024x4096xf16> -> tensor<4096x1024xf16>

    // CHECK:   [[SLICE_2_4D:%.+]] = IE.AffineReshape([[SLICE_2]])
    // CHECK-SAME:     shape_value = [16, 1024, 1, 1]
    // CHECK-SAME:  : tensor<16x1024xf16> -> tensor<16x1024x1x1xf16>
    // CHECK:   [[FQ_2_4D:%.+]] = IE.AffineReshape([[TRANSPOSE_FQ_2]])
    // CHECK-SAME:     shape_value = [4096, 1024, 1, 1]
    // CHECK-SAME:  : tensor<4096x1024xf16> -> tensor<4096x1024x1x1xf16>
    // CHECK:   [[GEMM_2:%.+]] = IE.Convolution([[SLICE_2_4D]], [[FQ_2_4D]])
    // CHECK-SAME:  tensor<16x1024x1x1xf16>, tensor<4096x1024x1x1xf16> -> tensor<16x4096x1x1xf16>
    // CHECK:   [[GEMM_2_2D:%.+]] = IE.AffineReshape([[GEMM_2]])
    // CHECK-SAME:     shape_value = [16, 4096]
    // CHECK-SAME:  : tensor<16x4096x1x1xf16> -> tensor<16x4096xf16>

    // CHECK:   [[ADD_0:%.+]] = IE.Accumulate([[GEMM_0_2D]], [[GEMM_1_2D]])
    // CHECK:   [[ADD_1:%.+]] = IE.Accumulate([[ADD_0]], [[GEMM_2_2D]])

    return %GEMM : tensor<16x4096xf16>
}

// -----

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @ConvertGPTQWithMergedMatMul
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x4096xf16>
func.func @ConvertGPTQWithMergedMatMul(%arg0: tensor<1x1x4096xf16>) -> (tensor<1x1x1024xf16>, tensor<1x1x512xf16>) {
    %cst0_in = const.Declare tensor<2x2048x1024xf16> = dense<1.0> : tensor<2x2048x1024xf32>, [#const.CastElemType<f16>]
    %cst0_inlow = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %cst0_inhigh = const.Declare tensor<1x1x1xf16> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %cst0_outlow = const.Declare tensor<2x1x1024xf16> = dense<-1.500000e+01> : tensor<2x1x1024xf32>, [#const.CastElemType<f16>]
    %cst0_outhigh = const.Declare tensor<2x1x1024xf16> = dense<4.500000e+01> : tensor<2x1x1024xf32>, [#const.CastElemType<f16>]

    %0 = IE.FakeQuantize(%cst0_in, %cst0_inlow, %cst0_inhigh, %cst0_outlow, %cst0_outhigh)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x2048x1024xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<2x1x1024xf16>, tensor<2x1x1024xf16> -> tensor<2x2048x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [1]], shape_value = [4096, 1024]} : tensor<2x2048x1024xf16> -> tensor<4096x1024xf16>
    %2 = IE.Transpose(%1) {order_value = #CN} : tensor<4096x1024xf16> -> tensor<1024x4096xf16>
    %3 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 4096]} : tensor<1x1x4096xf16> -> tensor<1x4096xf16>
    %4 = IE.FullyConnected(%3, %2) : tensor<1x4096xf16>, tensor<1024x4096xf16> -> tensor<1x1024xf16>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 1024]} : tensor<1x1024xf16> -> tensor<1x1x1024xf16>

    %cst1_in = const.Declare tensor<2x2048x512xf16> = dense<2.0> : tensor<2x2048x512xf32>, [#const.CastElemType<f16>]
    %cst1_inlow = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %cst1_inhigh = const.Declare tensor<1x1x1xf16> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    %cst1_outlow = const.Declare tensor<2x1x512xf16> = dense<-1.500000e+01> : tensor<2x1x512xf32>, [#const.CastElemType<f16>]
    %cst1_outhigh = const.Declare tensor<2x1x512xf16> = dense<4.500000e+01> : tensor<2x1x512xf32>, [#const.CastElemType<f16>]

    %6 = IE.FakeQuantize(%cst1_in, %cst1_inlow, %cst1_inhigh, %cst1_outlow, %cst1_outhigh)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
            : tensor<2x2048x512xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<2x1x512xf16>, tensor<2x1x512xf16> -> tensor<2x2048x512xf16>
    %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [0], [1]], shape_value = [4096, 512]} : tensor<2x2048x512xf16> -> tensor<4096x512xf16>
    %8 = IE.Transpose(%7) {order_value = #CN} : tensor<4096x512xf16> -> tensor<512x4096xf16>
    %9 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 4096]} : tensor<1x1x4096xf16> -> tensor<1x4096xf16>
    %10 = IE.FullyConnected(%9, %8) : tensor<1x4096xf16>, tensor<512x4096xf16> -> tensor<1x512xf16>
    %11 = IE.AffineReshape(%10) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 512]} : tensor<1x512xf16> -> tensor<1x1x512xf16>

    return  %5, %11 : tensor<1x1x1024xf16>, tensor<1x1x512xf16>

    // CHECK-DAG:   [[CST:%.+]]  = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1536xf16> =
    // CHECK-SAME:          tensor<2x1x1536xf16>, [#const.SubView<[1, 0, 0], [1, 1, 1536]>]
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1536xf16> =
    // CHECK-SAME:          tensor<2x1x1536xf16>, [#const.SubView<[0, 0, 0], [1, 1, 1536]>]
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1536xf16> =
    // CHECK-SAME:          tensor<2x1x1536xf16>, [#const.SubView<[1, 0, 0], [1, 1, 1536]>]
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x1x1536xf16> =
    // CHECK-SAME:          tensor<2x1x1536xf16>, [#const.SubView<[0, 0, 0], [1, 1, 1536]>]
    // CHECK-DAG:   [[CST_4:%.+]] = const.Declare tensor<1x2048x1536xf16> =
    // CHECK-SAME:          tensor<2x2048x1536xf16>, [#const.SubView<[1, 0, 0], [1, 2048, 1536]>]
    // CHECK-DAG:   [[CST_5:%.+]] = const.Declare tensor<1x2048x1536xf16> =
    // CHECK-SAME:         tensor<2x2048x1536xf16>, [#const.SubView<[0, 0, 0], [1, 2048, 1536]>]
    // CHECK-DAG:   [[CST_6:%.+]] = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_7:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.500000e+01> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:       [[ARESHAPE_IN:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:          : tensor<1x1x4096xf16> -> tensor<1x4096xf16>

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[CST_5]], [[CST_6]], [[CST_7]], [[CST_3]], [[CST_1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
    // CHECK-SAME:          : tensor<1x2048x1536xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1536xf16>, tensor<1x1x1536xf16>
    // CHECK-SAME:              -> tensor<1x2048x1536xf16>

    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[CST_4]], [[CST_6]], [[CST_7]], [[CST_2]], [[CST_0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64}
    // CHECK-SAME:          : tensor<1x2048x1536xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1536xf16>, tensor<1x1x1536xf16>
    // CHECK-SAME:              -> tensor<1x2048x1536xf16>

    // CHECK:       [[ARESHAPE_FQ_0:%.+]] = IE.AffineReshape([[FQ_0]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1536]} : tensor<1x2048x1536xf16> -> tensor<2048x1536xf16>
    // CHECK:       [[ARESHAPE_FQ_1:%.+]] = IE.AffineReshape([[FQ_1]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [2048, 1536]} : tensor<1x2048x1536xf16> -> tensor<2048x1536xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[ARESHAPE_IN]] [0, 0] [1, 2048] : tensor<1x4096xf16> to tensor<1x2048xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[ARESHAPE_IN]] [0, 2048] [1, 2048] : tensor<1x4096xf16> to tensor<1x2048xf16>

    // CHECK:       [[TRANSPOSE_0:%.+]] = IE.Transpose([[ARESHAPE_FQ_0]]) {order_value = #CN} : tensor<2048x1536xf16> -> tensor<1536x2048xf16>

    // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME:          : tensor<1x2048xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[TRANSPOSE_0]])
    // CHECK-SAME:          : tensor<1536x2048xf16> -> tensor<1536x2048x1x1xf16>

    // CHECK:       [[CONV_0:%.+]]  = IE.Convolution([[RESHAPE_0]], [[RESHAPE_1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x2048x1x1xf16>, tensor<1536x2048x1x1xf16> -> tensor<1x1536x1x1xf16>
    // CHECK:       [[ARESHAPE_2:%.+]] = IE.AffineReshape([[CONV_0]])
    // CHECK-SAME:          : tensor<1x1536x1x1xf16> -> tensor<1x1x1x1536xf16>

    // CHECK:       [[TRANSPOSE_1:%.+]] = IE.Transpose([[ARESHAPE_FQ_1]]) {order_value = #CN} : tensor<2048x1536xf16> -> tensor<1536x2048xf16>

    // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK:       : tensor<1x2048xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.AffineReshape([[TRANSPOSE_1]])
    // CHECK:       : tensor<1536x2048xf16> -> tensor<1536x2048x1x1xf16>

    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_3]], [[RESHAPE_4]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x2048x1x1xf16>, tensor<1536x2048x1x1xf16> -> tensor<1x1536x1x1xf16>
    // CHECK:       [[ARESHAPE_3:%.+]]  = IE.AffineReshape([[CONV_1]])
    // CHECK-SAME:          : tensor<1x1536x1x1xf16> -> tensor<1x1x1x1536xf16>

    // CHECK:       [[CONCAT:%.+]]  = IE.Concat([[ARESHAPE_2]], [[ARESHAPE_3]])
    // CHECK-SAME{LITERAL}:          {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}
    // CHECK-SAME:              : tensor<1x1x1x1536xf16>, tensor<1x1x1x1536xf16> -> tensor<1x2x1x1536xf16>

    // CHECK:       [[ARESHAPE_4:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME:          : tensor<1x2x1x1536xf16> -> tensor<1x2x96x16xf16>
    // CHECK:       [[TRANSPOSE_2:%.+]] = IE.Transpose([[ARESHAPE_4]])
    // CHECK-SAME:          : tensor<1x2x96x16xf16> -> tensor<1x16x2x96xf16>

    // CHECK:       [[AVEPOOL:%.+]] = IE.AvgPool([[TRANSPOSE_2]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:       rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:          : tensor<1x16x2x96xf16> -> tensor<1x16x1x96xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[AVEPOOL]], [[CST]])
    // CHECK-SAME:          : tensor<1x16x1x96xf16>, tensor<1xf16> -> tensor<1x16x1x96xf16>
    // CHECK:       [[TRANSPOSE_3:%.+]] = IE.Transpose([[MUL]])
    // CHECK-SAME:          : tensor<1x16x1x96xf16> -> tensor<1x1x96x16xf16>
    // CHECK:       [[ARESHAPE_5:%.+]]  = IE.AffineReshape([[TRANSPOSE_3]])
    // CHECK-SAME:       : tensor<1x1x96x16xf16> -> tensor<1x1x1536xf16>
    // CHECK:       [[ARESHAPE_6:%.+]]  = IE.AffineReshape([[ARESHAPE_5]])
    // CHECK-SAME{LITERAL}:  : tensor<1x1x1536xf16> -> tensor<1x1536xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[ARESHAPE_6]] [0, 0] [1, 512] : tensor<1x1536xf16> to tensor<1x512xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[ARESHAPE_6]] [0, 512] [1, 1024] : tensor<1x1536xf16> to tensor<1x1024xf16>
    // CHECK:       [[ARESHAPE_7:%.+]] = IE.AffineReshape([[SLICE_3]])
    // CHECK-SAME{LITERAL}:          : tensor<1x1024xf16> -> tensor<1x1x1024xf16>
    // CHECK:       [[ARESHAPE_8:%.+]] = IE.AffineReshape([[SLICE_2]])
    // CHECK-SAME{LITERAL}:          : tensor<1x512xf16> -> tensor<1x1x512xf16>
    // CHECK:       return [[ARESHAPE_7]], [[ARESHAPE_8]]
}