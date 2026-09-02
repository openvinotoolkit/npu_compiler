//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-weights-dynamic-dequantization=true compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-grouped-matmul=false enable-nce-eltwise-multiply=false" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @MoEQuantizedWeightsMatMul
module @MoEQuantizedWeightsMatMul {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2880xf16>
        DataInfo "w1_weights" : tensor<4x5760x2880x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
        DataInfo "w1_scale" : tensor<4x5760x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x1x5760xf32>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2880xf16>,
    // CHECK-SAME:              [[W1_WEIGHTS:%.+]]: tensor<4x5760x2880x!{{.+}}>, [[W1_SCALE:%.+]]: tensor<4x5760x1xf16>) -> tensor<1x4x1x5760xf32>
    func.func @main(%arg0: tensor<1x2880xf16>,
                    %arg1: tensor<4x5760x2880x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
                    %arg2: tensor<4x5760x1xf16>) -> tensor<1x4x1x5760xf32> {

        // Convert input and tile from 1x2880 to 4x2880
        %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x2880xf16> -> tensor<1x2880xf32>
        %1 = IE.Tile(%0) {repeats_values = [4, 1]} : tensor<1x2880xf32> -> tensor<4x2880xf32>
        %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2]], shape_value = [4, 1, 2880]} : tensor<4x2880xf32> -> tensor<4x1x2880xf32>

        // Convert -> Multiply -> Convert for quantized weights
        %3 = IE.Convert(%arg1) {dstElemType = f16} : tensor<4x5760x2880x!QuantileType.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<4x5760x2880xf16>
        %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x5760x2880xf16>, tensor<4x5760x1xf16> -> tensor<4x5760x2880xf16>
        %5 = IE.Convert(%4) {dstElemType = f32} : tensor<4x5760x2880xf16> -> tensor<4x5760x2880xf32>

        // Reshape for MatMul with batch dimension
        %6 = IE.Reshape(%2) {shape_value = [1, 4, 1, 2880]} : tensor<4x1x2880xf32> -> tensor<1x4x1x2880xf32>
        %7 = IE.Reshape(%5) {shape_value = [1, 4, 5760, 2880]} : tensor<4x5760x2880xf32> -> tensor<1x4x5760x2880xf32>
        %8 = IE.MatMul(%6, %7) {transpose_b} : tensor<1x4x1x2880xf32>, tensor<1x4x5760x2880xf32> -> tensor<1x4x1x5760xf32>

        return %8 : tensor<1x4x1x5760xf32>

        // CHECK-DAG:   [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[W1_SCALE]]) {
        // CHECK-SAME(LITERAL):      dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [23040, 1, 1, 1]}
        // CHECK-SAME:      : tensor<4x5760x1xf16> -> tensor<23040x1x1x1xf16>
        // CHECK:       [[SCALE_PERM:%.+]] = IE.PermuteCast([[SCALE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      : tensor<23040x1x1x1xf16> -> tensor<23040x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE0:%.+]] = IE.Slice [[SCALE_PERM]] [0, 0, 0, 0] [5760, 1, 1, 1]
        // CHECK-SAME:      : tensor<23040x1x1x1xf16, {order = #NHWC}> to tensor<5760x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE1:%.+]] = IE.Slice [[SCALE_PERM]] [5760, 0, 0, 0] [5760, 1, 1, 1]
        // CHECK-SAME:      : tensor<23040x1x1x1xf16, {order = #NHWC}> to tensor<5760x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE2:%.+]] = IE.Slice [[SCALE_PERM]] [11520, 0, 0, 0] [5760, 1, 1, 1]
        // CHECK-SAME:      : tensor<23040x1x1x1xf16, {order = #NHWC}> to tensor<5760x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE3:%.+]] = IE.Slice [[SCALE_PERM]] [17280, 0, 0, 0] [5760, 1, 1, 1]
        // CHECK-SAME:      : tensor<23040x1x1x1xf16, {order = #NHWC}> to tensor<5760x1x1x1xf16, {order = #NHWC}>

        // CHECK:       [[QCAST:%.+]] = IE.QuantizeCast([[W1_WEIGHTS]]) {dstElemType = !qElemType}
        // CHECK:       [[WEIGHT_RESHAPE:%.+]] = IE.AffineReshape([[QCAST]]) {
        // CHECK-SAME(LITERAL):      dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [23040, 2880, 1, 1]}
        // CHECK-SAME:      : tensor<4x5760x2880x!qElemType> -> tensor<23040x2880x1x1x!qElemType>
        // CHECK:       [[WEIGHT_PERM:%.+]] = IE.PermuteCast([[WEIGHT_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      : tensor<23040x2880x1x1x!qElemType> -> tensor<23040x2880x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[WEIGHT_SLICE3:%.+]] = IE.Slice [[WEIGHT_PERM]] [17280, 0, 0, 0] [5760, 2880, 1, 1]
        // CHECK-SAME:      : tensor<23040x2880x1x1x!qElemType, {order = #NHWC}> to tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[WEIGHT_SLICE2:%.+]] = IE.Slice [[WEIGHT_PERM]] [11520, 0, 0, 0] [5760, 2880, 1, 1]
        // CHECK-SAME:      : tensor<23040x2880x1x1x!qElemType, {order = #NHWC}> to tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[WEIGHT_SLICE1:%.+]] = IE.Slice [[WEIGHT_PERM]] [5760, 0, 0, 0] [5760, 2880, 1, 1]
        // CHECK-SAME:      : tensor<23040x2880x1x1x!qElemType, {order = #NHWC}> to tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[WEIGHT_SLICE0:%.+]] = IE.Slice [[WEIGHT_PERM]] [0, 0, 0, 0] [5760, 2880, 1, 1]
        // CHECK-SAME:      : tensor<23040x2880x1x1x!qElemType, {order = #NHWC}> to tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>

        // CHECK:       [[INPUT_RESHAPE0:%.+]] = IE.AffineReshape([[INPUT]]) {
        // CHECK-SAME(LITERAL):      dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 2880]}
        // CHECK-SAME:      : tensor<1x2880xf16> -> tensor<1x1x1x2880xf16>
        // CHECK:       [[INPUT_TILE:%.+]] = IE.Tile([[INPUT_RESHAPE0]]) {repeats_values = [1, 1, 4, 1]}
        // CHECK-SAME:      : tensor<1x1x1x2880xf16> -> tensor<1x1x4x2880xf16>
        // CHECK:       [[INPUT_RESHAPE1:%.+]] = IE.AffineReshape([[INPUT_TILE]]) {
        // CHECK-SAME(LITERAL):      dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [4, 2880, 1, 1]}
        // CHECK-SAME:      : tensor<1x1x4x2880xf16> -> tensor<4x2880x1x1xf16>
        // CHECK:       [[INPUT_PERM:%.+]] = IE.PermuteCast([[INPUT_RESHAPE1]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      : tensor<4x2880x1x1xf16> -> tensor<4x2880x1x1xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_SLICE0:%.+]] = IE.Slice [[INPUT_PERM]] [0, 0, 0, 0] [1, 2880, 1, 1]
        // CHECK-SAME:      : tensor<4x2880x1x1xf16, {order = #NHWC}> to tensor<1x2880x1x1xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_SLICE1:%.+]] = IE.Slice [[INPUT_PERM]] [1, 0, 0, 0] [1, 2880, 1, 1]
        // CHECK-SAME:      : tensor<4x2880x1x1xf16, {order = #NHWC}> to tensor<1x2880x1x1xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_SLICE2:%.+]] = IE.Slice [[INPUT_PERM]] [2, 0, 0, 0] [1, 2880, 1, 1]
        // CHECK-SAME:      : tensor<4x2880x1x1xf16, {order = #NHWC}> to tensor<1x2880x1x1xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_SLICE3:%.+]] = IE.Slice [[INPUT_PERM]] [3, 0, 0, 0] [1, 2880, 1, 1]
        // CHECK-SAME:      : tensor<4x2880x1x1xf16, {order = #NHWC}> to tensor<1x2880x1x1xf16, {order = #NHWC}>

        // CHECK:       [[CONV0:%.+]] = IE.Convolution([[INPUT_SLICE0]], [[WEIGHT_SLICE0]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GCONV0:%.+]] = IE.GroupConvolution([[CONV0]], [[SCALE_SLICE0]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 5760 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<5760x1x1x1xf16, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV1:%.+]] = IE.Convolution([[INPUT_SLICE1]], [[WEIGHT_SLICE1]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GCONV1:%.+]] = IE.GroupConvolution([[CONV1]], [[SCALE_SLICE1]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 5760 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<5760x1x1x1xf16, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV2:%.+]] = IE.Convolution([[INPUT_SLICE2]], [[WEIGHT_SLICE2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GCONV2:%.+]] = IE.GroupConvolution([[CONV2]], [[SCALE_SLICE2]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 5760 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<5760x1x1x1xf16, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV3:%.+]] = IE.Convolution([[INPUT_SLICE3]], [[WEIGHT_SLICE3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[GCONV3:%.+]] = IE.GroupConvolution([[CONV3]], [[SCALE_SLICE3]]) {
        // CHECK-SAME:      dilations = [1, 1], groups = 5760 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<5760x1x1x1xf16, {order = #NHWC}> -> tensor<1x5760x1x1xf16, {order = #NHWC}>

        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[GCONV0]], [[GCONV1]], [[GCONV2]], [[GCONV3]])
        // CHECK-SAME(LITERAL):      {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]}
        // CHECK-SAME:      : tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<1x5760x1x1xf16, {order = #NHWC}>, tensor<1x5760x1x1xf16, {order = #NHWC}> -> tensor<4x5760x1x1xf16, {order = #NHWC}>
        // CHECK:       [[PERM_BACK:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NCHW, mem_perm = #NWCH}
        // CHECK-SAME:      : tensor<4x5760x1x1xf16, {order = #NHWC}> -> tensor<4x5760x1x1xf16>
        // CHECK:       [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[PERM_BACK]]) {
        // CHECK-SAME(LITERAL):      dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4, 1, 5760]}
        // CHECK-SAME:      : tensor<4x5760x1x1xf16> -> tensor<1x4x1x5760xf16>
        // CHECK:       [[CONVERT_OUT:%.+]] = IE.Convert([[RESHAPE_OUT]]) {dstElemType = f32}
        // CHECK-SAME:      : tensor<1x4x1x5760xf16> -> tensor<1x4x1x5760xf32>

        // CHECK:       return [[CONVERT_OUT]] : tensor<1x4x1x5760xf32>
    }
}

// -----

// int4 embedding table WD chain (Const -> Multiply(per-row scale) -> Gather) is routed through
// DynamicDequantize. swap-operations-with-gather-and-slice hoists Gather before DynamicDequantize so that
// dequantization runs on the gathered rows only, not the full 262144-row table.

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt4WithDynamicDequantize
module @EmbeddingInt4WithDynamicDequantize {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x768xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x768xf32> {
      %cst_wt = const.Declare tensor<262144x768xf32> = dense<1> : tensor<262144x768xsi4>,
          [#const.ConvertElemType<si8>, #const.CastElemType<f32>]
      %cst_scale = const.Declare tensor<262144x1xf32> = dense<3.9215686e-3> : tensor<262144x1xf32>
      %cst_splat = const.Declare tensor<1x1xf32> = dense<2.0> : tensor<1x1xf32>

      %mul_wd = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<262144x768xf32>, tensor<262144x1xf32> -> tensor<262144x768xf32>
      %gather = IE.Gather(%mul_wd, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<262144x768xf32>, tensor<256xsi32> -> tensor<256x768xf32>
      %mul_out = IE.Multiply(%gather, %cst_splat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<256x768xf32>, tensor<1x1xf32> -> tensor<256x768xf32>
      return %mul_out : tensor<256x768xf32>

      // CHECK-NOT: IE.FakeQuantize
      // CHECK-DAG: [[GROUP_CONV_WT:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x262144x1x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x262144x1x768x!qElemType>
      // CHECK: [[GATHER_WT:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x262144x1x768x!qElemType>, tensor<256xsi32> -> tensor<1x256x1x768x!qElemType>
      // CHECK: [[GATHER_SCALE:%.+]] = IE.Gather([[SCALE]], [[INDICES]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x262144x1x1xf16>, tensor<256xsi32> -> tensor<1x256x1x1xf16>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER_WT]], [[GATHER_SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x256x1x768x!qElemType>, tensor<1x256x1x1xf16> -> tensor<1x256x1x768xf16>
      // CHECK: [[RESHAPE0:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
      // CHECK-SAME: tensor<1x256x1x768xf16> -> tensor<1x1x256x768xf16>
      // CHECK: [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[RESHAPE0]]) {dst_order = #NHWC}
      // CHECK-SAME: tensor<1x1x256x768xf16> -> tensor<1x1x256x768xf16, {order = #NHWC}>
      // CHECK: [[SHAPE_CAST_IN:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 48]} inputs([[LAYOUT_CAST_IN]]
      // CHECK-SAME: tensor<1x1x256x768xf16, {order = #NHWC}>) -> tensor<1x16x256x48xf16, {order = #NHWC}>
      // CHECK: [[GROUP_CONV:%.+]] = IE.GroupConvolution([[SHAPE_CAST_IN]], [[GROUP_CONV_WT]])
      // CHECK-SAME: groups = 16 : i64
      // CHECK-SAME: tensor<1x16x256x48xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x48xf32, {order = #NHWC}>
      // CHECK: [[SHAPE_CAST_OUT:%.+]] = IE.ShapeCast {shape = [1, 1, 256, 768]} inputs([[GROUP_CONV]]
      // CHECK-SAME: tensor<1x16x256x48xf32, {order = #NHWC}>) -> tensor<1x1x256x768xf32, {order = #NHWC}>
      // CHECK: [[LAYOUT_CAST_OUT:%.+]] = IE.LayoutCast([[SHAPE_CAST_OUT]]) {dst_order = #NCHW}
      // CHECK-SAME: tensor<1x1x256x768xf32, {order = #NHWC}> -> tensor<1x1x256x768xf32>
      // CHECK: [[RESHAPE1:%.+]] = IE.AffineReshape([[LAYOUT_CAST_OUT]])
      // CHECK-SAME: tensor<1x1x256x768xf32> -> tensor<256x768xf32>
      // CHECK: return [[RESHAPE1]] : tensor<256x768xf32>
    }
}
