//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-weights-dynamic-dequantization=true compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-grouped-matmul=false" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU6010

// CHECK-LABEL: @MoEQuantizedWeightsMatMul
module @MoEQuantizedWeightsMatMul {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2880xf16>
        DataInfo "w1_weights" : tensor<4x5760x2880x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
        DataInfo "w1_scale" : tensor<4x5760x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x1x5760xf32>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2880xf16>,
    // CHECK-SAME:              [[W1_WEIGHTS:%.+]]: tensor<4x5760x2880x!{{.+}}>, [[W1_SCALE:%.+]]: tensor<4x5760x1xf16>) -> tensor<1x4x1x5760xf32>
    func.func @main(%arg0: tensor<1x2880xf16>,
                    %arg1: tensor<4x5760x2880x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
                    %arg2: tensor<4x5760x1xf16>) -> tensor<1x4x1x5760xf32> {

        // Convert input and tile from 1x2880 to 4x2880
        %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x2880xf16> -> tensor<1x2880xf32>
        %1 = IE.Tile(%0) {repeats_values = [4, 1]} : tensor<1x2880xf32> -> tensor<4x2880xf32>
        %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2]], shape_value = [4, 1, 2880]} : tensor<4x2880xf32> -> tensor<4x1x2880xf32>

        // Convert -> Multiply -> Convert for quantized weights
        %3 = IE.Convert(%arg1) {dstElemType = f16} : tensor<4x5760x2880x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<4x5760x2880xf16>
        %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x5760x2880xf16>, tensor<4x5760x1xf16> -> tensor<4x5760x2880xf16>
        %5 = IE.Convert(%4) {dstElemType = f32} : tensor<4x5760x2880xf16> -> tensor<4x5760x2880xf32>

        // Reshape for MatMul with batch dimension
        %6 = IE.Reshape(%2) {shape_value = [1, 4, 1, 2880]} : tensor<4x1x2880xf32> -> tensor<1x4x1x2880xf32>
        %7 = IE.Reshape(%5) {shape_value = [1, 4, 5760, 2880]} : tensor<4x5760x2880xf32> -> tensor<1x4x5760x2880xf32>
        %8 = IE.MatMul(%6, %7) {transpose_b} : tensor<1x4x1x2880xf32>, tensor<1x4x5760x2880xf32> -> tensor<1x4x1x5760xf32>

        return %8 : tensor<1x4x1x5760xf32>

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

        // CHECK:       [[SCALE_SLICE0:%.+]] = IE.Slice [[W1_SCALE]] [0, 0, 0] [1, 5760, 1]
        // CHECK-SAME:      : tensor<4x5760x1xf16> to tensor<1x5760x1xf16>
        // CHECK:       [[SCALE_SLICE1:%.+]] = IE.Slice [[W1_SCALE]] [1, 0, 0] [1, 5760, 1]
        // CHECK-SAME:      : tensor<4x5760x1xf16> to tensor<1x5760x1xf16>
        // CHECK:       [[SCALE_SLICE2:%.+]] = IE.Slice [[W1_SCALE]] [2, 0, 0] [1, 5760, 1]
        // CHECK-SAME:      : tensor<4x5760x1xf16> to tensor<1x5760x1xf16>
        // CHECK:       [[SCALE_SLICE3:%.+]] = IE.Slice [[W1_SCALE]] [3, 0, 0] [1, 5760, 1]
        // CHECK-SAME:      : tensor<4x5760x1xf16> to tensor<1x5760x1xf16>


        // CHECK:       [[RESHAPE_SCALE_0:%.+]] = IE.Reshape([[SCALE_SLICE0]])
        // CHECK-SAME:      : tensor<1x5760x1xf16> -> tensor<1x192x5x6xf16>
        // CHECK:       [[LC_IN_SCALE_0:%.+]] = IE.LayoutCast([[RESHAPE_SCALE_0]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16> -> tensor<1x192x5x6xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_CONVERT_SCALE_0:%.+]] = IE.MaxPool([[LC_IN_SCALE_0]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16, {order = #NHWC}> -> tensor<1x192x5x6xf32, {order = #NHWC}>
        // CHECK:       [[LC_OUT_SCALE_0:%.+]] = IE.LayoutCast([[MAXPOOL_CONVERT_SCALE_0]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32, {order = #NHWC}> -> tensor<1x192x5x6xf32>
        // CHECK:       [[AR_SCALE_0:%.+]] = IE.AffineReshape([[LC_OUT_SCALE_0]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32> -> tensor<5760x1x1x1xf32>

        // CHECK:       [[CONV0:%.+]] = IE.Convolution([[INPUT_SLICE0]], [[WEIGHT_SLICE0]], [[AR_SCALE_0]]) {
        // CHECK-SAME:      dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 0>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>, tensor<5760x1x1x1xf32> -> tensor<1x5760x1x1xf16, {order = #NHWC}>


        // CHECK:       [[RESHAPE_SCALE_1:%.+]] = IE.Reshape([[SCALE_SLICE1]])
        // CHECK-SAME:      : tensor<1x5760x1xf16> -> tensor<1x192x5x6xf16>
        // CHECK:       [[LC_IN_SCALE_1:%.+]] = IE.LayoutCast([[RESHAPE_SCALE_1]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16> -> tensor<1x192x5x6xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_CONVERT_SCALE_1:%.+]] = IE.MaxPool([[LC_IN_SCALE_1]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16, {order = #NHWC}> -> tensor<1x192x5x6xf32, {order = #NHWC}>
        // CHECK:       [[LC_OUT_SCALE_1:%.+]] = IE.LayoutCast([[MAXPOOL_CONVERT_SCALE_1]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32, {order = #NHWC}> -> tensor<1x192x5x6xf32>
        // CHECK:       [[AR_SCALE_1:%.+]] = IE.AffineReshape([[LC_OUT_SCALE_1]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32> -> tensor<5760x1x1x1xf32>

        // CHECK:       [[CONV1:%.+]] = IE.Convolution([[INPUT_SLICE1]], [[WEIGHT_SLICE1]], [[AR_SCALE_1]]) {
        // CHECK-SAME:      dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 1>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>, tensor<5760x1x1x1xf32> -> tensor<1x5760x1x1xf16, {order = #NHWC}>


        // CHECK:       [[RESHAPE_SCALE_2:%.+]] = IE.Reshape([[SCALE_SLICE2]])
        // CHECK-SAME:      : tensor<1x5760x1xf16> -> tensor<1x192x5x6xf16>
        // CHECK:       [[LC_IN_SCALE_2:%.+]] = IE.LayoutCast([[RESHAPE_SCALE_2]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16> -> tensor<1x192x5x6xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_CONVERT_SCALE_2:%.+]] = IE.MaxPool([[LC_IN_SCALE_2]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16, {order = #NHWC}> -> tensor<1x192x5x6xf32, {order = #NHWC}>
        // CHECK:       [[LC_OUT_SCALE_2:%.+]] = IE.LayoutCast([[MAXPOOL_CONVERT_SCALE_2]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32, {order = #NHWC}> -> tensor<1x192x5x6xf32>
        // CHECK:       [[AR_SCALE_2:%.+]] = IE.AffineReshape([[LC_OUT_SCALE_2]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32> -> tensor<5760x1x1x1xf32>

        // CHECK:       [[CONV2:%.+]] = IE.Convolution([[INPUT_SLICE2]], [[WEIGHT_SLICE2]], [[AR_SCALE_2]]) {
        // CHECK-SAME:      dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 2>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>, tensor<5760x1x1x1xf32> -> tensor<1x5760x1x1xf16, {order = #NHWC}>


        // CHECK:       [[RESHAPE_SCALE_3:%.+]] = IE.Reshape([[SCALE_SLICE3]])
        // CHECK-SAME:      : tensor<1x5760x1xf16> -> tensor<1x192x5x6xf16>
        // CHECK:       [[LC_IN_SCALE_3:%.+]] = IE.LayoutCast([[RESHAPE_SCALE_3]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16> -> tensor<1x192x5x6xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_CONVERT_SCALE_3:%.+]] = IE.MaxPool([[LC_IN_SCALE_3]])
        // CHECK-SAME:      : tensor<1x192x5x6xf16, {order = #NHWC}> -> tensor<1x192x5x6xf32, {order = #NHWC}>
        // CHECK:       [[LC_OUT_SCALE_3:%.+]] = IE.LayoutCast([[MAXPOOL_CONVERT_SCALE_3]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32, {order = #NHWC}> -> tensor<1x192x5x6xf32>
        // CHECK:       [[AR_SCALE_3:%.+]] = IE.AffineReshape([[LC_OUT_SCALE_3]])
        // CHECK-SAME:      : tensor<1x192x5x6xf32> -> tensor<5760x1x1x1xf32>

        // CHECK:       [[CONV3:%.+]] = IE.Convolution([[INPUT_SLICE3]], [[WEIGHT_SLICE3]], [[AR_SCALE_3]]) {
        // CHECK-SAME:      dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 3>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      : tensor<1x2880x1x1xf16, {order = #NHWC}>, tensor<5760x2880x1x1x!qElemType, {order = #NHWC}>, tensor<5760x1x1x1xf32> -> tensor<1x5760x1x1xf16, {order = #NHWC}>


        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
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
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x262144x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x1x262144x768x!qElemType>
      // CHECK: [[GATHER_WT:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x262144x768x!qElemType>, tensor<256xsi32> -> tensor<1x1x256x768x!qElemType>
      // CHECK: [[GATHER_SCALE:%.+]] = IE.Gather([[SCALE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x262144x1xf16>, tensor<256xsi32> -> tensor<1x1x256x1xf16>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER_WT]], [[GATHER_SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x1x256x768x!qElemType>, tensor<1x1x256x1xf16> -> tensor<1x1x256x768xf16>
      // CHECK: [[LAYOUT_CAST_IN:%.+]] = IE.LayoutCast([[DYN_DEQUANT]]) {dst_order = #NHWC}
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

// -----

// CHECK-LABEL: @MoEQuantizedWeightsMatMul
module @MoEQuantizedWeightsMatMul {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2880x16x16xf16>
        DataInfo "w1_weights" : tensor<128x2880x1x1x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>
        DataInfo "w1_scale" : tensor<128x1x1x1xf16>
        DataInfo "multiply_scale" : tensor<1x128x1x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x16x16xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2880x16x16xf16>,
    // CHECK-SAME:              [[W1_WEIGHTS:%.+]]: tensor<128x2880x1x1x!{{.+}}>, [[W1_SCALE:%.+]]: tensor<128x1x1x1xf16>, [[SCALE:%.+]]: tensor<1x128x1x1xf16>) -> tensor<1x128x16x16xf16>
    func.func @main(%arg0: tensor<1x2880x16x16xf16>,
                    %arg1: tensor<128x2880x1x1x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>>,
                    %arg2: tensor<128x1x1x1xf16>, %arg3: tensor<1x128x1x1xf16>) -> tensor<1x128x16x16xf16> {

        // Convert -> Multiply -> Convert for quantized weights
        %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<128x2880x1x1x!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>> -> tensor<128x2880x1x1xf16>
        %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x2880x1x1xf16>, tensor<128x1x1x1xf16> -> tensor<128x2880x1x1xf16>

        %2 = IE.Convolution(%arg0, %1) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]} : tensor<1x2880x16x16xf16>, tensor<128x2880x1x1xf16> -> tensor<1x128x16x16xf16>
        %3 = IE.Multiply(%2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x16x16xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x16x16xf16>

        return %3 : tensor<1x128x16x16xf16>

      // CHECK: [[QUANT_CAST_W:%.+]] = IE.QuantizeCast([[W1_WEIGHTS]]) {dstElemType = !qElemType}

      // CHECK: [[AFFINE_RESHAPE_SCALE_IN:%.+]] = IE.AffineReshape([[W1_SCALE]])
      // CHECK-SAME:  shape_value = [1, 128, 1, 1]}
      // CHECK: [[LC_SCALE_OUT:%.+]] = IE.LayoutCast([[SCALE]]) {dst_order = #NHWC}
      // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {order = #NHWC}>
      // CHECK: [[LC_SCALE_IN:%.+]] = IE.LayoutCast([[AFFINE_RESHAPE_SCALE_IN]]) {dst_order = #NHWC}
      // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf16, {order = #NHWC}>
      // CHECK: [[MUL_SCALES:%.+]] = IE.Multiply([[LC_SCALE_OUT]], [[LC_SCALE_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
      // CHECK-SAME: tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1x128x1x1xf16, {order = #NHWC}> -> tensor<1x128x1x1xf32, {order = #NHWC}>
      // CHECK: [[LC_MUL:%.+]] = IE.LayoutCast([[MUL_SCALES]]) {dst_order = #NCHW}
      // CHECK-SAME: tensor<1x128x1x1xf32, {order = #NHWC}> -> tensor<1x128x1x1xf32>
      // CHECK: [[AFFINE_RESHAPE_SCALE:%.+]] = IE.AffineReshape([[LC_MUL]])
      // CHECK-SAME:  shape_value = [128, 1, 1, 1]}

      // CHECK: [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[INPUT]])
      // CHECK: [[PERMUTE_CAST_W:%.+]] = IE.PermuteCast([[QUANT_CAST_W]])
      // CHECK-SAME: {dst_order = #NHWC, mem_perm = #NHWC}

      // CHECK: [[CONV:%.+]] = IE.Convolution([[PERMUTE_QUANTIZE]], [[PERMUTE_CAST_W]], [[AFFINE_RESHAPE_SCALE]])
      // CHECK-SAME: tensor<1x2880x16x16xf16, {order = #NHWC}>, tensor<128x2880x1x1x!qElemType, {order = #NHWC}>, tensor<128x1x1x1xf32> -> tensor<1x128x16x16xf16>

      // CHECK: return [[CONV]] : tensor<1x128x16x16xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: ![[QTYPE:.+]] = !quant.uniform<!quant.quantile<ui4:f16,
// CHECK: @SmallICWithI4WeightsMatMulWAI
module @SmallICWithI4WeightsMatMulWAI {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingIDU : true
    config.Option @config.AutoPaddingODU : true
}

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x4xf32>
        DataInfo "weights" : tensor<16x4xsi4>
        DataInfo "scale" : tensor<16x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x16xf32>
    }

// CHECK: func.func @main([[ACT:%.+]]: tensor<1x1x4xf32>, [[WRAW:%.+]]: tensor<16x4xsi4>, [[SCALE:%.+]]: tensor<16x1xf16>)
func.func @main(%arg0: tensor<1x1x4xf32>, %arg1: tensor<16x4xsi4>, %arg2: tensor<16x1xf16>) -> tensor<1x1x16xf32> {
    // Second operand producer chain: raw si4 weights dequantized via Convert + Multiply + Convert.
    %0 = IE.Convert(%arg1) {dstElemType = f16} : tensor<16x4xsi4> -> tensor<16x4xf16>
    %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x4xf16>, tensor<16x1xf16> -> tensor<16x4xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<16x4xf16> -> tensor<16x4xf32>
    %3 = IE.MatMul(%arg0, %2) {transpose_b} : tensor<1x1x4xf32>, tensor<16x4xf32> -> tensor<1x1x16xf32>
    return %3 : tensor<1x1x16xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<16x12x1x1x![[QTYPE]], {order = #NHWC}> = dense<0>

    // CHECK: [[ARG_RESHAPE:%.+]] = IE.AffineReshape([[ACT]])
    // CHECK-SAME:    : tensor<1x1x4xf32> -> tensor<1x1x1x4xf32>
    // CHECK: [[ARG_CONVERT:%.+]] = IE.Convert([[ARG_RESHAPE]])
    // CHECK-SAME:    : tensor<1x1x1x4xf32> -> tensor<1x1x1x4xf16>
    // CHECK: [[ARG_RESHAPE2:%.+]] = IE.AffineReshape([[ARG_CONVERT]])
    // CHECK-SAME:    : tensor<1x1x1x4xf16> -> tensor<1x4x1x1xf16>

    // Weights processing
    // CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[WRAW]]) {dstElemType = ![[QTYPE]]}
    // CHECK-SAME: : tensor<16x4xsi4> -> tensor<16x4x![[QTYPE]]>
    // CHECK: [[W_RESHAPE:%.+]] = IE.AffineReshape([[QCAST]])
    // CHECK-SAME: : tensor<16x4x![[QTYPE]]> -> tensor<16x4x1x1x![[QTYPE]]>
    // CHECK: [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK-SAME: : tensor<16x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK: [[SCALE_PERM:%.+]] = IE.PermuteCast([[SCALE_RESHAPE]])
    // CHECK-SAME: : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK: [[SCALE_CONVERT:%.+]] = IE.MaxPool([[SCALE_PERM]])
    // CHECK-SAME: : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK: [[SCALE_PERM2:%.+]] = IE.PermuteCast([[SCALE_CONVERT]])
    // CHECK-SAME: : tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x1x1xf32>
    // CHECK: [[SCALE_FINAL:%.+]] = IE.AffineReshape([[SCALE_PERM2]])
    // CHECK-SAME: : tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>

    // CHECK: [[A_PERM:%.+]] = IE.PermuteCast([[ARG_RESHAPE2]])
    // CHECK-SAME:   : tensor<1x4x1x1xf16> -> tensor<1x4x1x1xf16, {order = #NHWC}>
    // CHECK: [[W_PERM:%.+]] = IE.PermuteCast([[W_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:   -> tensor<16x4x1x1x![[QTYPE]], {order = #NHWC}>

    // Activation and weights expansion
    // CHECK: [[ACT_INPUT:%.+]] = IE.Expand([[A_PERM]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:   : tensor<1x4x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK: [[W_CONCAT:%.+]] = IE.Concat([[W_PERM]], [[CST]])
    // CHECK-SAME:   -> tensor<16x16x1x1x![[QTYPE]], {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[ACT_INPUT]], [[W_CONCAT]], [[SCALE_FINAL]])
    // CHECK-SAME: input_padding = [0, 12, 0, 0], output_padding = [0, 0, 0, 0]
    // CHECK-SAME: : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<16x16x1x1x![[QTYPE]], {order = #NHWC}>, tensor<16x1x1x1xf32>
    // CHECK-SAME: -> tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK: [[OUT_PERM:%.+]] = IE.PermuteCast([[CONV]])
    // CHECK-SAME:    -> tensor<1x16x1x1xf16>
    // CHECK: [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[OUT_PERM]])
    // CHECK-SAME:    -> tensor<1x1x1x16xf16>
    // CHECK: [[OUT_CONVERT:%.+]] = IE.Convert([[OUT_RESHAPE]])
    // CHECK-SAME:    -> tensor<1x1x1x16xf32>
    // CHECK: [[OUT:%.+]] = IE.AffineReshape([[OUT_CONVERT]])
    // CHECK-SAME: -> tensor<1x1x16xf32>

    // CHECK: return [[OUT]] : tensor<1x1x16xf32>
}
}

// -----

// Batch MatMul with weights fetched via Gather + AffineReshape + DynamicDequantize (MoE pattern).
// matmul-inputs-to-2d-set splits the MatMul into per-batch Gather ops, inserting per-batch
// index slices. propagate-affine-reshape moves the batch-leading AffineReshape before each
// Gather (axis 0 -> 1) and then hoists the 4-D NCE layout reshape in turn (axis 1 -> 0),
// so both per-batch Gathers share a single QuantizeCast on the full weight tensor.

// CHECK: !qElemType = !quant.uniform<!quant.quantile<ui4:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>
// CHECK-LABEL: @MatMulWithDequantizeGatherChain
module @MatMulWithDequantizeGatherChain {
net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x1x32xf16>
        DataInfo "weights" : tensor<64x32xsi4>
        DataInfo "flat_indices" : tensor<32xi64>
        DataInfo "scale" : tensor<2x16x1xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x1x16xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2x1x32xf16>,
    // CHECK-SAME:              [[WEIGHTS:%.+]]: tensor<64x32xsi4>, [[FLAT_IDX:%.+]]: tensor<32xi64>, [[SCALE:%.+]]: tensor<2x16x1xf16>)
    func.func @main(%arg0: tensor<1x2x1x32xf16>,
                    %arg1: tensor<64x32xsi4>,
                    %arg2: tensor<32xi64>,
                    %arg3: tensor<2x16x1xf16>) -> tensor<1x2x1x16xf16> {
        %gather = IE.Gather(%arg1, %arg2) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
            : tensor<64x32xsi4>, tensor<32xi64> -> tensor<32x32xsi4>
        %reshape = IE.AffineReshape(%gather) {dim_mapping = [[0, 1], [2]], shape_value = [2, 16, 32]}
            : tensor<32x32xsi4> -> tensor<2x16x32xsi4>
        %dequant = IE.DynamicDequantize(%reshape, %arg3) {dstElemType = f16}
            : tensor<2x16x32xsi4>, tensor<2x16x1xf16> -> tensor<2x16x32xf16>
        %matmul = IE.MatMul(%arg0, %dequant) {transpose_b}
            : tensor<1x2x1x32xf16>, tensor<2x16x32xf16> -> tensor<1x2x1x16xf16>
        return %matmul : tensor<1x2x1x16xf16>

        // Scale sliced per batch.
        // CHECK:       [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[SCALE]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [2, 3]], shape_value = [2, 16, 1, 1]}
        // CHECK-SAME:      tensor<2x16x1xf16> -> tensor<2x16x1x1xf16>
        // CHECK:       [[SCALE_PERM:%.+]] = IE.PermuteCast([[SCALE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<2x16x1x1xf16> -> tensor<2x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE_0:%.+]] = IE.Slice [[SCALE_PERM]] [0, 0, 0, 0] [1, 16, 1, 1]
        // CHECK-SAME:      tensor<2x16x1x1xf16, {order = #NHWC}> to tensor<1x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SCALE_SLICE_1:%.+]] = IE.Slice [[SCALE_PERM]] [1, 0, 0, 0] [1, 16, 1, 1]
        // CHECK-SAME:      tensor<2x16x1x1xf16, {order = #NHWC}> to tensor<1x16x1x1xf16, {order = #NHWC}>

        // Activation reshaped to 4-D NHWC and sliced per batch.
        // CHECK:       [[IN_RESHAPE:%.+]] = IE.AffineReshape([[INPUT]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [2, 32, 1, 1]}
        // CHECK-SAME:      tensor<1x2x1x32xf16> -> tensor<2x32x1x1xf16>
        // CHECK:       [[IN_PERM:%.+]] = IE.PermuteCast([[IN_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<2x32x1x1xf16> -> tensor<2x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[IN_SLICE_0:%.+]] = IE.Slice [[IN_PERM]] [0, 0, 0, 0] [1, 32, 1, 1]
        // CHECK-SAME:      tensor<2x32x1x1xf16, {order = #NHWC}> to tensor<1x32x1x1xf16, {order = #NHWC}>
        // CHECK:       [[IN_SLICE_1:%.+]] = IE.Slice [[IN_PERM]] [1, 0, 0, 0] [1, 32, 1, 1]
        // CHECK-SAME:      tensor<2x32x1x1xf16, {order = #NHWC}> to tensor<1x32x1x1xf16, {order = #NHWC}>

        // Index slices for per-batch Gathers.
        // CHECK:       [[IDX_SLICE_0:%.+]] = IE.Slice [[FLAT_IDX]] [0] [16]
        // CHECK-SAME:      tensor<32xi64> to tensor<16xi64>

        // Weight reshaped to 3-D, quantcast, then reshaped to 4-D NCE layout.
        // Both per-batch Gathers share the same QuantizeCast result.
        // CHECK:       [[W_RESHAPE_3D:%.+]] = IE.AffineReshape([[WEIGHTS]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2]], shape_value = [1, 64, 32]}
        // CHECK-SAME:      tensor<64x32xsi4> -> tensor<1x64x32xsi4>
        // CHECK:       [[W_QCAST:%.+]] = IE.QuantizeCast([[W_RESHAPE_3D]]) {dstElemType = !qElemType}
        // CHECK-SAME:      tensor<1x64x32xsi4> -> tensor<1x64x32x!qElemType>
        // CHECK:       [[W_RESHAPE_4D_0:%.+]] = IE.AffineReshape([[W_QCAST]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [64, 32, 1, 1]}
        // CHECK-SAME:      tensor<1x64x32x!qElemType> -> tensor<64x32x1x1x!qElemType>

        // Per-batch Gather at axis 0 on the 4-D weight tensor.
        // CHECK:       [[GATHER_0:%.+]] = IE.Gather([[W_RESHAPE_4D_0]], [[IDX_SLICE_0]])
        // CHECK-SAME:      {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
        // CHECK-SAME:      tensor<64x32x1x1x!qElemType>, tensor<16xi64> -> tensor<16x32x1x1x!qElemType>

        // CHECK:       [[IDX_SLICE_1:%.+]] = IE.Slice [[FLAT_IDX]] [16] [16]
        // CHECK-SAME:      tensor<32xi64> to tensor<16xi64>
        // CHECK:       [[W_RESHAPE_4D_1:%.+]] = IE.AffineReshape([[W_QCAST]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [64, 32, 1, 1]}
        // CHECK-SAME:      tensor<1x64x32x!qElemType> -> tensor<64x32x1x1x!qElemType>
        // CHECK:       [[GATHER_1:%.+]] = IE.Gather([[W_RESHAPE_4D_1]], [[IDX_SLICE_1]])
        // CHECK-SAME:      {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
        // CHECK-SAME:      tensor<64x32x1x1x!qElemType>, tensor<16xi64> -> tensor<16x32x1x1x!qElemType>

        // Scale batch 0 converted to f32 via MaxPool for NCE Convolution.
        // CHECK:       [[SCALE_F32_0:%.+]] = IE.MaxPool([[SCALE_SLICE_0]])
        // CHECK-SAME:      {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
        // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
        // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf32, {order = #NHWC}>
        // CHECK:       [[SCALE_PERM_0:%.+]] = IE.PermuteCast([[SCALE_F32_0]])
        // CHECK-SAME:      tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x1x1xf32>
        // CHECK:       [[SCALE_RESHAPE_0:%.+]] = IE.AffineReshape([[SCALE_PERM_0]])
        // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>

        // Per-batch NCE Convolution for batch 0.
        // CHECK:       [[W_PERM_0:%.+]] = IE.PermuteCast([[GATHER_0]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<16x32x1x1x!qElemType> -> tensor<16x32x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[IN_SLICE_0]], [[W_PERM_0]], [[SCALE_RESHAPE_0]])
        // CHECK-SAME:      {dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 0>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      tensor<1x32x1x1xf16, {order = #NHWC}>, tensor<16x32x1x1x!qElemType, {order = #NHWC}>,
        // CHECK-SAME:      tensor<16x1x1x1xf32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

        // Scale batch 1 converted to f32 and per-batch NCE Convolution for batch 1.
        // CHECK:       [[SCALE_F32_1:%.+]] = IE.MaxPool([[SCALE_SLICE_1]])
        // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x1xf32, {order = #NHWC}>
        // CHECK:       [[SCALE_PERM_1:%.+]] = IE.PermuteCast([[SCALE_F32_1]])
        // CHECK-SAME:      tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x1x1xf32>
        // CHECK:       [[SCALE_RESHAPE_1:%.+]] = IE.AffineReshape([[SCALE_PERM_1]])
        // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
        // CHECK:       [[W_PERM_1:%.+]] = IE.PermuteCast([[GATHER_1]]) {dst_order = #NHWC, mem_perm = #NHWC}
        // CHECK-SAME:      tensor<16x32x1x1x!qElemType> -> tensor<16x32x1x1x!qElemType, {order = #NHWC}>
        // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[IN_SLICE_1]], [[W_PERM_1]], [[SCALE_RESHAPE_1]])
        // CHECK-SAME:      {dilations = [1, 1], loop_tile_coord = {axes = array<i64: 0>, indices = array<i64: 1>}, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], tiling_loop_index = 0 : i64}
        // CHECK-SAME:      tensor<1x32x1x1xf16, {order = #NHWC}>, tensor<16x32x1x1x!qElemType, {order = #NHWC}>,
        // CHECK-SAME:      tensor<16x1x1x1xf32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

        // Concat and reshape back to original output shape.
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CONV_0]], [[CONV_1]])
        // CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}
        // CHECK-SAME:      tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<2x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[PERM_BACK:%.+]] = IE.PermuteCast([[CONCAT]])
        // CHECK-SAME:      tensor<2x16x1x1xf16, {order = #NHWC}> -> tensor<2x16x1x1xf16>
        // CHECK:       [[OUT:%.+]] = IE.AffineReshape([[PERM_BACK]])
        // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 2, 1, 16]}
        // CHECK-SAME:      tensor<2x16x1x1xf16> -> tensor<1x2x1x16xf16>
        // CHECK:       return [[OUT]] : tensor<1x2x1x16xf16>
    }
}

// -----

// A per-channel runtime bias converted to f16 before Add must be recovered as f32 by the
// FuseAddToConvBias pass in the DefaultHW IE pipeline.
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @RuntimeGroupConvolutionBiasAdd
module @RuntimeGroupConvolutionBiasAdd {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
        DataInfo "bias" : tensor<1x1x1x16xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x8x8xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x16x8x8xf16, {order = #NHWC}>,
    // CHECK-SAME:              [[BIAS:%.+]]: tensor<1x1x1x16xf32>)
    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>,
                    %bias: tensor<1x1x1x16xf32>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        %filter = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> =
            dense<1.000000e+00> : tensor<16x1x1x1xf16>, [#const.Reorder<#NHWC>]
        %conv = IE.GroupConvolution(%input, %filter) {
                dilations = [1, 1],
                groups = 16 : i64,
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1]
            } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x1x1x1xf16, {order = #NHWC}>
                -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %biasF16 = IE.Convert(%bias) {dstElemType = f16} : tensor<1x1x1x16xf32> -> tensor<1x1x1x16xf16>
        %biasReshaped = IE.AffineReshape(%biasF16) {
                dim_mapping = [[0], [0], [0], [1, 2, 3]],
                shape_value = [1, 16, 1, 1]
            } : tensor<1x1x1x16xf16> -> tensor<1x16x1x1xf16>
        %biasPermuted = IE.PermuteCast(%biasReshaped) {dst_order = #NHWC, mem_perm = #NHWC}
            : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16, {order = #NHWC}>
        %tile = IE.Tile(%biasPermuted) {repeats_values = [1, 1, 8, 8]}
            : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %add = IE.Add(%conv, %tile) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
            -> tensor<1x16x8x8xf16, {order = #NHWC}>
        return %add : tensor<1x16x8x8xf16, {order = #NHWC}>
        // CHECK-NOT: IE.Add
        // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
        // CHECK: [[BIAS_RESHAPED:%.+]] = IE.AffineReshape([[BIAS]])
        // CHECK-SAME: tensor<1x1x1x16xf32> -> tensor<1x16x1x1xf32>
        // CHECK: [[BIAS:%.+]] = IE.PermuteCast([[BIAS_RESHAPED]])
        // CHECK: IE.GroupConvolution([[INPUT]], [[WEIGHTS]], [[BIAS]])
    }
}
