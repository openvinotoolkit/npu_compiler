//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x42840x17xf16>
    }
    outputsInfo : {
        DataInfo "reducemax" : tensor<1x42840x1xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x42840x17xf16>)
    func.func @main(%arg0: tensor<1x42840x17xf16>) -> tensor<1x42840x1xf16> {
        %0 = IE.ReduceMax(%arg0) {axes_value = [2], keep_dims} : tensor<1x42840x17xf16> -> tensor<1x42840x1xf16>
        return %0 : tensor<1x42840x1xf16>
    }

        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [1, 42840, 1] : tensor<1x42840x17xf16> to tensor<1x42840x1xf16>
        // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[ARG0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 17]]} : tensor<1x42840x17xf16>, tensor<1x42840x1xf16> -> tensor<1x42840x18xf16>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<1x42840x18xf16> -> tensor<1x42840x32xf16>
        // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 6, 7140, 32]} : tensor<1x42840x32xf16> -> tensor<1x6x7140x32xf16>
        // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[AFFINERESHAPE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x6x7140x32xf16> -> tensor<1x16x7140x32xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 16, 7140, 18] : tensor<1x16x7140x32xf16, {order = #NHWC}> to tensor<1x16x7140x18xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SLICE2]]) {kernel_size = [1, 6], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 6]} : tensor<1x16x7140x18xf16, {order = #NHWC}> -> tensor<1x16x7140x3xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7140x3xf16, {order = #NHWC}> -> tensor<1x16x7140x1xf16, {order = #NWCH}>
        // CHECK:       [[PERMUTE_0:%.+]] = IE.PermuteCast([[MAXPOOL2]]) {
        // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NHWC} : tensor<1x16x7140x1xf16, {order = #NWCH}> -> tensor<1x16x7140x1xf16>
        // CHECK:       [[SLICE3:%.+]] = IE.Slice [[PERMUTE_0]] [0, 0, 0, 0] [1, 6, 7140, 1] : tensor<1x16x7140x1xf16> to tensor<1x6x7140x1xf16
        // CHECK:       [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[SLICE3]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 42840, 1]} : tensor<1x6x7140x1xf16> -> tensor<1x42840x1xf16>
        // CHECK:       return [[AFFINERESHAPE2]] : tensor<1x42840x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReduceMinWithLargeTensorToPooling
module @ConvertReduceMinWithLargeTensorToPooling {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x128x512x16xf16, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x128x512x16xf16>) -> tensor<1xf16> {
    func.func @main(%arg0: tensor<1x128x512x16xf16>) -> tensor<1xf16> {
        %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x128x512x16xf16> -> tensor<1xf16>
        return %0 : tensor<1xf16>

        // CHECK:       [[NEGATIVE_IN_1:%.+]] = IE.Negative([[INPUT]]) : tensor<1x128x512x16xf16> -> tensor<1x128x512x16xf16>
        // CHECK:       [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[NEGATIVE_IN_1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x512x16xf16> -> tensor<1x128x512x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_1:%.+]] = IE.MaxPool([[PERMUTE_QUANTIZE]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x128x512x16xf16, {order = #NHWC}> -> tensor<1x128x64x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_2:%.+]] = IE.MaxPool([[MAXPOOL_1]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x128x64x16xf16, {order = #NHWC}> -> tensor<1x128x8x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_3:%.+]] = IE.MaxPool([[MAXPOOL_2]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x8x16xf16, {order = #NHWC}> -> tensor<1x128x1x16xf16, {order = #NCWH}>
        // CHECK:       [[PERMUTE_0:%.+]] = IE.PermuteCast([[MAXPOOL_3]]) {
        // CHECK-SAME:          dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x128x1x16xf16, {order = #NCWH}> -> tensor<1x128x1x16xf16>

        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[PERMUTE_0]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 128, 16]} : tensor<1x128x1x16xf16> -> tensor<1x1x128x16xf16>

        // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x128x16xf16> -> tensor<1x1x128x16xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 1]} inputs([[PERMUTE_CAST_2]] : tensor<1x1x128x16xf16, {order = #NHWC}>) -> tensor<1x16x128x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[AFFINE_RESHAPE_2]]
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x128x1xf16, {order = #NHWC}>, tensor<256x16x1x1xf16, {order = #NHWC}> -> tensor<1x256x128x1xf16, {order = #NHWC}>

        // CHECK:       [[AFFINE_RESHAPE_3:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 16]} inputs([[CONV_0]] : tensor<1x256x128x1xf16, {order = #NHWC}>) -> tensor<1x16x128x16xf16, {order = #NHWC}>

        // CHECK:       [[MAXPOOL_5:%.+]] = IE.MaxPool([[AFFINE_RESHAPE_3]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x128x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_6:%.+]] = IE.MaxPool([[MAXPOOL_5]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 1]} : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x2x16xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_7:%.+]] = IE.MaxPool([[MAXPOOL_6]]) {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x2x16xf16, {order = #NHWC}> -> tensor<1x16x1x16xf16, {order = #NHWC}>

        // CHECK:       [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs([[MAXPOOL_7]] : tensor<1x16x1x16xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL_8:%.+]] = IE.MaxPool([[SHAPECAST]]) {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[MAXPOOL_8]] [0, 0, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
        // CHECK:       [[NEGATIVE_OUT_3:%.+]] = IE.Negative([[SLICE_2]]) : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16, {order = #NHWC}>

        // CHECK:       [[PERMUTE_CAST_4:%.+]] = IE.PermuteCast([[NEGATIVE_OUT_3]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_5:%.+]] = IE.AffineReshape([[PERMUTE_CAST_4]])
        // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xf16> -> tensor<1xf16>
        // CHECK:       return [[AFFINE_RESHAPE_5]] : tensor<1xf16>
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

// -----

// int8 embedding table WD chain: i8 weights cast to f16, per-row f16 scale, Multiply outputs f16,
// a single-use ConvertOp (f16→f32) sits between Multiply and Gather.
// ConsolidateWeightsDequantization converts Multiply(i8_as_f16, per-row scale) to
// DynamicDequantize(i8_quant, f16_scale). SwapOperationsWithGatherAndSlice hoists Gather before DynamicDequantize.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt8PerRowF16WithConvertToGather
module @EmbeddingInt8PerRowF16WithConvertToGather {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x512xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
      %cst_wt = const.Declare tensor<65536x512xf16> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f16>]
      %cst_scale = const.Declare tensor<65536x1xf16> = dense<3.9215686e-3> : tensor<65536x1xf16>

      %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<65536x512xf16>, tensor<65536x1xf16> -> tensor<65536x512xf16>
      %convert = IE.Convert(%mul) {dstElemType = f32} : tensor<65536x512xf16> -> tensor<65536x512xf32>
      %gather = IE.Gather(%convert, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
      return %gather : tensor<256x512xf32>

      // Both weights and per-row scale are gathered; dequantization runs on the gathered slices only.
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x65536x1x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x65536x1x512x!qElemType>
      // CHECK: [[GATHER_WT:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x65536x1x512x!qElemType>, tensor<256xsi32> -> tensor<1x256x1x512x!qElemType>
      // CHECK: [[GATHER_SCALE:%.+]] = IE.Gather([[SCALE]], [[INDICES]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x65536x1x1xf16>, tensor<256xsi32> -> tensor<1x256x1x1xf16>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER_WT]], [[GATHER_SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x256x1x512x!qElemType>, tensor<1x256x1x1xf16> -> tensor<1x256x1x512xf16>
      // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[DYN_DEQUANT]])
      // CHECK-SAME: tensor<1x256x1x512xf16> -> tensor<1x1x256x512xf16>
      // CHECK: [[CONVERT:%.+]] = IE.Convert([[RESHAPE]]) {dstElemType = f32}
      // CHECK-SAME: tensor<1x1x256x512xf16> -> tensor<1x1x256x512xf32>
      // CHECK: [[RESHAPE2:%.+]] = IE.AffineReshape([[CONVERT]])
      // CHECK-SAME: tensor<1x1x256x512xf32> -> tensor<256x512xf32>
      // CHECK: return [[RESHAPE2]] : tensor<256x512xf32>
    }
}

// -----

// int8 embedding table WD chain: i8 weights, per-tensor f32 scale, Multiply outputs f32,
// no ConvertOp before Gather.
// ConsolidateWeightsDequantization converts Multiply(i8_as_f32, per-tensor scale) to
// DynamicDequantize(i8_quant, f16_scale). SwapOperationsWithGatherAndSlice then hoists Gather before
// DynamicDequantize because the per-tensor scale [1x1x1x1] is invariant to row selection.

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @EmbeddingInt8PerTensorF32ToGather
module @EmbeddingInt8PerTensorF32ToGather {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "indices" : tensor<256xsi32>
    } outputsInfo : {
        DataInfo "output" : tensor<256x512xf32>
    }

    // CHECK: func.func @main([[INDICES:%.+]]: tensor<256xsi32>)
    func.func @main(%indices: tensor<256xsi32>) -> tensor<256x512xf32> {
      %cst_wt = const.Declare tensor<65536x512xf32> = dense<1> : tensor<65536x512xsi8>, [#const.CastElemType<f32>]
      %cst_scale = const.Declare tensor<1x1xf32> = dense<3.9215686e-3> : tensor<1x1xf32>

      %mul = IE.Multiply(%cst_wt, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
          : tensor<65536x512xf32>, tensor<1x1xf32> -> tensor<65536x512xf32>
      %gather = IE.Gather(%mul, %indices) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
          : tensor<65536x512xf32>, tensor<256xsi32> -> tensor<256x512xf32>
      return %gather : tensor<256x512xf32>

      // Only weights are gathered; per-tensor scale [1x1x1x1] is row-selection-invariant and reused directly.
      // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf16>
      // CHECK-DAG: [[WT_QTYPE:%.+]] = const.Declare tensor<1x1x65536x512x!qElemType>
      // CHECK: [[GATHER:%.+]] = IE.Gather([[WT_QTYPE]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
      // CHECK-SAME: tensor<1x1x65536x512x!qElemType>, tensor<256xsi32> -> tensor<1x1x256x512x!qElemType>
      // CHECK: [[DYN_DEQUANT:%.+]] = IE.DynamicDequantize([[GATHER]], [[SCALE]]) {dstElemType = f16}
      // CHECK-SAME: tensor<1x1x256x512x!qElemType>, tensor<1x1x1x1xf16> -> tensor<1x1x256x512xf16>
      // CHECK: [[CONVERT:%.+]] = IE.Convert([[DYN_DEQUANT]]) {dstElemType = f32}
      // CHECK-SAME: tensor<1x1x256x512xf16> -> tensor<1x1x256x512xf32>
      // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONVERT]])
      // CHECK-SAME: tensor<1x1x256x512xf32> -> tensor<256x512xf32>
      // CHECK: return [[RESHAPE]] : tensor<256x512xf32>
    }
}
