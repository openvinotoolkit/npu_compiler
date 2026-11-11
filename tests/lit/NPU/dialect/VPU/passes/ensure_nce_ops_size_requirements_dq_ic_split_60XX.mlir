//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --ensure-nce-ops-size-requirements="enable-split-channel-for-dynamic-dequantize=true" --canonicalize --mlir-elide-elementsattrs-if-larger 2048 --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @SplitMemPermuteAndDynamicDequantizeLargeIC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x14336x256x4xf16, {order = #NHWC}>
func.func @SplitMemPermuteAndDynamicDequantizeLargeIC(%arg0: tensor<1x14336x256x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x112x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x4096x112x1xf16> = dense<1.0> : tensor<1x4096x112x1xf16>

  %weights_mem_perm = VPU.MemPermute(%weights) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x112x4096x128x!qElemType> -> tensor<1x4096x112x128x!qElemType>
  %dequant = VPU.DynamicDequantize(%weights_mem_perm, %scales) {dstElemType = f16} : tensor<1x4096x112x128x!qElemType>, tensor<1x4096x112x1xf16> -> tensor<1x4096x112x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 14336, 1, 1]} : tensor<1x4096x112x128xf16> -> tensor<4096x14336x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<4096x14336x1x1xf16> -> tensor<4096x14336x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [4096, 14336, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x14336x256x4xf16, {order = #NHWC}>, tensor<4096x14336x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  %output_reshape = VPU.AffineReshape(%output) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 1024, 1]} : tensor<1x4096x256x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
  return %output_reshape : tensor<1x4096x1024x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x4096x56x1xf16> = dense<1.000000e+00> : tensor<1x4096x112x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 56, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x56x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 56, 4096, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x4096x56x1xf16> = dense<1.000000e+00> : tensor<1x4096x112x1xf16>, [#const.SubView<[0, 0, 56, 0], [1, 4096, 56, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x56x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.SubView<[0, 56, 0, 0], [1, 56, 4096, 128]>, #const.CastElemType<!qElemType>]
  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 7168, 256, 4]
  // CHECK-SAME:          to tensor<1x7168x256x4xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE_TILE0:%.+]] = VPU.MemPermute([[WEIGHTS_TILE0]])
  // CHECK-SAME:          -> tensor<1x4096x56x128x!qElemType>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[MEM_PERMUTE_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x56x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [4096, 7168, 1, 1]
  // CHECK-SAME:          -> tensor<4096x7168x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<4096x7168x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[PERM_TILE0]]) rawFilterShape [4096, 7168, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 7168, 0, 0] [1, 7168, 256, 4]
  // CHECK-SAME:          to tensor<1x7168x256x4xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE_TILE1:%.+]] = VPU.MemPermute([[WEIGHTS_TILE1]])
  // CHECK-SAME:          -> tensor<1x4096x56x128x!qElemType>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[MEM_PERMUTE_TILE1]], [[SCALES_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x56x128xf16>
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [4096, 7168, 1, 1]
  // CHECK-SAME:          -> tensor<4096x7168x1x1xf16>
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK-SAME:          -> tensor<4096x7168x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[PERM_TILE1]]) rawFilterShape [4096, 7168, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       VPU.AffineReshape([[ADD]])
  // CHECK-SAME:          shape_value = [1, 4096, 1024, 1]
  // CHECK-SAME:          -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitDynamicDequantizeMergedICAlignedToW
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x8320x128x8xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeMergedICAlignedToW(%arg0: tensor<1x8320x128x8xf16, {order = #NHWC}>) -> tensor<1x3072x128x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x3072x65x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x3072x65x1xf16> = dense<1.0> : tensor<1x3072x65x1xf16>
  %weights_table = const.Declare tensor<3072x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<3072x1x1x4xsi32>

  %dequant = VPU.DynamicDequantize(%weights, %scales) {dstElemType = f16} : tensor<1x3072x65x128x!qElemType>, tensor<1x3072x65x1xf16> -> tensor<1x3072x65x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [3072, 8320, 1, 1]} : tensor<1x3072x65x128xf16> -> tensor<3072x8320x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<3072x8320x1x1xf16> -> tensor<3072x8320x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast, %weights_table) rawFilterShape [3072, 8320, 1, 1] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x8320x128x8xf16, {order = #NHWC}>, tensor<3072x8320x1x1xf16, {order = #NHWC}>, tensor<3072x1x1x4xsi32, {order = #NCHW}> -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  return %output : tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x3072x22x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 3072, 22, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x3072x22x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 3072, 22, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x3072x22x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 22, 0], [1, 3072, 22, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x3072x22x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 22, 0], [1, 3072, 22, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x3072x21x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 44, 0], [1, 3072, 21, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x3072x21x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 44, 0], [1, 3072, 21, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2816, 128, 8]
  // CHECK-SAME:          to tensor<1x2816x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x22x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[PERM_TILE0]], {{%.+}}) rawFilterShape [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2816, 0, 0] [1, 2816, 128, 8]
  // CHECK-SAME:          to tensor<1x2816x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x22x128xf16>
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16>
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[PERM_TILE1]], {{%.+}}) rawFilterShape [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 5632, 0, 0] [1, 2688, 128, 8]
  // CHECK-SAME:          to tensor<1x2688x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE2]], [[SCALES_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x21x128xf16>
  // CHECK:       [[RESHAPE_TILE2:%.+]] = VPU.AffineReshape([[DQ_TILE2]])
  // CHECK-SAME:          shape_value = [3072, 2688, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2688x1x1xf16>
  // CHECK:       [[PERM_TILE2:%.+]] = VPU.PermuteCast([[RESHAPE_TILE2]])
  // CHECK-SAME:          -> tensor<3072x2688x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[PERM_TILE2]], {{%.+}}) rawFilterShape [3072, 2688, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]]) {op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>
  // CHECK:       [[FINAL_OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT]], [[OUTPUT_TILE2]]) {op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>
  // CHECK:       return [[FINAL_OUTPUT]] : tensor<1x3072x128x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @SplitDynamicDequantizeLargeIC
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x24576x256x4xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeLargeIC(%arg0: tensor<1x24576x256x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x4096x192x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x4096x192x1xf16> = dense<1.0> : tensor<1x4096x192x1xf16>

  %dequant = VPU.DynamicDequantize(%weights, %scales) {dstElemType = f16} : tensor<1x4096x192x128x!qElemType>, tensor<1x4096x192x1xf16> -> tensor<1x4096x192x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 24576, 1, 1]} : tensor<1x4096x192x128xf16> -> tensor<4096x24576x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<4096x24576x1x1xf16> -> tensor<4096x24576x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [4096, 24576, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x24576x256x4xf16, {order = #NHWC}>, tensor<4096x24576x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  %output_reshape = VPU.AffineReshape(%output) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 1024, 1]} : tensor<1x4096x256x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
  return %output_reshape : tensor<1x4096x1024x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x4096x48x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 48, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x4096x48x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 48, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x4096x48x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 48, 0], [1, 4096, 48, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x4096x48x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 48, 0], [1, 4096, 48, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x4096x48x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 96, 0], [1, 4096, 48, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x4096x48x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 96, 0], [1, 4096, 48, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE3:%.+]] = const.Declare tensor<1x4096x48x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 144, 0], [1, 4096, 48, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE3:%.+]] = const.Declare tensor<1x4096x48x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 144, 0], [1, 4096, 48, 128]>, #const.CastElemType<!qElemType>]
  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 6144, 256, 4]
  // CHECK-SAME:          to tensor<1x6144x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x48x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[PERM_TILE0]]) rawFilterShape [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 6144, 0, 0] [1, 6144, 256, 4]
  // CHECK-SAME:          to tensor<1x6144x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x48x128xf16>
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16>
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[PERM_TILE1]]) rawFilterShape [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 12288, 0, 0] [1, 6144, 256, 4]
  // CHECK-SAME:          to tensor<1x6144x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE2]], [[SCALES_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x48x128xf16>
  // CHECK:       [[RESHAPE_TILE2:%.+]] = VPU.AffineReshape([[DQ_TILE2]])
  // CHECK-SAME:          shape_value = [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16>
  // CHECK:       [[PERM_TILE2:%.+]] = VPU.PermuteCast([[RESHAPE_TILE2]])
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], [[PERM_TILE2]]) rawFilterShape [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 18432, 0, 0] [1, 6144, 256, 4]
  // CHECK-SAME:          to tensor<1x6144x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE3:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE3]], [[SCALES_TILE3]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x48x128xf16>
  // CHECK:       [[RESHAPE_TILE3:%.+]] = VPU.AffineReshape([[DQ_TILE3]])
  // CHECK-SAME:          shape_value = [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16>
  // CHECK:       [[PERM_TILE3:%.+]] = VPU.PermuteCast([[RESHAPE_TILE3]])
  // CHECK-SAME:          -> tensor<4096x6144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[INPUT_TILE3]], [[PERM_TILE3]]) rawFilterShape [4096, 6144, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK:       [[ADD:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD_1:%.+]] = VPU.NCE.Eltwise([[ADD]], [[OUTPUT_TILE2]])
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD_2:%.+]] = VPU.NCE.Eltwise([[ADD_1]], [[OUTPUT_TILE3]])
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       VPU.AffineReshape([[ADD_2]])
  // CHECK-SAME:          shape_value = [1, 4096, 1024, 1]
  // CHECK-SAME:          -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u2:f16, 1.000000e+00>

// CHECK-LABEL:   @SplitDynamicDequantizeWithZeroPointUI2
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x17920x64x8xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeWithZeroPointUI2(%arg0: tensor<1x17920x64x8xf16, {order = #NHWC}>) -> tensor<1x5120x64x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x5120x280x64x!qElemType> = dense<1> : tensor<1x5120x280x64xui2>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x5120x280x1xf16> = dense<1.0> : tensor<1x5120x280x1xf16>
  %zp = const.Declare tensor<1x5120x280x1xui2> = dense<0> : tensor<1x5120x280x1xui2>

  %dequant = VPU.DynamicDequantize(%weights, %scales, %zp) {dstElemType = f16} : tensor<1x5120x280x64x!qElemType>, tensor<1x5120x280x1xf16>, tensor<1x5120x280x1xui2> -> tensor<1x5120x280x64xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [5120, 17920, 1, 1]} : tensor<1x5120x280x64xf16> -> tensor<5120x17920x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<5120x17920x1x1xf16> -> tensor<5120x17920x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [5120, 17920, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x17920x64x8xf16, {order = #NHWC}>, tensor<5120x17920x1x1xf16, {order = #NHWC}> -> tensor<1x5120x64x8xf16, {order = #NHWC}>

  return %output : tensor<1x5120x64x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x5120x72x1xf16>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[ZP_TILE0:%.+]] = const.Declare tensor<1x5120x72x1xui2>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x5120x72x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 5120, 72, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x5120x72x1xf16>{{.*}}[#const.SubView<[0, 0, 72, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[ZP_TILE1:%.+]] = const.Declare tensor<1x5120x72x1xui2>{{.*}}[#const.SubView<[0, 0, 72, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x5120x72x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 72, 0], [1, 5120, 72, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x5120x72x1xf16>{{.*}}[#const.SubView<[0, 0, 144, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[ZP_TILE2:%.+]] = const.Declare tensor<1x5120x72x1xui2>{{.*}}[#const.SubView<[0, 0, 144, 0], [1, 5120, 72, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x5120x72x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 144, 0], [1, 5120, 72, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE3:%.+]] = const.Declare tensor<1x5120x64x1xf16>{{.*}}[#const.SubView<[0, 0, 216, 0], [1, 5120, 64, 1]>]
  // CHECK-DAG:   [[ZP_TILE3:%.+]] = const.Declare tensor<1x5120x64x1xui2>{{.*}}[#const.SubView<[0, 0, 216, 0], [1, 5120, 64, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE3:%.+]] = const.Declare tensor<1x5120x64x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 216, 0], [1, 5120, 64, 64]>, #const.CastElemType<!qElemType>]

  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4608, 64, 8]
  // CHECK-SAME:          to tensor<1x4608x64x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]], [[ZP_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x5120x72x64xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [5120, 4608, 1, 1]
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[PERM_TILE0]]) rawFilterShape [5120, 4608, 1, 1]
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 4608, 0, 0] [1, 4608, 64, 8]
  // CHECK-SAME:          to tensor<1x4608x64x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]], [[ZP_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x5120x72x64xf16>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], {{%.+}}) rawFilterShape [5120, 4608, 1, 1]
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 9216, 0, 0] [1, 4608, 64, 8]
  // CHECK-SAME:          to tensor<1x4608x64x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE2]], [[SCALES_TILE2]], [[ZP_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x5120x72x64xf16>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], {{%.+}}) rawFilterShape [5120, 4608, 1, 1]
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 13824, 0, 0] [1, 4096, 64, 8]
  // CHECK-SAME:          to tensor<1x4096x64x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE3:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE3]], [[SCALES_TILE3]], [[ZP_TILE3]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x5120x64x64xf16>
  // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[INPUT_TILE3]], {{%.+}}) rawFilterShape [5120, 4096, 1, 1]
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>

  // CHECK:       [[ADD:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>
  // CHECK:       [[ADD_1:%.+]] = VPU.NCE.Eltwise([[ADD]], [[OUTPUT_TILE2]])
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>
  // CHECK:       [[ADD_2:%.+]] = VPU.NCE.Eltwise([[ADD_1]], [[OUTPUT_TILE3]])
  // CHECK-SAME:          -> tensor<1x5120x64x8xf16, {order = #NHWC}>
  // CHECK:       return [[ADD_2]] : tensor<1x5120x64x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 81664 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitDynamicDequantizeWithBroadcastZeroPoint
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x320x8x8xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeWithBroadcastZeroPoint(%arg0: tensor<1x320x8x8xf16, {order = #NHWC}>) -> tensor<1x64x8x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x64x2x160xsi4> = dense<1> : tensor<1x64x2x160xsi4>
  %scales = const.Declare tensor<1x64x2x1xf16> = dense<1.0> : tensor<1x64x2x1xf16>
  %zp = const.Declare tensor<1x64x1x1xsi4> = dense<0> : tensor<1x64x1x1xsi4>

  %dequant = VPU.DynamicDequantize(%weights, %scales, %zp) {dstElemType = f16} : tensor<1x64x2x160xsi4>, tensor<1x64x2x1xf16>, tensor<1x64x1x1xsi4> -> tensor<1x64x2x160xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 320, 1, 1]} : tensor<1x64x2x160xf16> -> tensor<64x320x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x320x1x1xf16> -> tensor<64x320x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [64, 320, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x320x8x8xf16, {order = #NHWC}>, tensor<64x320x1x1xf16, {order = #NHWC}> -> tensor<1x64x8x8xf16, {order = #NHWC}>

  return %output : tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[ZP:%.+]] = const.Declare tensor<1x64x1x1xsi4> = dense<0> : tensor<1x64x1x1xsi4>
  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x64x1x1xf16>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 64, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x64x1x160xsi4>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 64, 1, 160]>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x64x1x1xf16>{{.*}}[#const.SubView<[0, 0, 1, 0], [1, 64, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x64x1x160xsi4>{{.*}}[#const.SubView<[0, 0, 1, 0], [1, 64, 1, 160]>]

  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 160, 8, 8]
  // CHECK-SAME:          to tensor<1x160x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]], [[ZP]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x1x160xf16>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], {{%.+}}) rawFilterShape [64, 160, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 160, 0, 0] [1, 160, 8, 8]
  // CHECK-SAME:          to tensor<1x160x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]], [[ZP]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x1x160xf16>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], {{%.+}}) rawFilterShape [64, 160, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       return [[OUTPUT]] : tensor<1x64x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u2:f16, 1.000000e+00>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 1859584 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @AvoindSplittingOverMaxTileIC
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x19200x8x8xf16, {order = #NHWC}>
func.func @AvoindSplittingOverMaxTileIC(%arg0: tensor<1x19200x8x8xf16, {order = #NHWC}>) -> tensor<1x64x8x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x64x300x64x!qElemType> = dense<1> : tensor<1x64x300x64xui2>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x64x300x1xf16> = dense<1.0> : tensor<1x64x300x1xf16>
  %zp = const.Declare tensor<1x64x300x1xui2> = dense<0> : tensor<1x64x300x1xui2>

  %dequant = VPU.DynamicDequantize(%weights, %scales, %zp) {dstElemType = f16} : tensor<1x64x300x64x!qElemType>, tensor<1x64x300x1xf16>, tensor<1x64x300x1xui2> -> tensor<1x64x300x64xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 19200, 1, 1]} : tensor<1x64x300x64xf16> -> tensor<64x19200x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x19200x1x1xf16> -> tensor<64x19200x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [64, 19200, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x19200x8x8xf16, {order = #NHWC}>, tensor<64x19200x1x1xf16, {order = #NHWC}> -> tensor<1x64x8x8xf16, {order = #NHWC}>

  return %output : tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x64x60x1xf16>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[ZP_TILE0:%.+]] = const.Declare tensor<1x64x60x1xui2>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x64x60x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 0, 0], [1, 64, 60, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x64x60x1xf16>{{.*}}[#const.SubView<[0, 0, 60, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[ZP_TILE1:%.+]] = const.Declare tensor<1x64x60x1xui2>{{.*}}[#const.SubView<[0, 0, 60, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x64x60x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 60, 0], [1, 64, 60, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x64x60x1xf16>{{.*}}[#const.SubView<[0, 0, 120, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[ZP_TILE2:%.+]] = const.Declare tensor<1x64x60x1xui2>{{.*}}[#const.SubView<[0, 0, 120, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x64x60x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 120, 0], [1, 64, 60, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE3:%.+]] = const.Declare tensor<1x64x60x1xf16>{{.*}}[#const.SubView<[0, 0, 180, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[ZP_TILE3:%.+]] = const.Declare tensor<1x64x60x1xui2>{{.*}}[#const.SubView<[0, 0, 180, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE3:%.+]] = const.Declare tensor<1x64x60x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 180, 0], [1, 64, 60, 64]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE4:%.+]] = const.Declare tensor<1x64x60x1xf16>{{.*}}[#const.SubView<[0, 0, 240, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[ZP_TILE4:%.+]] = const.Declare tensor<1x64x60x1xui2>{{.*}}[#const.SubView<[0, 0, 240, 0], [1, 64, 60, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE4:%.+]] = const.Declare tensor<1x64x60x64x!qElemType>{{.*}}[#const.SubView<[0, 0, 240, 0], [1, 64, 60, 64]>, #const.CastElemType<!qElemType>]

  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3840, 8, 8]
  // CHECK-SAME:          to tensor<1x3840x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]], [[ZP_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x60x64xf16>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], {{%.+}}) rawFilterShape [64, 3840, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 3840, 0, 0] [1, 3840, 8, 8]
  // CHECK-SAME:          to tensor<1x3840x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]], [[ZP_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x60x64xf16>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], {{%.+}}) rawFilterShape [64, 3840, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 7680, 0, 0] [1, 3840, 8, 8]
  // CHECK-SAME:          to tensor<1x3840x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE2]], [[SCALES_TILE2]], [[ZP_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x60x64xf16>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], {{%.+}}) rawFilterShape [64, 3840, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 11520, 0, 0] [1, 3840, 8, 8]
  // CHECK-SAME:          to tensor<1x3840x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE3:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE3]], [[SCALES_TILE3]], [[ZP_TILE3]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x60x64xf16>
  // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[INPUT_TILE3]], {{%.+}}) rawFilterShape [64, 3840, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 15360, 0, 0] [1, 3840, 8, 8]
  // CHECK-SAME:          to tensor<1x3840x8x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE4:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE4]], [[SCALES_TILE4]], [[ZP_TILE4]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x64x60x64xf16>
  // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.Convolution([[INPUT_TILE4]], {{%.+}}) rawFilterShape [64, 3840, 1, 1]
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[ADD0:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       [[ADD1:%.+]] = VPU.NCE.Eltwise([[ADD0]], [[OUTPUT_TILE2]])
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[OUTPUT_TILE3]])
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       [[ADD3:%.+]] = VPU.NCE.Eltwise([[ADD2]], [[OUTPUT_TILE4]])
  // CHECK-SAME:          -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       return [[ADD3]] : tensor<1x64x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 71680 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitDynamicDequantizeICWithCombinedAlignment
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x720x8x8xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeICWithCombinedAlignment(%arg0: tensor<1x720x8x8xf16, {order = #NHWC}>) -> tensor<1x64x8x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x64x72x10x!qElemType> = dense<1> : tensor<1x64x72x10xi8>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x64x72x1xf16> = dense<1.0> : tensor<1x64x72x1xf16>

  %dequant = VPU.DynamicDequantize(%weights, %scales) {dstElemType = f16} : tensor<1x64x72x10x!qElemType>, tensor<1x64x72x1xf16> -> tensor<1x64x72x10xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [64, 720, 1, 1]} : tensor<1x64x72x10xf16> -> tensor<64x720x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<64x720x1x1xf16> -> tensor<64x720x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [64, 720, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x720x8x8xf16, {order = #NHWC}>, tensor<64x720x1x1xf16, {order = #NHWC}> -> tensor<1x64x8x8xf16, {order = #NHWC}>

  return %output : tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 80, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 160, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 240, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[INPUT_TILE3]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 320, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.Convolution([[INPUT_TILE4]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 400, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.Convolution([[INPUT_TILE5]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 480, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.Convolution([[INPUT_TILE6]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT]] [0, 560, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE7:%.+]] = VPU.NCE.Convolution([[INPUT_TILE7]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
  // CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT]] [0, 640, 0, 0] [1, 80, 8, 8]
  // CHECK-SAME:          to tensor<1x80x8x8xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE8:%.+]] = VPU.NCE.Convolution([[INPUT_TILE8]], {{%.+}}) rawFilterShape [64, 80, 1, 1]
}
