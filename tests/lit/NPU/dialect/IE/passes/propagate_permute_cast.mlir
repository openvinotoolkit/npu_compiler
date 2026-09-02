//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --propagate-permute-cast %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.006552408723270192:147>

// CHECK-LABEL: @OptimizeDequantizeMemPermuteReorder()
func.func @OptimizeDequantizeMemPermuteReorder() -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
   %cst = const.Declare tensor<320x1280x1x1x!qElemType> = dense<1> : tensor<320x1280xui8>, [#const.Reshape<[1, 1, 320, 1280]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[320, 1280, 1, 1]>]
   %dequantize = IE.Dequantize(%cst) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   %permute_cast = IE.PermuteCast(%dequantize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

   return %permute_cast : tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
   // CHECK:        [[CST:%.+]] =  const.Declare tensor<320x1280x1x1x!qElemType, {order = #NHWC}>
   // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType, {order = #NHWC}> -> tensor<320x1280x1x1xf16, {order = #NHWC}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.006552408723270192:147>

// CHECK-LABEL: @IncorrectLayoutDontPropagatePermuteThroughDequantize()
func.func @IncorrectLayoutDontPropagatePermuteThroughDequantize() -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> {
   %cst = const.Declare tensor<320x1280x1x1x!qElemType> = dense<1> : tensor<320x1280xui8>, [#const.Reshape<[1, 1, 320, 1280]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[320, 1280, 1, 1]>]
   %dequantize = IE.Dequantize(%cst) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   %permute_cast = IE.PermuteCast(%dequantize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>

   return %permute_cast : tensor<320x1280x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
   // CHECK:        [[CST:%.+]] =  const.Declare tensor<320x1280x1x1x!qElemType>
   // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<320x1280x1x1x!qElemType> -> tensor<320x1280x1x1xf16>
   // CHECK:        [[PERMUTECAST:%.+]] = IE.PermuteCast([[DEQUANT]]) {dst_order = #NWHC, mem_perm = #NHWC} : tensor<320x1280x1x1xf16> -> tensor<320x1280x1x1xf16, {order = #NWHC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: func.func @PropagatePermuteCastThroughConcat
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x1x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x1x16xf16>)
func.func @PropagatePermuteCastThroughConcat(%arg0: tensor<1x4x1x16xf16>, %arg1: tensor<1x4x1x16xf16>) -> tensor<1x8x1x16xf16> {
    %permute_cast_0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x16xf16> -> tensor<1x1x4x16xf16>
    %permute_cast_1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x16xf16> -> tensor<1x1x4x16xf16>
    %concat = IE.Concat(%permute_cast_0, %permute_cast_1) {static_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]]} : tensor<1x1x4x16xf16>, tensor<1x1x4x16xf16> -> tensor<1x1x8x16xf16>
    %permute_cast = IE.PermuteCast(%concat) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x8x16xf16> -> tensor<1x8x1x16xf16>
    return %permute_cast : tensor<1x8x1x16xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]}
    // CHECK-SAME:  tensor<1x4x1x16xf16>, tensor<1x4x1x16xf16> -> tensor<1x8x1x16xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PropagatePermuteCastThroughConcatPerAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x1x16xf16>, [[ARG_1:%[^:]+]]: tensor<1x4x1x16xf16>)
func.func @PropagatePermuteCastThroughConcatPerAxis(%arg0: tensor<1x4x1x16xf16>, %arg1: tensor<1x4x1x16xf16>) -> tensor<1x8x1x16xf16> {
    %permute_cast_0 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x16xf16> -> tensor<1x1x4x16xf16>
    %permute_cast_1 = IE.PermuteCast(%arg1) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x4x1x16xf16> -> tensor<1x1x4x16xf16>
    %concat = IE.Concat(%permute_cast_0, %permute_cast_1) {per_axis = #IE.Concat<axis = 2>} : tensor<1x1x4x16xf16>, tensor<1x1x4x16xf16> -> tensor<1x1x8x16xf16>
    %permute_cast = IE.PermuteCast(%concat) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x1x8x16xf16> -> tensor<1x8x1x16xf16>
    return %permute_cast : tensor<1x8x1x16xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]}
    // CHECK-SAME:  tensor<1x4x1x16xf16>, tensor<1x4x1x16xf16> -> tensor<1x8x1x16xf16>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DontPropagateThroughConcatWithTooManyInputs
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1x64xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x64xf16>, [[ARG_2:%[^:]+]]: tensor<1x1x1x64xf16>)
func.func @DontPropagateThroughConcatWithTooManyInputs(%arg0: tensor<1x1x1x64xf16>, %arg1: tensor<1x1x1x64xf16>, %arg2: tensor<1x1x1x64xf16>) -> tensor<1x1x1x192xf16, {order = #NHWC}> {
    %concat = IE.Concat(%arg0, %arg1, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 64], [0, 0, 0, 128]]} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x192xf16>
    %permute_cast = IE.PermuteCast(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x1x192xf16> -> tensor<1x1x1x192xf16, {order = #NHWC}>
    return %permute_cast : tensor<1x1x1x192xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]], [[ARG_2]])
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[CONCAT]])
    // CHECK:       return [[PERMUTECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DontPropagateThroughConcatWithNoPermuteCastInputs
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1x128xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x128xf16>)
func.func @DontPropagateThroughConcatWithNoPermuteCastInputs(%arg0: tensor<1x1x1x128xf16>, %arg1: tensor<1x1x1x128xf16>) -> tensor<1x1x1x256xf16, {order = #NHWC}> {
    %concat = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 128]]} : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x1x1x256xf16>
    %permute_cast = IE.PermuteCast(%concat) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16, {order = #NHWC}>
    return %permute_cast : tensor<1x1x1x256xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[ARG_0]], [[ARG_1]])
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[CONCAT]])
    // CHECK:       return [[PERMUTECAST]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @PropagatePermuteCastThroughSlice
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x7x55xf16>)
func.func @PropagatePermuteCastThroughSlice(%arg0: tensor<1x16x7x55xf16>) -> tensor<1x1x7x55xf16> {
    %permute_cast = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x16x7x55xf16> -> tensor<1x16x55x7xf16, {order = #NCWH}>
    %slice = IE.Slice %permute_cast [0, 0, 0, 0] [1, 1, 55, 7] : tensor<1x16x55x7xf16, {order = #NCWH}> to tensor<1x1x55x7xf16, {order = #NCWH}>
    %permute_cast_0 = IE.PermuteCast(%slice) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x55x7xf16, {order = #NCWH}> -> tensor<1x1x7x55xf16>
    return %permute_cast_0 : tensor<1x1x7x55xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 7, 55] : tensor<1x16x7x55xf16> to tensor<1x1x7x55xf16>
    // CHECK:       return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DontPropagateThroughSliceNonInversePermuteCasts
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x7x55xf16>)
func.func @DontPropagateThroughSliceNonInversePermuteCasts(%arg0: tensor<1x16x7x55xf16>) -> tensor<1x55x1x7xf16, {order = #NHWC}> {
    %permute_cast = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} : tensor<1x16x7x55xf16> -> tensor<1x16x55x7xf16, {order = #NCWH}>
    %slice = IE.Slice %permute_cast [0, 0, 0, 0] [1, 1, 55, 7] : tensor<1x16x55x7xf16, {order = #NCWH}> to tensor<1x1x55x7xf16, {order = #NCWH}>
    %permute_cast_0 = IE.PermuteCast(%slice) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1x55x7xf16, {order = #NCWH}> -> tensor<1x55x1x7xf16, {order = #NHWC}>
    return %permute_cast_0 : tensor<1x55x1x7xf16, {order = #NHWC}>

    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[ARG_0]])
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERMUTECAST]] [0, 0, 0, 0] [1, 1, 55, 7]
    // CHECK:       [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[SLICE]])
    // CHECK:       return [[PERMUTECAST_0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DontPropagateThroughSliceNoParentPermuteCast
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x55x55xf16, {order = #NHWC}>)
func.func @DontPropagateThroughSliceNoParentPermuteCast(%arg0: tensor<1x16x55x55xf16, {order = #NHWC}>) -> tensor<1x55x55x1xf16> {
    %slice = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 55, 55] : tensor<1x16x55x55xf16, {order = #NHWC}> to tensor<1x1x55x55xf16, {order = #NHWC}>
    %permute_cast = IE.PermuteCast(%slice) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x55x55xf16, {order = #NHWC}> -> tensor<1x55x55x1xf16>
    return %permute_cast : tensor<1x55x55x1xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 55, 55]
    // CHECK:       [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]])
    // CHECK:       return [[PERMUTECAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#NCHW_TO_NHWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

// CHECK-LABEL: func.func @PropagateThroughSwish
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<4x4x1x1xf16>)
func.func @PropagateThroughSwish(%arg0: tensor<4x4x1x1xf16>) -> tensor<1x4x4x1xf16, {order = #NHWC}> {
    %swish = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xf16>
    %permute_cast = IE.PermuteCast(%swish) {dst_order = #NHWC, mem_perm = #NCHW_TO_NHWC} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    return %permute_cast : tensor<1x4x4x1xf16, {order = #NHWC}>

    // PermuteCast is pushed before Swish; Swish now runs in NHWC layout
    // CHECK:       [[PC:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #map} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    // CHECK:       [[SW:%.+]] = IE.Swish([[PC]]) {beta_value = {{.+}}} : tensor<1x4x4x1xf16, {order = #NHWC}> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    // CHECK:       return [[SW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCHW_TO_NHWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

// CHECK-LABEL: func.func @PropagateThroughSwishWithAffineReshape
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<4x4x1x1xf16>)
func.func @PropagateThroughSwishWithAffineReshape(%arg0: tensor<4x4x1x1xf16>) -> tensor<1x4x2x2xf16, {order = #NHWC}> {
    %swish = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xf16>
    %permute_cast = IE.PermuteCast(%swish) {dst_order = #NHWC, mem_perm = #NCHW_TO_NHWC} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    %reshape = IE.AffineReshape(%permute_cast) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 4, 2, 2]} : tensor<1x4x4x1xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    return %reshape : tensor<1x4x2x2xf16, {order = #NHWC}>

    // AffineReshape moves before Swish; Swish runs at the Multiply shape enabling VerticalFusion
    // CHECK:       [[PC:%.+]] = IE.PermuteCast([[ARG_0]]) {dst_order = #NHWC, mem_perm = #map} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    // CHECK:       [[AR:%.+]] = IE.AffineReshape([[PC]]) {{{.+}} shape_value = [1, 4, 2, 2]} : tensor<1x4x4x1xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    // CHECK:       [[SW:%.+]] = IE.Swish([[AR]]) {beta_value = {{.+}}} : tensor<1x4x2x2xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[SW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#NHWC_TO_NCHW = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#NCHW_TO_NHWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

// CHECK-LABEL: func.func @SwiGLUGateOptimization
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x8x4x1xf16, {order = #NHWC}>)
func.func @SwiGLUGateOptimization(%arg0: tensor<1x8x4x1xf16, {order = #NHWC}>) -> tensor<1x4x2x2xf16, {order = #NHWC}> {
    // Gate path: PermuteCast(NHWC→NCHW) → Slice → Swish → PermuteCast(NCHW→NHWC) → AffineReshape
    // PropagateThroughSwish absorbs AffineReshape into Swish input,
    // then PropagateThroughSlice cancels the two PermuteCasts,
    // yielding: Slice(NHWC) → AffineReshape → Swish(NHWC)
    %pc1 = IE.PermuteCast(%arg0) {dst_order = #NCHW, mem_perm = #NHWC_TO_NCHW} : tensor<1x8x4x1xf16, {order = #NHWC}> -> tensor<4x8x1x1xf16>
    %slice = IE.Slice %pc1 [0, 0, 0, 0] [4, 4, 1, 1] : tensor<4x8x1x1xf16> to tensor<4x4x1x1xf16>
    %swish = IE.Swish(%slice) {beta_value = 1.000000e+00 : f64} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xf16>
    %pc2 = IE.PermuteCast(%swish) {dst_order = #NHWC, mem_perm = #NCHW_TO_NHWC} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    %reshape = IE.AffineReshape(%pc2) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 4, 2, 2]} : tensor<1x4x4x1xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    return %reshape : tensor<1x4x2x2xf16, {order = #NHWC}>

    // Both PermuteCasts are eliminated; Slice moves to NHWC; AffineReshape precedes Swish
    // CHECK-NOT:   IE.PermuteCast
    // CHECK:       [[SL:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 4, 1] : tensor<1x8x4x1xf16, {order = #NHWC}> to tensor<1x4x4x1xf16, {order = #NHWC}>
    // CHECK:       [[AR:%.+]] = IE.AffineReshape([[SL]]) {{{.+}} shape_value = [1, 4, 2, 2]} : tensor<1x4x4x1xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    // CHECK:       [[SW:%.+]] = IE.Swish([[AR]]) {beta_value = {{.+}}} : tensor<1x4x2x2xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[SW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCHW_TO_NHWC = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>

// CHECK-LABEL: func.func @DontPropagateThroughSwishMultipleUses
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<4x4x1x1xf16>)
func.func @DontPropagateThroughSwishMultipleUses(%arg0: tensor<4x4x1x1xf16>) -> (tensor<1x4x4x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16>) {
    %swish = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<4x4x1x1xf16> -> tensor<4x4x1x1xf16>
    %permute_cast = IE.PermuteCast(%swish) {dst_order = #NHWC, mem_perm = #NCHW_TO_NHWC} : tensor<4x4x1x1xf16> -> tensor<1x4x4x1xf16, {order = #NHWC}>
    // %swish has 2 users (%permute_cast and the direct return) → propagation must not fire
    return %permute_cast, %swish : tensor<1x4x4x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16>

    // CHECK:       [[SW:%.+]] = IE.Swish([[ARG_0]])
    // CHECK:       [[PC:%.+]] = IE.PermuteCast([[SW]])
    // CHECK:       return [[PC]], [[SW]]
}
