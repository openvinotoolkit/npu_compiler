//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --apply-tiling="enable-scf-tiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
 // CHECK-LABEL: @ApplyConvCTiling
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func.func @ApplyConvCTiling(
            %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
                -> tensor<1x512x14x14xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
        %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<1> : tensor<512x1x1x4xsi32>

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [512, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            
            strides = [1, 1],
            tilingStrategy = [1, 2, 1, 1]
        } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NCHW}> -> tensor<1x512x14x14xf16, {order = #NHWC}>

        return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<1> : tensor<512x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 512 : index
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 256 : index
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x512x14x14xf16, {order = #NHWC}>) {

    //CHECK:      [[SLICE_WEIGHTS:%.+]]  = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER]], 0, 0, 0] [256, 256, 3, 3] [1, 1, 1, 1] : tensor<512x256x3x3xf16, {order = #NHWC}> to tensor<256x256x3x3xf16, {order = #NHWC}>
    //CHECK:      [[SLICE_WEIGHTS_TABLE:%.+]] = tensor.extract_slice [[WEIGHTS_TABLE]][[[LOOP_ITER]], 0, 0, 0] [256, 1, 1, 4] [1, 1, 1, 1] : tensor<512x1x1x4xsi32, {order = #NCHW}> to tensor<256x1x1x4xsi32>
    //CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICE_WEIGHTS]], [[SLICE_WEIGHTS_TABLE]])

    //CHECK:      [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, 256, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf16, {order = #NHWC}> into tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: scf.yield [[INSERT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 - 1, 0)>
//CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-(d0 - 1), 0)>
//CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0, 1)>
//CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (d0 - 54, 0)>
//CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (-d0 - d1 + 10)>

// CHECK-LABEL:   @ConvChannel2DTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @ConvChannel2DTiling(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [256, 512, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        
        strides = [1, 1],
        tilingStrategy = [1, 2, 8, 1]
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_END_C:%.+]] = arith.constant 256 : index
    //CHECK: [[LOOP_END_H:%.+]] = arith.constant 64 : index
    //CHECK: [[LOOP_STEP_C:%.+]] = arith.constant 128 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 8 : index

    //CHECK: [[LOOP_C:%.+]] = scf.for
    //CHECK-SAME:          [[LOOP_ITER_C:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_C]] step [[LOOP_STEP_C]]
    //CHECK-SAME:          iter_args([[LOOP_OUT_C:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>)

    //CHECK:                [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:                           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:                           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUT_C]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>)


    //CHECK:                                  [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                                  [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER_H]])
    //CHECK:                                  [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE0]]]
    //CHECK:                                  [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    //CHECK:                                  [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE1]]]
    //CHECK:                                  [[INPUT_SIZE:%.+]] = affine.apply #[[$MAP4]]([[PAD_LOW]], [[PAD_HIGH]])

    //CHECK:                                  [[SLICE_INPUT:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 512, [[INPUT_SIZE]], 64] [1, 1, 1, 1] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                  [[SLICE_WEIGHTS:%.+]] = tensor.extract_slice [[WEIGHTS]][[[LOOP_ITER_C]], 0, 0, 0] [128, 512, 3, 3] [1, 1, 1, 1] : tensor<256x512x3x3xf16, {order = #NHWC}> to tensor<128x512x3x3xf16, {order = #NHWC}>
    //CHECK:                                  [[SLICE_WEIGHTS_TABLE:%.+]] = tensor.extract_slice [[WEIGHTS_TABLE]][[[LOOP_ITER_C]], 0, 0, 0] [128, 1, 1, 4] [1, 1, 1, 1] : tensor<256x1x1x4xsi32> to tensor<128x1x1x4xsi32>
    //CHECK:                                  [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK:                                  [[PAD:%.+]] = tensor.pad [[SLICE_INPUT]] low[0, 0, [[PAD_LOW]], 1] high[0, 0, [[PAD_HIGH]], 1] {
    //CHECK:                                  tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                                  tensor<1x512x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 64, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x512x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 66, 66]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                                  [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[SLICE_WEIGHTS]], [[SLICE_WEIGHTS_TABLE]])
    //CHECK:                                  [[PADDED_DIM:%.+]] = arith.constant 8 : index
    //CHECK:                                  [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT]][0, [[LOOP_ITER_C]], [[LOOP_ITER_H]], 0] [1, 128, [[PADDED_DIM]], 64] [1, 1, 1, 1] : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[LOOP_H]]
    //CHECK:  return [[LOOP_C]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

// Regression guard for the STATIC DepthToSpace SCF-tiling block-size alignment fix
// (updateDepthToSpaceBlockAlignmentMultiplier in src/vpux_compiler/src/core/tiling.cpp: alignment is now
// applied for the single-layer static case as well, while remaining disabled for static fused VF chains).
//
// The output H (12) is tiled into 4 pieces. Without alignment the tile size would be 3,
// is then non-integer and DepthToSpace fails to legalize. With the fix the output tile is
// snapped up to 4 (a multiple of block_size), so:
//   * the loop step is 4 (even) and 12 / 4 = 3 evenly-divisible tiles, no remainder,
//   * every input tile is an exact 2 (= 4 / block_size) with integer offset iv floordiv 2,
//   * no tile can overrun the input extent on unroll.
//
// The test feeds a VPU.DepthToSpace straight into apply-tiling, so it exercises the generic SCF
// tiling path independently of platform (the D2S->TransposedConv rewrite is an IE-dialect step that
// does not run here); before the fix `apply-tiling` reports "failed to legalize 'VPU.DepthToSpace'".

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @D2SStaticBlockAligned
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x6x4xf16, {order = #NHWC}>
func.func @D2SStaticBlockAligned(%arg0: tensor<1x16x6x4xf16, {order = #NHWC}>) -> tensor<1x4x12x8xf32, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {
        block_size = 2 : i64,
        dstElemType = f32,
        mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        tilingStrategy = [1, 1, 4, 1]
    } : tensor<1x16x6x4xf16, {order = #NHWC}> -> tensor<1x4x12x8xf32, {order = #NHWC}>

    return %0 : tensor<1x4x12x8xf32, {order = #NHWC}>

    // Output loop step is block-aligned: %c4 (a multiple of block_size=2), and 12 / 4 = 3
    // tiles divide the output H evenly (no remainder tile to overflow).
    // CHECK:       [[C0:%.+]] = arith.constant 0 : index
    // CHECK:       [[C12:%.+]] = arith.constant 12 : index
    // CHECK:       [[C4:%.+]] = arith.constant 4 : index
    // CHECK:       scf.for [[IV:%.+]] = [[C0]] to [[C12]] step [[C4]]
    // Input offset is an integer (iv floordiv block_size) and the input tile size is an exact 2
    // (= output tile 4 / block_size 2) — the property the alignment fix restores for static shapes.
    // CHECK:         [[OFF:%.+]] = affine.apply {{.*}}([[IV]])
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[OFF]], 0] [1, 16, 2, 4]
    // CHECK-SAME:        to tensor<1x16x2x4xf16, {order = #NHWC}>
    // CHECK:         [[D2S:%.+]] = VPU.DepthToSpace([[SLICE]]
    // CHECK-SAME:        tensor<1x16x2x4xf16, {order = #NHWC}> -> tensor<1x4x4x8xf32, {order = #NHWC}>
    // CHECK:         tensor.insert_slice [[D2S]] into {{.*}}[0, 0, [[IV]], 0] [1, 4, 4, 8]
}
