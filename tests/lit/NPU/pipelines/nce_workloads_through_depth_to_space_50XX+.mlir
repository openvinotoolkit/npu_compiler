//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cost-model-analysis-construct --nce-workloads="scf-tiling=true enable-print-statistics=false" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#tile_width = affine_map<(d0) -> (-d0 + 2560, 366)>
#input_offset = affine_map<(d0) -> (d0 floordiv 2)>
#input_width = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK-LABEL: @WorkloadShapeThroughDepthToSpace
func.func @WorkloadShapeThroughDepthToSpace(%arg0: tensor<1x96x16x1280xf16, {order = #NHWC}>)
        -> tensor<1x4x32x2560xf32, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c2560 = arith.constant 2560 : index
    %c366 = arith.constant 366 : index
    %weights = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %empty = tensor.empty() : tensor<1x4x32x2560xf32, {order = #NHWC}>

    %result = scf.for %width = %c0 to %c2560 step %c366 iter_args(%output = %empty) -> (tensor<1x4x32x2560xf32, {order = #NHWC}>) {
        %output_width = affine.min #tile_width(%width)
        %offset = affine.apply #input_offset(%width)
        %conv_width = affine.apply #input_width(%output_width)
        %slice = tensor.extract_slice %arg0[0, 0, 0, %offset] [1, 96, 16, %conv_width] [1, 1, 1, 1]
            : tensor<1x96x16x1280xf16, {order = #NHWC}>
            to tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %input_cmx = VPU.Copy(%slice) {out_mem_space = [@CMX_NN, 0]}
            : tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
            -> tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %weights_cmx = VPU.Copy(%weights) {out_mem_space = [@CMX_NN, 0]}
            : tensor<16x96x1x1xf16, {order = #NHWC}> -> tensor<16x96x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %conv = VPU.NCE.Convolution(%input_cmx, %weights_cmx) rawFilterShape [16, 96, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
            : tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>,
              tensor<16x96x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
            -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %conv_ddr = VPU.Copy(%conv)
            : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
            -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %upsampled = VPU.DepthToSpace(%conv_ddr) {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
            : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
            -> tensor<1x4x32x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %inserted = tensor.insert_slice %upsampled into %output[0, 0, 0, %width] [1, 4, 32, %output_width] [1, 1, 1, 1]
            : tensor<1x4x32x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 2560]> : tensor<4xsi64>, order = #NHWC}>
            into tensor<1x4x32x2560xf32, {order = #NHWC}>
        scf.yield %inserted : tensor<1x4x32x2560xf32, {order = #NHWC}>
    }

    return %result : tensor<1x4x32x2560xf32, {order = #NHWC}>
}

// The tail D2S output tile has width 364, so its input convolution and workload
// must have width 182 rather than the bounded width 1280.
// CHECK: VPU.Slice {{%.+}} [0, 0, 0, 1098] [1, 96, 16, 182]
// CHECK: VPU.NCE.Convolution
// CHECK-SAME: -> tensor<1x16x16x182xf16
// CHECK: VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 96, 16, 182] outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 182] pad [0, 0, 0, 0] <CUBOID_16x16>
