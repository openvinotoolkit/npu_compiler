//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling="enable-scf-tiling=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

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

        %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 256, 3, 3],
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

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 512, 3, 3],
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
