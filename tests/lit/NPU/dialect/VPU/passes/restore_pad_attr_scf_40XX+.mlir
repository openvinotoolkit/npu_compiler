//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --restore-tensor-pad-after-scf-tiling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 30)>

// CHECK-LABEL:   @ApplyTilingNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
module {
  func.func @ApplyTilingNCEConv(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg1)
      %3 = affine.max #map1(%arg1)
      %4 = affine.min #map2()[%3]
      %5 = affine.max #map3(%2)
      %6 = affine.min #map2()[%5]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
      %padded = tensor.pad %extracted_slice low[0, 0, %4, 1] high[0, 0, %6, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 35, 66]> : tensor<4xsi64>, order = #NHWC}>
      %7 = VPU.NCE.Convolution(%padded, %cst_0) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 35, 66]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 33, 64]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %7 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 33, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x32x64xf16, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }

    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 64 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {

    //CHECK:                [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP]]([[LOOP_ITER]])
    //CHECK:                [[DIFF1:%.+]] = affine.max #[[$MAP1]](%arg1)
    //CHECK:                [[PAD_LOW:%.+]] = affine.min #[[$MAP2]]()[[[DIFF1]]]
    //CHECK:                [[DIFF2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET]])
    //CHECK:                [[PAD_HIGH:%.+]] = affine.min #[[$MAP2]]()[[[DIFF2]]]

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    //CHECK:                [[CAST0:%.+]] = tensor.cast [[SLICE]] : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 33, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV:%.+]] = VPU.NCE.Convolution([[CAST0]], [[WEIGHTS]])
    //CHECK-SAME:           {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 33, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>
    //CHECK-SAME:           -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 33, 64]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CAST1:%.+]] = tensor.cast [[CONV]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 33, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x32x64xf16, {order = #NHWC}>
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[CAST1]] into [[LOOP_OUT]][0, 0, [[LOOP_ITER]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
    //CHECK: return [[LOOP]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
 // CHECK-LABEL: @NoPaddingDWCONV
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x32x200x200xf16, {order = #NHWC}>,
 // CHECK-SAME:      [[WEIGHTS:%arg[0-9]]]: tensor<32x16x1x1xf16, {order = #NHWC}>

module {
  func.func @NoPaddingDWCONV(%arg0: tensor<1x32x200x200xf16, {order = #NHWC}>, %arg1: tensor<32x16x1x1xf16, {order = #NHWC}>) -> tensor<1x32x200x200xf16, {order = #NHWC}> {
    %c50 = arith.constant 50 : index
    %c200 = arith.constant 200 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x32x200x200xf16, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c200 step %c50 iter_args(%arg3 = %0) -> (tensor<1x32x200x200xf16, {order = #NHWC}>) {
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x200xf16, {order = #NHWC}> to tensor<1x32x200x50xf16, {order = #NHWC}>
      %2 = VPU.NCE.DepthConvolution(%extracted_slice, %arg1) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x200x50xf16, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %2 into %arg3[0, 0, 0, %arg2] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x50xf16, {order = #NHWC}> into tensor<1x32x200x200xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x200x200xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x32x200x200xf16, {order = #NHWC}>
  }

  //CHECK: [[LOOP_STEP:%.+]] = arith.constant 50 : index
  //CHECK: [[LOOP_END:%.+]] = arith.constant 200 : index
  //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
  //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x200x200xf16, {order = #NHWC}>
  //CHECK: [[LOOP:%.+]] = scf.for
  //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
  //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x200x200xf16, {order = #NHWC}>) {

  //CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x200xf16, {order = #NHWC}> to tensor<1x32x200x50xf16, {order = #NHWC}>
  //CHECK:      [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS]])

  //CHECK: [[INSERT:%.+]] = tensor.insert_slice [[DWCONV]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 200, 50] [1, 1, 1, 1] : tensor<1x32x200x50xf16, {order = #NHWC}> into tensor<1x32x200x200xf16, {order = #NHWC}>
  //CHECK: scf.yield [[INSERT]] : tensor<1x32x200x200xf16, {order = #NHWC}>
  //CHECK: return [[LOOP]] : tensor<1x32x200x200xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 254)>
#map4 = affine_map<(d0) -> (0, d0 - 358)>
#map5 = affine_map<(d0, d1) -> (-d0 - d1 + 122)>

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0) -> (0, d0 - 254)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (0, d0 - 358)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (-d0 - d1 + 122)>

// CHECK-LABEL:   @Tiling2DPaddedMaxPool
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x512x480xf16, {order = #NHWC}>
module {
  func.func @Tiling2DPaddedMaxPool(%arg0: tensor<1x16x512x480xf16, {order = #NHWC}>) -> tensor<1x16x512x480xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c120 = arith.constant 120 : index
    %c480 = arith.constant 480 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c512 step %c256 iter_args(%arg2 = %0) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %c480 step %c120 iter_args(%arg4 = %arg2) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
        %3 = affine.max #map(%arg1)
        %4 = affine.max #map1(%arg1)
        %5 = affine.min #map2()[%4]
        %6 = affine.max #map3(%3)
        %7 = affine.min #map2()[%6]
        %8 = affine.max #map(%arg3)
        %9 = affine.max #map1(%arg3)
        %10 = affine.min #map2()[%9]
        %11 = affine.max #map4(%8)
        %12 = affine.min #map2()[%11]
        %13 = affine.apply #map5(%10, %12)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, %8] [1, 16, 257, %13] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %padded = tensor.pad %extracted_slice low[0, 0, %5, %10] high[0, 0, %7, %12] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %cst : f16
        } : tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 259, 482]> : tensor<4xsi64>, order = #NHWC}>
        %14 = VPU.NCE.MaxPool(%padded) {kernel_size = [3, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %cast = tensor.cast %14 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg4[0, 0, %arg1, %arg3] [1, 16, 256, 120] [1, 1, 1, 1] : tensor<1x16x256x120xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x512x480xf16, {order = #NHWC}>
      }
      scf.yield %2 : tensor<1x16x512x480xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x512x480xf16, {order = #NHWC}>
  }

    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 120 : index
    //CHECK: [[LOOP_END_W:%.+]] = arith.constant 480 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 256 : index
    //CHECK: [[LOOP_END_H:%.+]] = arith.constant 512 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:                            [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:                            iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x512x480xf16, {order = #NHWC}>)

    //CHECK:                                 [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                                 [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER_H]])
    //CHECK:                                 [[PAD_LOW_H:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE0]]]
    //CHECK:                                 [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_H]])
    //CHECK:                                 [[PAD_HIGH_H:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE1]]]
    //CHECK:                                 [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_W]])
    //CHECK:                                 [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER_W]])
    //CHECK:                                 [[PAD_LOW_W:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE2]]]
    //CHECK:                                 [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP4]]([[SLICE_OFFSET_W]])
    //CHECK:                                 [[PAD_HIGH_W:%.+]] = affine.min #[[$MAP2]]()[[[TEMP_VALUE3]]]
    //CHECK:                                 [[W_SIZE:%.+]] = affine.apply #map5([[PAD_LOW_W]], [[PAD_HIGH_W]])

    //CHECK:                                 [[SLICE:%.+]]  = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 16, 257, [[W_SIZE]]] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                 [[CAST0:%.+]] = tensor.cast [[SLICE]] : tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                 [[POOL:%.+]] = VPU.NCE.MaxPool([[CAST0]])
    //CHECK-SAME:                                           {kernel_size = [3, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                                           -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                                 [[CAST1:%.+]] = tensor.cast [[POOL]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
    //CHECK:                                 [[INSERT:%.+]] = tensor.insert_slice [[CAST1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, 256, 120] [1, 1, 1, 1]
    //CHECK-SAME:                            tensor<1x16x256x120xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK:  scf.yield [[INSERT]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  scf.yield [[LOOP_W]] : tensor<1x16x512x480xf16, {order = #NHWC}>
    //CHECK:  return [[LOOP_H]] : tensor<1x16x512x480xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 960, 26)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 + d10 - d11 - d12 + 12)>

//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 26)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>

// CHECK-LABEL: @MergeVFChain3Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x540x120xf16, {order = #NHWC}>)
module {
  func.func @MergeVFChain3Tiles(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c26 = arith.constant 26 : index
    %c960 = arith.constant 960 : index
    %c0 = arith.constant 0 : index
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = scf.for %arg1 = %c0 to %c960 step %c26 iter_args(%arg2 = %1) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
      %4 = affine.min #map(%arg1)
      %5 = affine.max #map1(%arg1)
      %6 = affine.max #map2(%arg1)
      %7 = affine.min #map3()[%6]
      %8 = affine.max #map4(%4, %5)
      %9 = affine.min #map3()[%8]
      %10 = affine.max #map1(%5)
      %11 = affine.max #map2(%5)
      %12 = affine.min #map3()[%11]
      %13 = affine.max #map5(%10, %4, %7, %9)
      %14 = affine.min #map3()[%13]
      %15 = affine.max #map1(%10)
      %16 = affine.max #map2(%10)
      %17 = affine.min #map3()[%16]
      %18 = affine.max #map6(%15, %12, %14, %4, %7, %9)
      %19 = affine.min #map3()[%18]
      %20 = affine.max #map1(%15)
      %21 = affine.max #map2(%15)
      %22 = affine.min #map3()[%21]
      %23 = affine.max #map7(%20, %17, %19, %12, %14, %4, %7, %9)
      %24 = affine.min #map3()[%23]
      %25 = affine.max #map1(%20)
      %26 = affine.max #map2(%20)
      %27 = affine.min #map3()[%26]
      %28 = affine.max #map8(%25, %22, %24, %17, %19, %12, %14, %4, %7, %9)
      %29 = affine.min #map3()[%28]
      %30 = affine.max #map1(%25)
      %31 = affine.max #map2(%25)
      %32 = affine.min #map3()[%31]
      %33 = affine.max #map9(%30, %27, %29, %22, %24, %17, %19, %12, %14, %4, %7, %9)
      %34 = affine.min #map3()[%33]
      %35 = affine.apply #map10(%32, %34, %27, %29, %22, %24, %17, %19, %12, %14, %4, %7, %9)
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, %30] [1, 32, 540, %35] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded = tensor.pad %extracted_slice low[0, 0, 1, %32] high[0, 0, 1, %34] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %36 = VPU.NCE.Convolution(%padded, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_2 = tensor.pad %36 low[0, 0, 1, %27] high[0, 0, 1, %29] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %37 = VPU.NCE.Convolution(%padded_2, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_3 = tensor.pad %37 low[0, 0, 1, %22] high[0, 0, 1, %24] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %38 = VPU.NCE.Convolution(%padded_3, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %39 = VPU.NCE.DepthConvolution(%38, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_4 = tensor.pad %39 low[0, 0, 1, %17] high[0, 0, 1, %19] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %40 = VPU.NCE.Convolution(%padded_4, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_5 = tensor.pad %40 low[0, 0, 1, %12] high[0, 0, 1, %14] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %41 = VPU.NCE.Convolution(%padded_5, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_6 = tensor.pad %41 low[0, 0, 1, %7] high[0, 0, 1, %9] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %42 = VPU.NCE.Convolution(%padded_6, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %43 = VPU.NCE.DepthConvolution(%42, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %43 into %arg2[0, 0, 0, %arg1] [1, 32, 540, %4] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x540x960xf16, {order = #NHWC}>
    }
    %3 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%2 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
    return %3 : tensor<1x128x540x240xf16, {order = #NHWC}>
  }
    //CHECK: [[PAD_CONST:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 26 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}>
    //CHECK: [[DW_WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK: [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {

    //CHECK:                [[TILE_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])
    //CHECK:                [[SLICE_OFFSET1:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:                [[TEMP_VAL1:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:                [[PAD_LEFT1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL1]]]
    //CHECK:                [[TEMP_VAL2:%.+]] = affine.max #[[$MAP4]]([[TILE_SIZE]], [[SLICE_OFFSET1]])
    //CHECK:                [[PAD_RIGHT1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL2]]]
    //CHECK:                [[SLICE_OFFSET2:%.+]] = affine.max #[[$MAP1]]([[SLICE_OFFSET1]])
    //CHECK:                [[TEMP_VAL3:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET1]])
    //CHECK:                [[PAD_LEFT2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL3]]]
    //CHECK:                [[TEMP_VAL4:%.+]] = affine.max #[[$MAP5]]([[SLICE_OFFSET2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])
    //CHECK:                [[PAD_RIGHT2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL4]]]
    //CHECK:                [[SLICE_OFFSET3:%.+]] = affine.max #[[$MAP1]]([[SLICE_OFFSET2]])
    //CHECK:                [[TEMP_VAL5:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET2]])
    //CHECK:                [[PAD_LEFT3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL5]]]
    //CHECK:                [[TEMP_VAL6:%.+]] = affine.max #[[$MAP6]]([[SLICE_OFFSET3]], [[PAD_LEFT2]], [[PAD_RIGHT2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])
    //CHECK:                [[PAD_RIGHT3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL6]]]
    //CHECK:                [[SLICE_OFFSET4:%.+]] = affine.max #[[$MAP1]]([[SLICE_OFFSET3]])
    //CHECK:                [[TEMP_VAL7:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET3]])
    //CHECK:                [[PAD_LEFT4:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL7]]]
    //CHECK:                [[TEMP_VAL8:%.+]] = affine.max #[[$MAP7]]([[SLICE_OFFSET4]], [[PAD_LEFT3]], [[PAD_RIGHT3]], [[PAD_LEFT2]], [[PAD_RIGHT2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])
    //CHECK:                [[PAD_RIGHT4:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL8]]]
    //CHECK:                [[SLICE_OFFSET5:%.+]] = affine.max #[[$MAP1]]([[SLICE_OFFSET4]])
    //CHECK:                [[TEMP_VAL9:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET4]])
    //CHECK:                [[PAD_LEFT5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL9]]]
    //CHECK:                [[TEMP_VAL10:%.+]] = affine.max #[[$MAP8]]([[SLICE_OFFSET5]], [[PAD_LEFT4]], [[PAD_RIGHT4]], [[PAD_LEFT3]], [[PAD_RIGHT3]], [[PAD_LEFT2]], [[PAD_RIGHT2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])
    //CHECK:                [[PAD_RIGHT5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL10]]]
    //CHECK:                [[SLICE_OFFSET6:%.+]] = affine.max #[[$MAP1]]([[SLICE_OFFSET5]])
    //CHECK:                [[TEMP_VAL11:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET5]])
    //CHECK:                [[PAD_LEFT6:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL11]]]
    //CHECK:                [[TEMP_VAL12:%.+]] = affine.max #[[$MAP9]]([[SLICE_OFFSET6]], [[PAD_LEFT5]], [[PAD_RIGHT5]], [[PAD_LEFT4]], [[PAD_RIGHT4]], [[PAD_LEFT3]], [[PAD_RIGHT3]], [[PAD_LEFT2]], [[PAD_RIGHT2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])
    //CHECK:                [[PAD_RIGHT6:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VAL12]]]
    //CHECK:                [[SLICE_WIDTH:%.+]] = affine.apply #map10([[PAD_LEFT6]], [[PAD_RIGHT6]], [[PAD_LEFT5]], [[PAD_RIGHT5]], [[PAD_LEFT4]], [[PAD_RIGHT4]], [[PAD_LEFT3]], [[PAD_RIGHT3]], [[PAD_LEFT2]], [[PAD_RIGHT2]], [[TILE_SIZE]], [[PAD_LEFT1]], [[PAD_RIGHT1]])

    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[SHAPE_CAST]][0, 0, 0, [[SLICE_OFFSET6]]] [1, 32, 540, [[SLICE_WIDTH]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[CONV3:%.+]] = VPU.NCE.Convolution([[CONV2]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV3]], [[DW_WEIGHTS]])
    //CHECK-SAME:           {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:                [[CONV4:%.+]] = VPU.NCE.Convolution([[DWCONV1]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[CONV5:%.+]] = VPU.NCE.Convolution([[CONV4]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[CONV6:%.+]] = VPU.NCE.Convolution([[CONV5]], [[WEIGHTS]])
    //CHECK-SAME:           {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK:                [[DWCONV2:%.+]] = VPU.NCE.DepthConvolution([[CONV6]], [[DW_WEIGHTS]])
    //CHECK-SAME:           {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[DWCONV2]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[TILE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[FINAL_CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]] : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
    //CHECK: return [[FINAL_CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}
