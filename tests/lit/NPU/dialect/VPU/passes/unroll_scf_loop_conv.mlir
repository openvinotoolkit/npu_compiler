//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,1,2" --canonicalize --cse %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
#map2 = affine_map<(d0) -> (0, d0 - 1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
#map7 = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>
#map8 = affine_map<(d0)[s0] -> (d0 + s0 - 160)>
#map9 = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
#map10 = affine_map<(d0)[s0] -> (d0 + s0 - 162)>
#map11 = affine_map<(d0)[s0] -> (d0 + s0 - 258)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP4:.*]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
// CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>
// CHECK: #[[$MAP8:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 258)>
// CHECK: #[[$MAP9:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 162)>
// CHECK: #[[$MAP10:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
// CHECK: #[[$MAP11:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 160)>

module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn2D inputsInfo : {
    DataInfo "input" : tensor<1x32x?x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x?xf16>
  }

  // CHECK: func.func @merged_vpu_func_10_11([[ARG0:%.+]]: tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 162] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 257, 161] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_10_10([[ARG0:%.+]]: tensor<1x32x257x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 162] : tensor<1x32x257x324xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 257, 162] : tensor<1x32x257x324xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_12_11([[ARG0:%.+]]: tensor<1x32x257x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 161] : tensor<1x32x257x322xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 257, 161] : tensor<1x32x257x322xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_12_10([[ARG0:%.+]]: tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 161] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 257, 162] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_00_01([[ARG0:%.+]]: tensor<1x32x258x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 258, 162] : tensor<1x32x258x323xf16, {order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 258, 161] : tensor<1x32x258x323xf16, {order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_00_00([[ARG0:%.+]]: tensor<1x32x258x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 258, 162] : tensor<1x32x258x324xf16, {order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 258, 162] : tensor<1x32x258x324xf16, {order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_02_01([[ARG0:%.+]]: tensor<1x32x258x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 258, 161] : tensor<1x32x258x322xf16, {order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 258, 161] : tensor<1x32x258x322xf16, {order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_02_00([[ARG0:%.+]]: tensor<1x32x258x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 258, 161] : tensor<1x32x258x323xf16, {order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 258, 162] : tensor<1x32x258x323xf16, {order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_20_21([[ARG0:%.+]]: tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 162] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 257, 161] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_20_20([[ARG0:%.+]]: tensor<1x32x257x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 162] : tensor<1x32x257x324xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 162] [1, 32, 257, 162] : tensor<1x32x257x324xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_22_21([[ARG0:%.+]]: tensor<1x32x257x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 161] : tensor<1x32x257x322xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 257, 161] : tensor<1x32x257x322xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @merged_vpu_func_22_20([[ARG0:%.+]]: tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 32, 257, 161] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:   [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 161] [1, 32, 257, 162] : tensor<1x32x257x323xf16, {order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:   [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {static_offsets =
  // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 160]]}
  // CHECK-SAME:  tensor<1x256x256x160xf16, {order = #NHWC}>, tensor<1x256x256x160xf16, {order = #NHWC}> -> tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK:   return [[CONCAT]] : tensor<1x256x256x320xf16, {order = #NHWC}>
  // CHECK: }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static([[ARG0:%.+]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static(%arg0: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static([[ARG0:%.+]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static(%arg0: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static([[ARG0:%.+]]: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static(%arg0: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static([[ARG0:%.+]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static(%arg0: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static([[ARG0:%.+]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static(%arg0: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static([[ARG0:%.+]]: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static(%arg0: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static([[ARG0:%.+]]: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static(%arg0: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static([[ARG0:%.+]]: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static(%arg0: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x161xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static([[ARG0:%.+]]: tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: }
  func.func @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static(%arg0: tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x162xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x160xf16, {order = #NHWC}>
    return %0 : tensor<1x256x256x160xf16, {order = #NHWC}>
  }

  func.func @ApplyTilingNCEConvDyn2D(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %false = arith.constant false
    %c161 = arith.constant 161 : index
    %c257 = arith.constant 257 : index
    %c1 = arith.constant 1 : index
    %c160 = arith.constant 160 : index
    %c256 = arith.constant 256 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %1 = arith.remui %dim, %c256 : index
    %2 = arith.remui %dim_0, %c160 : index
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %3 = arith.remui %dim_1, %c257 : index
    %4 = arith.remui %dim_0, %c161 : index
    %5 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
      %6 = scf.for %arg3 = %c0 to %dim_0 step %c160 iter_args(%arg4 = %arg2) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
        %7 = affine.min #map(%arg1)[%dim]
        %8 = affine.min #map1(%arg3)[%dim_0]
        %9 = affine.max #map2(%arg1)
        %10 = affine.max #map3(%arg1)
        %11 = affine.min #map4()[%10]
        %12 = affine.max #map5(%7, %9)
        %13 = affine.min #map4()[%12]
        %14 = affine.apply #map6(%7, %11, %13)
        %15 = affine.max #map2(%arg3)
        %16 = affine.max #map3(%arg3)
        %17 = affine.min #map4()[%16]
        %18 = affine.max #map7(%8, %15)
        %19 = affine.min #map4()[%18]
        %20 = affine.apply #map6(%8, %17, %19)
        %21 = arith.cmpi eq, %arg1, %c0 : index
        %22 = arith.cmpi eq, %arg3, %c0 : index
        %23 = scf.if %22 -> (index) {
          %32 = arith.cmpi sge, %8, %c160 : index
          cf.assert %32, "Not enough elements to backtrack in scf.for loop"
          scf.yield %arg3 : index
        } else {
          %32 = arith.addi %arg3, %c160 : index
          %33 = arith.cmpi slt, %32, %dim_0 : index
          %34 = scf.if %33 -> (index) {
            scf.yield %arg3 : index
          } else {
            %35 = arith.cmpi eq, %32, %dim_0 : index
            %36 = scf.if %35 -> (index) {
              scf.yield %arg3 : index
            } else {
              %37 = affine.apply #map8(%arg3)[%2]
              scf.yield %37 : index
            }
            scf.yield %36 : index
          }
          scf.yield %34 : index
        }
        %24 = scf.if %21 -> (index) {
          %32 = arith.cmpi sge, %7, %c256 : index
          cf.assert %32, "Not enough elements to backtrack in scf.for loop"
          scf.yield %arg1 : index
        } else {
          %32 = arith.addi %arg1, %c256 : index
          %33 = arith.cmpi slt, %32, %dim : index
          %34 = scf.if %33 -> (index) {
            scf.yield %arg1 : index
          } else {
            %35 = arith.cmpi eq, %32, %dim : index
            %36 = scf.if %35 -> (index) {
              scf.yield %arg1 : index
            } else {
              %37 = affine.apply #map9(%arg1)[%1]
              scf.yield %37 : index
            }
            scf.yield %36 : index
          }
          scf.yield %34 : index
        }
        %25 = arith.cmpi eq, %9, %c0 : index
        %26 = arith.cmpi eq, %15, %c0 : index
        %27:2 = scf.if %26 -> (index, index) {
          %32 = arith.cmpi sge, %20, %c161 : index
          cf.assert %32, "Not enough elements to backtrack in scf.for loop"
          %33 = arith.cmpi eq, %20, %dim_0 : index
          %34 = arith.select %33, %c3, %c2 : index
          scf.yield %34, %15 : index, index
        } else {
          %32 = arith.addi %15, %c161 : index
          %33 = arith.cmpi slt, %32, %dim_0 : index
          %34 = arith.select %33, %c0, %c1 : index
          %35 = scf.if %33 -> (index) {
            scf.yield %15 : index
          } else {
            %36 = arith.cmpi eq, %32, %dim_0 : index
            %37 = scf.if %36 -> (index) {
              scf.yield %15 : index
            } else {
              %38 = affine.apply #map10(%15)[%4]
              scf.yield %38 : index
            }
            scf.yield %37 : index
          }
          scf.yield %34, %35 : index, index
        }
        %28:2 = scf.if %25 -> (index, index) {
          %32 = arith.cmpi sge, %14, %c257 : index
          cf.assert %32, "Not enough elements to backtrack in scf.for loop"
          %33 = arith.cmpi eq, %14, %dim_1 : index
          %34 = arith.select %33, %c3, %c2 : index
          scf.yield %34, %9 : index, index
        } else {
          %32 = arith.addi %9, %c257 : index
          %33 = arith.cmpi slt, %32, %dim_1 : index
          %34 = arith.select %33, %c0, %c1 : index
          %35 = scf.if %33 -> (index) {
            scf.yield %9 : index
          } else {
            %36 = arith.cmpi eq, %32, %dim_1 : index
            %37 = scf.if %36 -> (index) {
              scf.yield %9 : index
            } else {
              %38 = affine.apply #map11(%9)[%3]
              scf.yield %38 : index
            }
            scf.yield %37 : index
          }
          scf.yield %34, %35 : index, index
        }
        %29 = arith.shli %28#0, %c2 : index
        %30 = arith.ori %29, %27#0 : index
        %31 = scf.index_switch %30 -> tensor<1x256x256x160xf16, {order = #NHWC}>
        case 0 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static(%extracted_slice) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 1 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static(%extracted_slice) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 2 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static(%extracted_slice) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 4 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static(%extracted_slice) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 5 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static(%extracted_slice) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 6 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static(%extracted_slice) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 8 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static(%extracted_slice) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 9 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static(%extracted_slice) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        case 10 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static(%extracted_slice) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        default {
          cf.assert %false, "Unsupported case"
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
          %cast = tensor.cast %extracted_slice : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static(%extracted_slice) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
          scf.yield %32 : tensor<1x256x256x160xf16, {order = #NHWC}>
        }
        %inserted_slice = tensor.insert_slice %31 into %arg4[0, 0, %24, %23] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %6 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %5 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: func.func @ApplyTilingNCEConvDyn2D(
  // CHECK-SAME:    [[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK-SAME:  ) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK:   [[C320:%.+]] = arith.constant 320 : index
    // CHECK:   [[C159:%.+]] = arith.constant 159 : index
    // CHECK:   [[FALSE:%.+]] = arith.constant false
    // CHECK:   [[C161:%.+]] = arith.constant 161 : index
    // CHECK:   [[C257:%.+]] = arith.constant 257 : index
    // CHECK:   [[C1:%.+]] = arith.constant 1 : index
    // CHECK:   [[C160:%.+]] = arith.constant 160 : index
    // CHECK:   [[C256:%.+]] = arith.constant 256 : index
    // CHECK:   [[C3:%.+]] = arith.constant 3 : index
    // CHECK:   [[C0:%.+]] = arith.constant 0 : index
    // CHECK:   [[C2:%.+]] = arith.constant 2 : index
    // CHECK:   [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[DIM_0:%.+]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[EMPTY:%.+]] = tensor.empty([[DIM]], [[DIM_0]]) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   [[REM1:%.+]] = arith.remui [[DIM]], [[C256]] : index
    // CHECK:   [[REM2:%.+]] = arith.remui [[DIM_0]], [[C160]] : index
    // CHECK:   [[REM3:%.+]] = arith.remui [[DIM]], [[C257]] : index
    // CHECK:   [[REM4:%.+]] = arith.remui [[DIM_0]], [[C161]] : index
    // CHECK:   [[FOR1:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[DIM]] step [[C256]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK:     [[ADD1:%.+]] = arith.addi [[DIM_0]], [[C159]] : index
    // CHECK:     [[DIV:%.+]] = arith.divui [[ADD1]], [[C160]] : index
    // CHECK:     [[REM_DIV:%.+]] = arith.remsi [[DIV]], [[C2]] : index
    // CHECK:     [[SUB:%.+]] = arith.subi [[DIV]], [[REM_DIV]] : index
    // CHECK:     [[MUL:%.+]] = arith.muli [[SUB]], [[C160]] : index
    // CHECK:     [[FOR2:%.+]] = scf.for [[ARG3:%.+]] = [[C0]] to [[MUL]] step [[C320]] iter_args([[ARG4:%.+]] = [[ARG2]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK:       [[MIN1:%.+]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM]]]
    // CHECK:       [[MIN2:%.+]] = affine.min #[[$MAP1]]([[ARG3]])[[[DIM_0]]]
    // CHECK:       [[MAX1:%.+]] = affine.max #[[$MAP2]]([[ARG1]])
    // CHECK:       [[MAX2:%.+]] = affine.max #[[$MAP3]]([[ARG1]])
    // CHECK:       [[MIN3:%.+]] = affine.min #[[$MAP4]]()[[[MAX2]]]
    // CHECK:       [[MAX3:%.+]] = affine.max #[[$MAP5]]([[MIN1]], [[MAX1]])
    // CHECK:       [[MIN4:%.+]] = affine.min #[[$MAP4]]()[[[MAX3]]]
    // CHECK:       [[APPLY1:%.+]] = affine.apply #[[$MAP6]]([[MIN1]], [[MIN3]], [[MIN4]])
    // CHECK:       [[MAX4:%.+]] = affine.max #[[$MAP2]]([[ARG3]])
    // CHECK:       [[MAX5:%.+]] = affine.max #[[$MAP3]]([[ARG3]])
    // CHECK:       [[MIN5:%.+]] = affine.min #[[$MAP4]]()[[[MAX5]]]
    // CHECK:       [[MAX6:%.+]] = affine.max #[[$MAP7]]([[MIN2]], [[MAX4]])
    // CHECK:       [[MIN6:%.+]] = affine.min #[[$MAP4]]()[[[MAX6]]]
    // CHECK:       [[APPLY2:%.+]] = affine.apply #[[$MAP6]]([[MIN2]], [[MIN5]], [[MIN6]])
    // CHECK:       [[CMP1:%.+]] = arith.cmpi eq, [[ARG1]], [[C0]] : index
    // CHECK:       [[CMP2:%.+]] = arith.cmpi eq, [[ARG3]], [[C0]] : index
    // CHECK:       [[CMP3:%.+]] = arith.cmpi eq, [[MAX1]], [[C0]] : index
    // CHECK:       [[CMP4:%.+]] = arith.cmpi eq, [[MAX4]], [[C0]] : index
    // CHECK:       [[ADD2:%.+]] = arith.addi [[ARG3]], [[C160]] : index
    // CHECK:       [[MIN7:%.+]] = affine.min #[[$MAP1]]([[ADD2]])[[[DIM_0]]]
    // CHECK:       [[MAX7:%.+]] = affine.max #[[$MAP2]]([[ADD2]])
    // CHECK:       [[MAX8:%.+]] = affine.max #[[$MAP3]]([[ADD2]])
    // CHECK:       [[MIN8:%.+]] = affine.min #[[$MAP4]]()[[[MAX8]]]
    // CHECK:       [[MAX9:%.+]] = affine.max #[[$MAP7]]([[MIN7]], [[MAX7]])
    // CHECK:       [[MIN9:%.+]] = affine.min #[[$MAP4]]()[[[MAX9]]]
    // CHECK:       [[APPLY3:%.+]] = affine.apply #[[$MAP6]]([[MIN7]], [[MIN8]], [[MIN9]])
    // CHECK:       [[CMP5:%.+]] = arith.cmpi eq, [[ADD2]], [[C0]] : index
    // CHECK:       [[CMP6:%.+]] = arith.cmpi eq, [[MAX7]], [[C0]] : index
    // CHECK:       [[IF1:%.+]] = scf.if [[CMP3]] -> (index) {
    // CHECK:         [[CMP_SGE:%.+]] = arith.cmpi sge, [[APPLY1]], [[C257]] : index
    // CHECK:         cf.assert [[CMP_SGE]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:         [[CMP_EQ:%.+]] = arith.cmpi eq, [[APPLY1]], [[DIM]] : index
    // CHECK:         [[SELECT:%.+]] = arith.select [[CMP_EQ]], [[C3]], [[C2]] : index
    // CHECK:         scf.yield [[SELECT]] : index
    // CHECK:       } else {
    // CHECK:         [[ADD:%.+]] = arith.addi [[MAX1]], [[C257]] : index
    // CHECK:         [[CMP_SLT:%.+]] = arith.cmpi slt, [[ADD]], [[DIM]] : index
    // CHECK:         [[SELECT2:%.+]] = arith.select [[CMP_SLT]], [[C0]], [[C1]] : index
    // CHECK:         scf.yield [[SELECT2]] : index
    // CHECK:       }
    // CHECK:       [[SHLI:%.+]] = arith.shli [[IF1]], [[C2]] : index
    // CHECK:       [[IF2:%.+]] = scf.if [[CMP6]] -> (index) {
    // CHECK:         [[CMP_SGE2:%.+]] = arith.cmpi sge, [[APPLY3]], [[C161]] : index
    // CHECK:         cf.assert [[CMP_SGE2]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:         [[CMP_EQ2:%.+]] = arith.cmpi eq, [[APPLY3]], [[DIM_0]] : index
    // CHECK:         [[SELECT3:%.+]] = arith.select [[CMP_EQ2]], [[C3]], [[C2]] : index
    // CHECK:         scf.yield [[SELECT3]] : index
    // CHECK:       } else {
    // CHECK:         [[ADD3:%.+]] = arith.addi [[MAX7]], [[C161]] : index
    // CHECK:         [[CMP_SLT2:%.+]] = arith.cmpi slt, [[ADD3]], [[DIM_0]] : index
    // CHECK:         [[SELECT4:%.+]] = arith.select [[CMP_SLT2]], [[C0]], [[C1]] : index
    // CHECK:         scf.yield [[SELECT4]] : index
    // CHECK:       }
    // CHECK:       [[ORI:%.+]] = arith.ori [[SHLI]], [[IF2]] : index
    // CHECK:       scf.if [[CMP1]] {
    // CHECK:         [[CMP_SGE3:%.+]] = arith.cmpi sge, [[MIN1]], [[C256]] : index
    // CHECK:         cf.assert [[CMP_SGE3]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:       }
    // CHECK:       scf.if [[CMP5]] {
    // CHECK:         [[CMP_SGE4:%.+]] = arith.cmpi sge, [[MIN7]], [[C160]] : index
    // CHECK:         cf.assert [[CMP_SGE4]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:       }
    // CHECK:       [[IF3:%.+]]:2 = scf.if [[CMP3]] -> (index, index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[SHLI2:%.+]] = arith.shli [[IF3]]#0, [[C2]] : index
    // CHECK:       [[IF6:%.+]]:2 = scf.if [[CMP4]] -> (index, index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[ORI2:%.+]] = arith.ori [[SHLI2]], [[IF6]]#0 : index
    // CHECK:       [[IF9:%.+]] = scf.if [[CMP1]] -> (index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[IF12:%.+]] = scf.if [[CMP2]] -> (index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[ANDI1:%.+]] = arith.andi [[ORI2]], [[C3]] : index
    // CHECK:       [[ANDI2:%.+]] = arith.andi [[ORI]], [[C3]] : index
    // CHECK:       [[SHLI3:%.+]] = arith.shli [[ANDI2]], [[C2]] : index
    // CHECK:       [[ORI3:%.+]] = arith.ori [[ANDI1]], [[SHLI3]] : index
    // CHECK:       [[SWITCH:%.+]] = scf.index_switch [[ORI3]] -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       case 162 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_22_20([[SLICE]]) : (tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 166 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 322] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x322xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_22_21([[SLICE]]) : (tensor<1x32x257x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 160 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 324] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x324xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_20_20([[SLICE]]) : (tensor<1x32x257x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 164 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_20_21([[SLICE]]) : (tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 2 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 258, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_02_00([[SLICE]]) : (tensor<1x32x258x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 6 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 258, 322] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x322xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_02_01([[SLICE]]) : (tensor<1x32x258x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 0 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 258, 324] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x324xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_00_00([[SLICE]]) : (tensor<1x32x258x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 4 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 258, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_00_01([[SLICE]]) : (tensor<1x32x258x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 82 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_12_10([[SLICE]]) : (tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 86 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 322] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x322xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_12_11([[SLICE]]) : (tensor<1x32x257x322xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 80 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 324] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x324xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_10_10([[SLICE]]) : (tensor<1x32x257x324xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 84 {
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_10_11([[SLICE]]) : (tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       default {
    // CHECK:         cf.assert [[FALSE]], "Invalid block position"
    // CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF3]]#1, [[IF6]]#1] [1, 32, 257, 323] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x323xf16, {order = #NHWC}>
    // CHECK:         [[CALL:%.+]] = func.call @merged_vpu_func_22_20([[SLICE]]) : (tensor<1x32x257x323xf16, {order = #NHWC}>) -> tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL]] : tensor<1x256x256x320xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[SWITCH]] into [[ARG4]][0, 0, [[IF9]], [[IF12]]] [1, 256, 256, 320] [1, 1, 1, 1] : tensor<1x256x256x320xf16, {order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       scf.yield [[INSERT]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     }
    // CHECK:     [[FOR3:%.+]] = scf.for [[ARG3:%.+]] = [[MUL]] to [[DIM_0]] step [[C160]] iter_args([[ARG4:%.+]] = [[FOR2]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK:       [[MIN1:%.+]] = affine.min #[[$MAP]]([[ARG1]])[[[DIM]]]
    // CHECK:       [[MIN2:%.+]] = affine.min #[[$MAP1]]([[ARG3]])[[[DIM_0]]]
    // CHECK:       [[MAX1:%.+]] = affine.max #[[$MAP2]]([[ARG1]])
    // CHECK:       [[MAX2:%.+]] = affine.max #[[$MAP3]]([[ARG1]])
    // CHECK:       [[MIN3:%.+]] = affine.min #[[$MAP4]]()[[[MAX2]]]
    // CHECK:       [[MAX3:%.+]] = affine.max #[[$MAP5]]([[MIN1]], [[MAX1]])
    // CHECK:       [[MIN4:%.+]] = affine.min #[[$MAP4]]()[[[MAX3]]]
    // CHECK:       [[APPLY1:%.+]] = affine.apply #[[$MAP6]]([[MIN1]], [[MIN3]], [[MIN4]])
    // CHECK:       [[MAX4:%.+]] = affine.max #[[$MAP2]]([[ARG3]])
    // CHECK:       [[MAX5:%.+]] = affine.max #[[$MAP3]]([[ARG3]])
    // CHECK:       [[MIN5:%.+]] = affine.min #[[$MAP4]]()[[[MAX5]]]
    // CHECK:       [[MAX6:%.+]] = affine.max #[[$MAP7]]([[MIN2]], [[MAX4]])
    // CHECK:       [[MIN6:%.+]] = affine.min #[[$MAP4]]()[[[MAX6]]]
    // CHECK:       [[APPLY2:%.+]] = affine.apply #[[$MAP6]]([[MIN2]], [[MIN5]], [[MIN6]])
    // CHECK:       [[CMP1:%.+]] = arith.cmpi eq, [[ARG1]], [[C0]] : index
    // CHECK:       [[CMP2:%.+]] = arith.cmpi eq, [[ARG3]], [[C0]] : index
    // CHECK:       [[IF15:%.+]] = scf.if [[CMP2]] -> (index) {
    // CHECK:         [[CMP_SGE9:%.+]] = arith.cmpi sge, [[MIN2]], [[C160]] : index
    // CHECK:         cf.assert [[CMP_SGE9]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:         scf.yield [[ARG3]] : index
    // CHECK:       } else {
    // CHECK:         [[ADD_NEXT:%.+]] = arith.addi [[ARG3]], [[C160]] : index
    // CHECK:         [[CMP_LT_NEXT:%.+]] = arith.cmpi slt, [[ADD_NEXT]], [[DIM_0]] : index
    // CHECK:         [[IF_NEXT:%.+]] = scf.if [[CMP_LT_NEXT]] -> (index) {
    // CHECK:           scf.yield [[ARG3]] : index
    // CHECK:         } else {
    // CHECK:           [[CMP_EQ_NEXT:%.+]] = arith.cmpi eq, [[ADD_NEXT]], [[DIM_0]] : index
    // CHECK:           [[IF_EQ_NEXT:%.+]] = scf.if [[CMP_EQ_NEXT]] -> (index) {
    // CHECK:             scf.yield [[ARG3]] : index
    // CHECK:           } else {
    // CHECK:             [[APPLY_NEXT:%.+]] = affine.apply #[[$MAP11]]([[ARG3]])[[[REM2]]]
    // CHECK:             scf.yield [[APPLY_NEXT]] : index
    // CHECK:           }
    // CHECK:           scf.yield [[IF_EQ_NEXT]] : index
    // CHECK:         }
    // CHECK:         scf.yield [[IF_NEXT]] : index
    // CHECK:       }
    // CHECK:       [[IF16:%.+]] = scf.if [[CMP1]] -> (index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[CMP7:%.+]] = arith.cmpi eq, [[MAX1]], [[C0]] : index
    // CHECK:       [[CMP8:%.+]] = arith.cmpi eq, [[MAX4]], [[C0]] : index
    // CHECK:       [[IF19:%.+]]:2 = scf.if [[CMP8]] -> (index, index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[IF22:%.+]]:2 = scf.if [[CMP7]] -> (index, index) {
    // CHECK:         {{.*}}
    // CHECK:       }
    // CHECK:       [[SHLI4:%.+]] = arith.shli [[IF22]]#0, [[C2]] : index
    // CHECK:       [[ORI4:%.+]] = arith.ori [[SHLI4]], [[IF19]]#0 : index
    // CHECK:       [[SWITCH2:%.+]] = scf.index_switch [[ORI4]] -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       case 0 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static([[SLICE2]]) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 1 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static([[SLICE2]]) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 2 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static([[SLICE2]]) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 4 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static([[SLICE2]]) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 5 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static([[SLICE2]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 6 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static([[SLICE2]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 8 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static([[SLICE2]]) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 9 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static([[SLICE2]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       case 10 {
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static([[SLICE2]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       default {
    // CHECK:         cf.assert [[FALSE]], "Unsupported case"
    // CHECK:         [[SLICE2:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IF22]]#1, [[IF19]]#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
    // CHECK:         [[CALL2:%.+]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static([[SLICE2]]) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:         scf.yield [[CALL2]] : tensor<1x256x256x160xf16, {order = #NHWC}>
    // CHECK:       }
    // CHECK:       [[INSERT2:%.+]] = tensor.insert_slice [[SWITCH2]] into [[ARG4]][0, 0, [[IF16]], [[IF15]]] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       scf.yield [[INSERT2]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:     }
    // CHECK:     scf.yield [[FOR3]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   }
    // CHECK:   return [[FOR1]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:   }
}
