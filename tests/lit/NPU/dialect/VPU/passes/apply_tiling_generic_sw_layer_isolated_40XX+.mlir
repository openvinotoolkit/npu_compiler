//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
}
func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
    >
}

// CHECK-LABEL: func.func @IsolatedGenericSwLayerNHWCLargeC
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
func.func @IsolatedGenericSwLayerNHWCLargeC(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
  %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x640x1x960xf16, {order = #NHWC}>
  return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>

// CHECK:         [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 640, 1, 480] : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[ARG1]] : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [480, 640], offsets = [0, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[SLICE1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 480] [1, 640, 1, 480] : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE1]], [[ARG1]] : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [480, 640], offsets = [480, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 480]]} : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x480xf16, {order = #NHWC}> -> tensor<1x640x1x960xf16, {order = #NHWC}>
// CHECK:         return [[CONCAT]] : tensor<1x640x1x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
}
func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
    >
}

// CHECK-LABEL: func.func @IsolatedGenericSwLayerNHWCLargeH
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>)
func.func @IsolatedGenericSwLayerNHWCLargeH(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x960x1x1xf16, {order = #NHWC}>) -> tensor<1x960x1x640xf16, {order = #NHWC}> {
  %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 960], offsets = [0, 0]) {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x960x1x640xf16, {order = #NHWC}>
  return %0 : tensor<1x960x1x640xf16, {order = #NHWC}>

// CHECK:         [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 480, 1, 640] : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE1:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 480, 1, 1] : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[SLICE1]] : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 480], offsets = [0, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x480x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE2:%.+]] = VPU.Slice [[ARG0]] [0, 480, 0, 0] [1, 480, 1, 640] : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE3:%.+]] = VPU.Slice [[ARG1]] [0, 480, 0, 0] [1, 480, 1, 1] : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE2]], [[SLICE3]] : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 480], offsets = [0, 480]) {tiling_loop_index = 0 : i64} -> tensor<1x480x1x640xf16, {order = #NHWC}>
// CHECK:         [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 480, 0, 0]]} : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x640xf16, {order = #NHWC}> -> tensor<1x960x1x640xf16, {order = #NHWC}>
// CHECK:         return [[CONCAT]] : tensor<1x960x1x640xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (960, s0)>
#map1 = affine_map<()[s0, s1] -> (-s0 + 960, s1)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = affine.min #map()[%arg3]
  %1 = affine.min #map1()[%0, %arg1]
  return %c1, %c1, %arg0, %1, %c0, %c0, %arg2, %0, %c1, %c1, %c1, %1, %c0, %c0, %c0, %0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
}
func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
    >
}

// CHECK-LABEL: func.func @Padded
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>)
func.func @Padded(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x960x1x1xf16, {order = #NHWC}>) -> tensor<1x1000x1x640xf16, {order = #NHWC}> {
  %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 1000], offsets = [0, 0]) {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x1000x1x640xf16, {order = #NHWC}>
  return %0 : tensor<1x1000x1x640xf16, {order = #NHWC}>

// CHECK:         [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 512, 1, 640] : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x512x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE1:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 512, 1, 1] : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[SLICE1]] : tensor<1x512x1x640xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 512], offsets = [0, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x512x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE2:%.+]] = VPU.Slice [[ARG0]] [0, 512, 0, 0] [1, 448, 1, 640] : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x448x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE3:%.+]] = VPU.Slice [[ARG1]] [0, 512, 0, 0] [1, 448, 1, 1] : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x448x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE2]], [[SLICE3]] : tensor<1x448x1x640xf16, {order = #NHWC}>, tensor<1x448x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 488], offsets = [0, 512]) {tiling_loop_index = 0 : i64} -> tensor<1x488x1x640xf16, {order = #NHWC}>
// CHECK:         [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 512, 0, 0]]} : tensor<1x512x1x640xf16, {order = #NHWC}>, tensor<1x488x1x640xf16, {order = #NHWC}> -> tensor<1x1000x1x640xf16, {order = #NHWC}>
// CHECK:         return [[CONCAT]] : tensor<1x1000x1x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (960, s0)>
#map1 = affine_map<()[s0, s1] -> (-s0 + 960, s1)>

func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = affine.min #map()[%arg3]
  %1 = affine.min #map1()[%0, %arg1]
  return %c1, %c1, %arg0, %1, %c0, %c0, %arg2, %0, %c1, %c1, %c1, %1, %c0, %c0, %c0, %0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
}
func.func nested @generated_0(tensor<1x1x?x?xf16>, tensor<1x1x1x?xf16>, index, index, index, index) -> tensor<1x1x?x?xf16> attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
    >
}

// CHECK-LABEL: func.func @PaddedNonZeroOffsets
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x448x1x640xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x448x1x1xf16, {order = #NHWC}>)
func.func @PaddedNonZeroOffsets(%arg0: tensor<1x448x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x448x1x1xf16, {order = #NHWC}>) -> tensor<1x488x1x640xf16, {order = #NHWC}> {
   %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x448x1x640xf16, {order = #NHWC}>, tensor<1x448x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 488], offsets = [0, 512]) {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x488x1x640xf16, {order = #NHWC}>
   return %0 : tensor<1x488x1x640xf16, {order = #NHWC}>

// CHECK:         [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 256, 1, 640] : tensor<1x448x1x640xf16, {order = #NHWC}> to tensor<1x256x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE1:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 256, 1, 1] : tensor<1x448x1x1xf16, {order = #NHWC}> to tensor<1x256x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[SLICE1]] : tensor<1x256x1x640xf16, {order = #NHWC}>, tensor<1x256x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 256], offsets = [0, 512]) {tiling_loop_index = 0 : i64} -> tensor<1x256x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE2:%.+]] = VPU.Slice [[ARG0]] [0, 256, 0, 0] [1, 192, 1, 640] : tensor<1x448x1x640xf16, {order = #NHWC}> to tensor<1x192x1x640xf16, {order = #NHWC}>
// CHECK:         [[SLICE3:%.+]] = VPU.Slice [[ARG1]] [0, 256, 0, 0] [1, 192, 1, 1] : tensor<1x448x1x1xf16, {order = #NHWC}> to tensor<1x192x1x1xf16, {order = #NHWC}>
// CHECK:         [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE2]], [[SLICE3]] : tensor<1x192x1x640xf16, {order = #NHWC}>, tensor<1x192x1x1xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [640, 232], offsets = [0, 768]) {tiling_loop_index = 0 : i64} -> tensor<1x232x1x640xf16, {order = #NHWC}>
// CHECK:         [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x640xf16, {order = #NHWC}>, tensor<1x232x1x640xf16, {order = #NHWC}> -> tensor<1x488x1x640xf16, {order = #NHWC}>
// CHECK:         return [[CONCAT]] : tensor<1x488x1x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  return %c1, %c1, %arg0, %arg1,
          %c0, %c0, %arg2, %arg3,
        %c1, %c1, %c1, %arg1,
          %c0, %c0, %c0, %arg3,
        %c1, %c1, %arg0, %arg1,
          %c0, %c0, %arg2, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
}

func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
    kernelInfo = #VPU.KernelInfo<
        tilingInfoFunc = @generated_info_0,
        tilingAxes = [2, 3],
        numSlicedInputs = 2 : i64
    >
}

// CHECK-LABEL: func.func @IsolatedGenericSwLayerNHWCLargeCWithScratch
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
func.func @IsolatedGenericSwLayerNHWCLargeCWithScratch(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
  %scratch = VPU.Empty : tensor<1x640x1x960xf16, {order = #NHWC}>
  %0 = VPU.GenericSwLayer(%arg0, %arg1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>)
         scratch(%scratch : tensor<1x640x1x960xf16, {order = #NHWC}>)
         @generated_0
         tiling(sizes = [960, 640], offsets = [0, 0]) {
          tilingStrategy = [1, 1, 1, 2]
        } -> tensor<1x640x1x960xf16, {order = #NHWC}>
  return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>

// CHECK:         [[SCRATCH0:%.+]] = VPU.Empty : tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[SCRATCH1:%.+]] = VPU.Empty : tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 640, 1, 480] : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[ARG1]] : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH1]] : tensor<1x640x1x480xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [480, 640], offsets = [0, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[SLICE1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 480] [1, 640, 1, 480] : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE1]], [[ARG1]] : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH0]] : tensor<1x640x1x480xf16, {order = #NHWC}>) @generated_0 tiling(sizes = [480, 640], offsets = [480, 0]) {tiling_loop_index = 0 : i64} -> tensor<1x640x1x480xf16, {order = #NHWC}>
// CHECK:         [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 480]]} : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x480xf16, {order = #NHWC}> -> tensor<1x640x1x960xf16, {order = #NHWC}>
// CHECK:         return [[CONCAT]] : tensor<1x640x1x960xf16, {order = #NHWC}>
}
