//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --outline-codegen-capsules --canonicalize %s | FileCheck %s --check-prefix=TILED
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --outline-codegen-capsules="force-unroll=true" --canonicalize %s | FileCheck %s --check-prefix=UNROLLED
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (s0 + 3)>

// TILED: [[MAP:#.+]] = affine_map<()[s0] -> (s0 + 3)>
// TILED: module @OutlineTiledWithPaddedInputAndLayout
// TILED: module @VPU.SW
// TILED: func.func @generated_info_0([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index)
// TILED-SAME: -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index)
// TILED-DAG:  [[C1:%.+]] = arith.constant 1 : index
// TILED-DAG:  [[C0:%.+]] = arith.constant 0 : index
// TILED-DAG:  [[OFFSET_ADJ:%.+]] = affine.apply [[MAP]]()[%arg5]
// TILED-DAG:  return [[C1]], [[ARG0]], [[ARG1]], [[ARG2]], [[C0]], [[ARG3]], [[ARG4]], [[OFFSET_ADJ]], [[C1]], [[ARG0]], [[ARG1]], [[ARG2]], [[C0]], [[ARG3]], [[ARG4]], [[ARG5]]

// TILED: func.func @generated_0(
// TILED-SAME:    [[ARG0:%.+]]: tensor<1x?x?x?xi32>, [[ARG1:%.+]]: tensor<1x?x?x?xi32>,
// TILED-SAME:    [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index, [[ARG7:%.+]]: index)
// TILED-SAME: -> tensor<1x?x?x?xi32>
// TILED-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 2 : i64>
// TILED:      [[EMPTY:%.+]] = tensor.empty([[ARG2]], [[ARG3]], [[ARG4]]) : tensor<1x?x?x?xi32>
// TILED:      [[OP:%.+]] = linalg.generic
// TILED-SAME:     ins([[ARG0]], [[ARG1]] : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>)
// TILED-SAME:     outs([[EMPTY]] : tensor<1x?x?x?xi32>)
// TILED:      return [[OP]] : tensor<1x?x?x?xi32>

// UNROLLED: module @OutlineTiledWithPaddedInputAndLayout
// UNROLLED: module @VPU.SW
// UNROLLED: func.func @generated_0([[ARG0:%.+]]: tensor<1x28x29x10xi32>, [[ARG1:%.+]]: tensor<1x28x29x3xi32>) -> tensor<1x28x29x3xi32>
// UNROLLED-DAG:  [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 3] [1, 28, 29, 3] [1, 1, 1, 1] : tensor<1x28x29x10xi32> to tensor<1x28x29x3xi32>
// UNROLLED-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x28x29x3xi32>
// UNROLLED:      [[OP:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[SLICE]], [[ARG1]] : tensor<1x28x29x3xi32>, tensor<1x28x29x3xi32>)
// UNROLLED-SAME:     outs([[EMPTY]] : tensor<1x28x29x3xi32>)
// UNROLLED:      return [[OP]] : tensor<1x28x29x3xi32>

module @OutlineTiledWithPaddedInputAndLayout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x10x28x29xsi32, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x29xsi32, {order = #NHWC}>
  }
  // TILED: func.func @main([[ARG0:%.+]]: tensor<1x10x28x29xsi32, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x3x28x29xsi32, {order = #NHWC}>)
  // UNROLLED: func.func @main([[ARG0:%.+]]: tensor<1x10x28x29xsi32, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x3x28x29xsi32, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x10x28x29xsi32, {order = #NHWC}>, %arg1: tensor<1x3x28x29xsi32, {order = #NHWC}>) -> tensor<1x3x28x29xsi32, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x28x29x10xi32>, %arg1 as %arg3: tensor<1x28x29x3xi32>) {
      %c0 = arith.constant 0 : index
      %1 = tensor.empty() : tensor<1x28x29x3xi32>
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c0_0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %3 = scf.for %arg4 = %c0_0 to %2 step %c1 iter_args(%arg5 = %1) -> (tensor<1x28x29x3xi32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg4) -> sizes(index, index, index), offsets(index, index, index)
        %5 = affine.apply #map()[%offsets#2]
        %extracted_slice = tensor.extract_slice %arg2[0, %offsets#0, %offsets#1, %5] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x28x29x10xi32> to tensor<1x?x?x?xi32>
        %extracted_slice_1 = tensor.extract_slice %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x28x29x3xi32> to tensor<1x?x?x?xi32>
        %6 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xi32>
        %7 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_1 : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>) outs(%6 : tensor<1x?x?x?xi32>) {
        ^bb0(%in: i32, %in_2: i32, %out: i32):
          %8 = arith.divsi %in, %in_2 : i32
          linalg.yield %8 : i32
        } -> tensor<1x?x?x?xi32>
        %inserted_slice = tensor.insert_slice %7 into %arg5[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xi32> into tensor<1x28x29x3xi32>
        scf.yield %inserted_slice : tensor<1x28x29x3xi32>
      }
      IE.CGCYield %3 : tensor<1x28x29x3xi32>
    } -> tensor<1x3x28x29xsi32, {order = #NHWC}>
    return %0 : tensor<1x3x28x29xsi32, {order = #NHWC}>

// TILED:    [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 3, 0, 0] [1, 3, 28, 29] : tensor<1x10x28x29xsi32, {order = #NHWC}> to tensor<1x3x28x29xsi32, {order = #NHWC}>
// TILED:    [[OP:%.+]] = VPU.GenericSwLayer([[SLICE]], [[ARG1]] : tensor<1x3x28x29xsi32, {order = #NHWC}>, tensor<1x3x28x29xsi32, {order = #NHWC}>)
// TILED-SAME:      @VPU.SW::@generated_0
// TILED-SAME:      tiling(sizes = [28, 29, 3], offsets = [0, 0, 0])
// TILED-SAME:      -> tensor<1x3x28x29xsi32, {order = #NHWC}>
// TILED:    return [[OP]] : tensor<1x3x28x29xsi32, {order = #NHWC}>

// UNROLLED: [[OP:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]] : tensor<1x10x28x29xsi32, {order = #NHWC}>, tensor<1x3x28x29xsi32, {order = #NHWC}>)
// UNROLLED-SAME:      @VPU.SW::@generated_0
// UNROLLED-NOT:       tiling
// UNROLLED-SAME:      -> tensor<1x3x28x29xsi32, {order = #NHWC}>
// UNROLLED: return [[OP]] : tensor<1x3x28x29xsi32, {order = #NHWC}>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

// TILED: module @PaddedSoftmax
// TILED: module @VPU.SW
// TILED: func.func @generated_info_0(
// TILED-SAME: [[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index)
// TILED-SAME: -> (index, index, index, index, index, index, index, index) {
// TILED-DAG:      [[C1:%.+]] = arith.constant 1 : index
// TILED-DAG:      [[C4000:%.+]] = arith.constant 4000 : index
// TILED-DAG:      [[C0:%.+]] = arith.constant 0 : index
// TILED-NEXT:     return [[C1]], [[ARG0]], [[C4000]], [[ARG1]], [[C0]], [[ARG2]], [[C0]], [[ARG3]]

// TILED:    func.func @generated_0(
// TILED-SAME: [[ARG0:%.+]]: tensor<1x?x4000x?xf32>, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index) -> tensor<1x?x4000x?xf32>
// TILED-SAME: kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 3], numSlicedInputs = 1 : i64>}
// TILED-DAG:      [[CST:%.+]] = arith.constant -3.40282347E+38 : f32
// TILED-DAG:      [[C0:%.+]] = arith.constant 0.000000e+00 : f32
// TILED-DAG:      [[EMPTY_MAX:%.+]] = tensor.empty([[ARG1]], [[ARG2]]) : tensor<1x?x?xf32>
// TILED-DAG:      [[FILL_MAX:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY_MAX]] : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// TILED:          [[MAX:%.+]] = linalg.generic
// TILED-SAME:         ins([[ARG0]] : tensor<1x?x4000x?xf32>)
// TILED-SAME:         outs([[FILL_MAX]] : tensor<1x?x?xf32>)
// TILED:          [[EMPTY_SUBEXP:%.+]] = tensor.empty([[ARG1]], [[ARG2]]) : tensor<1x?x4000x?xf32>
// TILED:          [[SUBEXP:%.+]] = linalg.generic
// TILED-SAME:         ins([[ARG0]], [[MAX]] : tensor<1x?x4000x?xf32>, tensor<1x?x?xf32>)
// TILED-SAME:         outs([[EMPTY_SUBEXP]] : tensor<1x?x4000x?xf32>)
// TILED:          [[FILL_ADD:%.+]] = linalg.fill ins([[C0]] : f32) outs([[EMPTY_MAX]] : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// TILED:          [[REDUCE_ADD:%.+]] = linalg.generic
// TILED-SAME:         ins([[SUBEXP]] : tensor<1x?x4000x?xf32>)
// TILED-SAME:         outs([[FILL_ADD]] : tensor<1x?x?xf32>)
// TILED:          [[DIV:%.+]] = linalg.generic
// TILED-SAME:         ins([[SUBEXP]], [[REDUCE_ADD]] : tensor<1x?x4000x?xf32>, tensor<1x?x?xf32>)
// TILED-SAME:         outs([[EMPTY_SUBEXP]] : tensor<1x?x4000x?xf32>)
// TILED:          return [[DIV]] : tensor<1x?x4000x?xf32>

// UNROLLED: module @PaddedSoftmax
// UNROLLED: module @VPU.SW

// UNROLLED: func.func @generated_0([[ARG0:%.+]]: tensor<1x16x4010x200xf32>) -> tensor<1x16x4000x200xf32>
// UNROLLED-DAG:  [[CST:%.+]] = arith.constant -3.40282347E+38 : f32
// UNROLLED-DAG:  [[C0:%.+]] = arith.constant 0.000000e+00 : f32
// UNROLLED-DAG:  [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 0] [1, 16, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4010x200xf32> to tensor<1x16x4000x200xf32>
// UNROLLED-DAG:  [[MAX_EMPTY:%.+]] = tensor.empty() : tensor<1x16x200xf32>
// UNROLLED:      [[MAX_FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[MAX_EMPTY]] : tensor<1x16x200xf32>) -> tensor<1x16x200xf32>
// UNROLLED:      [[MAX:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[SLICE]] : tensor<1x16x4000x200xf32>)
// UNROLLED-SAME:     outs([[MAX_FILL]] : tensor<1x16x200xf32>)
// UNROLLED:      [[SUBEXP_EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// UNROLLED:      [[SUBEXP:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[SLICE]], [[MAX]] : tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>)
// UNROLLED-SAME:     outs([[SUBEXP_EMPTY]] : tensor<1x16x4000x200xf32>)
// UNROLLED:      [[REDUCE_ADD_FILL:%.+]] = linalg.fill ins([[C0]] : f32) outs([[MAX_EMPTY]] : tensor<1x16x200xf32>) -> tensor<1x16x200xf32>
// UNROLLED:      [[REDUCE_ADD:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[SUBEXP]] : tensor<1x16x4000x200xf32>) outs([[REDUCE_ADD_FILL]] : tensor<1x16x200xf32>)
// UNROLLED:      [[DIV:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[SUBEXP]], [[REDUCE_ADD]] : tensor<1x16x4000x200xf32>, tensor<1x16x200xf32>)
// UNROLLED-SAME:     outs([[SUBEXP_EMPTY]] : tensor<1x16x4000x200xf32>)
// UNROLLED:      return [[DIV]] : tensor<1x16x4000x200xf32>

module @PaddedSoftmax {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x200x16x4010xf32, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x4000x200xf32>
  }

// TILED: func.func @main([[ARG0:%.+]]: tensor<1x200x16x4010xf32, {order = #NHWC}>) -> tensor<1x16x4000x200xf32>
// UNROLLED:  func.func @main([[ARG0:%.+]]: tensor<1x200x16x4010xf32, {order = #NHWC}>) -> tensor<1x16x4000x200xf32> {
  func.func @main(%arg0: tensor<1x200x16x4010xf32, {order = #NHWC}>) -> tensor<1x16x4000x200xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4010x200xf32>) {
      %cst = arith.constant -3.40282347E+38 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %1 = tensor.empty() : tensor<1x16x4000x200xf32>
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %3 = scf.for %arg2 = %c0_1 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x16x4000x200xf32>) {
        %sizes:2, %offsets:2 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index), offsets(index, index)
        %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, 0, %offsets#1] [1, %sizes#0, 4000, %sizes#1] [1, 1, 1, 1] : tensor<1x16x4010x200xf32> to tensor<1x?x4000x?xf32>
        %5 = tensor.empty(%sizes#0, %sizes#1) : tensor<1x?x?xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
        %7 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%extracted_slice : tensor<1x?x4000x?xf32>) outs(%6 : tensor<1x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.maximumf %in, %out fastmath<nnan,nsz> : f32
          linalg.yield %13 : f32
        } -> tensor<1x?x?xf32>
        %8 = tensor.empty(%sizes#0, %sizes#1) : tensor<1x?x4000x?xf32>
        %9 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %7 : tensor<1x?x4000x?xf32>, tensor<1x?x?xf32>) outs(%8 : tensor<1x?x4000x?xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %13 = arith.subf %in, %in_2 : f32
          %14 = math.exp %13 fastmath<afn> : f32
          linalg.yield %14 : f32
        } -> tensor<1x?x4000x?xf32>
        %10 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
        %11 = linalg.generic {indexing_maps = [#NCHW, #map], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%9 : tensor<1x?x4000x?xf32>) outs(%10 : tensor<1x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.addf %out, %in fastmath<reassoc> : f32
          linalg.yield %13 : f32
        } -> tensor<1x?x?xf32>
        %12 = linalg.generic {indexing_maps = [#NCHW, #map, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %11 : tensor<1x?x4000x?xf32>, tensor<1x?x?xf32>) outs(%8 : tensor<1x?x4000x?xf32>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %13 = arith.divf %in, %in_2 fastmath<arcp> : f32
          linalg.yield %13 : f32
        } -> tensor<1x?x4000x?xf32>
        %inserted_slice = tensor.insert_slice %12 into %arg3[0, %offsets#0, 0, %offsets#1] [1, %sizes#0, 4000, %sizes#1] [1, 1, 1, 1] : tensor<1x?x4000x?xf32> into tensor<1x16x4000x200xf32>
        scf.yield %inserted_slice : tensor<1x16x4000x200xf32>
      }
      IE.CGCYield %3 : tensor<1x16x4000x200xf32>
    } -> tensor<1x16x4000x200xf32>
    return %0 : tensor<1x16x4000x200xf32>

// TILED:      [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 200, 16, 4000] : tensor<1x200x16x4010xf32, {order = #NHWC}> to tensor<1x200x16x4000xf32, {order = #NHWC}>
// TILED-NEXT: [[OP:%.+]] = VPU.GenericSwLayer([[SLICE]] : tensor<1x200x16x4000xf32, {order = #NHWC}>)
// TILED-SAME:     @VPU.SW::@generated_0
// TILED-SAME:     tiling(sizes = [16, 200], offsets = [0, 0])
// TILED-SAME:     -> tensor<1x16x4000x200xf32>
// TILED-NEXT: return [[OP]] : tensor<1x16x4000x200xf32>

// UNROLLED:    [[OP:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x200x16x4010xf32, {order = #NHWC}>)
// UNROLLED-SAME:  @VPU.SW::@generated_0
// UNROLLED-SAME:  -> tensor<1x16x4000x200xf32>
// UNROLLED:    return [[OP]] : tensor<1x16x4000x200xf32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0, 200)>
#map1 = affine_map<(d0, d1) -> (-d0 + 200, d1)>
#map2 = affine_map<(d0, d1) -> (d0 - d1)>

// TILED: [[MAP:#.+]] = affine_map<()[s0] -> (200, s0)>
// TILED: [[MAP1:#.+]] = affine_map<()[s0, s1] -> (-s0 + 200, s1)>
// TILED: [[MAP2:#.+]] = affine_map<()[s0, s1] -> (s0 - s1)>

// TILED: module @PaddedEltwise
// TILED: module @VPU.SW
// TILED: func.func @generated_info_0(
// TILED-SAME: [[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index)
// TILED-SAME: -> (index, index, index, index, index, index, index, index)
// TILED-DAG:  [[C1:%.+]] = arith.constant 1 : index
// TILED-DAG:  [[C0:%.+]] = arith.constant 0 : index
// TILED-DAG:  [[MIN0:%.+]] = affine.min [[MAP]](){{\[}}[[ARG5]]{{\]}}
// TILED-DAG:  [[MIN1:%.+]] = affine.min [[MAP1]](){{\[}}[[MIN0]], [[ARG2]]{{\]}}
// TILED:      return [[C1]], [[ARG0]], [[ARG1]], [[MIN1]], [[C0]], [[ARG3]], [[ARG4]], [[MIN0]] : index, index, index, index, index, index, index, index

// TILED:    func.func @generated_0(
// TILED-SAME:    [[ARG0:%.+]]: tensor<1x?x?x?xf32>, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index)
// TILED-SAME: -> tensor<1x?x?x?xf32>
// TILED-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64>
// TILED-DAG:   [[MIN0:%.+]] = affine.min [[MAP]](){{\[}}[[ARG6]]{{\]}}
// TILED-DAG:   [[MIN1:%.+]] = affine.min [[MAP1]](){{\[}}[[MIN0]], [[ARG3]]{{\]}}
// TILED-DAG:   [[PADSZ:%.+]] = affine.apply [[MAP2]](){{\[}}[[ARG3]], [[MIN1]]{{\]}}
// TILED-DAG:   [[EMPTY:%.+]] = tensor.empty([[ARG1]], [[ARG2]], [[MIN1]]) : tensor<1x?x?x?xf32>
// TILED:       [[OP:%.+]] = linalg.generic
// TILED-SAME:      ins([[ARG0]] : tensor<1x?x?x?xf32>) outs([[EMPTY]] : tensor<1x?x?x?xf32>)
// TILED:       [[PADDED:%.+]] = tensor.pad [[OP]] low[0, 0, 0, 0] high[0, 0, 0, [[PADSZ]]]
// TILED:       return [[PADDED]] : tensor<1x?x?x?xf32>

// UNROLLED: module @PaddedEltwise
// UNROLLED: module @VPU.SW

// UNROLLED:    func.func @generated_0([[ARG0:%.+]]: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
// UNROLLED-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// UNROLLED:      [[OP:%.+]] = linalg.generic
// UNROLLED-SAME:     ins([[ARG0]] : tensor<1x16x4000x200xf32>)
// UNROLLED-SAME:     outs([[EMPTY]] : tensor<1x16x4000x200xf32>)
// UNROLLED:      [[PADDED:%.+]] = tensor.pad [[OP]] low[0, 0, 0, 0] high[0, 0, 0, 56]
// UNROLLED:      return [[PADDED]] : tensor<1x16x4000x256xf32>

module @PaddedEltwise {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x4000x200xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x4000x256xf32>
  }

// TILED: func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32>
// UNROLLED: func.func @main([[ARG0:%.+]]: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32>
  func.func @main(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %1 = tensor.empty() : tensor<1x16x4000x256xf32>
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c0_0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_1 = arith.constant 0 : index
      %3 = scf.for %arg2 = %c0_0 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x16x4000x256xf32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index, index), offsets(index, index, index)
        %5 = affine.min #map(%offsets#2)
        %6 = affine.min #map1(%5, %sizes#2)
        %7 = arith.cmpi eq, %6, %c0_1 : index
        %8 = affine.apply #map2(%sizes#2, %6)
        %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, %offsets#1, %5] [1, %sizes#0, %sizes#1, %6] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
        %10 = tensor.empty(%sizes#0, %sizes#1, %6) : tensor<1x?x?x?xf32>
        %11 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%10 : tensor<1x?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %12 = math.exp %in fastmath<afn> : f32
          linalg.yield %12 : f32
        } -> tensor<1x?x?x?xf32>
        %padded = tensor.pad %11 low[0, 0, 0, 0] high[0, 0, 0, %8] {
        ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
          tensor.yield %cst : f32
        } : tensor<1x?x?x?xf32> to tensor<1x?x?x?xf32>
        %inserted_slice = tensor.insert_slice %padded into %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x256xf32>
        scf.yield %inserted_slice : tensor<1x16x4000x256xf32>
      }
      IE.CGCYield %3 : tensor<1x16x4000x256xf32>
    } -> tensor<1x16x4000x256xf32>
    return %0 : tensor<1x16x4000x256xf32>

// TILED:  [[OP:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>)
// TILED-SAME:     @VPU.SW::@generated_0
// TILED-SAME:     tiling(sizes = [16, 4000, 256], offsets = [0, 0, 0])
// TILED-SAME:     -> tensor<1x16x4000x256xf32>
// TILED:  return [[OP]] : tensor<1x16x4000x256xf32>

// UNROLLED: [[OP:%.+]] = VPU.GenericSwLayer([[ARG0]] : tensor<1x16x4000x200xf32>)
// UNROLLED-SAME:   @VPU.SW::@generated_0
// UNROLLED-SAME:   -> tensor<1x16x4000x256xf32>
// UNROLLED: return [[OP]] : tensor<1x16x4000x256xf32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (s0 + 3)>

// Requires a dynamic slice of the original tensor so must fall back
// through unrolling.

// TILED: module @DynamicSlice
// TILED-NOT: func.func @generated_info_0
// TILED: func.func @generated_0(
// TILED-SAME: [[ARG0:%.+]]: tensor<1x28x?x10xi32>, [[ARG1:%.+]]: tensor<1x28x?x3xi32>) -> tensor<1x28x?x3xi32>
// TILED-DAG:  [[C2:%.+]] = arith.constant 2 : index
// TILED-DAG:  [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x28x?x10xi32>
// TILED-DAG:  [[EMPTY1:%.+]] = tensor.empty([[DIM]]) : tensor<1x28x?x3xi32>
// TILED-DAG:  [[SLICE1:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 3] [1, 28, [[DIM]], 3] [1, 1, 1, 1] : tensor<1x28x?x10xi32> to tensor<1x28x?x3xi32>
// TILED-DAG:  [[SLICE2:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 28, [[DIM]], 3] [1, 1, 1, 1] : tensor<1x28x?x3xi32> to tensor<1x28x?x3xi32>
// TILED:      [[EMPTY2:%.+]] = tensor.empty([[DIM]]) : tensor<1x28x?x3xi32>
// TILED:      [[OP:%.+]] = linalg.generic
// TILED-SAME:     ins([[SLICE1]], [[SLICE2]] : tensor<1x28x?x3xi32>, tensor<1x28x?x3xi32>)
// TILED-SAME:     outs([[EMPTY2]] : tensor<1x28x?x3xi32>)
// TILED:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[OP]] into [[EMPTY1]][0, 0, 0, 0] [1, 28, [[DIM]], 3] [1, 1, 1, 1] : tensor<1x28x?x3xi32> into tensor<1x28x?x3xi32>
// TILED:      return [[INSERT_SLICE]] : tensor<1x28x?x3xi32>

module @DynamicSlice {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x10x28x?xsi32, {order = #NHWC}>
    DataInfo "input1" : tensor<1x3x28x?xsi32, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x28x?xsi32, {order = #NHWC}>
  }
// TILED:  func.func @main(
// TILED-SAME: [[ARG0:%.+]]: tensor<1x10x28x?xsi32, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x3x28x?xsi32, {order = #NHWC}>)
  func.func @main(%arg0: tensor<1x10x28x?xsi32, {order = #NHWC}>, %arg1: tensor<1x3x28x?xsi32, {order = #NHWC}>) -> tensor<1x3x28x?xsi32, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x28x?x10xi32>, %arg1 as %arg3: tensor<1x28x?x3xi32>) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %dim = tensor.dim %arg2, %c2 : tensor<1x28x?x10xi32>
      %1 = tensor.empty(%dim) : tensor<1x28x?x3xi32>
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c0_0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %3 = scf.for %arg4 = %c0_0 to %2 step %c1 iter_args(%arg5 = %1) -> (tensor<1x28x?x3xi32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg4) -> sizes(index, index, index), offsets(index, index, index)
        %5 = affine.apply #map()[%offsets#2]
        %extracted_slice = tensor.extract_slice %arg2[0, %offsets#0, %offsets#1, %5] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x28x?x10xi32> to tensor<1x?x?x?xi32>
        %extracted_slice_1 = tensor.extract_slice %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x28x?x3xi32> to tensor<1x?x?x?xi32>
        %6 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xi32>
        %7 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_1 : tensor<1x?x?x?xi32>, tensor<1x?x?x?xi32>) outs(%6 : tensor<1x?x?x?xi32>) {
        ^bb0(%in: i32, %in_2: i32, %out: i32):
          %8 = arith.divsi %in, %in_2 : i32
          linalg.yield %8 : i32
        } -> tensor<1x?x?x?xi32>
        %inserted_slice = tensor.insert_slice %7 into %arg5[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xi32> into tensor<1x28x?x3xi32>
        scf.yield %inserted_slice : tensor<1x28x?x3xi32>
      }
      IE.CGCYield %3 : tensor<1x28x?x3xi32>
    } -> tensor<1x3x28x?xsi32, {order = #NHWC}>
    return %0 : tensor<1x3x28x?xsi32, {order = #NHWC}>

// TILED: [[OP:%.+]] = VPU.GenericSwLayer([[ARG0]], [[ARG1]] : tensor<1x10x28x?xsi32, {order = #NHWC}>, tensor<1x3x28x?xsi32, {order = #NHWC}>)
// TILED-SAME:   @VPU.SW::@generated_0
// TILED-SAME: -> tensor<1x3x28x?xsi32, {order = #NHWC}>
// TILED: return [[OP]] : tensor<1x3x28x?xsi32, {order = #NHWC}>
  }
}

// -----

// TILED-DAG: [[QT:!.+]] = !quant.uniform<u8:f16:1, {1.{{.*}},2.{{.*}},3.{{.*}},4.{{.*}}}>
// TILED-DAG: [[QT1:!.+]] = !quant.uniform<u8:f16:1, {2.{{.*}},3.{{.*}},4.{{.*}}}>
// TILED-DAG: [[MAP:#.+]] = affine_map<()[s0] -> (s0 + 1)>
// TILED-DAG: [[MAP1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// TILED: module @PaddedInputDequantize
// TILED: module @VPU.SW
// TILED: func.func @generated_info_0(
// TILED-SAME: [[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index)
// TILED-DAG:  [[C1:%.+]] = arith.constant 1 : index
// TILED-DAG:  [[C0:%.+]] = arith.constant 0 : index
// TILED-DAG:  [[INC:%.+]] = affine.apply [[MAP]]()[%arg5]
// TILED:      return [[C1]], [[ARG0]], [[ARG1]], [[ARG2]], [[C0]], [[ARG3]], [[ARG4]], [[INC]] : index, index, index, index, index, index, index, index

// TILED: func.func @generated_0(
// TILED-SAME: [[ARG0:%.+]]: tensor<1x?x?x?xi8>, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index, [[ARG6:%.+]]: index)
// TILED-SAME: -> tensor<1x?x?x?xf16>
// TILED-SAME:  kernelInfo = #VPU.KernelInfo<tilingInfoFunc = @generated_info_0, tilingAxes = [1, 2, 3], numSlicedInputs = 1 : i64>
// TILED-DAG:   [[CI:%.+]] = arith.constant dense<0> : tensor<3xi8>
// TILED-DAG:   [[CF:%.+]] = arith.constant dense<[{{.*}}]> : tensor<3xf16>
// TILED-DAG:   [[SLICE_F:%.+]] = tensor.extract_slice [[CF]]{{\[}}[[ARG6]]{{\]}} {{\[}}[[ARG3]]{{\]}} [1] : tensor<3xf16> to tensor<?xf16>
// TILED-DAG:   [[SLICE_I:%.+]] = tensor.extract_slice [[CI]]{{\[}}[[ARG6]]{{\]}} {{\[}}[[ARG3]]{{\]}} [1] : tensor<3xi8> to tensor<?xi8>
// TILED-DAG:   [[EMPTY:%.+]] = tensor.empty([[ARG1]], [[ARG2]], [[ARG3]]) : tensor<1x?x?x?xf16>
// TILED:       [[OP:%.+]] = linalg.generic
// TILED-SAME:      ins(%arg0, [[SLICE_F]], [[SLICE_I]] : tensor<1x?x?x?xi8>, tensor<?xf16>, tensor<?xi8>)
// TILED-SAME:      outs([[EMPTY]] : tensor<1x?x?x?xf16>)
// TILED:       return [[OP:%.+]] : tensor<1x?x?x?xf16>

// UNROLLED: [[QT:!.+]] = !quant.uniform<u8:f16:1, {1.{{.*}},2.{{.*}},3.{{.*}},4.{{.*}}}>
// UNROLLED: module @PaddedInputDequantize
// UNROLLED: module @VPU.SW
// UNROLLED: func.func @generated_0([[ARG0:%.+]]: tensor<1x8x32x4xi8>) -> tensor<1x8x32x3xf16>
// UNROLLED-DAG:   [[CSTI:%.+]] = arith.constant dense<0> : tensor<3xi8>
// UNROLLED-DAG:   [[CSTF:%.+]] = arith.constant dense<[1.999510e-01, 3.000490e-01, 3.999020e-01]> : tensor<3xf16>
// UNROLLED-DAG:   [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, 1] [1, 8, 32, 3] [1, 1, 1, 1] : tensor<1x8x32x4xi8> to tensor<1x8x32x3xi8>
// UNROLLED-DAG:   [[EMPTY:%.+]] = tensor.empty() : tensor<1x8x32x3xf16>
// UNROLLED:       [[OP:%.+]] = linalg.generic
// UNROLLED-SAME:      ins([[SLICE]], [[CSTF]], [[CSTI]] : tensor<1x8x32x3xi8>, tensor<3xf16>, tensor<3xi8>)
// UNROLLED-SAME:      outs([[EMPTY]] : tensor<1x8x32x3xf16>)
// UNROLLED:       return [[OP]] : tensor<1x8x32x3xf16>

!qElemType = !quant.uniform<u8:f16:1, {0.1, 0.2, 0.3, 0.4}>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>

module @PaddedInputDequantize {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x4x8x32xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x8x32xf16, {order = #NHWC}>
  }

// TILED:  func.func @main([[ARG0:%.+]]: tensor<1x4x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
// UNROLLED:  func.func @main([[ARG0:%.+]]: tensor<1x4x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
  func.func @main(%arg0: tensor<1x4x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x4x8x32xf16, {order = #NHWC}> -> tensor<1x4x8x32x!qElemType, {order = #NHWC}>
    %1 = IE.CodeGenCapsule inputs(%0 as %arg1: tensor<1x8x32x4xi8>) {
      %cst = arith.constant dense<0> : tensor<3xi8>
      %cst_0 = arith.constant dense<[0.2, 0.3, 0.4]> : tensor<3xf16>
      %c0 = arith.constant 0 : index
      %2 = tensor.empty() : tensor<1x8x32x3xf16>
      %3 = Shave.LoopTripCount(%c0) : index -> index
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %4 = scf.for %arg2 = %c0_1 to %3 step %c1 iter_args(%arg3 = %2) -> (tensor<1x8x32x3xf16>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index, index), offsets(index, index, index)
        %6 = affine.apply #map()[%offsets#2]
        %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, %offsets#1, %6] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x8x32x4xi8> to tensor<1x?x?x?xi8>
        %extracted_slice_2 = tensor.extract_slice %cst_0[%offsets#2] [%sizes#2] [1] : tensor<3xf16> to tensor<?xf16>
        %extracted_slice_3 = tensor.extract_slice %cst[%offsets#2] [%sizes#2] [1] : tensor<3xi8> to tensor<?xi8>
        %7 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf16>
        %8 = linalg.generic {indexing_maps = [#NCHW, #map1, #map1, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_2, %extracted_slice_3 : tensor<1x?x?x?xi8>, tensor<?xf16>, tensor<?xi8>) outs(%7 : tensor<1x?x?x?xf16>) {
        ^bb0(%in: i8, %in_4: f16, %in_5: i8, %out: f16):
          %9 = arith.uitofp %in : i8 to f16
          %10 = arith.uitofp %in_5 : i8 to f16
          %11 = arith.subf %9, %10 : f16
          %12 = arith.mulf %11, %in_4 : f16
          linalg.yield %12 : f16
        } -> tensor<1x?x?x?xf16>
        %inserted_slice = tensor.insert_slice %8 into %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf16> into tensor<1x8x32x3xf16>
        scf.yield %inserted_slice : tensor<1x8x32x3xf16>
      }
      IE.CGCYield %4 : tensor<1x8x32x3xf16>
    } -> tensor<1x3x8x32xf16, {order = #NHWC}>
    return %1 : tensor<1x3x8x32xf16, {order = #NHWC}>

// TILED:       [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QT]]} : tensor<1x4x8x32xf16, {order = #NHWC}> -> tensor<1x4x8x32x[[QT]], {order = #NHWC}>
// TILED-NEXT:  [[SLICE:%.+]] = IE.Slice [[QUANT]] [0, 1, 0, 0] [1, 3, 8, 32] : tensor<1x4x8x32x[[QT]], {order = #NHWC}> to tensor<1x3x8x32x[[QT1]], {order = #NHWC}>
// TILED-NEXT:  [[OP:%.+]] = VPU.GenericSwLayer([[SLICE]] : tensor<1x3x8x32x[[QT1]], {order = #NHWC}>)
// TILED-SAME:      @VPU.SW::@generated_0
// TILED-SAME:      tiling(sizes = [8, 32, 3], offsets = [0, 0, 0])
// TILED-SAME:  -> tensor<1x3x8x32xf16, {order = #NHWC}>
// TILED-NEXT:  return [[OP]] : tensor<1x3x8x32xf16, {order = #NHWC}>

// UNROLLED:    [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = [[QT]]} : tensor<1x4x8x32xf16, {order = #NHWC}> -> tensor<1x4x8x32x[[QT]], {order = #NHWC}>
// UNROLLED:    [[OP:%.+]] = VPU.GenericSwLayer([[QUANT]] : tensor<1x4x8x32x[[QT]], {order = #NHWC}>) @VPU.SW::@generated_0 -> tensor<1x3x8x32xf16, {order = #NHWC}>
// UNROLLED:    return [[OP]] : tensor<1x3x8x32xf16, {order = #NHWC}>
  }
}
