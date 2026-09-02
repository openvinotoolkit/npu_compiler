//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --split-codegen-capsules %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SimpleEltwise(%arg0: tensor<1x200x16x4000xf32, {order = #NHWC}>) -> tensor<1x200x16x4000xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32>) {
    %1 = Shave.KernelRegion(%arg1 : tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    ^bb0(%arg2: tensor<1x16x4000x200xf32>):
      %c0 = arith.constant 0 : index
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c1 = arith.constant 1 : index
      %3 = tensor.empty() : tensor<1x16x4000x200xf32>
      %4 = scf.for %arg3 = %c0 to %2 step %c1 iter_args(%arg4 = %3) -> (tensor<1x16x4000x200xf32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg3) -> sizes(index, index, index), offsets(index, index, index)
        %extracted_slice = tensor.extract_slice %arg2[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
        %5 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
        %6 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%5 : tensor<1x?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = math.exp %in fastmath<afn> : f32
          linalg.yield %7 : f32
        } -> tensor<1x?x?x?xf32>
        %inserted_slice = tensor.insert_slice %6 into %arg4[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
        scf.yield %inserted_slice : tensor<1x16x4000x200xf32>
      }
      Shave.KernelRegion.Yield %4 : tensor<1x16x4000x200xf32>
    }
    IE.CGCYield %1 : tensor<1x16x4000x200xf32>
  } -> tensor<1x200x16x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x200x16x4000xf32, {order = #NHWC}>

// CHECK: func.func @SimpleEltwise([[ARG0:%.+]]: tensor<1x200x16x4000xf32, {order = #NHWC}>) -> tensor<1x200x16x4000xf32, {order = #NHWC}> {
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x16x4000x200xf32>) {
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG3:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x200xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG2]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG1]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG3]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x200xf32>
// CHECK:     IE.CGCYield [[FOR]] : tensor<1x16x4000x200xf32>
// CHECK: } -> tensor<1x200x16x4000xf32, {order = #NHWC}>
// CHECK: return [[CAPSULE]] : tensor<1x200x16x4000xf32, {order = #NHWC}>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @LargeReduction(%arg0: tensor<1x100000000x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x100000000xf32>) {
    %1 = Shave.KernelRegion(%arg1 : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
    ^bb0(%arg2: tensor<1x1x1x100000000xf32>):
      %cst = arith.constant 0.000000e+00 : f32
      %2 = tensor.empty() : tensor<1x1x1x1xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%arg2 : tensor<1x1x1x100000000xf32>) outs(%3 : tensor<1x1x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %out, %in fastmath<reassoc> : f32
        linalg.yield %5 : f32
      } -> tensor<1x1x1x1xf32>
      Shave.KernelRegion.Yield %4 : tensor<1x1x1x1xf32>
    }
    IE.CGCYield %1 : tensor<1x1x1x1xf32>
  } -> tensor<1x1x1x1xf32, {order = #NHWC}>
  return %0 : tensor<1x1x1x1xf32, {order = #NHWC}>

// CHECK: func.func @LargeReduction([[ARG0:%.+]]: tensor<1x100000000x1x1xf32, {order = #NHWC}>) -> tensor<1x1x1x1xf32, {order = #NHWC}> {
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>) {
// CHECK-DAG:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK:         [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:         [[GEN:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG1]] : tensor<1x1x1x100000000xf32>)
// CHECK-SAME:        outs([[FILL]] : tensor<1x1x1x1xf32>)
// CHECK:     IE.CGCYield [[GEN]] : tensor<1x1x1x1xf32>
// CHECK: } -> tensor<1x1x1x1xf32, {order = #NHWC}>
// CHECK: return [[CAPSULE]] : tensor<1x1x1x1xf32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>

func.func @ThreePartGraph(%arg0: tensor<1x100000000x1x1xf32, {order = #NHWC}>) -> tensor<1x100000000x1x1xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x100000000xf32>) {
    %1 = Shave.KernelRegion(%arg1 : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x100000000xf32> {
    ^bb0(%arg2: tensor<1x1x1x100000000xf32>):
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %4 = Shave.LoopTripCount(%c1) : index -> index
      %5 = tensor.empty() : tensor<1x1x1x100000000xf32>
      %6 = scf.for %arg3 = %c0 to %4 step %c1 iter_args(%arg4 = %5) -> (tensor<1x1x1x100000000xf32>) {
        %sizes, %offsets = Shave.OutSliceInfo(%c1, %arg3) -> sizes(index), offsets(index)
        %extracted_slice = tensor.extract_slice %arg2[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
        %7 = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
        %8 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x1x1x?xf32>) outs(%7 : tensor<1x1x1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = math.exp %in fastmath<afn> : f32
          linalg.yield %9 : f32
        } -> tensor<1x1x1x?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg4[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
        scf.yield %inserted_slice : tensor<1x1x1x100000000xf32>
      }
      Shave.KernelRegion.Yield %6 : tensor<1x1x1x100000000xf32>
    }
    %2 = Shave.KernelRegion(%1 : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
    ^bb0(%arg2: tensor<1x1x1x100000000xf32>):
      %cst = arith.constant 0.000000e+00 : f32
      %4 = tensor.empty() : tensor<1x1x1x1xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
      %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%arg2 : tensor<1x1x1x100000000xf32>) outs(%5 : tensor<1x1x1x1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %out, %in fastmath<reassoc> : f32
        linalg.yield %7 : f32
      } -> tensor<1x1x1x1xf32>
      Shave.KernelRegion.Yield %6 : tensor<1x1x1x1xf32>
    }
    %3 = Shave.KernelRegion(%arg1, %2 : tensor<1x1x1x100000000xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x100000000xf32> {
    ^bb0(%arg2: tensor<1x1x1x100000000xf32>, %arg3: tensor<1x1x1x1xf32>):
      %c0 = arith.constant 0 : index
      %4 = Shave.LoopTripCount(%c0) : index -> index
      %c1 = arith.constant 1 : index
      %5 = tensor.empty() : tensor<1x1x1x100000000xf32>
      %6 = scf.for %arg4 = %c0 to %4 step %c1 iter_args(%arg5 = %5) -> (tensor<1x1x1x100000000xf32>) {
        %sizes, %offsets = Shave.OutSliceInfo(%c0, %arg4) -> sizes(index), offsets(index)
        %extracted_slice = tensor.extract_slice %arg2[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
        %7 = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
        %8 = linalg.generic {indexing_maps = [#NCHW, #map2, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %arg3 : tensor<1x1x1x?xf32>, tensor<1x1x1x1xf32>) outs(%7 : tensor<1x1x1x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %9 = arith.addf %in, %in_0 : f32
          linalg.yield %9 : f32
        } -> tensor<1x1x1x?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg5[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
        scf.yield %inserted_slice : tensor<1x1x1x100000000xf32>
      }
      Shave.KernelRegion.Yield %6 : tensor<1x1x1x100000000xf32>
    }
    IE.CGCYield %3 : tensor<1x1x1x100000000xf32>
  } -> tensor<1x100000000x1x1xf32, {order = #NHWC}>
  return %0 : tensor<1x100000000x1x1xf32, {order = #NHWC}>

// CHECK: func.func @ThreePartGraph([[ARG0:%.+]]: tensor<1x100000000x1x1xf32, {order = #NHWC}>) -> tensor<1x100000000x1x1xf32, {order = #NHWC}> {
// CHECK: [[C1_EXP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>) {
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP1:%.+]] = Shave.LoopTripCount([[C1]]) : index -> index
// CHECK-DAG:     [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK:         [[FOR1:%.+]] = scf.for [[IT1:%.+]] = [[C0]] to [[TRIP1]] step [[C1]] iter_args([[ACC1:%.+]] = [[EMPTY1]]) -> (tensor<1x1x1x100000000xf32>) {
// CHECK:             [[SIZES1:%.+]], [[OFFSETS1:%.+]] = Shave.OutSliceInfo([[C1]], [[IT1]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSETS1]]] [1, 1, 1, [[SIZES1]]] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN1:%.+]] = tensor.empty([[SIZES1]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN1:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT1]] : tensor<1x1x1x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN1]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS1:%.+]] = tensor.insert_slice [[GEN1]] into [[ACC1]][0, 0, 0, [[OFFSETS1]]] [1, 1, 1, [[SIZES1]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
// CHECK:             scf.yield [[INS1]] : tensor<1x1x1x100000000xf32>
// CHECK:     IE.CGCYield [[FOR1]] : tensor<1x1x1x100000000xf32>
// CHECK: } -> tensor<1x1x1x100000000xf32>
// CHECK: [[C2_RED:%.+]] = IE.CodeGenCapsule inputs([[C1_EXP]] as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>) {
// CHECK-DAG:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK:         [[FILL2:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY2]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:         [[GEN2:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG1]] : tensor<1x1x1x100000000xf32>)
// CHECK-SAME:        outs([[FILL2]] : tensor<1x1x1x1xf32>)
// CHECK:     IE.CGCYield [[GEN2]] : tensor<1x1x1x1xf32>
// CHECK: } -> tensor<1x1x1x1xf32>
// CHECK: [[C3_OUT:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>, [[C2_RED]] as [[ARG2:%.+]]: tensor<1x1x1x1xf32>) {
// CHECK-DAG:     [[C0_3:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP3:%.+]] = Shave.LoopTripCount([[C0_3]]) : index -> index
// CHECK-DAG:     [[C1_3:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK:         [[FOR3:%.+]] = scf.for [[IT3:%.+]] = [[C0_3]] to [[TRIP3]] step [[C1_3]] iter_args([[ACC3:%.+]] = [[EMPTY3]]) -> (tensor<1x1x1x100000000xf32>) {
// CHECK:             [[SIZES3:%.+]], [[OFFSETS3:%.+]] = Shave.OutSliceInfo([[C0_3]], [[IT3]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT3:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSETS3]]] [1, 1, 1, [[SIZES3]]] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN3:%.+]] = tensor.empty([[SIZES3]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN3:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT3]], [[ARG2]] : tensor<1x1x1x?xf32>, tensor<1x1x1x1xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN3]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS3:%.+]] = tensor.insert_slice [[GEN3]] into [[ACC3]][0, 0, 0, [[OFFSETS3]]] [1, 1, 1, [[SIZES3]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
// CHECK:             scf.yield [[INS3]] : tensor<1x1x1x100000000xf32>
// CHECK:     IE.CGCYield [[FOR3]] : tensor<1x1x1x100000000xf32>
// CHECK: } -> tensor<1x100000000x1x1xf32, {order = #NHWC}>
// CHECK: return [[C3_OUT]] : tensor<1x100000000x1x1xf32, {order = #NHWC}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SimpleEltwiseDynamic(%arg0: tensor<1x?x16x4000xf32, {order = #NHWC}>) -> tensor<1x?x16x4000xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x?xf32>) {
    %1 = Shave.KernelRegion(%arg1 : tensor<1x16x4000x?xf32>) -> tensor<1x16x4000x?xf32> {
    ^bb0(%arg2: tensor<1x16x4000x?xf32>):
      %c0 = arith.constant 0 : index
      %2 = Shave.LoopTripCount(%c0) : index -> index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %dim = tensor.dim %arg2, %c3 : tensor<1x16x4000x?xf32>
      %3 = tensor.empty(%dim) : tensor<1x16x4000x?xf32>
      %4 = scf.for %arg3 = %c0 to %2 step %c1 iter_args(%arg4 = %3) -> (tensor<1x16x4000x?xf32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg3) -> sizes(index, index, index), offsets(index, index, index)
        %5 = affine.min #NC(%sizes#2, %dim)
        %extracted_slice = tensor.extract_slice %arg2[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %5] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
        %6 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
        %7 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%6 : tensor<1x?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %8 = math.exp %in fastmath<afn> : f32
          linalg.yield %8 : f32
        } -> tensor<1x?x?x?xf32>
        %inserted_slice = tensor.insert_slice %7 into %arg4[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
        scf.yield %inserted_slice : tensor<1x16x4000x?xf32>
      }
      Shave.KernelRegion.Yield %4 : tensor<1x16x4000x?xf32>
    }
    IE.CGCYield %1 : tensor<1x16x4000x?xf32>
  } -> tensor<1x?x16x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x?x16x4000xf32, {order = #NHWC}>

// CHECK: func.func @SimpleEltwiseDynamic([[ARG0:%.+]]: tensor<1x?x16x4000xf32, {order = #NHWC}>) -> tensor<1x?x16x4000xf32, {order = #NHWC}> {
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x16x4000x?xf32>) {
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:     [[DIM:%.+]] = tensor.dim [[ARG1]], [[C3]] : tensor<1x16x4000x?xf32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x4000x?xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG3:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x?xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG2]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[MIN:%.+]] = affine.min {{.+}}([[SIZES]]#2, [[DIM]])
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG1]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG3]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x?xf32>
// CHECK:     IE.CGCYield [[FOR]] : tensor<1x16x4000x?xf32>
// CHECK: } -> tensor<1x?x16x4000xf32, {order = #NHWC}>
// CHECK: return [[CAPSULE]] : tensor<1x?x16x4000xf32, {order = #NHWC}>
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 + 1)>

func.func @SimpleEltwiseDynamicExternalScalar(%arg0: tensor<1x?x16x4000xf32, {order = #NHWC}>, %arg1: tensor<1x?x16x4000xf32, {order = #NHWC}>) -> tensor<1x?x16x4000xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x16x4000x?xf32>, %arg1 as %arg3: tensor<1x16x4000x?xf32>) {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg3, %c3 : tensor<1x16x4000x?xf32>
    %1 = Shave.KernelRegion(%arg2, %dim : tensor<1x16x4000x?xf32>, index) -> tensor<1x16x4000x?xf32> {
    ^bb0(%arg4: tensor<1x16x4000x?xf32>, %arg5: index):
      %c0 = arith.constant 0 : index
      %2 = affine.apply #map(%arg5)
      %3 = Shave.LoopTripCount(%c0) : index -> index
      %c1 = arith.constant 1 : index
      %4 = tensor.empty(%2) : tensor<1x16x4000x?xf32>
      %5 = scf.for %arg6 = %c0 to %3 step %c1 iter_args(%arg7 = %4) -> (tensor<1x16x4000x?xf32>) {
        %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg6) -> sizes(index, index, index), offsets(index, index, index)
        %6 = affine.min #NC(%sizes#2, %2)
        %extracted_slice = tensor.extract_slice %arg4[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %6] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
        %7 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
        %8 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%7 : tensor<1x?x?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = math.exp %in fastmath<afn> : f32
          linalg.yield %9 : f32
        } -> tensor<1x?x?x?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg7[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
        scf.yield %inserted_slice : tensor<1x16x4000x?xf32>
      }
      Shave.KernelRegion.Yield %5 : tensor<1x16x4000x?xf32>
    }
    IE.CGCYield %1 : tensor<1x16x4000x?xf32>
  } -> tensor<1x?x16x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x?x16x4000xf32, {order = #NHWC}>

// CHECK: func.func @SimpleEltwiseDynamicExternalScalar([[ARG0:%.+]]: tensor<1x?x16x4000xf32, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x?x16x4000xf32, {order = #NHWC}>) -> tensor<1x?x16x4000xf32, {order = #NHWC}> {
// The dim is extracted from the NHWC outer tensor at dim 1
// CHECK-DAG: [[CDIM:%.+]] = arith.constant 1 : index
// CHECK-DAG: [[DIM:%.+]] = tensor.dim [[ARG1]], [[CDIM]] : tensor<1x?x16x4000xf32, {order = #NHWC}>
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG2:%.+]]: tensor<1x16x4000x?xf32>, [[DIM]] as [[ARG3:%.+]]: index) {
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[D_PLUS_1:%.+]] = affine.apply {{.+}}([[ARG3]])
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty([[D_PLUS_1]]) : tensor<1x16x4000x?xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG4:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG5:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x?xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG4]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[MIN:%.+]] = affine.min {{.+}}([[SIZES]]#2, [[D_PLUS_1]])
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG2]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG5]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x?xf32>
// CHECK:     IE.CGCYield [[FOR]] : tensor<1x16x4000x?xf32>
// CHECK: } -> tensor<1x?x16x4000xf32, {order = #NHWC}>
// CHECK: return [[CAPSULE]] : tensor<1x?x16x4000xf32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DiamondGraph(%arg0: tensor<1x1000x1x1xf32, {order = #NHWC}>) -> tensor<1x1000x1x1xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
    %1 = Shave.KernelRegion(%arg1 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    ^bb0(%arg2: tensor<1x1x1x1000xf32>):
      %4 = tensor.empty() : tensor<1x1x1x1000xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x1x1x1000xf32>) outs(%4 : tensor<1x1x1x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in fastmath<afn> : f32
        linalg.yield %6 : f32
      } -> tensor<1x1x1x1000xf32>
      Shave.KernelRegion.Yield %5 : tensor<1x1x1x1000xf32>
    }
    %2 = Shave.KernelRegion(%1 : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    ^bb0(%arg2: tensor<1x1x1x1000xf32>):
      %c0 = arith.constant 0 : index
      %4 = Shave.LoopTripCount(%c0) : index -> index
      %c1 = arith.constant 1 : index
      %5 = tensor.empty() : tensor<1x1x1x1000xf32>
      %6 = scf.for %arg3 = %c0 to %4 step %c1 iter_args(%arg4 = %5) -> (tensor<1x1x1x1000xf32>) {
        %sizes, %offsets = Shave.OutSliceInfo(%c0, %arg3) -> sizes(index), offsets(index)
        %extracted_slice = tensor.extract_slice %arg2[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x1000xf32> to tensor<1x1x1x?xf32>
        %7 = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
        %8 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x1x1x?xf32>) outs(%7 : tensor<1x1x1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.negf %in : f32
          linalg.yield %9 : f32
        } -> tensor<1x1x1x?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg4[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x1000xf32>
        scf.yield %inserted_slice : tensor<1x1x1x1000xf32>
      }
      Shave.KernelRegion.Yield %6 : tensor<1x1x1x1000xf32>
    }
    %3 = Shave.KernelRegion(%1, %2 : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    ^bb0(%arg2: tensor<1x1x1x1000xf32>, %arg3: tensor<1x1x1x1000xf32>):
      %4 = tensor.empty() : tensor<1x1x1x1000xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x1x1x1000xf32>) outs(%4 : tensor<1x1x1x1000xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.absf %in : f32
        linalg.yield %7 : f32
      } -> tensor<1x1x1x1000xf32>
      %6 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %5 : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) outs(%4 : tensor<1x1x1x1000xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %7 = arith.addf %in, %in_0 : f32
        linalg.yield %7 : f32
      } -> tensor<1x1x1x1000xf32>
      Shave.KernelRegion.Yield %6 : tensor<1x1x1x1000xf32>
    }
    IE.CGCYield %3 : tensor<1x1x1x1000xf32>
  } -> tensor<1x1000x1x1xf32, {order = #NHWC}>
  return %0 : tensor<1x1000x1x1xf32, {order = #NHWC}>

// CHECK: func.func @DiamondGraph([[ARG0:%.+]]: tensor<1x1000x1x1xf32, {order = #NHWC}>) -> tensor<1x1000x1x1xf32, {order = #NHWC}> {
// CHECK: [[C1_EXP:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>) {
// CHECK-DAG:     [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[GEN1:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG1]] : tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY1]] : tensor<1x1x1x1000xf32>)
// CHECK:     IE.CGCYield [[GEN1]] : tensor<1x1x1x1000xf32>
// CHECK: } -> tensor<1x1x1x1000xf32>
// CHECK: [[C2_NEG:%.+]] = IE.CodeGenCapsule inputs([[C1_EXP]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>) {
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP2:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[FOR2:%.+]] = scf.for [[IT2:%.+]] = [[C0]] to [[TRIP2]] step [[C1]] iter_args([[ACC2:%.+]] = [[EMPTY2]]) -> (tensor<1x1x1x1000xf32>) {
// CHECK:             [[SIZES2:%.+]], [[OFFSETS2:%.+]] = Shave.OutSliceInfo([[C0]], [[IT2]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT2:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSETS2]]] [1, 1, 1, [[SIZES2]]] [1, 1, 1, 1] : tensor<1x1x1x1000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN2:%.+]] = tensor.empty([[SIZES2]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN2:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT2]] : tensor<1x1x1x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN2]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS2:%.+]] = tensor.insert_slice [[GEN2]] into [[ACC2]][0, 0, 0, [[OFFSETS2]]] [1, 1, 1, [[SIZES2]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x1000xf32>
// CHECK:             scf.yield [[INS2]] : tensor<1x1x1x1000xf32>
// CHECK:     IE.CGCYield [[FOR2]] : tensor<1x1x1x1000xf32>
// CHECK: } -> tensor<1x1x1x1000xf32>
// CHECK: [[C3_OUT:%.+]] = IE.CodeGenCapsule inputs([[C1_EXP]] as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>, [[C2_NEG]] as [[ARG2:%.+]]: tensor<1x1x1x1000xf32>) {
// CHECK-DAG:     [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[GEN3A:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG1]] : tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY3]] : tensor<1x1x1x1000xf32>)
// CHECK:         [[GEN3B:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG2]], [[GEN3A]] : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY3]] : tensor<1x1x1x1000xf32>)
// CHECK:     IE.CGCYield [[GEN3B]] : tensor<1x1x1x1000xf32>
// CHECK: } -> tensor<1x1000x1x1xf32, {order = #NHWC}>
// CHECK: return [[C3_OUT]] : tensor<1x1000x1x1xf32, {order = #NHWC}>
}
