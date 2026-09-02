//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --wrap-in-kernel-region %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SimpleEltwise(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<1x16x4000x200xf32>
    %2 = Shave.LoopTripCount(%c0) : index -> index
    %3 = scf.for %arg2 = %c0 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x16x4000x200xf32>) {
      %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index, index), offsets(index, index, index)
      %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
      %4 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%4 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in fastmath<afn> : f32
        linalg.yield %6 : f32
      } -> tensor<1x?x?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
      scf.yield %inserted_slice : tensor<1x16x4000x200xf32>
    }
    IE.CGCYield %3 : tensor<1x16x4000x200xf32>
  } -> tensor<1x16x4000x200xf32>
  return %0 : tensor<1x16x4000x200xf32>

// CHECK: func.func @SimpleEltwise(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x16x4000x200xf32>) {
// CHECK:     [[KERNEL:%.+]] = Shave.KernelRegion([[ARG1]] : tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
// CHECK:     ^bb0([[ARG2:%.+]]: tensor<1x16x4000x200xf32>):
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG3:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG4:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x200xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG3]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG2]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG4]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x200xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR]] : tensor<1x16x4000x200xf32>
// CHECK:     IE.CGCYield [[KERNEL]] : tensor<1x16x4000x200xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
func.func @LargeReduction(%arg0: tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x100000000xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<1x1x1x1xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%arg1 : tensor<1x1x1x100000000xf32>) outs(%2 : tensor<1x1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %out, %in fastmath<reassoc> : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x1x1xf32>
    IE.CGCYield %3 : tensor<1x1x1x1xf32>
  } -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>

// CHECK: func.func @LargeReduction(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>) {
// CHECK:     [[KERNEL:%.+]] = Shave.KernelRegion([[ARG1]] : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:     ^bb0([[ARG2:%.+]]: tensor<1x1x1x100000000xf32>):
// CHECK-DAG:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK:         [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:         [[GEN:%.+]] = linalg.generic
// CHECK-SAME:        ins([[ARG2]] : tensor<1x1x1x100000000xf32>)
// CHECK-SAME:        outs([[FILL]] : tensor<1x1x1x1xf32>)
// CHECK:         Shave.KernelRegion.Yield [[GEN]] : tensor<1x1x1x1xf32>
// CHECK-NOT:     Shave.KernelRegion
// CHECK:     IE.CGCYield [[KERNEL]] : tensor<1x1x1x1xf32>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>

func.func @ThreePartGraph(%arg0: tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x100000000xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x100000000xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = tensor.empty() : tensor<1x1x1x100000000xf32>
    %2 = Shave.LoopTripCount(%c1) : index -> index
    %3 = scf.for %arg2 = %c0 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x1x1x100000000xf32>) {
      %sizes, %offsets = Shave.OutSliceInfo(%c1, %arg2) -> sizes(index), offsets(index)
      %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
      %10 = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
      %11 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x1x1x?xf32>) outs(%10 : tensor<1x1x1x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = math.exp %in fastmath<afn> : f32
        linalg.yield %12 : f32
      } -> tensor<1x1x1x?xf32>
      %inserted_slice = tensor.insert_slice %11 into %arg3[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
      scf.yield %inserted_slice : tensor<1x1x1x100000000xf32>
    }
    %4 = tensor.empty() : tensor<1x1x1x1xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} ins(%3 : tensor<1x1x1x100000000xf32>) outs(%5 : tensor<1x1x1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.addf %out, %in fastmath<reassoc> : f32
      linalg.yield %10 : f32
    } -> tensor<1x1x1x1xf32>
    %7 = tensor.empty() : tensor<1x1x1x100000000xf32>
    %8 = Shave.LoopTripCount(%c0) : index -> index
    %9 = scf.for %arg2 = %c0 to %8 step %c1 iter_args(%arg3 = %7) -> (tensor<1x1x1x100000000xf32>) {
      %sizes, %offsets = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index), offsets(index)
      %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
      %10 = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
      %11 = linalg.generic {indexing_maps = [#NCHW, #map2, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %6 : tensor<1x1x1x?xf32>, tensor<1x1x1x1xf32>) outs(%10 : tensor<1x1x1x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %12 = arith.addf %in, %in_0 : f32
        linalg.yield %12 : f32
      } -> tensor<1x1x1x?xf32>
      %inserted_slice = tensor.insert_slice %11 into %arg3[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
      scf.yield %inserted_slice : tensor<1x1x1x100000000xf32>
    }
    IE.CGCYield %9 : tensor<1x1x1x100000000xf32>
  } -> tensor<1x1x1x100000000xf32>
  return %0 : tensor<1x1x1x100000000xf32>

// CHECK: func.func @ThreePartGraph(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x1x1x100000000xf32>) {
// CHECK:     [[KERNEL1:%.+]] = Shave.KernelRegion([[ARG1]] : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x100000000xf32> {
// CHECK:     ^bb0([[K1_ARG:%.+]]: tensor<1x1x1x100000000xf32>):
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP1:%.+]] = Shave.LoopTripCount([[C1]]) : index -> index
// CHECK-DAG:     [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK:         [[FOR1:%.+]] = scf.for [[IT1:%.+]] = [[C0]] to [[TRIP1]] step [[C1]] iter_args([[ACC1:%.+]] = [[EMPTY1]]) -> (tensor<1x1x1x100000000xf32>) {
// CHECK:             [[SIZES1:%.+]], [[OFFSETS1:%.+]] = Shave.OutSliceInfo([[C1]], [[IT1]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT1:%.+]] = tensor.extract_slice [[K1_ARG]][0, 0, 0, [[OFFSETS1]]] [1, 1, 1, [[SIZES1]]] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN1:%.+]] = tensor.empty([[SIZES1]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN1:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT1]] : tensor<1x1x1x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN1]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS1:%.+]] = tensor.insert_slice [[GEN1]] into [[ACC1]][0, 0, 0, [[OFFSETS1]]] [1, 1, 1, [[SIZES1]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
// CHECK:             scf.yield [[INS1]] : tensor<1x1x1x100000000xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR1]] : tensor<1x1x1x100000000xf32>
// CHECK:     [[KERNEL2:%.+]] = Shave.KernelRegion([[KERNEL1]] : tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:     ^bb0([[K2_ARG:%.+]]: tensor<1x1x1x100000000xf32>):
// CHECK-DAG:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK:         [[FILL2:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY2]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK:         [[GEN2:%.+]] = linalg.generic
// CHECK-SAME:        ins([[K2_ARG]] : tensor<1x1x1x100000000xf32>)
// CHECK-SAME:        outs([[FILL2]] : tensor<1x1x1x1xf32>)
// CHECK:         Shave.KernelRegion.Yield [[GEN2]] : tensor<1x1x1x1xf32>
// CHECK:     [[KERNEL3:%.+]] = Shave.KernelRegion([[ARG1]], [[KERNEL2]] : tensor<1x1x1x100000000xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x100000000xf32> {
// CHECK:     ^bb0([[K3_ARG0:%.+]]: tensor<1x1x1x100000000xf32>, [[K3_ARG1:%.+]]: tensor<1x1x1x1xf32>):
// CHECK-DAG:     [[C0_3:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP3:%.+]] = Shave.LoopTripCount([[C0_3]]) : index -> index
// CHECK-DAG:     [[C1_3:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK:         [[FOR3:%.+]] = scf.for [[IT3:%.+]] = [[C0_3]] to [[TRIP3]] step [[C1_3]] iter_args([[ACC3:%.+]] = [[EMPTY3]]) -> (tensor<1x1x1x100000000xf32>) {
// CHECK:             [[SIZES3:%.+]], [[OFFSETS3:%.+]] = Shave.OutSliceInfo([[C0_3]], [[IT3]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT3:%.+]] = tensor.extract_slice [[K3_ARG0]][0, 0, 0, [[OFFSETS3]]] [1, 1, 1, [[SIZES3]]] [1, 1, 1, 1] : tensor<1x1x1x100000000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN3:%.+]] = tensor.empty([[SIZES3]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN3:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT3]], [[K3_ARG1]] : tensor<1x1x1x?xf32>, tensor<1x1x1x1xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN3]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS3:%.+]] = tensor.insert_slice [[GEN3]] into [[ACC3]][0, 0, 0, [[OFFSETS3]]] [1, 1, 1, [[SIZES3]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x100000000xf32>
// CHECK:             scf.yield [[INS3]] : tensor<1x1x1x100000000xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR3]] : tensor<1x1x1x100000000xf32>
// CHECK:     IE.CGCYield [[KERNEL3]] : tensor<1x1x1x100000000xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SimpleEltwiseDynamic(%arg0: tensor<1x16x4000x?xf32>) -> tensor<1x16x4000x?xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x?xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %d = tensor.dim %arg1, %c3 : tensor<1x16x4000x?xf32>
    %1 = tensor.empty(%d) : tensor<1x16x4000x?xf32>
    %2 = Shave.LoopTripCount(%c0) : index -> index
    %3 = scf.for %arg2 = %c0 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x16x4000x?xf32>) {
      %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index, index), offsets(index, index, index)
      %min = affine.min affine_map<(d0, d1) -> (d0, d1)>(%sizes#2, %d)
      %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %min] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
      %4 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%4 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in fastmath<afn> : f32
        linalg.yield %6 : f32
      } -> tensor<1x?x?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
      scf.yield %inserted_slice : tensor<1x16x4000x?xf32>
    }
    IE.CGCYield %3 : tensor<1x16x4000x?xf32>
  } -> tensor<1x16x4000x?xf32>
  return %0 : tensor<1x16x4000x?xf32>

// CHECK: func.func @SimpleEltwiseDynamic(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x16x4000x?xf32>) {
// CHECK:     [[KERNEL:%.+]] = Shave.KernelRegion([[ARG1]] : tensor<1x16x4000x?xf32>) -> tensor<1x16x4000x?xf32> {
// CHECK:     ^bb0([[ARG2:%.+]]: tensor<1x16x4000x?xf32>):
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:     [[DIM:%.+]] = tensor.dim [[ARG2]], [[C3]] : tensor<1x16x4000x?xf32>
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x4000x?xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG3:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG4:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x?xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG3]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[MIN:%.+]] = affine.min {{.+}}([[SIZES]]#2, [[DIM]])
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG2]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG4]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x?xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR]] : tensor<1x16x4000x?xf32>
// CHECK:     IE.CGCYield [[KERNEL]] : tensor<1x16x4000x?xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SimpleEltwiseDynamicExternalScalar(%arg0: tensor<1x16x4000x?xf32>, %external: tensor<1x16x4000x?xf32>) -> tensor<1x16x4000x?xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x?xf32>, %external as %external0: tensor<1x16x4000x?xf32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %d = tensor.dim %external0, %c3 : tensor<1x16x4000x?xf32>
    %d_plus_1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%d)
    %1 = tensor.empty(%d_plus_1) : tensor<1x16x4000x?xf32>
    %2 = Shave.LoopTripCount(%c0) : index -> index
    %3 = scf.for %arg2 = %c0 to %2 step %c1 iter_args(%arg3 = %1) -> (tensor<1x16x4000x?xf32>) {
      %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %arg2) -> sizes(index, index, index), offsets(index, index, index)
      %min = affine.min affine_map<(d0, d1) -> (d0, d1)>(%sizes#2, %d_plus_1)
      %extracted_slice = tensor.extract_slice %arg1[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %min] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
      %4 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%4 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in fastmath<afn> : f32
        linalg.yield %6 : f32
      } -> tensor<1x?x?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg3[0, %offsets#0, %offsets#1, %offsets#2] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
      scf.yield %inserted_slice : tensor<1x16x4000x?xf32>
    }
    IE.CGCYield %3 : tensor<1x16x4000x?xf32>
  } -> tensor<1x16x4000x?xf32>
  return %0 : tensor<1x16x4000x?xf32>

// CHECK: func.func @SimpleEltwiseDynamicExternalScalar(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG2:%.+]]: tensor<1x16x4000x?xf32>, {{.*}} as [[ARG3:%.+]]: tensor<1x16x4000x?xf32>) {
// CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG: [[DIM:%.+]] = tensor.dim [[ARG3]], [[C3]] : tensor<1x16x4000x?xf32>
// CHECK:     [[KERNEL:%.+]] = Shave.KernelRegion([[ARG2]], [[DIM]] : tensor<1x16x4000x?xf32>, index) -> tensor<1x16x4000x?xf32> {
// CHECK:     ^bb0([[ARG4:%.+]]: tensor<1x16x4000x?xf32>, [[ARG5:%.+]]: index):
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[D_PLUS_1:%.+]] = affine.apply {{.+}}([[ARG5]])
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty([[D_PLUS_1]]) : tensor<1x16x4000x?xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[ARG6:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ARG7:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x?xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[ARG6]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[MIN:%.+]] = affine.min {{.+}}([[SIZES]]#2, [[D_PLUS_1]])
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[ARG4]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x4000x?xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ARG7]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x?xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x?xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR]] : tensor<1x16x4000x?xf32>
// CHECK:     IE.CGCYield [[KERNEL]] : tensor<1x16x4000x?xf32>
}

// -----

// Verify that tensor block arguments are listed before scalar ones in
// the KernelRegionOp signature.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @TensorArgBeforeScalarArg(%arg0: tensor<1x16x4000x200xf32>, %arg1: index) -> tensor<1x16x4000x200xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %t: tensor<1x16x4000x200xf32>, %arg1 as %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %trip = Shave.LoopTripCount(%c0) : index -> index
    %empty = tensor.empty() : tensor<1x16x4000x200xf32>
    %result = scf.for %iv = %c0 to %trip step %c1 iter_args(%acc = %empty) -> (tensor<1x16x4000x200xf32>) {
      %sizes:3, %offsets:3 = Shave.OutSliceInfo(%c0, %iv) -> sizes(index, index, index), offsets(index, index, index)
      %shifted_offset = arith.addi %offsets#2, %n : index
      %extracted_slice = tensor.extract_slice %t[0, %offsets#0, %offsets#1, %shifted_offset] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
      %4 = tensor.empty(%sizes#0, %sizes#1, %sizes#2) : tensor<1x?x?x?xf32>
      %5 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x?x?x?xf32>) outs(%4 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = math.exp %in fastmath<afn> : f32
        linalg.yield %6 : f32
      } -> tensor<1x?x?x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %acc[0, %offsets#0, %offsets#1, %shifted_offset] [1, %sizes#0, %sizes#1, %sizes#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
      scf.yield %inserted_slice : tensor<1x16x4000x200xf32>
    }
    IE.CGCYield %result : tensor<1x16x4000x200xf32>
  } -> tensor<1x16x4000x200xf32>
  return %0 : tensor<1x16x4000x200xf32>

// CHECK: func.func @TensorArgBeforeScalarArg(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[T:%.+]]: tensor<1x16x4000x200xf32>, {{.*}} as [[N:%.+]]: index) {
// CHECK:     [[KERNEL:%.+]] = Shave.KernelRegion([[T]], [[N]] : tensor<1x16x4000x200xf32>, index) -> tensor<1x16x4000x200xf32> {
// CHECK:     ^bb0([[BB_T:%.+]]: tensor<1x16x4000x200xf32>, [[BB_N:%.+]]: index):
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK:         [[FOR:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[TRIP]] step [[C1]] iter_args([[ACC:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x200xf32>) {
// CHECK:             [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo([[C0]], [[IV]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:             [[SHIFTED:%.+]] = arith.addi [[OFFSETS]]#2, [[BB_N]] : index
// CHECK:             [[EXT:%.+]] = tensor.extract_slice [[BB_T]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[SHIFTED]]] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x?x?x?xf32>
// CHECK:             [[EMPTY_DYN:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
// CHECK:             [[GEN:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN]] : tensor<1x?x?x?xf32>)
// CHECK:             [[INS:%.+]] = tensor.insert_slice [[GEN]] into [[ACC]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[SHIFTED]]] [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1] : tensor<1x?x?x?xf32> into tensor<1x16x4000x200xf32>
// CHECK:             scf.yield [[INS]] : tensor<1x16x4000x200xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR]] : tensor<1x16x4000x200xf32>
// CHECK:     IE.CGCYield [[KERNEL]] : tensor<1x16x4000x200xf32>
}

// -----

// Diamond-shaped graph: node 1 feeds both node 2 (tiled scf.for) and
// node 3; node 4 consumes both node 2 and node 3 outputs.
// Verifies that node 1 is not pulled into node 3/4's subgraph because it also
// has a user in a different subgraph (the KernelRegion wrapping node 2).
// Note that merging 1 with 3 and 4 would additionally cause a cyclical dependency
// with node 2.
//
//      1
//    /   \
//   2     3
//    \   /
//      4

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @DiamondGraph(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %buf = tensor.empty() : tensor<1x1x1x1000xf32>
    %out1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x1x1000xf32>) outs(%buf : tensor<1x1x1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %e = math.exp %in fastmath<afn> : f32
      linalg.yield %e : f32
    } -> tensor<1x1x1x1000xf32>

    %trip2 = Shave.LoopTripCount(%c0) : index -> index
    %out2 = scf.for %iv = %c0 to %trip2 step %c1 iter_args(%acc = %buf) -> (tensor<1x1x1x1000xf32>) {
      %sizes, %offsets = Shave.OutSliceInfo(%c0, %iv) -> sizes(index), offsets(index)
      %slice_in = tensor.extract_slice %out1[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x1000xf32> to tensor<1x1x1x?xf32>
      %slice_buf = tensor.empty(%sizes) : tensor<1x1x1x?xf32>
      %slice_out = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%slice_in : tensor<1x1x1x?xf32>) outs(%slice_buf : tensor<1x1x1x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %n = arith.negf %in : f32
        linalg.yield %n : f32
      } -> tensor<1x1x1x?xf32>
      %acc_new = tensor.insert_slice %slice_out into %acc[0, 0, 0, %offsets] [1, 1, 1, %sizes] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x1000xf32>
      scf.yield %acc_new : tensor<1x1x1x1000xf32>
    }

    %out3 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%out1 : tensor<1x1x1x1000xf32>) outs(%buf : tensor<1x1x1x1000xf32>) {
    ^bb0(%in: f32, %out: f32):
      %a = math.absf %in : f32
      linalg.yield %a : f32
    } -> tensor<1x1x1x1000xf32>

    %out4 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%out2, %out3 : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) outs(%buf : tensor<1x1x1x1000xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %s = arith.addf %in0, %in1 : f32
      linalg.yield %s : f32
    } -> tensor<1x1x1x1000xf32>

    IE.CGCYield %out4 : tensor<1x1x1x1000xf32>
  } -> tensor<1x1x1x1000xf32>
  return %0 : tensor<1x1x1x1000xf32>

// CHECK: func.func @DiamondGraph(
// CHECK: [[CAPSULE:%.+]] = IE.CodeGenCapsule inputs({{.*}} as [[ARG1:%.+]]: tensor<1x1x1x1000xf32>) {
// First KernelRegion: exp (node 1)
// CHECK:     [[KERNEL1:%.+]] = Shave.KernelRegion([[ARG1]] : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
// CHECK:     ^bb0([[K1_ARG:%.+]]: tensor<1x1x1x1000xf32>):
// CHECK-DAG:     [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[GEN1:%.+]] = linalg.generic
// CHECK-SAME:        ins([[K1_ARG]] : tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY1]] : tensor<1x1x1x1000xf32>)
// CHECK:         Shave.KernelRegion.Yield [[GEN1]] : tensor<1x1x1x1000xf32>
// CHECK:     [[KERNEL2:%.+]] = Shave.KernelRegion([[KERNEL1]] : tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
// CHECK:     ^bb0([[K2_ARG:%.+]]: tensor<1x1x1x1000xf32>):
// CHECK-DAG:     [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[TRIP2:%.+]] = Shave.LoopTripCount([[C0]]) : index -> index
// CHECK-DAG:     [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:     [[EMPTY2:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[FOR2:%.+]] = scf.for [[IT2:%.+]] = [[C0]] to [[TRIP2]] step [[C1]] iter_args([[ACC2:%.+]] = [[EMPTY2]]) -> (tensor<1x1x1x1000xf32>) {
// CHECK:             [[SIZES2:%.+]], [[OFFSETS2:%.+]] = Shave.OutSliceInfo([[C0]], [[IT2]]) -> sizes(index), offsets(index)
// CHECK:             [[EXT2:%.+]] = tensor.extract_slice [[K2_ARG]][0, 0, 0, [[OFFSETS2]]] [1, 1, 1, [[SIZES2]]] [1, 1, 1, 1] : tensor<1x1x1x1000xf32> to tensor<1x1x1x?xf32>
// CHECK:             [[EMPTY_DYN2:%.+]] = tensor.empty([[SIZES2]]) : tensor<1x1x1x?xf32>
// CHECK:             [[GEN2:%.+]] = linalg.generic
// CHECK-SAME:            ins([[EXT2]] : tensor<1x1x1x?xf32>)
// CHECK-SAME:            outs([[EMPTY_DYN2]] : tensor<1x1x1x?xf32>)
// CHECK:             [[INS2:%.+]] = tensor.insert_slice [[GEN2]] into [[ACC2]][0, 0, 0, [[OFFSETS2]]] [1, 1, 1, [[SIZES2]]] [1, 1, 1, 1] : tensor<1x1x1x?xf32> into tensor<1x1x1x1000xf32>
// CHECK:             scf.yield [[INS2]] : tensor<1x1x1x1000xf32>
// CHECK:         Shave.KernelRegion.Yield [[FOR2]] : tensor<1x1x1x1000xf32>
// CHECK:     [[KERNEL3:%.+]] = Shave.KernelRegion([[KERNEL1]], [[KERNEL2]] : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
// CHECK:     ^bb0([[K3_ARG0:%.+]]: tensor<1x1x1x1000xf32>, [[K3_ARG1:%.+]]: tensor<1x1x1x1000xf32>):
// CHECK-DAG:     [[EMPTY3:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:         [[GEN3A:%.+]] = linalg.generic
// CHECK-SAME:        ins([[K3_ARG0]] : tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY3]] : tensor<1x1x1x1000xf32>)
// CHECK:         [[GEN3B:%.+]] = linalg.generic
// CHECK-SAME:        ins([[K3_ARG1]], [[GEN3A]] : tensor<1x1x1x1000xf32>, tensor<1x1x1x1000xf32>)
// CHECK-SAME:        outs([[EMPTY3]] : tensor<1x1x1x1000xf32>)
// CHECK:         Shave.KernelRegion.Yield [[GEN3B]] : tensor<1x1x1x1000xf32>
// CHECK:     IE.CGCYield [[KERNEL3]] : tensor<1x1x1x1000xf32>
}
