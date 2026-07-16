//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --scg-parametric-capsule-tiling %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @SimpleEltwise
module @SimpleEltwise {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x4000x200xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x4000x200xf32>
  }

// CHECK:  func.func @main([[ARG:%.+]]: tensor<1x16x4000x200xf32>)
  func.func @main(%argi: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %caps = IE.CodeGenCapsule inputs(%argi as %arg0: tensor<1x16x4000x200xf32>) {
        %0 = tensor.empty() : tensor<1x16x4000x200xf32>
        %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4000x200xf32>) outs(%0 : tensor<1x16x4000x200xf32>) {
        ^bb0(%in: f32, %out: f32):
          %2 = math.exp %in fastmath<afn> : f32
          linalg.yield %2 : f32
        } -> tensor<1x16x4000x200xf32>
        IE.CGCYield %1 : tensor<1x16x4000x200xf32>
    } -> tensor<1x16x4000x200xf32>
    return %caps : tensor<1x16x4000x200xf32>

  // CHECK: IE.CodeGenCapsule inputs([[ARG]] as [[ARG0:%.+]]: tensor<1x16x4000x200xf32>)
  // CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
  // CHECK-DAG:  [[TRIPCOUNT:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
  // CHECK-DAG:  [[ONE:%.+]] = arith.constant 1 : index
  // CHECK:      [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIPCOUNT]] step [[ONE]] iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x200xf32>) {
  // CHECK:          [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index, index, index), offsets(index, index, index)
  // CHECK:          [[SLICE:%.+]] = tensor.extract_slice
  // CHECK-SAME:         [[ARG0]][0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2]
  // CHECK-SAME:         [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2]
  // CHECK-SAME:         to tensor<1x?x?x?xf32>
  // CHECK:          [[LOOP_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2) : tensor<1x?x?x?xf32>
  // CHECK:          [[OP:%.+]] = linalg.generic
  // CHECK-SAME:        ins([[SLICE]] : tensor<1x?x?x?xf32>)
  // CHECK-SAME:        outs([[LOOP_EMPTY]] : tensor<1x?x?x?xf32>)
  // CHECK:          [[INSERT:%.+]] = tensor.insert_slice [[OP]] into [[ACCUM]]
  // CHECK-SAME:         [0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2]
  // CHECK-SAME:         [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1]
  // CHECK-SAME:         into tensor<1x16x4000x200xf32>
  // CHECK-NEXT:     scf.yield [[INSERT]] : tensor<1x16x4000x200xf32>
  // CHECK:       IE.CGCYield [[FOR]] : tensor<1x16x4000x200xf32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: [[MAP:#.+]] = affine_map<(d0) -> (d0, 200)>
// CHECK: [[MAP1:#.+]] = affine_map<(d0, d1) -> (-d0 + 200, d1)>
// CHECK: [[MAP2:#.+]] = affine_map<(d0, d1) -> (d0 - d1)>

// CHECK: module @EltwisePad
module @EltwisePad {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x4000x200xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x4000x256xf32>
  }

// CHECK: func.func @main([[ARG:%.+]]: tensor<1x16x4000x200xf32>)
  func.func @main(%argi: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x256xf32> {
    %caps = IE.CodeGenCapsule inputs(%argi as %arg0: tensor<1x16x4000x200xf32>) {
        %0 = tensor.empty() : tensor<1x16x4000x200xf32>
        %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4000x200xf32>) outs(%0 : tensor<1x16x4000x200xf32>) {
        ^bb0(%in: f32, %out: f32):
          %2 = math.exp %in fastmath<afn> : f32
          linalg.yield %2 : f32
        } -> tensor<1x16x4000x200xf32>
        %padv = arith.constant 0.0 : f32
        %2 = tensor.pad %1 low[0, 0, 0, 0] high[0, 0, 0, 56] {
        ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
          tensor.yield %padv : f32
        } : tensor<1x16x4000x200xf32> to tensor<1x16x4000x256xf32>
        IE.CGCYield %2 : tensor<1x16x4000x256xf32>
    } -> tensor<1x16x4000x256xf32>
    return %caps : tensor<1x16x4000x256xf32>

// CHECK: IE.CodeGenCapsule inputs([[ARG]] as [[ARG0:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK-DAG:   [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x256xf32>
// CHECK-DAG:   [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK:       [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x256xf32>) {
// CHECK:         [[SIZES:%.+]]:3, [[OFFSETS:%.+]]:3 = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index, index, index), offsets(index, index, index)
// CHECK:         [[B1:%.+]] = affine.min [[MAP]]([[OFFSETS]]#2)
// CHECK:         [[B2:%.+]] = affine.min [[MAP1]]([[B1]], [[SIZES]]#2)
// CHECK:         [[B4:%.+]] = affine.apply [[MAP2]]([[SIZES]]#2, [[B2]])
// CHECK:         [[SLICE:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:        [0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[B1]]]
// CHECK-SAME:        [1, [[SIZES]]#0, [[SIZES]]#1, [[B2]]] [1, 1, 1, 1]
// CHECK-SAME:        to tensor<1x?x?x?xf32>
// CHECK:         [[LOOP_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1, [[B2]]) : tensor<1x?x?x?xf32>
// CHECK:         [[OP:%.+]] = linalg.generic
// CHECK-SAME:        ins([[SLICE]] : tensor<1x?x?x?xf32>)
// CHECK-SAME:        outs([[LOOP_EMPTY]] : tensor<1x?x?x?xf32>)
// CHECK:         [[PAD:%.+]] = tensor.pad [[OP]] low[0, 0, 0, 0] high[0, 0, 0, [[B4]]]
// CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[PAD]] into [[ACCUM]]
// CHECK-SAME:        [0, [[OFFSETS]]#0, [[OFFSETS]]#1, [[OFFSETS]]#2]
// CHECK-SAME:        [1, [[SIZES]]#0, [[SIZES]]#1, [[SIZES]]#2] [1, 1, 1, 1]
// CHECK-SAME:        into tensor<1x16x4000x256xf32>
// CHECK:         scf.yield [[INSERT]] : tensor<1x16x4000x256xf32>
// CHECK:       IE.CGCYield [[FOR]] : tensor<1x16x4000x256xf32>
  }
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>

// CHECK: module @ReduceAndEltwise
module @ReduceAndEltwise {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<20x1x1x175x512xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<20x1x1x175x512xf32>
  }

// CHECK: func.func @main([[ARG:%.+]]: tensor<20x1x1x175x512xf32>)
  func.func @main(%arg0: tensor<20x1x1x175x512xf32>) -> tensor<20x1x1x175x512xf32> {
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<20x1x1x175x512xf32>) {
         %cst = arith.constant 0.000000e+00 : f32
         %padv = arith.constant 0.0 : f32
         %0 = tensor.empty() : tensor<1x1x175x512xf32>
         %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x175x512xf32>) -> tensor<1x1x175x512xf32>
         %2 = linalg.generic {
             indexing_maps = [#NCDHW, #map],
             iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel"]}
            ins(%arg2 : tensor<20x1x1x175x512xf32>)
            outs(%1 : tensor<1x1x175x512xf32>) {
         ^bb0(%in: f32, %out: f32):
           %7 = arith.addf %out, %in fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x175x512xf32>
         %binit = tensor.empty() : tensor<20x1x1x175x512xf32>
         %3 = linalg.generic {
             indexing_maps = [#NCDHW, #map1, #NCDHW],
             iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
             ins(%arg2, %2 : tensor<20x1x1x175x512xf32>, tensor<1x1x175x512xf32>)
             outs(%binit : tensor<20x1x1x175x512xf32>) {
         ^bb0(%in: f32, %in1: f32,  %out: f32):
           %7 = arith.addf %in1, %in fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<20x1x1x175x512xf32>

         IE.CGCYield %3 : tensor<20x1x1x175x512xf32>
     } -> tensor<20x1x1x175x512xf32>
     return %0 : tensor<20x1x1x175x512xf32>

// CHECK:  IE.CodeGenCapsule inputs({{.*}} as [[ARG0:%.+]]: tensor<20x1x1x175x512xf32>) {
// CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<20x1x1x175x512xf32>
// CHECK-DAG:  [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK:      [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<20x1x1x175x512xf32>) {
// CHECK:         [[SIZES:%.+]]:2, [[OFFSETS:%.+]]:2 = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index, index), offsets(index, index)
// CHECK:         [[SLICE]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:        [0, 0, 0, [[OFFSETS]]#0, [[OFFSETS]]#1]
// CHECK-SAME:        [20, 1, 1, [[SIZES]]#0, [[SIZES]]#1]
// CHECK-SAME:        to tensor<20x1x1x?x?xf32>
// CHECK:         [[FILL_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1) : tensor<1x1x?x?xf32>
// CHECK:         [[FILL:%.+]] = linalg.fill ins({{.*}} : f32) outs([[FILL_EMPTY]] : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
// CHECK:         [[OP1:%.+]] = linalg.generic
// CHECK-SAME:        ins([[SLICE]] : tensor<20x1x1x?x?xf32>)
// CHECK-SAME:        outs([[FILL]] : tensor<1x1x?x?xf32>)
// CHECK:         [[OP2_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1) : tensor<20x1x1x?x?xf32>
// CHECK:         [[OP2:%.+]] = linalg.generic
// CHECK-SAME:        ins([[SLICE]], [[OP1]] : tensor<20x1x1x?x?xf32>, tensor<1x1x?x?xf32>)
// CHECK-SAME:        outs([[OP2_EMPTY]] : tensor<20x1x1x?x?xf32>)
// CHECK:         [[INSERT:%.+]] = tensor.insert_slice [[OP2]] into [[ACCUM]]
// CHECK-SAME:        [0, 0, 0, [[OFFSETS]]#0, [[OFFSETS]]#1]
// CHECK-SAME:        [20, 1, 1, [[SIZES]]#0, [[SIZES]]#1] [1, 1, 1, 1, 1]
// CHECK-SAME:        into tensor<20x1x1x175x512xf32>
// CHECK:         scf.yield [[INSERT]] : tensor<20x1x1x175x512xf32>
// CHECK:      IE.CGCYield [[FOR]] : tensor<20x1x1x175x512xf32>
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>

// CHECK: module @LargeReduction
module @LargeReduction {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x100000000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1xf32>
  }

// Negative test, this is a reduction op which cannot be parallel tiled

// CHECK: func.func @main({{.*}}: tensor<1x1x1x100000000xf32>)
  func.func @main(%arg0: tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x1xf32> {
// CHECK-NOT: Shave.LoopTripCount
// CHECK-NOT: scf.for
// CHECK-NOT: Shave.OutSliceInfo
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x100000000xf32>) {
         %cst = arith.constant 0.000000e+00 : f32
         %padv = arith.constant 0.0 : f32
         %0 = tensor.empty() : tensor<1x1x1x1xf32>
         %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
         %2 = linalg.generic {
             indexing_maps = [#map, #map1],
             iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
            ins(%arg2 : tensor<1x1x1x100000000xf32>)
            outs(%1 : tensor<1x1x1x1xf32>) {
         ^bb0(%in: f32, %out: f32):
           %7 = arith.addf %out, %in fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x1x1xf32>

         IE.CGCYield %2 : tensor<1x1x1x1xf32>
     } -> tensor<1x1x1x1xf32>
     return %0 : tensor<1x1x1x1xf32>
  }
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>

// CHECK: module @RankAdjustedSoftmax

module @RankAdjustedSoftmax {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x16x4000x200xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x4000x200xf32>
  }
// CHECK:   func.func @main(
  func.func @main(%argi: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
    %caps = IE.CodeGenCapsule inputs(%argi as %arg0: tensor<1x16x4000x200xf32>) {
        %0 = tensor.empty() : tensor<1x16x4000x200xf32>
        %cst = arith.constant -3.40282347E+38 : f32
        %1 = tensor.empty() : tensor<1x16x1x200xf32>
        %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x16x1x200xf32>) -> tensor<1x16x1x200xf32>
        %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x4000x200xf32>) outs(%2 : tensor<1x16x1x200xf32>) {
        ^bb0(%in: f32, %out: f32):
          %8 = arith.maximumf %in, %out fastmath<nnan,nsz> : f32
          linalg.yield %8 : f32
        } -> tensor<1x16x1x200xf32>
        %4 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<1x16x4000x200xf32>, tensor<1x16x1x200xf32>) outs(%0 : tensor<1x16x4000x200xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.subf %in, %in_1 : f32
          %9 = math.exp %8 fastmath<afn> : f32
          linalg.yield %9 : f32
        } -> tensor<1x16x4000x200xf32>
        %cst_0 = arith.constant 0.000000e+00 : f32
        %5 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x16x1x200xf32>) -> tensor<1x16x1x200xf32>
        %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%4 : tensor<1x16x4000x200xf32>) outs(%5 : tensor<1x16x1x200xf32>) {
        ^bb0(%in: f32, %out: f32):
          %8 = arith.addf %out, %in fastmath<reassoc> : f32
          linalg.yield %8 : f32
        } -> tensor<1x16x1x200xf32>
        %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%4, %6 : tensor<1x16x4000x200xf32>, tensor<1x16x1x200xf32>) outs(%0 : tensor<1x16x4000x200xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.divf %in, %in_1 fastmath<arcp> : f32
          linalg.yield %8 : f32
        } -> tensor<1x16x4000x200xf32>
        IE.CGCYield %7 : tensor<1x16x4000x200xf32>
    } -> tensor<1x16x4000x200xf32>
    return %caps : tensor<1x16x4000x200xf32>

// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG0]]: tensor<1x16x4000x200xf32>) {
// CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK-DAG:  [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK:      [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x16x4000x200xf32>) {
// CHECK-DAG:    [[SIZES:%.+]]:2, [[OFFSETS:%.+]]:2 = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index, index), offsets(index, index)
// CHECK-DAG:    [[SLICE:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:       [0, [[OFFSETS]]#0, 0, [[OFFSETS]]#1]
// CHECK-SAME:       [1, [[SIZES]]#0, 4000, [[SIZES]]#1] [1, 1, 1, 1]
// CHECK-SAME:       to tensor<1x?x4000x?xf32>
// CHECK-DAG:    [[FILL_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1) : tensor<1x?x1x?xf32>
// CHECK-DAG:    [[FILL:%.+]] = linalg.fill ins({{.*}} : f32) outs([[FILL_EMPTY]] : tensor<1x?x1x?xf32>) -> tensor<1x?x1x?xf32>
// CHECK-DAG:    [[MAX:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SLICE]] : tensor<1x?x4000x?xf32>)
// CHECK-SAME:       outs([[FILL]] : tensor<1x?x1x?xf32>) {
// CHECK-DAG:    [[SUBEXP_EMPTY:%.+]] = tensor.empty([[SIZES]]#0, [[SIZES]]#1) : tensor<1x?x4000x?xf32>
// CHECK-DAG:    [[SUBEXP:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SLICE]], [[MAX]] : tensor<1x?x4000x?xf32>, tensor<1x?x1x?xf32>)
// CHECK-SAME:       outs([[SUBEXP_EMPTY]] : tensor<1x?x4000x?xf32>)
// CHECK-DAG:    [[ADD_FILL_EMPTY:%.+]] = linalg.fill ins({{.*}}) outs([[FILL_EMPTY]] : tensor<1x?x1x?xf32>) -> tensor<1x?x1x?xf32>
// CHECK-DAG:    [[REDUCE_ADD:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SUBEXP]] : tensor<1x?x4000x?xf32>)
// CHECK-SAME:       outs([[ADD_FILL_EMPTY]] : tensor<1x?x1x?xf32>)
// CHECK-DAG:    [[DIV:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SUBEXP]], [[REDUCE_ADD]] : tensor<1x?x4000x?xf32>, tensor<1x?x1x?xf32>)
// CHECK-SAME:       outs([[SUBEXP_EMPTY]] : tensor<1x?x4000x?xf32>)
// CHECK:        [[SLICE:%.+]] = tensor.insert_slice [[DIV]] into [[ACCUM]]
// CHECK-SAME:       [0, [[OFFSETS]]#0, 0, [[OFFSETS]]#1]
// CHECK-SAME:       [1, [[SIZES]]#0, 4000, [[SIZES]]#1] [1, 1, 1, 1]
// CHECK-SAME:       into tensor<1x16x4000x200xf32>
// CHECK-NEXT:   scf.yield [[SLICE]] : tensor<1x16x4000x200xf32>
// CHECK: IE.CGCYield [[FOR]] : tensor<1x16x4000x200xf32>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>

// CHECK: module @ThreePartGraph
module @ThreePartGraph {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x100000000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x100000000xf32>
  }

// CHECK: func.func @main(
  func.func @main(%arg0: tensor<1x1x1x100000000xf32>) -> tensor<1x1x1x100000000xf32> {
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x100000000xf32>) {
         // Large element-wise layer (can tile).
         %empty = tensor.empty() : tensor<1x1x1x100000000xf32>
         %elt = linalg.generic {
             indexing_maps = [#NCHW, #NCHW],
             iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%arg2 : tensor<1x1x1x100000000xf32>)
            outs(%empty : tensor<1x1x1x100000000xf32>) {
         ^bb0(%in: f32, %out: f32):
           %res = math.exp %in fastmath<afn> : f32
           linalg.yield %res : f32
         } -> tensor<1x1x1x100000000xf32>

         // Large reduction layer which does not fit (cannot tile).
         %0 = tensor.empty() : tensor<1x1x1x1xf32>
         %cst = arith.constant 0.000000e+00 : f32
         %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
         %2 = linalg.generic {
             indexing_maps = [#map, #map1],
             iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
            ins(%elt : tensor<1x1x1x100000000xf32>)
            outs(%1 : tensor<1x1x1x1xf32>) {
         ^bb0(%in: f32, %out: f32):
           %7 = arith.addf %out, %in fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x1x1xf32>

         // A large broadcasting layer
         %3 = linalg.generic {
             indexing_maps = [#NCHW, #map2, #NCHW],
             iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%arg2, %2 : tensor<1x1x1x100000000xf32>, tensor<1x1x1x1xf32>)
            outs(%empty : tensor<1x1x1x100000000xf32>) {
         ^bb0(%in: f32, %in1: f32, %out: f32):
           %7 = arith.addf %in, %in1 : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x1x100000000xf32>

         IE.CGCYield %3 : tensor<1x1x1x100000000xf32>
     } -> tensor<1x1x1x100000000xf32>
     return %0 : tensor<1x1x1x100000000xf32>

// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG0:%.+]]: tensor<1x1x1x100000000xf32>)
// CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK-DAG:  [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK:      [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x1x1x100000000xf32>)
// CHECK-DAG:    [[SIZES:%.+]], [[OFFSETS:%.+]] = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index), offsets(index)
// CHECK-DAG:    [[SLICE:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]]]
// CHECK-SAME:       [1, 1, 1, [[SIZES]]] [1, 1, 1, 1]
// CHECK-SAME:       to tensor<1x1x1x?xf32>
// CHECK-DAG:    [[OP_EMPTY:%.+]] = tensor.empty([[SIZES]]) : tensor<1x1x1x?xf32>
// CHECK-DAG:    [[OP:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SLICE]] : tensor<1x1x1x?xf32>)
// CHECK-SAME:       outs([[OP_EMPTY]] : tensor<1x1x1x?xf32>)
// CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[OP]] into [[ACCUM]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]]]
// CHECK-SAME:       [1, 1, 1, [[SIZES]]] [1, 1, 1, 1]
// CHECK-SAME:       into tensor<1x1x1x100000000xf32>
// CHECK-NEXT:   scf.yield [[INSERT]] : tensor<1x1x1x100000000xf32>
// CHECK-DAG:  [[REDUCE_EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1xf32>
// CHECK-DAG:  [[REDUCE_FILL:%.+]] = linalg.fill ins({{.*}} : f32) outs([[REDUCE_EMPTY]] : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:  [[REDUCE:%.+]] = linalg.generic
// CHECK-SAME:     ins([[FOR]] : tensor<1x1x1x100000000xf32>)
// CHECK-SAME:     outs([[REDUCE_FILL]] : tensor<1x1x1x1xf32>)
// CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x100000000xf32>
// CHECK-DAG:  [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK:      [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x1x1x100000000xf32>) {
// CHECK-DAG:    [[SIZES:%.+]], [[OFFSETS:%.+]] = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index), offsets(index)
// CHECK-DAG:    [[SLICE:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]]]
// CHECK-SAME:       [1, 1, 1, [[SIZES]]] [1, 1, 1, 1]
// CHECK-SAME:       to tensor<1x1x1x?xf32>
// CHECK-DAG:    [[OP_EMPTY:%.+]] = tensor.empty([[SIZES]]) : tensor<1x1x1x?xf32>
// CHECK-DAG:    [[OP:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SLICE]], [[REDUCE]] : tensor<1x1x1x?xf32>, tensor<1x1x1x1xf32>)
// CHECK-SAME:       outs([[OP_EMPTY]] : tensor<1x1x1x?xf32>) {
// CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[OP]] into [[ACCUM]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]]]
// CHECK-SAME:       [1, 1, 1, [[SIZES]]] [1, 1, 1, 1]
// CHECK-SAME:       into tensor<1x1x1x100000000xf32>
// CHECK-NEXT:   scf.yield [[INSERT]] : tensor<1x1x1x100000000xf32>
// CHECK:  IE.CGCYield [[FOR]] : tensor<1x1x1x100000000xf32>
  }
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK: module @CascadeReduce
module @CascadeReduce {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<20x1x1x175x512xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x175xf32>
  }

// CHECK: func.func @main(
  func.func @main(%arg0: tensor<20x1x1x175x512xf16>) -> tensor<1x1x1x175xf32> {
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<20x1x1x175x512xf16>) {
         %cst = arith.constant 0.000000e+00 : f32
         %0 = tensor.empty() : tensor<1x1x175x512xf32>
         %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x175x512xf32>) -> tensor<1x1x175x512xf32>
         %2 = linalg.generic {indexing_maps = [#NCDHW, #map], iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<20x1x1x175x512xf16>) outs(%1 : tensor<1x1x175x512xf32>) {
         ^bb0(%in: f16, %out: f32):
           %5 = math.absf %in : f16
           %6 = arith.extf %5 : f16 to f32
           %7 = arith.addf %out, %6 fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x175x512xf32>
         %red_init = tensor.empty() : tensor<1x1x1x175xf32>
         %red = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<1x1x175x512xf32>) outs(%red_init : tensor<1x1x1x175xf32>) {
         ^bb0(%in: f32, %out: f32):
           %7 = arith.addf %out, %in fastmath<reassoc> : f32
           linalg.yield %7 : f32
         } -> tensor<1x1x1x175xf32>

         IE.CGCYield %red : tensor<1x1x1x175xf32>
     } -> tensor<1x1x1x175xf32>
     return %0 : tensor<1x1x1x175xf32>

// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG0:%.+]]: tensor<20x1x1x175x512xf16>)
// CHECK-DAG:  [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x175xf32>
// CHECK-DAG:  [[TRIP:%.+]] = Shave.LoopTripCount({{.*}}) : index -> index
// CHECK-DAG:  [[FOR:%.+]] = scf.for [[IT:%.+]] = {{.*}} to [[TRIP]] step {{.*}} iter_args([[ACCUM:%.+]] = [[EMPTY]]) -> (tensor<1x1x1x175xf32>) {
// CHECK-DAG:    [[SIZES:%.+]], [[OFFSETS:%.+]] = Shave.OutSliceInfo({{.*}}, [[IT]]) -> sizes(index), offsets(index)
// CHECK-DAG:    [[SLICE:%.+]] = tensor.extract_slice [[ARG0]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]], 0]
// CHECK-SAME:       [20, 1, 1, [[SIZES]], 512] [1, 1, 1, 1, 1]
// CHECK-SAME:       to tensor<20x1x1x?x512xf16>
// CHECK-DAG:    [[FILL_EMPTY:%.+]] = tensor.empty([[SIZES]]) : tensor<1x1x?x512xf32>
// CHECK-DAG:    [[FILL:%.+]] = linalg.fill ins({{.*}} : f32) outs([[FILL_EMPTY]] : tensor<1x1x?x512xf32>) -> tensor<1x1x?x512xf32>
// CHECK-DAG:    [[REDUCE1:%.+]] = linalg.generic
// CHECK-SAME:       ins([[SLICE]] : tensor<20x1x1x?x512xf16>)
// CHECK-SAME:       outs([[FILL]] : tensor<1x1x?x512xf32>) {
// CHECK-DAG:    [[REDUCE2_EMPTY:%.+]] = tensor.empty([[SIZES]]) : tensor<1x1x1x?xf32>
// CHECK-DAG:    [[REDUCE2:%.+]] = linalg.generic
// CHECK-SAME:       ins([[REDUCE1]] : tensor<1x1x?x512xf32>)
// CHECK-SAME:       outs([[REDUCE2_EMPTY]] : tensor<1x1x1x?xf32>)
// CHECK-DAG:    [[INSERT:%.+]] = tensor.insert_slice [[REDUCE2]] into [[ACCUM]]
// CHECK-SAME:       [0, 0, 0, [[OFFSETS]]]
// CHECK-SAME:       [1, 1, 1, [[SIZES]]] [1, 1, 1, 1]
// CHECK-SAME:       into tensor<1x1x1x175xf32>
// CHECK:        scf.yield [[INSERT]] : tensor<1x1x1x175xf32>
// CHECK:  IE.CGCYield [[FOR]] : tensor<1x1x1x175xf32>
  }
}

// -----

// CHECK: module @OpSoftmax
module @OpSoftmax {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x10000x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x10000x1000xf16>
  }

// CHECK: func.func @main(
  func.func @main(%arg0: tensor<1x1x10000x1000xf16>) -> tensor<1x1x10000x1000xf16> {
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x10000x1000xf16>) {
         %empty = tensor.empty() : tensor<1x1x10000x1000xf16>
         %sm = linalg.softmax dimension(3) ins(%arg2 : tensor<1x1x10000x1000xf16>) outs(%empty : tensor<1x1x10000x1000xf16>) -> tensor<1x1x10000x1000xf16>
         IE.CGCYield %sm : tensor<1x1x10000x1000xf16>
     } -> tensor<1x1x10000x1000xf16>
     return %0 : tensor<1x1x10000x1000xf16>

// CHECK-NOT: Shave.LoopTripCount
// CHECK-NOT: scf.for
// CHECK-NOT: Shave.OutSliceInfo
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 floordiv 7)>

// CHECK: module @InvalidOp
module @InvalidOp {

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

// CHECK: func.func @main(
  func.func @main(%arg0: tensor<1x1x1x1000000xf32>) -> tensor<1x1x1x1000xf32> {
     %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000000xf32>) {
         %empty = tensor.empty() : tensor<1x1x1x1000xf32>
         %1 = linalg.generic {
             indexing_maps = [#map, #NCHW],
             iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
           ins(%arg2 : tensor<1x1x1x1000000xf32>) outs(%empty : tensor<1x1x1x1000xf32>) {
         ^bb0(%in: f32, %out: f32):
           %2 = math.exp %in fastmath<afn> : f32
           linalg.yield %2 : f32
         } -> tensor<1x1x1x1000xf32>
         IE.CGCYield %1 : tensor<1x1x1x1000xf32>
     } -> tensor<1x1x1x1000xf32>
     return %0 : tensor<1x1x1x1000xf32>

// CHECK-NOT: Shave.LoopTripCount
// CHECK-NOT: scf.for
// CHECK-NOT: Shave.OutSliceInfo
  }
}
