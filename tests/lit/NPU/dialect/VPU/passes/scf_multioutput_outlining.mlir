//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --scf-compute-ops-outlining  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @OutliningLSTMGates
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Outlining: LSTMGates (2 outputs) inside scf.for loop body.
// The compute op should be outlined into a nested function returning both outputs.

module @OutliningLSTMGates {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "gatesInput" : tensor<1x1x128x512xf16>
        DataInfo "cellState" : tensor<1x1x128x128xf16>
    }
    outputsInfo : {
        DataInfo "hiddenState" : tensor<1x1x128x128xf16>
        DataInfo "cellState_out" : tensor<1x1x128x128xf16>
    }

  func.func @main(
      %arg0: tensor<1x1x128x512xf16>,
      %arg1: tensor<1x1x128x128xf16>
  ) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out0 = tensor.empty() : tensor<1x1x128x128xf16>
    %out1 = tensor.empty() : tensor<1x1x128x128xf16>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg2 = %out0, %arg3 = %out1) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {

      %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, 32, 512] [1, 1, 1, 1]
          : tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
      %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>

      %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell) {
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>
        -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

      %ins_h = tensor.insert_slice %h into %arg2[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
      %ins_c = tensor.insert_slice %c into %arg3[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>

      scf.yield %ins_h, %ins_c : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
    }

    return %result#0, %result#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

// CHECK:      func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x1x32x512xf16>, [[INPUT1:%.+]]: tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:        [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[INPUT0]], [[INPUT1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16> -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
// CHECK-NEXT: return [[H]], [[C]] : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x1x128x512xf16>, [[ARG1:%.+]]: tensor<1x1x128x128xf16>) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>)
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[EMPTY0:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
// CHECK-DAG:    [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[EMPTY0]], [[ACC1:%.+]] = [[EMPTY1]]) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>)
// CHECK-NEXT:      [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 1, 32, 512] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
// CHECK-NEXT:      [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>
// CHECK-NEXT:      [[FUNC:%.+]]:2 = func.call @main_func0([[EXTRACT0]], [[EXTRACT1]]) :
// CHECK-SAME:          (tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[ACC0]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[ACC1]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

  }
}


// -----

// CHECK-LABEL: @OutliningLSTMGatesWithCondition
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>

// Outlining: LSTMGates with scf.if for first/last tile padding variation.
// Tests multi-output outlining with conditional execution (scf.if inside scf.for).

module @OutliningLSTMGatesWithCondition {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "gatesInput" : tensor<1x1x128x512xf16>
        DataInfo "cellState" : tensor<1x1x128x128xf16>
    }
    outputsInfo : {
        DataInfo "hiddenState" : tensor<1x1x128x128xf16>
        DataInfo "cellState_out" : tensor<1x1x128x128xf16>
    }

  func.func @main(
      %arg0: tensor<1x1x128x512xf16>,
      %arg1: tensor<1x1x128x128xf16>
  ) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out0 = tensor.empty() : tensor<1x1x128x128xf16>
    %out1 = tensor.empty() : tensor<1x1x128x128xf16>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg2 = %out0, %arg3 = %out1) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {

      %c0_0 = arith.constant 0 : index
      %is_first = arith.cmpi eq, %iv, %c0_0 : index

      %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, 32, 512] [1, 1, 1, 1]
          : tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
      %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>

      %tile_results:2 = scf.if %is_first -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
        %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>
          -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
        scf.yield %h, %c : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
      } else {
        %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>
          -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
        scf.yield %h, %c : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
      }

      %ins_h = tensor.insert_slice %tile_results#0 into %arg2[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
      %ins_c = tensor.insert_slice %tile_results#1 into %arg3[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
          : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>

      scf.yield %ins_h, %ins_c : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
    }

    return %result#0, %result#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

// CHECK:      func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x1x32x512xf16>, [[INPUT1:%.+]]: tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:        [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[INPUT0]], [[INPUT1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16> -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
// CHECK-NEXT: return [[H]], [[C]] : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

// CHECK:      func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x1x32x512xf16>, [[INPUT1:%.+]]: tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:        [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[INPUT0]], [[INPUT1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16> -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
// CHECK-NEXT: return [[H]], [[C]] : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x1x128x512xf16>, [[ARG1:%.+]]: tensor<1x1x128x128xf16>) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>)
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[EMPTY0:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
// CHECK-DAG:    [[EMPTY1:%.+]] = tensor.empty() : tensor<1x1x128x128xf16>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[EMPTY0]], [[ACC1:%.+]] = [[EMPTY1]]) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>)
// CHECK-NEXT:    [[C0_0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpi eq, [[IDX]], [[C0_0]] : index
// CHECK-NEXT:      [[EXTRACT0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 1, 32, 512] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
// CHECK-NEXT:      [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>
// CHECK-NEXT:      [[IF:%.+]]:2 = scf.if [[CMP]] -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:          [[FUNC:%.+]]:2 = func.call @main_func0([[EXTRACT0]], [[EXTRACT1]]) :
// CHECK-SAME:              (tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:          scf.yield [[FUNC]]#0, [[FUNC]]#1 : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
// CHECK-NEXT:      else
// CHECK-NEXT:          [[FUNC:%.+]]:2 = func.call @main_func1([[EXTRACT0]], [[EXTRACT1]]) :
// CHECK-SAME:              (tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>) -> (tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
// CHECK-NEXT:          scf.yield [[FUNC]]#0, [[FUNC]]#1 : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
// CHECK:           [[INSERT0:%.+]] = tensor.insert_slice [[IF]]#0 into [[ACC0]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[IF]]#1 into [[ACC1]][0, 0, [[IDX]], 0] [1, 1, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

  }
}

// -----

// CHECK-LABEL: @OutliningTopK
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Outlining: TopK (2 outputs: f32 values + si32 indices) inside scf.for loop body.
// The compute op should be outlined into a nested function returning both outputs.
// Tests that mixed-type multi-output ops are outlined correctly.

module @OutliningTopK {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x128x128xf32>
    }
    outputsInfo : {
        DataInfo "values" : tensor<1x8x128x128xf32>
        DataInfo "indices" : tensor<1x8x128x128xsi32>
    }

  func.func @main(
      %arg0: tensor<1x64x128x128xf32>
  ) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
    %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out_vals = tensor.empty() : tensor<1x8x128x128xf32>
    %out_inds = tensor.empty() : tensor<1x8x128x128xsi32>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg1 = %out_vals, %arg2 = %out_inds) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {

      %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, 32, 128] [1, 1, 1, 1]
          : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>

      %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
          axis = 1 : i64,
          element_type = si32,
          k_value = 8 : i64,
          mode = #IE.topk_mode<MAX>,
          sort = #IE.topk_sort_type<SORT_INDICES>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
      } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
        -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

      %ins_vals = tensor.insert_slice %vals into %arg1[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
      %ins_inds = tensor.insert_slice %inds into %arg2[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>

      scf.yield %ins_vals, %ins_inds : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
    }

    return %result#0, %result#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>


// CHECK:      func.func nested @main_func0() -> tensor<1x1x1x1024xui8>
// CHECK-NEXT:        [[V:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
// CHECK-NEXT: return [[V]]

// CHECK:      func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x64x32x128xf32>, [[INPUT1:%.+]]: tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:        [[V:%.+]], [[S:%.+]] = VPU.TopK([[INPUT0]], [[INPUT1]]) {axis = 1 : i64, element_type = si32, k_value = 8 : i64, mode = #IE.topk_mode<MAX>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, sort = #IE.topk_sort_type<SORT_INDICES>}
// CHECK-SAME:            tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8> -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
// CHECK-NEXT: return [[V]], [[S]] : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x64x128x128xf32>) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>)
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[EMPTY:%.+]] = call @main_func0() : () -> tensor<1x1x1x1024xui8>
// CHECK-DAG:    [[B0:%.+]] = tensor.empty() : tensor<1x8x128x128xf32>
// CHECK-DAG:    [[B1:%.+]] = tensor.empty() : tensor<1x8x128x128xsi32>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[B0]], [[ACC1:%.+]] = [[B1]]) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>)
// CHECK-NEXT:      [[EXTRACT:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 64, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>
// CHECK-NEXT:      [[FUNC:%.+]]:2 = func.call @main_func1([[EXTRACT]], [[EMPTY]]) :
// CHECK-SAME:          (tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:      [[INSERT0:%.+]] = tensor.insert_slice [[FUNC]]#0 into [[ACC0]][0, 0, [[IDX]], 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[FUNC]]#1 into [[ACC1]][0, 0, [[IDX]], 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

  }
}


// -----

// CHECK-LABEL: @OutliningTopKWithCondition
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>

// Outlining: TopK with scf.if for first/last tile handling.
// Tests mixed-type multi-output (f32+si32) outlining with conditional execution.

module @OutliningTopKWithCondition {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x128x128xf32>
    }
    outputsInfo : {
        DataInfo "values" : tensor<1x8x128x128xf32>
        DataInfo "indices" : tensor<1x8x128x128xsi32>
    }

  func.func @main(
      %arg0: tensor<1x64x128x128xf32>
  ) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
    %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %out_vals = tensor.empty() : tensor<1x8x128x128xf32>
    %out_inds = tensor.empty() : tensor<1x8x128x128xsi32>

    %result:2 = scf.for %iv = %c0 to %c128 step %c32
        iter_args(%arg1 = %out_vals, %arg2 = %out_inds) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {

      %c0_0 = arith.constant 0 : index
      %is_first = arith.cmpi eq, %iv, %c0_0 : index

      %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, 32, 128] [1, 1, 1, 1]
          : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>

      %tile_results:2 = scf.if %is_first -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>) {
        %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
            axis = 1 : i64,
            element_type = si32,
            k_value = 8 : i64,
            mode = #IE.topk_mode<MAX>,
            sort = #IE.topk_sort_type<SORT_INDICES>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
          -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
        scf.yield %vals, %inds : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
      } else {
        %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
            axis = 1 : i64,
            element_type = si32,
            k_value = 8 : i64,
            mode = #IE.topk_mode<MAX>,
            sort = #IE.topk_sort_type<SORT_INDICES>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
          -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
        scf.yield %vals, %inds : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
      }

      %ins_vals = tensor.insert_slice %tile_results#0 into %arg1[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
      %ins_inds = tensor.insert_slice %tile_results#1 into %arg2[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
          : tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>

      scf.yield %ins_vals, %ins_inds : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
    }

    return %result#0, %result#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

// CHECK:      func.func nested @main_func0() -> tensor<1x1x1x1024xui8>
// CHECK-NEXT:        [[V:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>
// CHECK-NEXT: return [[V]]

// CHECK:      func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x64x32x128xf32>, [[INPUT1:%.+]]: tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:        [[V:%.+]], [[S:%.+]] = VPU.TopK([[INPUT0]], [[INPUT1]]) {axis = 1 : i64, element_type = si32, k_value = 8 : i64, mode = #IE.topk_mode<MAX>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, sort = #IE.topk_sort_type<SORT_INDICES>}
// CHECK-SAME:            tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8> -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
// CHECK-NEXT: return [[V]], [[S]] : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

// CHECK:      func.func nested @main_func2([[INPUT0:%.+]]: tensor<1x64x32x128xf32>, [[INPUT1:%.+]]: tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:        [[V:%.+]], [[S:%.+]] = VPU.TopK([[INPUT0]], [[INPUT1]]) {axis = 1 : i64, element_type = si32, k_value = 8 : i64, mode = #IE.topk_mode<MAX>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, sort = #IE.topk_sort_type<SORT_INDICES>}
// CHECK-SAME:            tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8> -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
// CHECK-NEXT: return [[V]], [[S]] : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x64x128x128xf32>) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>)
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C128:%.+]] = arith.constant 128 : index
// CHECK-DAG:    [[C32:%.+]] = arith.constant 32 : index
// CHECK-DAG:    [[EMPTY:%.+]] = call @main_func0() : () -> tensor<1x1x1x1024xui8>
// CHECK-DAG:    [[B0:%.+]] = tensor.empty() : tensor<1x8x128x128xf32>
// CHECK-DAG:    [[B1:%.+]] = tensor.empty() : tensor<1x8x128x128xsi32>
// CHECK:        [[SCF:%.+]]:2 = scf.for [[IDX:%.+]] = [[C0]] to [[C128]] step [[C32]] iter_args([[ACC0:%.+]] = [[B0]], [[ACC1:%.+]] = [[B1]]) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>)
// CHECK-NEXT:      [[C0_0:%.+]] = arith.constant 0 : index
// CHECK-NEXT:      [[CMP:%.+]] = arith.cmpi eq, [[IDX]], [[C0_0]] : index
// CHECK-NEXT:      [[EXTRACT:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IDX]], 0] [1, 64, 32, 128] [1, 1, 1, 1] : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>
// CHECK-NEXT:      [[IF:%.+]]:2 = scf.if [[CMP]] -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:          [[FUNC:%.+]]:2 = func.call @main_func1([[EXTRACT]], [[EMPTY]]) : (tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:          scf.yield [[FUNC]]#0, [[FUNC]]#1 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
// CHECK-NEXT:      else
// CHECK-NEXT:          [[FUNC:%.+]]:2 = func.call @main_func2([[EXTRACT]], [[EMPTY]]) : (tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>) -> (tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>)
// CHECK-NEXT:          scf.yield [[FUNC]]#0, [[FUNC]]#1 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>
// CHECK:           [[INSERT0:%.+]] = tensor.insert_slice [[IF]]#0 into [[ACC0]][0, 0, [[IDX]], 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
// CHECK-NEXT:      [[INSERT1:%.+]] = tensor.insert_slice [[IF]]#1 into [[ACC1]][0, 0, [[IDX]], 0] [1, 8, 32, 128] [1, 1, 1, 1]
// CHECK-SAME:          tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>
// CHECK-NEXT:      scf.yield [[INSERT0]], [[INSERT1]] : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
// CHECK:        return [[SCF]]#0, [[SCF]]#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

  }
}
