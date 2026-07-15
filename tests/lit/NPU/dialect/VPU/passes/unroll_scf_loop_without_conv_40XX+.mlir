//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,2,1 enable-cascaded-unrolling=false" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @SimpleLoopUnroll inputsInfo : {
    DataInfo "input" : tensor<1x32x64x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x64xf16>
  }

  // CHECK-LABEL: func.func @SimpleLoopUnroll
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}>
  func.func @SimpleLoopUnroll(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index

    %0 = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
    %1 = scf.for %i = %c0 to %c32 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
        %2 = tensor.extract_slice %arg0[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>

        %3 = tensor.insert_slice %2 into %arg1[0, 0, %i, 0] [1, 32, 2, 64] [1, 1, 1, 1]
            : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>

        scf.yield %3 : tensor<1x32x64x64xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x32x64x64xf16, {order = #NHWC}>
  }

  // CHECK-DAG: [[C4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C32:%.+]] = arith.constant 32 : index
  // CHECK-DAG: [[C2:%.+]] = arith.constant 2 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: [[RESULT:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[C32]] step [[C4]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
  // CHECK:   [[ADD:%.+]] = arith.addi [[ARG1]], [[C2]] : index
  // CHECK:   [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[ARG1]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK:   [[INSERT1:%.+]] = tensor.insert_slice [[EXTRACT1]] into [[ARG2]][0, 0, [[ARG1]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK:   [[EXTRACT2:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[ADD]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK:   [[INSERT2:%.+]] = tensor.insert_slice [[EXTRACT2]] into [[INSERT1]][0, 0, [[ADD]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK:   scf.yield [[INSERT2]] : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @TwoDimLoopUnroll inputsInfo : {
    DataInfo "input" : tensor<1x32x64x96xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x96xf16>
  }

  // CHECK-LABEL: func.func @TwoDimLoopUnroll
  // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x64x96xf16, {order = #NHWC}>)
  func.func @TwoDimLoopUnroll(%arg0: tensor<1x32x64x96xf16, {order = #NHWC}>) -> tensor<1x32x64x96xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.empty() : tensor<1x32x64x96xf16, {order = #NHWC}>
    %1 = scf.for %h = %c0 to %c64 step %c2 iter_args(%arg1 = %0) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
        %2 = scf.for %w = %c0 to %c96 step %c3 iter_args(%arg2 = %arg1) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
            %3 = tensor.extract_slice %arg0[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>

            %4 = tensor.insert_slice %3 into %arg2[0, 0, %h, %w] [1, 32, 2, 3] [1, 1, 1, 1]
                : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>

            scf.yield %4 : tensor<1x32x64x96xf16, {order = #NHWC}>
        }
        scf.yield %2 : tensor<1x32x64x96xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x32x64x96xf16, {order = #NHWC}>
  }

  // CHECK-DAG: [[C4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[C64:%.+]] = arith.constant 64 : index
  // CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index
  // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
  // CHECK: [[EMPTY_0:%.+]] = tensor.empty() : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK: [[RESULT_0:%.+]] = scf.for [[ARG1:%.+]] = [[C0]] to [[C64]] step [[C4]] iter_args([[ARG2:%.+]] = [[EMPTY_0]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // CHECK:   [[INNER_RESULT:%.+]] = scf.for [[ARG3:%.+]] = [[C0]] to [[C96]] step [[C3]] iter_args([[ARG4:%.+]] = [[ARG2]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // CHECK:     [[EXTRACT1:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[ARG1]], [[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     [[INSERT1:%.+]] = tensor.insert_slice [[EXTRACT1]] into [[ARG4]][0, 0, [[ARG1]], [[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     scf.yield [[INSERT1]] : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:   }
  // CHECK:   scf.yield [[INNER_RESULT]] : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return [[RESULT_0]] : tensor<1x32x64x96xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
#map1 = affine_map<(d0, d1) -> (d0 - d1)>
module @StaticEltwiseNHWC {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }

  // CHECK:    func.func @merged_vpu_func_0([[ARG0:%.+]]: tensor<1x16x200x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}, [[ARG1:%.+]]: tensor<1x16x200x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x200x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) {
  // CHECK:      [[SLICE0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[SLICE2:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE2]])
  // CHECK:      [[SLICE1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 100, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[SLICE3:%.+]] = VPU.Slice [[ARG1]] [0, 0, 100, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[SLICE1]], [[SLICE3]])
  // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[ELTWISE0]], [[ELTWISE1]])
  // CHECK:      return [[CONCAT]] : tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main_func0_static([[ARG0:%.+]]: tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}, [[ARG1:%.+]]: tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) {
  // CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {{{.+}}op_type = #VPU.eltwise_type<ADD>{{.+}}} -> tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:    }

  func.func @main_func0_static(%arg0: tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}, %arg1: tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) -> (tensor<1x16x100x1000xf16, {order = #NHWC}> {func.dynamicStrides = true}) {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x100x1000xf16, {order = #NHWC}>
    return %0 : tensor<1x16x100x1000xf16, {order = #NHWC}>
  }

  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim]
      %3 = arith.cmpi eq, %arg2, %c0 : index
      %4 = scf.if %3 -> (index) {
        %7 = arith.cmpi sge, %2, %c100 : index
        cf.assert %7, "Not enough elements to backtrack in scf.for loop"
        scf.yield %c0 : index
      } else {
        %7 = arith.addi %arg2, %c100 : index
        %8 = arith.cmpi slt, %7, %dim : index
        %9 = scf.if %8 -> (index) {
          scf.yield %c0 : index
        } else {
          %10 = arith.cmpi eq, %7, %dim : index
          %11 = scf.if %10 -> (index) {
            scf.yield %c0 : index
          } else {
            %12 = arith.subi %dim, %arg2 : index
            %13 = arith.subi %12, %c100 : index
            scf.yield %13 : index
          }
          scf.yield %11 : index
        }
        scf.yield %9 : index
      }
      %5 = affine.apply #map1(%arg2, %4)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %5, 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %5, 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
      %6 = func.call @main_func0_static(%extracted_slice, %extracted_slice_0) : (tensor<1x16x100x1000xf16, {order = #NHWC}>, tensor<1x16x100x1000xf16, {order = #NHWC}>) -> tensor<1x16x100x1000xf16, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %6 into %arg3[0, 0, %5, 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x100x1000xf16, {order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }


  // CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK-DAG: [[C720:%.+]] = arith.constant 720 : index
  // CHECK-DAG: [[C200:%.+]] = arith.constant 200 : index
  // CHECK-DAG: [[C100:%.+]] = arith.constant 100 : index
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[EMPTY:%.+]] = tensor.empty
  // CHECK: [[FOR10:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C200]] iter_args([[ARG3_1:.+]] = [[EMPTY]])
  // CHECK:       [[OFFSET1:%.+]] = affine.min
  // CHECK:       [[EXTRACT0:%.+]] = tensor.extract_slice {{.+}}[0, 0, [[OFFSET1]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {{.+}}> to tensor<1x16x200x1000xf16, {{.+}}>
  // CHECK:       [[EXTRACT1:%.+]] = tensor.extract_slice {{.+}}[0, 0, [[OFFSET1]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {{.+}}> to tensor<1x16x200x1000xf16, {{.+}}>
  // CHECK:       [[OUTPUT:%.+]] = func.call @merged_vpu_func_0([[EXTRACT0]], [[EXTRACT1]]) : (tensor<1x16x200x1000xf16, {{.+}}>, tensor<1x16x200x1000xf16, {{.+}}>) -> tensor<1x16x200x1000xf16, {{.+}}>
  // CHECK:       [[INSERTED:%.+]] = tensor.insert_slice [[OUTPUT]] into {{%.+}}[0, 0, [[OFFSET1]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x200x1000xf16, {{.+}}> into tensor<1x16x720x?xf16, {{.+}}>
  // CHECK:       [[CAST:%.+]] = tensor.cast [[INSERTED]]
  // CHECK:       scf.yield [[CAST]]
  // CHECK: } {no_await_all = true}

  // CHECK: return
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK:       func.func @merged_vpu_func_1([[INPUT:%.+]]: tensor<1x8x4x16xf16>)
// CHECK-SAME:    -> tensor<1x8x4x16xf16>
// CHECK-DAG:   [[SLICE_FIRST:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0]  [1, 4, 4, 16]
// CHECK:       VPU.PermuteCast
// CHECK:       VPU.MaxPool
// CHECK:       VPU.PermuteCast
// CHECK:       VPU.Slice [[INPUT]] [0, 4, 0, 0] [1, 4, 4, 16]
// CHECK:       VPU.PermuteCast
// CHECK:       VPU.MaxPool
// CHECK:       VPU.PermuteCast
// CHECK:       VPU.Concat

module @testNHWCtoNCHWPermuteTransformation {
  net.NetworkInfo entryPoint : @permutetest inputsInfo : {
    DataInfo "input" : tensor<1x40x?x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x40x?xf16>
  }

  func.func private @tile_func(%arg0: tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
                                  mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}
         : tensor<1x4x4x16xf16> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %1 = VPU.MaxPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
                           rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
         : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    %2 = VPU.PermuteCast(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                  mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}
         : tensor<1x16x4x4xf16, {order = #NHWC}> -> tensor<1x4x4x16xf16>
    return %2 : tensor<1x4x4x16xf16>
  }

  // CHECK-LABEL: func.func @permutetest
  // CHECK-SAME:  ([[ARG0:%[^:]+]]: tensor<1x40x?x16xf16, {{.+}}>) -> tensor<1x16x40x?xf16, {{.+}}>
  func.func @permutetest(%arg0: tensor<1x40x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 40, 40, 16]> : tensor<4xsi64>}>)
      -> tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}> {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
    %c4  = arith.constant 4 : index
    %c40 = arith.constant 40 : index
    %h_dim = tensor.dim %arg0, %c1 : tensor<1x40x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 40, 40, 16]> : tensor<4xsi64>}>
    %w_dim = tensor.dim %arg0, %c2 : tensor<1x40x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 40, 40, 16]> : tensor<4xsi64>}>
    %init = tensor.empty(%w_dim) : tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>
    %result = scf.for %h = %c0 to %h_dim step %c4 iter_args(%acc = %init)
                -> (tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>) {
      %inner = scf.for %w = %c0 to %w_dim step %c4 iter_args(%acc2 = %acc)
                 -> (tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>) {
        %slice = tensor.extract_slice %arg0[0, %h, %w, 0] [1, 4, 4, 16] [1, 1, 1, 1]
                 : tensor<1x40x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 40, 40, 16]> : tensor<4xsi64>}> to tensor<1x4x4x16xf16>
        %out = func.call @tile_func(%slice) : (tensor<1x4x4x16xf16>) -> tensor<1x4x4x16xf16>
        %inserted = tensor.insert_slice %out into %acc2[0, %h, %w, 0] [1, 4, 4, 16] [1, 1, 1, 1]
                    : tensor<1x4x4x16xf16> into tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted : tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %inner : tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %result : tensor<1x16x40x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 40, 40]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-DAG:       [[C40:%.+]] = arith.constant 40 : index
  // CHECK-DAG:       [[C8:%.+]] = arith.constant 8 : index
  // CHECK-DAG:       [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
  // CHECK-DAG:       [[C4:%.+]] = arith.constant 4 : index
  // CHECK:       [[DIM:%.+]] = tensor.dim [[ARG0]], {{%.+}}
  // CHECK:       [[INIT:%.+]] = tensor.empty([[DIM]])
  // CHECK:       [[MERGED_RESULT:%.+]] = scf.for {{%.+}} = [[C0]] to [[C40]] step [[C8]] iter_args({{%.+}} = [[INIT]])
  // CHECK:         {{%.+}} = scf.for {{%.+}} = [[C0]] to [[DIM]] step [[C4]] iter_args({{%.+}} = {{%.+}})
  // CHECK:           tensor.extract_slice [[ARG0]][0, {{%.+}}, {{%.+}}, 0] [1, 8, 4, 16] [1, 1, 1, 1]
  // CHECK:           func.call @merged_vpu_func_1
  // CHECK:           tensor.insert_slice {{%.+}} into {{%.+}}[0, {{%.+}}, {{%.+}}, 0] [1, 8, 4, 16] [1, 1, 1, 1]
  // CHECK:         }
  // CHECK:       }
}
