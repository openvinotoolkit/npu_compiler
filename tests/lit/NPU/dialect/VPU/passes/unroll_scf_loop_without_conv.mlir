//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,2,1" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module {
  net.NetworkInfo entryPoint : @SimpleLoopUnroll inputsInfo : {
    DataInfo "input" : tensor<1x32x64x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x64xf16>
  }

  // CHECK-LABEL: @SimpleLoopUnroll
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

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[RESULT:.*]] = scf.for %[[ARG1:.*]] = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%[[ARG2:.*]] = %[[EMPTY]]) -> (tensor<1x32x64x64xf16, {order = #NHWC}>) {
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[MUL:.*]] = arith.muli %[[C2]], %[[C1]] : index
  // CHECK:   %[[ADD:.*]] = arith.addi %[[ARG1]], %[[MUL]] : index
  // CHECK:   %[[EXTRACT1:.*]] = tensor.extract_slice %arg0[0, 0, %[[ARG1]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK:   %[[INSERT1:.*]] = tensor.insert_slice %[[EXTRACT1]] into %[[ARG2]][0, 0, %[[ARG1]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK:   %[[EXTRACT2:.*]] = tensor.extract_slice %arg0[0, 0, %[[ADD]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x2x64xf16, {order = #NHWC}>
  // CHECK:   %[[INSERT2:.*]] = tensor.insert_slice %[[EXTRACT2]] into %[[INSERT1]][0, 0, %[[ADD]], 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf16, {order = #NHWC}> into tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK:   scf.yield %[[INSERT2]] : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return %[[RESULT]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @TwoDimLoopUnroll inputsInfo : {
    DataInfo "input" : tensor<1x32x64x96xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x32x64x96xf16>
  }

  // CHECK-LABEL: @TwoDimLoopUnroll
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

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: %[[C64:.*]] = arith.constant 64 : index
  // CHECK: %[[C96:.*]] = arith.constant 96 : index
  // CHECK: %[[C2_0:.*]] = arith.constant 2 : index
  // CHECK: %[[C3_0:.*]] = arith.constant 3 : index
  // CHECK: %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK: %[[C4_0:.*]] = arith.constant 4 : index
  // CHECK: %[[RESULT_0:.*]] = scf.for %[[ARG1_0:.*]] = %[[C0_0]] to %[[C64]] step %[[C4_0]] iter_args(%[[ARG2_0:.*]] = %[[EMPTY_0]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // CHECK:   %[[C1_0:.*]] = arith.constant 1 : index
  // CHECK:   %[[MUL_0:.*]] = arith.muli %[[C2_0]], %[[C1_0]] : index
  // CHECK:   %[[ADD_0:.*]] = arith.addi %[[ARG1_0]], %[[MUL_0]] : index
  // CHECK:   %[[INNER_RESULT:.*]] = scf.for %[[ARG3:.*]] = %[[C0_0]] to %[[C96]] step %[[C3_0]] iter_args(%[[ARG4:.*]] = %[[ARG2_0]]) -> (tensor<1x32x64x96xf16, {order = #NHWC}>) {
  // CHECK:     %[[EXTRACT1_0:.*]] = tensor.extract_slice %arg0[0, 0, %[[ARG1_0]], %[[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     %[[INSERT1_0:.*]] = tensor.insert_slice %[[EXTRACT1_0]] into %[[ARG4]][0, 0, %[[ARG1_0]], %[[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     %[[EXTRACT2_0:.*]] = tensor.extract_slice %arg0[0, 0, %[[ADD_0]], %[[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x64x96xf16, {order = #NHWC}> to tensor<1x32x2x3xf16, {order = #NHWC}>
  // CHECK:     %[[INSERT2_0:.*]] = tensor.insert_slice %[[EXTRACT2_0]] into %[[ARG4]][0, 0, %[[ADD_0]], %[[ARG3]]] [1, 32, 2, 3] [1, 1, 1, 1] : tensor<1x32x2x3xf16, {order = #NHWC}> into tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:     scf.yield %[[INSERT1_0]] : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK:   }
  // CHECK:   scf.yield %[[INNER_RESULT]] : tensor<1x32x64x96xf16, {order = #NHWC}>
  // CHECK: }
  // CHECK: return %[[RESULT_0]] : tensor<1x32x64x96xf16, {order = #NHWC}>
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

  // CHECK:    func.func @merged_vpu_func_0([[ARG0:%.*]]: tensor<1x16x200x1000xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x200x1000xf16, {order = #NHWC}>) -> tensor<1x16x200x1000xf16, {order = #NHWC}> {
  // CHECK:      [[SLICE0:%.*]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[SLICE2:%.*]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE0:%.*]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[SLICE1:%.*]] = VPU.Slice [[ARG0]] [0, 0, 100, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[SLICE3:%.*]] = VPU.Slice [[ARG1]] [0, 0, 100, 0] [1, 16, 100, 1000] : tensor<1x16x200x1000xf16, {order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[ELTWISE1:%.*]] = VPU.NCE.Eltwise([[SLICE1]], [[SLICE3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      [[CONCAT:%.*]] = VPU.Concat([[ELTWISE0]], [[ELTWISE1]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x100x1000xf16, {order = #NHWC}>, tensor<1x16x100x1000xf16, {order = #NHWC}> -> tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:      return [[CONCAT]] : tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:    }

  // CHECK:    func.func @main_func0_static([[ARG0:%.*]]: tensor<1x16x100x1000xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x100x1000xf16, {order = #NHWC}>) -> tensor<1x16x100x1000xf16, {order = #NHWC}> {
  // CHECK:      [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {{{.*}}op_type = #VPU.eltwise_type<ADD>{{.*}}} -> tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:      return [[ELTWISE]] : tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:    }
  func.func @main_func0_static(%arg0: tensor<1x16x100x1000xf16, {order = #NHWC}>, %arg1: tensor<1x16x100x1000xf16, {order = #NHWC}>) -> tensor<1x16x100x1000xf16, {order = #NHWC}> {
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

  // CHECK:   func.func @main(%[[ARG0:.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %[[ARG1:.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:     %[[C100:.*]] = arith.constant 100 : index
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[C2:.*]] = arith.constant 2 : index
  // CHECK:     %[[DIM:.*]] = tensor.dim %arg0, %[[C2]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     %[[SUB1:.*]] = arith.subi %[[DIM]], %[[C0]] : index
  // CHECK:     %[[C1:.*]] = arith.constant 1 : index
  // CHECK:     %[[SUB2:.*]] = arith.subi %[[C100]], %[[C1]] : index
  // CHECK:     %[[ADD:.*]] = arith.addi %[[SUB1]], %[[SUB2]] : index
  // CHECK:     %[[DIV:.*]] = arith.divui %[[ADD]], %[[C100]] : index
  // CHECK:     %[[C2_0:.*]] = arith.constant 2 : index
  // CHECK:     %[[REM:.*]] = arith.remsi %[[DIV]], %[[C2_0]] : index
  // CHECK:     %[[SUB3:.*]] = arith.subi %[[DIV]], %[[REM]] : index
  // CHECK:     %[[MUL1:.*]] = arith.muli %[[SUB3]], %[[C100]] : index
  // CHECK:     %[[ADD2:.*]] = arith.addi %[[C0]], %[[MUL1]] : index
  // CHECK:     %[[MUL2:.*]] = arith.muli %[[C100]], %[[C2_0]] : index
  // CHECK:     %[[RESULT_1:.*]] = scf.for %[[ARG2_1:.*]] = %[[C0]] to %[[ADD2]] step %[[MUL2]] iter_args(%[[ARG3_1:.*]] = %[[EMPTY]]) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:       %[[MIN1:.*]] = affine.min #map(%[[ARG2_1]])[%[[DIM]]]
  // CHECK:       %[[CMPI1:.*]] = arith.cmpi eq, %[[ARG2_1]], %[[C0]] : index
  // CHECK:       %[[C1_1:.*]] = arith.constant 1 : index
  // CHECK:       %[[MUL3:.*]] = arith.muli %[[C100]], %[[C1_1]] : index
  // CHECK:       %[[ADD3:.*]] = arith.addi %[[ARG2_1]], %[[MUL3]] : index
  // CHECK:       %[[MIN2:.*]] = affine.min #map(%[[ADD3]])[%[[DIM]]]
  // CHECK:       %[[CMPI2:.*]] = arith.cmpi eq, %[[ADD3]], %[[C0]] : index
  // CHECK:       %[[IF1:.*]] = scf.if %[[CMPI2]] -> (index) {
  // CHECK:         {{.*}}
  // CHECK:       }
  // CHECK:       %[[APPLY1:.*]] = affine.apply #map1(%[[ADD3]], %[[IF1]])
  // CHECK:       %[[IF4:.*]] = scf.if %[[CMPI1]] -> (index) {
  // CHECK:         {{.*}}
  // CHECK:       }
  // CHECK:       %[[APPLY2:.*]] = affine.apply #map1(%[[ARG2_1]], %[[IF4]])
  // CHECK:       %[[EXTRACT1_1:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[APPLY2]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:       %[[EXTRACT2_1:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[APPLY2]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:       %[[CALL:.*]] = func.call @merged_vpu_func_0(%[[EXTRACT1_1]], %[[EXTRACT2_1]]) : (tensor<1x16x200x1000xf16, {order = #NHWC}>, tensor<1x16x200x1000xf16, {order = #NHWC}>) -> tensor<1x16x200x1000xf16, {order = #NHWC}>
  // CHECK:       %[[INSERT:.*]] = tensor.insert_slice %[[CALL]] into %[[ARG3_1]][0, 0, %[[APPLY2]], 0] [1, 16, 200, 1000] [1, 1, 1, 1] : tensor<1x16x200x1000xf16, {order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       scf.yield %[[INSERT]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     }
  // CHECK:     %[[RESULT_2:.*]] = scf.for %[[ARG2_2:.*]] = %[[ADD2]] to %[[DIM]] step %[[C100]] iter_args(%[[ARG3_2:.*]] = %[[RESULT_1]]) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:       %[[MIN3:.*]] = affine.min #map(%[[ARG2_2]])[%[[DIM]]]
  // CHECK:       %[[CMPI3:.*]] = arith.cmpi eq, %[[ARG2_2]], %[[C0]] : index
  // CHECK:       %[[IF5:.*]] = scf.if %[[CMPI3]] -> (index) {
  // CHECK:         {{.*}}
  // CHECK:       }
  // CHECK:       %[[APPLY3:.*]] = affine.apply #map1(%[[ARG2_2]], %[[IF5]])
  // CHECK:       %[[EXTRACT3:.*]] = tensor.extract_slice %[[ARG0]][0, 0, %[[APPLY3]], 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:       %[[EXTRACT4:.*]] = tensor.extract_slice %[[ARG1]][0, 0, %[[APPLY3]], 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:       %[[CALL2:.*]] = func.call @main_func0_static(%[[EXTRACT3]], %[[EXTRACT4]]) : (tensor<1x16x100x1000xf16, {order = #NHWC}>, tensor<1x16x100x1000xf16, {order = #NHWC}>) -> tensor<1x16x100x1000xf16, {order = #NHWC}>
  // CHECK:       %[[INSERT2:.*]] = tensor.insert_slice %[[CALL2]] into %[[ARG3_2]][0, 0, %[[APPLY3]], 0] [1, 16, 100, 1000] [1, 1, 1, 1] : tensor<1x16x100x1000xf16, {order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       scf.yield %[[INSERT2]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     }
  // CHECK:     return %[[RESULT_2]] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   }
}
