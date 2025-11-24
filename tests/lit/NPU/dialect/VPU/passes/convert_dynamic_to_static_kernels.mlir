//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-dynamic-to-static-kernels  %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
#map1 = affine_map<(d0, d1) -> (d0 - d1)>

// CHECK: #map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0 - d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }

  // CHECK:    func.func @main_func0_static([[ARG0:%.*]]: tensor<1x16x720x100xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x720x100xf16, {order = #NHWC}>) -> tensor<1x16x720x100xf16, {order = #NHWC}> {
  // CHECK:      [[OUTPUT:%.*]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:      return [[OUTPUT]] : tensor<1x16x720x100xf16, {order = #NHWC}>
  // CHECK:    }

  func.func private @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
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
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %5] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %extracted_slice : tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %5] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %cast_2 = tensor.cast %extracted_slice_1 : tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %6 = func.call @main_func0(%cast, %cast_2) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %cast_3 = tensor.cast %6 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast_3 into %arg3[0, 0, 0, %5] [1, 16, 720, 100] [1, 1, 1, 1] : tensor<1x16x720x100xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }
  // CHECK:    func.func @main([[ARG0:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, [[ARG1:%.*]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:      [[C100:%.*]] = arith.constant 100 : index
  // CHECK:      [[C0:%.*]] = arith.constant 0 : index
  // CHECK:      [[C3:%.*]] = arith.constant 3 : index
  // CHECK:      [[DIM:%.*]] = tensor.dim [[ARG0]], [[C3]]
  // CHECK:      [[EMPTY:%.*]] = tensor.empty([[DIM]])
  // CHECK:      [[FOR:%.*]] = scf.for [[ARG2:%.*]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[ARG3:%.*]] = [[EMPTY]])
  // CHECK:        [[MIN:%.*]] = affine.min #map([[ARG2]]){{\[}}[[DIM]]{{\]}}
  // CHECK:        [[CMPI:%.*]] = arith.cmpi eq, [[ARG2]], [[C0]]
  // CHECK:        [[IF:%.*]] = scf.if [[CMPI]]
  // CHECK:          [[CMPI_1:%.*]] = arith.cmpi sge, [[MIN]], [[C100]]
  // CHECK:          cf.assert [[CMPI_1]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:          scf.yield [[C0]]
  // CHECK:        } else {
  // CHECK:          [[ADDI:%.*]] = arith.addi [[ARG2]], [[C100]]
  // CHECK:          [[CMPI_2:%.*]] = arith.cmpi slt, [[ADDI]], [[DIM]]
  // CHECK:          [[IF_1:%.*]] = scf.if [[CMPI_2]]
  // CHECK:            scf.yield [[C0]]
  // CHECK:          } else {
  // CHECK:            [[CMPI_3:%.*]] = arith.cmpi eq, [[ADDI]], [[DIM]]
  // CHECK:            [[IF_2:%.*]] = scf.if [[CMPI_3]]
  // CHECK:              scf.yield [[C0]]
  // CHECK:            } else {
  // CHECK:              [[SUBI:%.*]] = arith.subi [[DIM]], [[ARG2]]
  // CHECK:              [[SUBI_1:%.*]] = arith.subi [[SUBI]], [[C100]]
  // CHECK:              scf.yield [[SUBI_1]]
  // CHECK:            }
  // CHECK:            scf.yield [[IF_2]]
  // CHECK:          }
  // CHECK:          scf.yield [[IF_1]]
  // CHECK:        }
  // CHECK:        [[APPLY:%.*]] = affine.apply #map1([[ARG2]], [[IF]])
  // CHECK:        [[SLICE:%.*]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[APPLY]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK:        [[SLICE_1:%.*]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[APPLY]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK:        [[CALL:%.*]] = func.call @main_func0_static([[SLICE]], [[SLICE_1]])
  // CHECK:        [[INSERT:%.*]] = tensor.insert_slice [[CALL]] into [[ARG3]][0, 0, 0, [[APPLY]]] [1, 16, 720, 100] [1, 1, 1, 1]
  // CHECK:        scf.yield [[INSERT]]
  // CHECK:      }
  // CHECK:      return [[FOR]]
  // CHECK:    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map5 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
#map6 = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
#map7 = affine_map<(d0)[s0] -> (d0 + s0 - 258)>

// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
// CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
// CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
// CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
// CHECK: #[[$MAP6:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
// CHECK: #[[$MAP7:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 258)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn inputsInfo : {
    DataInfo "input" : tensor<1x32x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x64xf16>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn_func0_dims_H_cases_0_static([[ARG0:%.*]]: tensor<1x32x258x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x258x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK: }

  func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn_func0_dims_H_cases_2_static([[ARG0:%.*]]: tensor<1x32x257x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK: }

  func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_2(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn_func0_dims_H_cases_1_static([[ARG0:%.*]]: tensor<1x32x257x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}> {
  // CHECK:   [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:   [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x257x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:   return [[CONV]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK: }

  func.func private @ApplyTilingNCEConvDyn_func0_dims_H_cases_1(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

func.func @ApplyTilingNCEConvDyn(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %false = arith.constant false
    %c257 = arith.constant 257 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %1 = arith.remui %dim, %c256 : index
    %2 = arith.remui %dim, %c257 : index
    %3 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %4 = affine.min #map(%arg1)[%dim]
      %5 = affine.max #map1(%arg1)
      %6 = affine.max #map2(%arg1)
      %7 = affine.min #map3()[%6]
      %8 = affine.max #map4(%4, %5)
      %9 = affine.min #map3()[%8]
      %10 = affine.apply #map5(%4, %7, %9)
      %11 = arith.cmpi eq, %arg1, %c0 : index
      %12 = scf.if %11 -> (index) {
        %16 = arith.cmpi sge, %4, %c256 : index
        cf.assert %16, "Not enough elements to backtrack in scf.for loop"
        scf.yield %arg1 : index
      } else {
        %16 = arith.addi %arg1, %c256 : index
        %17 = arith.cmpi slt, %16, %dim : index
        %18 = scf.if %17 -> (index) {
          scf.yield %arg1 : index
        } else {
          %19 = arith.cmpi eq, %16, %dim : index
          %20 = scf.if %19 -> (index) {
            scf.yield %arg1 : index
          } else {
            %21 = affine.apply #map6(%arg1)[%1]
            scf.yield %21 : index
          }
          scf.yield %20 : index
        }
        scf.yield %18 : index
      }
      %13 = arith.cmpi eq, %5, %c0 : index
      %14:2 = scf.if %13 -> (index, index) {
        %16 = arith.cmpi sge, %10, %c257 : index
        cf.assert %16, "Not enough elements to backtrack in scf.for loop"
        %17 = arith.cmpi eq, %10, %dim : index
        %18 = arith.select %17, %c3, %c2 : index
        scf.yield %18, %5 : index, index
      } else {
        %16 = arith.addi %5, %c257 : index
        %17 = arith.cmpi slt, %16, %dim : index
        %18 = arith.select %17, %c0, %c1 : index
        %19 = scf.if %17 -> (index) {
          scf.yield %5 : index
        } else {
          %20 = arith.cmpi eq, %16, %dim : index
          %21 = scf.if %20 -> (index) {
            scf.yield %5 : index
          } else {
            %22 = affine.apply #map7(%5)[%2]
            scf.yield %22 : index
          }
          scf.yield %21 : index
        }
        scf.yield %18, %19 : index, index
      }
      %15 = scf.index_switch %14#0 -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      case 0 {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %14#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
        %cast_2 = tensor.cast %extracted_slice : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
        %16 = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0(%cast_2) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %16 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      }
      case 1 {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %14#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
        %cast_2 = tensor.cast %extracted_slice : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
        %16 = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_1(%cast_2) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %16 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      }
      case 2 {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %14#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
        %cast_2 = tensor.cast %extracted_slice : tensor<1x32x257x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>
        %16 = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_2(%cast_2) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %16 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      }
      default {
        cf.assert %false, "Unsupported case"
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %14#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
        %cast_2 = tensor.cast %extracted_slice : tensor<1x32x258x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>
        %16 = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0(%cast_2) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %16 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      }
      %cast = tensor.cast %15 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %12, 0] [1, 256, 256, 64] [1, 1, 1, 1] : tensor<1x256x256x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %3 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: func.func @ApplyTilingNCEConvDyn([[ARG0:%.*]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
  // CHECK:   [[FALSE:%.*]] = arith.constant false
  // CHECK:   [[C257:%.*]] = arith.constant 257 : index
  // CHECK:   [[C3:%.*]] = arith.constant 3 : index
  // CHECK:   [[C1:%.*]] = arith.constant 1 : index
  // CHECK:   [[C256:%.*]] = arith.constant 256 : index
  // CHECK:   [[C0:%.*]] = arith.constant 0 : index
  // CHECK:   [[C2:%.*]] = arith.constant 2 : index
  // CHECK:   [[DIM:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   [[EMPTY:%.*]] = tensor.empty([[DIM]]) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   [[REM1:%.*]] = arith.remui [[DIM]], [[C256]] : index
  // CHECK:   [[REM2:%.*]] = arith.remui [[DIM]], [[C257]] : index
  // CHECK:   [[FOR:%.*]] = scf.for [[IV:%.*]] = [[C0]] to [[DIM]] step [[C256]] iter_args([[ITER_ARG:%.*]] = [[EMPTY]]) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:     [[MIN1:%.*]] = affine.min #[[$MAP]]([[IV]])[[[DIM]]]
  // CHECK:     [[MAX1:%.*]] = affine.max #[[$MAP1]]([[IV]])
  // CHECK:     [[MAX2:%.*]] = affine.max #[[$MAP2]]([[IV]])
  // CHECK:     [[MIN2:%.*]] = affine.min #[[$MAP3]]()[[[MAX2]]]
  // CHECK:     [[MAX3:%.*]] = affine.max #[[$MAP4]]([[MIN1]], [[MAX1]])
  // CHECK:     [[MIN3:%.*]] = affine.min #[[$MAP3]]()[[[MAX3]]]
  // CHECK:     [[APPLY:%.*]] = affine.apply #[[$MAP5]]([[MIN1]], [[MIN2]], [[MIN3]])
  // CHECK:     [[CMP1:%.*]] = arith.cmpi eq, [[IV]], [[C0]] : index
  // CHECK:     [[IF1:%.*]] = scf.if [[CMP1]] -> (index) {
  // CHECK:       [[CMP2:%.*]] = arith.cmpi sge, [[MIN1]], [[C256]] : index
  // CHECK:       cf.assert [[CMP2]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:       scf.yield [[IV]] : index
  // CHECK:     } else {
  // CHECK:       [[ADD:%.*]] = arith.addi [[IV]], [[C256]] : index
  // CHECK:       [[CMP3:%.*]] = arith.cmpi slt, [[ADD]], [[DIM]] : index
  // CHECK:       [[IF2:%.*]] = scf.if [[CMP3]] -> (index) {
  // CHECK:         scf.yield [[IV]] : index
  // CHECK:       } else {
  // CHECK:         [[CMP4:%.*]] = arith.cmpi eq, [[ADD]], [[DIM]] : index
  // CHECK:         [[IF3:%.*]] = scf.if [[CMP4]] -> (index) {
  // CHECK:           scf.yield [[IV]] : index
  // CHECK:         } else {
  // CHECK:           [[APPLY2:%.*]] = affine.apply #[[$MAP6]]([[IV]])[[[REM1]]]
  // CHECK:           scf.yield [[APPLY2]] : index
  // CHECK:         }
  // CHECK:         scf.yield [[IF3]] : index
  // CHECK:       }
  // CHECK:       scf.yield [[IF2]] : index
  // CHECK:     }
  // CHECK:     [[CMP5:%.*]] = arith.cmpi eq, [[MAX1]], [[C0]] : index
  // CHECK:     [[IF4:%.*]]:2 = scf.if [[CMP5]] -> (index, index) {
  // CHECK:       [[CMP6:%.*]] = arith.cmpi sge, [[APPLY]], [[C257]] : index
  // CHECK:       cf.assert [[CMP6]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:       [[CMP7:%.*]] = arith.cmpi eq, [[APPLY]], [[DIM]] : index
  // CHECK:       [[SELECT1:%.*]] = arith.select [[CMP7]], [[C3]], [[C2]] : index
  // CHECK:       scf.yield [[SELECT1]], [[MAX1]] : index, index
  // CHECK:     } else {
  // CHECK:       [[ADD2:%.*]] = arith.addi [[MAX1]], [[C257]] : index
  // CHECK:       [[CMP8:%.*]] = arith.cmpi slt, [[ADD2]], [[DIM]] : index
  // CHECK:       [[SELECT2:%.*]] = arith.select [[CMP8]], [[C0]], [[C1]] : index
  // CHECK:       [[IF5:%.*]] = scf.if [[CMP8]] -> (index) {
  // CHECK:         scf.yield [[MAX1]] : index
  // CHECK:       } else {
  // CHECK:         [[CMP9:%.*]] = arith.cmpi eq, [[ADD2]], [[DIM]] : index
  // CHECK:         [[IF6:%.*]] = scf.if [[CMP9]] -> (index) {
  // CHECK:           scf.yield [[MAX1]] : index
  // CHECK:         } else {
  // CHECK:           [[APPLY3:%.*]] = affine.apply #[[$MAP7]]([[MAX1]])[[[REM2]]]
  // CHECK:           scf.yield [[APPLY3]] : index
  // CHECK:         }
  // CHECK:         scf.yield [[IF6]] : index
  // CHECK:       }
  // CHECK:       scf.yield [[SELECT2]], [[IF5]] : index, index
  // CHECK:     }
  // CHECK:     [[INDEX_SWITCH:%.*]] = scf.index_switch [[IF4]]#0 -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:     case 0 {
  // CHECK:       [[SLICE_0:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[IF4]]#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:       [[CALL_0:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0_static([[SLICE_0]]) : (tensor<1x32x258x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_0]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 1 {
  // CHECK:       [[SLICE_1:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[IF4]]#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
  // CHECK:       [[CALL_1:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_1_static([[SLICE_1]]) : (tensor<1x32x257x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_1]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 2 {
  // CHECK:       [[SLICE_2:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[IF4]]#1, 0] [1, 32, 257, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x64xf16, {order = #NHWC}>
  // CHECK:       [[CALL_2:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_2_static([[SLICE_2]]) : (tensor<1x32x257x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_2]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     default {
  // CHECK:       cf.assert [[FALSE]], "Unsupported case"
  // CHECK:       [[SLICE_DEFAULT:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[IF4]]#1, 0] [1, 32, 258, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x64xf16, {order = #NHWC}>
  // CHECK:       [[CALL_DEFAULT:%.*]] = func.call @ApplyTilingNCEConvDyn_func0_dims_H_cases_0_static([[SLICE_DEFAULT]]) : (tensor<1x32x258x64xf16, {order = #NHWC}>) -> tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_DEFAULT]] : tensor<1x256x256x64xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     [[INSERT:%.*]] = tensor.insert_slice [[INDEX_SWITCH]] into [[ITER_ARG]][0, 0, [[IF1]], 0] [1, 256, 256, 64] [1, 1, 1, 1] : tensor<1x256x256x64xf16, {order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     scf.yield [[INSERT]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   }
  // CHECK:   return [[FOR]] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: }
}

// -----

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
// CHECK: #[[$MAP8:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 160)>
// CHECK: #[[$MAP9:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
// CHECK: #[[$MAP10:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 162)>
// CHECK: #[[$MAP11:.*]] = affine_map<(d0)[s0] -> (d0 + s0 - 258)>

// CHECK-LABEL: @ApplyTilingNCEConvDyn2D
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn2D inputsInfo : {
    DataInfo "input" : tensor<1x32x?x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x?xf16>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK: [[CST:%.*]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK: [[CONV:%.*]] = VPU.NCE.Convolution([[ARG0]], [[CST]]) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK: return [[CONV]]
  // CHECK: }
  func.func private @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
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
    %3 = arith.remui %dim, %c257 : index
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
          %33 = arith.cmpi eq, %14, %dim : index
          %34 = arith.select %33, %c3, %c2 : index
          scf.yield %34, %9 : index, index
        } else {
          %32 = arith.addi %9, %c257 : index
          %33 = arith.cmpi slt, %32, %dim : index
          %34 = arith.select %33, %c0, %c1 : index
          %35 = scf.if %33 -> (index) {
            scf.yield %9 : index
          } else {
            %36 = arith.cmpi eq, %32, %dim : index
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
        %31 = scf.index_switch %30 -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        case 0 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 1 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 2 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x258x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 4 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 5 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 6 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 8 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 9 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        case 10 {
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x257x161xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 257, 161]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        default {
          cf.assert %false, "Unsupported case"
          %extracted_slice = tensor.extract_slice %arg0[0, 0, %28#1, %27#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
          %cast_3 = tensor.cast %extracted_slice : tensor<1x32x258x162xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>
          %32 = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00(%cast_3) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 258, 162]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
          scf.yield %32 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        }
        %cast = tensor.cast %31 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg4[0, 0, %24, %23] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %6 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %5 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK-LABEL: @ApplyTilingNCEConvDyn2D
  // CHECK-SAME: ([[ARG0:%.*]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: [[FALSE:%.*]] = arith.constant false
  // CHECK: [[C162:%.*]] = arith.constant 161 : index
  // CHECK: [[C258:%.*]] = arith.constant 257 : index
  // CHECK: [[C1:%.*]] = arith.constant 1 : index
  // CHECK: [[C160:%.*]] = arith.constant 160 : index
  // CHECK: [[C256:%.*]] = arith.constant 256 : index
  // CHECK: [[C3:%.*]] = arith.constant 3 : index
  // CHECK: [[C0:%.*]] = arith.constant 0 : index
  // CHECK: [[C2:%.*]] = arith.constant 2 : index
  // CHECK: [[DIM_H:%.*]] = tensor.dim [[ARG0]], [[C2]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: [[DIM_W:%.*]] = tensor.dim [[ARG0]], [[C3]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: [[EMPTY:%.*]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: [[REM_H256:%.*]] = arith.remui [[DIM_H]], [[C256]] : index
  // CHECK: [[REM_W160:%.*]] = arith.remui [[DIM_W]], [[C160]] : index
  // CHECK: [[REM_H258:%.*]] = arith.remui [[DIM_H]], [[C258]] : index
  // CHECK: [[REM_W162:%.*]] = arith.remui [[DIM_W]], [[C162]] : index
  // CHECK: [[OUTER_FOR:%.*]] = scf.for [[IV_H:.*]] = [[C0]] to [[DIM_H]] step [[C256]] iter_args([[ACC0:.*]] = [[EMPTY]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:   [[INNER_FOR:%.*]] = scf.for [[IV_W:.*]] = [[C0]] to [[DIM_W]] step [[C160]] iter_args([[ACC1:.*]] = [[ACC0]]) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:     [[H_MIN:%.*]] = affine.min #[[$MAP]]([[IV_H]])[[[DIM_H]]]
  // CHECK:     [[W_MIN:%.*]] = affine.min #[[$MAP1]]([[IV_W]])[[[DIM_W]]]
  // CHECK:     [[H_MAX0:%.*]] = affine.max #[[$MAP2]]([[IV_H]])
  // CHECK:     [[H_MAX1:%.*]] = affine.max #[[$MAP3]]([[IV_H]])
  // CHECK:     [[H_MIN1:%.*]] = affine.min #[[$MAP4]]()[[[H_MAX1]]]
  // CHECK:     [[H_MAX2:%.*]] = affine.max #[[$MAP5]]([[H_MIN]], [[H_MAX0]])
  // CHECK:     [[H_MIN2:%.*]] = affine.min #[[$MAP4]]()[[[H_MAX2]]]
  // CHECK:     [[H_APPLY:%.*]] = affine.apply #[[$MAP6]]([[H_MIN]], [[H_MIN1]], [[H_MIN2]])
  // CHECK:     [[W_MAX0:%.*]] = affine.max #[[$MAP2]]([[IV_W]])
  // CHECK:     [[W_MAX1:%.*]] = affine.max #[[$MAP3]]([[IV_W]])
  // CHECK:     [[W_MIN1:%.*]] = affine.min #[[$MAP4]]()[[[W_MAX1]]]
  // CHECK:     [[W_MAX2:%.*]] = affine.max #[[$MAP7]]([[W_MIN]], [[W_MAX0]])
  // CHECK:     [[W_MIN2:%.*]] = affine.min #[[$MAP4]]()[[[W_MAX2]]]
  // CHECK:     [[W_APPLY:%.*]] = affine.apply #[[$MAP6]]([[W_MIN]], [[W_MIN1]], [[W_MIN2]])
  // CHECK:     [[IS_H0:%.*]] = arith.cmpi eq, [[IV_H]], [[C0]] : index
  // CHECK:     [[IS_W0:%.*]] = arith.cmpi eq, [[IV_W]], [[C0]] : index
  // CHECK:     [[IF_W:%.*]] = scf.if [[IS_W0]] -> (index) {
  // CHECK:       [[ASSERT_W:%.*]] = arith.cmpi sge, [[W_MIN]], [[C160]] : index
  // CHECK:       cf.assert [[ASSERT_W]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:       scf.yield [[IV_W]] : index
  // CHECK:     } else {
  // CHECK:       [[W_PLUS:%.*]] = arith.addi [[IV_W]], [[C160]] : index
  // CHECK:       [[LT_W:%.*]] = arith.cmpi slt, [[W_PLUS]], [[DIM_W]] : index
  // CHECK:       [[IF_W1:%.*]] = scf.if [[LT_W]] -> (index) {
  // CHECK:         scf.yield [[IV_W]] : index
  // CHECK:       } else {
  // CHECK:         [[EQ_W:%.*]] = arith.cmpi eq, [[W_PLUS]], [[DIM_W]] : index
  // CHECK:         [[IF_W2:%.*]] = scf.if [[EQ_W]] -> (index) {
  // CHECK:           scf.yield [[IV_W]] : index
  // CHECK:         } else {
  // CHECK:           [[MAP8_RES:%.*]] = affine.apply #[[$MAP8]]([[IV_W]])[[[REM_W160]]]
  // CHECK:           scf.yield [[MAP8_RES]] : index
  // CHECK:         }
  // CHECK:         scf.yield [[IF_W2]] : index
  // CHECK:       }
  // CHECK:       scf.yield [[IF_W1]] : index
  // CHECK:     }
  // CHECK:     [[IF_H:%.*]] = scf.if [[IS_H0]] -> (index) {
  // CHECK:       [[ASSERT_H:%.*]] = arith.cmpi sge, [[H_MIN]], [[C256]] : index
  // CHECK:       cf.assert [[ASSERT_H]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:       scf.yield [[IV_H]] : index
  // CHECK:     } else {
  // CHECK:       [[H_PLUS:%.*]] = arith.addi [[IV_H]], [[C256]] : index
  // CHECK:       [[LT_H:%.*]] = arith.cmpi slt, [[H_PLUS]], [[DIM_H]] : index
  // CHECK:       [[IF_H1:%.*]] = scf.if [[LT_H]] -> (index) {
  // CHECK:         scf.yield [[IV_H]] : index
  // CHECK:       } else {
  // CHECK:         [[EQ_H:%.*]] = arith.cmpi eq, [[H_PLUS]], [[DIM_H]] : index
  // CHECK:         [[IF_H2:%.*]] = scf.if [[EQ_H]] -> (index) {
  // CHECK:           scf.yield [[IV_H]] : index
  // CHECK:         } else {
  // CHECK:           [[MAP9_RES:%.*]] = affine.apply #[[$MAP9]]([[IV_H]])[[[REM_H256]]]
  // CHECK:           scf.yield [[MAP9_RES]] : index
  // CHECK:         }
  // CHECK:         scf.yield [[IF_H2]] : index
  // CHECK:       }
  // CHECK:       scf.yield [[IF_H1]] : index
  // CHECK:     }
  // CHECK:     [[IS_HMAX0_0:%.*]] = arith.cmpi eq, [[H_MAX0]], [[C0]] : index
  // CHECK:     [[IS_WMAX0_0:%.*]] = arith.cmpi eq, [[W_MAX0]], [[C0]] : index
  // CHECK-NEXT:     [[SW_W:%.*]]:2 = scf.if [[IS_WMAX0_0]] -> (index, index) {
  // CHECK-NEXT:       [[ASSERT_W_TILE:%.*]] = arith.cmpi sge, [[W_APPLY]], [[C162]] : index
  // CHECK-NEXT:       cf.assert [[ASSERT_W_TILE]], "Not enough elements to backtrack in scf.for loop"
  // CHECK-NEXT:       [[EQ_WAPPLY:%.*]] = arith.cmpi eq, [[W_APPLY]], [[DIM_W]] : index
  // CHECK-NEXT:       [[SEL_W_CASE:%.*]] = arith.select [[EQ_WAPPLY]], [[C3]], [[C2]] : index
  // CHECK-NEXT:       scf.yield [[SEL_W_CASE]], [[W_MAX0]] : index, index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[W_PLUS162:%.*]] = arith.addi [[W_MAX0]], [[C162]] : index
  // CHECK-NEXT:       [[LT_W2:%.*]] = arith.cmpi slt, [[W_PLUS162]], [[DIM_W]] : index
  // CHECK-NEXT:       [[SEL_W_DEF:%.*]] = arith.select [[LT_W2]], [[C0]], [[C1]] : index
  // CHECK-NEXT:       [[IF_W3:%.*]] = scf.if [[LT_W2]] -> (index) {
  // CHECK-NEXT:         scf.yield [[W_MAX0]] : index
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         [[EQ_W2:%.*]] = arith.cmpi eq, [[W_PLUS162]], [[DIM_W]] : index
  // CHECK-NEXT:         [[IF_W4:%.*]] = scf.if [[EQ_W2]] -> (index) {
  // CHECK-NEXT:           scf.yield [[W_MAX0]] : index
  // CHECK-NEXT:         } else {
  // CHECK-NEXT:           [[MAP10_RES:%.*]] = affine.apply #[[$MAP10]]([[W_MAX0]])[[[REM_W162]]]
  // CHECK-NEXT:           scf.yield [[MAP10_RES]] : index
  // CHECK-NEXT:         }
  // CHECK-NEXT:         scf.yield [[IF_W4]] : index
  // CHECK-NEXT:       }
  // CHECK-NEXT:       scf.yield [[SEL_W_DEF]], [[IF_W3]] : index, index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[SW_H:%.*]]:2 = scf.if [[IS_HMAX0_0]] -> (index, index) {
  // CHECK:       [[ASSERT_H_TILE:%.*]] = arith.cmpi sge, [[H_APPLY]], [[C258]] : index
  // CHECK:       cf.assert [[ASSERT_H_TILE]], "Not enough elements to backtrack in scf.for loop"
  // CHECK:       [[EQ_HAPPLY:%.*]] = arith.cmpi eq, [[H_APPLY]], [[DIM_H]] : index
  // CHECK:       [[SEL_H_CASE:%.*]] = arith.select [[EQ_HAPPLY]], [[C3]], [[C2]] : index
  // CHECK:       scf.yield [[SEL_H_CASE]], [[H_MAX0]] : index, index
  // CHECK:     } else {
  // CHECK:       [[H_PLUS258:%.*]] = arith.addi [[H_MAX0]], [[C258]] : index
  // CHECK:       [[LT_H2:%.*]] = arith.cmpi slt, [[H_PLUS258]], [[DIM_H]] : index
  // CHECK:       [[SEL_H_DEF:%.*]] = arith.select [[LT_H2]], [[C0]], [[C1]] : index
  // CHECK:       [[IF_H3:%.*]] = scf.if [[LT_H2]] -> (index) {
  // CHECK:         scf.yield [[H_MAX0]] : index
  // CHECK:       } else {
  // CHECK:         [[EQ_H2:%.*]] = arith.cmpi eq, [[H_PLUS258]], [[DIM_H]] : index
  // CHECK:         [[IF_H4:%.*]] = scf.if [[EQ_H2]] -> (index) {
  // CHECK:           scf.yield [[H_MAX0]] : index
  // CHECK:         } else {
  // CHECK:           [[MAP11_RES:%.*]] = affine.apply #[[$MAP11]]([[H_MAX0]])[[[REM_H258]]]
  // CHECK:           scf.yield [[MAP11_RES]] : index
  // CHECK:         }
  // CHECK:         scf.yield [[IF_H4]] : index
  // CHECK:       }
  // CHECK:       scf.yield [[SEL_H_DEF]], [[IF_H3]] : index, index
  // CHECK:     }
  // CHECK:     [[SHLI:%.*]] = arith.shli [[SW_H]]#0, [[C2]] : index
  // CHECK:     [[ORI:%.*]] = arith.ori [[SHLI]], [[SW_W]]#0 : index
  // CHECK:     [[INDEX_SWITCH:%.*]] = scf.index_switch [[ORI]] -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     case 0 {
  // CHECK:       [[SLICE_00:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:       [[CALL_00:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static([[SLICE_00]]) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_00]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 1 {
  // CHECK:       [[SLICE_01:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_01:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_01_static([[SLICE_01]]) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_01]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 2 {
  // CHECK:       [[SLICE_02:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 258, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_02:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_02_static([[SLICE_02]]) : (tensor<1x32x258x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_02]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 4 {
  // CHECK:       [[SLICE_10:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:       [[CALL_10:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_10_static([[SLICE_10]]) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_10]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 5 {
  // CHECK:       [[SLICE_11:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_11:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_11_static([[SLICE_11]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_11]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 6 {
  // CHECK:       [[SLICE_12:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_12:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_12_static([[SLICE_12]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_12]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 8 {
  // CHECK:       [[SLICE_20:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x162xf16, {order = #NHWC}>
  // CHECK:       [[CALL_20:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_20_static([[SLICE_20]]) : (tensor<1x32x257x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_20]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 9 {
  // CHECK:       [[SLICE_21:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_21:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_21_static([[SLICE_21]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_21]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     case 10 {
  // CHECK:       [[SLICE_22:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 257, 161] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x257x161xf16, {order = #NHWC}>
  // CHECK:       [[CALL_22:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_22_static([[SLICE_22]]) : (tensor<1x32x257x161xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_22]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     default {
  // CHECK:       cf.assert [[FALSE]], "Unsupported case"
  // CHECK:       [[SLICE_DEFAULT:%.*]] = tensor.extract_slice [[ARG0]][0, 0, [[SW_H]]#1, [[SW_W]]#1] [1, 32, 258, 162] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x258x162xf16, {order = #NHWC}>
  // CHECK:       [[CALL_DEFAULT:%.*]] = func.call @ApplyTilingNCEConvDyn2D_func0_dims_HW_cases_00_static([[SLICE_DEFAULT]]) : (tensor<1x32x258x162xf16, {order = #NHWC}>) -> tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[CALL_DEFAULT]] : tensor<1x256x256x160xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     [[INSERT:%.*]] = tensor.insert_slice [[INDEX_SWITCH]] into [[ACC1]][0, 0, [[IF_H]], [[IF_W]]] [1, 256, 256, 160] [1, 1, 1, 1] : tensor<1x256x256x160xf16, {order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:     scf.yield [[INSERT]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:   }
  // CHECK:   scf.yield [[INNER_FOR]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK: }
  // CHECK: return [[OUTER_FOR]] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
}
