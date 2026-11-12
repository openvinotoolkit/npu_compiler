//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --adjust-block-size-for-scf-tiling="enable-sub-tile-fallback=true" --cse --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>

// CHECK-DAG: #[[$MAP_ORIG:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 100)>
// CHECK-DAG: #[[$MAP_BACKTRACK:.+]] = affine_map<(d0)[s0] -> (s0 - 100, d0)>

// CHECK-LABEL: @SubTileFallbackEltwise
module @SubTileFallbackEltwise {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
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
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // Original kernel is kept for the adjusted path, a dedicated dynamic copy serves the sub-tile path
  // CHECK:      func.func nested @main_func0(
  // CHECK:        VPU.NCE.Eltwise
  // CHECK:      func.func nested @main_func0_sub_tile([[ST_ARG0:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, [[ST_ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> attributes {sub_tile_fallback} {
  // CHECK:        [[ST_ELTWISE:%.+]] = VPU.NCE.Eltwise([[ST_ARG0]], [[ST_ARG1]])
  // CHECK:        return [[ST_ELTWISE]]

  // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x16x720x?xf16, {{.*}}>, [[ARG1:%.+]]: tensor<1x16x720x?xf16, {{.*}}>)
  // CHECK-DAG:    [[C100:%.+]] = arith.constant 100 : index
  // CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
  // CHECK:        [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]]
  // CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM]])
  // CHECK-NOT:    cf.assert
  // CHECK:        [[IS_SUB_TILE:%.+]] = arith.cmpi slt, [[DIM]], [[C100]] : index
  // CHECK:        [[RESULT:%.+]] = scf.if [[IS_SUB_TILE]] -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {

  // Sub-tile path: original dynamic loop, dedicated dynamic kernel
  // CHECK:          [[SUB_FOR:%.+]] = scf.for [[SUB_IV:%.+]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[SUB_ACC:%.+]] = [[EMPTY]])
  // CHECK:            [[SUB_SIZE:%.+]] = affine.min #[[$MAP_ORIG]]([[SUB_IV]])[[[DIM]]]
  // CHECK:            [[SUB_SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[SUB_IV]]] [1, 16, 720, [[SUB_SIZE]]]
  // CHECK:            [[SUB_SLICE1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[SUB_IV]]] [1, 16, 720, [[SUB_SIZE]]]
  // CHECK:            [[SUB_CALL:%.+]] = func.call @main_func0_sub_tile([[SUB_SLICE0]], [[SUB_SLICE1]])
  // CHECK:            [[SUB_INSERTED:%.+]] = tensor.insert_slice [[SUB_CALL]] into [[SUB_ACC]][0, 0, 0, [[SUB_IV]]] [1, 16, 720, [[SUB_SIZE]]]
  // CHECK:            scf.yield [[SUB_INSERTED]]
  // CHECK:          } {sub_tile_fallback}
  // CHECK:          scf.yield [[SUB_FOR]]
  // CHECK:        } else {

  // Adjusted path: backtracked offset, static tiles
  // CHECK:          [[FOR:%.+]] = scf.for [[IV:%.+]] = [[C0]] to [[DIM]] step [[C100]] iter_args([[ACC:%.+]] = [[EMPTY]])
  // CHECK:            [[OFFSET:%.+]] = affine.min #[[$MAP_BACKTRACK]]([[IV]])[[[DIM]]]
  // CHECK:            [[SLICE0:%.+]] = tensor.extract_slice [[ARG0]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100]
  // CHECK:            [[CAST0:%.+]] = tensor.cast [[SLICE0]]
  // CHECK:            [[SLICE1:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100]
  // CHECK:            [[CAST1:%.+]] = tensor.cast [[SLICE1]]
  // CHECK:            [[CALL:%.+]] = func.call @main_func0([[CAST0]], [[CAST1]])
  // CHECK:            [[CAST2:%.+]] = tensor.cast [[CALL]]
  // CHECK:            [[INSERTED:%.+]] = tensor.insert_slice [[CAST2]] into [[ACC]][0, 0, 0, [[OFFSET]]] [1, 16, 720, 100]
  // CHECK:            scf.yield [[INSERTED]]
  // CHECK:          }
  // CHECK:          scf.yield [[FOR]]
  // CHECK:        }
  // CHECK:        return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>

// A pre-existing symbol with the generated name is unrelated to the kernel and must not be reused:
// the clone gets a uniquified name and the fallback loop calls the clone.
// CHECK-LABEL: @SubTileFallbackNameCollision
module @SubTileFallbackNameCollision {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }
  func.func nested @main_func0_sub_tile(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    return %arg0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
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
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:      func.func nested @main_func0_sub_tile([[PRE_ARG:%.+]]: tensor<1x16x720x?xf16, {{.*}}>) -> tensor<1x16x720x?xf16, {{.*}}> {
  // CHECK-NEXT:   return [[PRE_ARG]]
  // CHECK:      func.func nested @main_func0(
  // CHECK:        VPU.NCE.Eltwise
  // CHECK:      func.func nested @[[CLONE:main_func0_sub_tile_[0-9]+]]([[ST_ARG0:%.+]]: tensor<1x16x720x?xf16, {{.*}}>, [[ST_ARG1:%.+]]: tensor<1x16x720x?xf16, {{.*}}>) -> tensor<1x16x720x?xf16, {{.*}}> attributes {sub_tile_fallback} {
  // CHECK:        [[ST_ELTWISE:%.+]] = VPU.NCE.Eltwise([[ST_ARG0]], [[ST_ARG1]])
  // CHECK:        return [[ST_ELTWISE]]

  // CHECK:      func.func @main(
  // CHECK:        scf.if
  // CHECK:          scf.for
  // CHECK:            func.call @[[CLONE]](
  // CHECK:          } {sub_tile_fallback}
  // CHECK:        } else {
  // CHECK:          scf.for
  // CHECK:            func.call @main_func0(
  // CHECK:          }
  // CHECK:        }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map5 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @SubTileFallbackPaddedConv
module @SubTileFallbackPaddedConv {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x32x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x64xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg1)[%dim]
      %3 = affine.max #map1(%arg1)
      %4 = affine.max #map2(%arg1)
      %5 = affine.min #map3()[%4]
      %6 = affine.max #map4(%2, %3)
      %7 = affine.min #map3()[%6]
      %8 = affine.apply #map5(%2, %5, %7)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, 0] [1, 32, %8, 64] [1, 1, 1, 1] : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %extracted_slice : tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %9 = func.call @main_func0(%cast) : (tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %9 into %arg2[0, 0, %arg1, 0] [1, 256, %2, 64] [1, 1, 1, 1] : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
  }

  // Original kernel is erased after specialization; the dedicated dynamic copy keeps the full padding
  // CHECK-NOT:  func.func nested @main_func0(
  // CHECK-DAG:  func.func nested @main_func0_dims_H_cases_2(
  // CHECK-DAG:  func.func nested @main_func0_dims_H_cases_1(
  // CHECK-DAG:  func.func nested @main_func0_dims_H_cases_0(
  // FULLBLK kernel (single tile at the bound, low resolution): full padding
  // CHECK-DAG:  func.func nested @main_func0_dims_H_cases_3({{%.+}}: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>)
  // CHECK-DAG:    VPU.NCE.Convolution({{.*}}) {{.*}} {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK:      func.func nested @main_func0_sub_tile([[ST_ARG0:%.+]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 64]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x64xf16, {{.*}}> attributes {sub_tile_fallback} {
  // CHECK:        VPU.NCE.Convolution([[ST_ARG0]], {{%.+}}) {{.*}} {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>

  // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x32x?x64xf16, {{.*}}>)
  // CHECK-DAG:    [[C256:%.+]] = arith.constant 256 : index
  // CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
  // CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
  // CHECK-DAG:    [[C257:%.+]] = arith.constant 257 : index
  // CHECK:        [[DIM:%.+]] = tensor.dim [[ARG0]], [[C2]]
  // CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM]])
  // The backtrack assert is replaced by the sub-tile dispatch; only the low resolution check of the FULLBLK case is
  // left
  // CHECK-NOT:    Not enough elements to backtrack
  // CHECK:        [[LOW_RES:%.+]] = arith.cmpi ule, [[DIM]], [[C257]] {vpux.lowres_pred = 0 : i64} : index
  // CHECK:        cf.assert {{%.+}}, "Low resolution is supported only when the dimension fits in a single tile" {vpux.lowres_assert = 0 : i64}
  // CHECK-NOT:    cf.assert
  // CHECK:        [[IS_SUB_TILE:%.+]] = arith.cmpi slt, [[DIM]], [[C256]] : index
  // CHECK:        [[RESULT:%.+]] = scf.if [[IS_SUB_TILE]] -> (tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>) {
  // CHECK:          [[SUB_FOR:%.+]] = scf.for {{.*}} iter_args({{.*}} = [[EMPTY]])
  // CHECK:            func.call @main_func0_sub_tile(
  // CHECK:          } {sub_tile_fallback}
  // CHECK:          scf.yield [[SUB_FOR]]
  // CHECK:        } else {
  // CHECK:          [[FOR:%.+]] = scf.for {{.*}} iter_args({{.*}} = [[EMPTY]])
  // CHECK:            [[POS:%.+]] = arith.select [[LOW_RES]], [[C3]], {{%.+}} : index
  // CHECK:            [[CASE_IDX:%.+]] = arith.shli [[POS]], [[C2]] : index
  // CHECK:            scf.index_switch [[CASE_IDX]]
  // CHECK:              func.call @main_func0_dims_H_cases_0(
  // CHECK:              func.call @main_func0_dims_H_cases_1(
  // CHECK:              func.call @main_func0_dims_H_cases_2(
  // CHECK:            case 12 {
  // CHECK:              tensor.extract_slice [[ARG0]][0, 0, {{%.+}}, 0] [1, 32, 256, 64] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 0>}
  // CHECK:              func.call @main_func0_dims_H_cases_3(
  // CHECK:          } {vpux.lowres = #VPU.LowRes<id = 0, dim = 2, step = 256, invalid_output_left = 1, invalid_output_right = 1>}
  // CHECK:          scf.yield [[FOR]]
  // CHECK:        }
  // CHECK:        return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 100)>

// Second nest consumes the result of the first one. Snapshots are detached clones, so the use in the second
// fallback loop must follow the first nest when it is wrapped into scf.if (RAUW updates detached uses as well).
// Canonicalization then merges the two adjacent scf.if ops with the same condition.
// CHECK-LABEL: @SubTileFallbackDependentNests
module @SubTileFallbackDependentNests {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x?xf16>
    DataInfo "input2" : tensor<1x16x720x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x?xf16>
  }
  func.func nested @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
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
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = func.call @main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    %2 = scf.for %arg4 = %c0 to %dim step %c100 iter_args(%arg5 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %6 = affine.min #map(%arg4)[%dim]
      %extracted_slice_2 = tensor.extract_slice %1[0, 0, 0, %arg4] [1, 16, 720, %6] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_3 = tensor.extract_slice %arg1[0, 0, 0, %arg4] [1, 16, 720, %6] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %7 = func.call @main_func0(%extracted_slice_2, %extracted_slice_3) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice_4 = tensor.insert_slice %7 into %arg5[0, 0, 0, %arg4] [1, 16, 720, %6] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice_4 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %2 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK:      func.func nested @main_func0(
  // CHECK:      func.func nested @main_func0_sub_tile(
  // CHECK:      func.func @main([[ARG0:%.+]]: tensor<1x16x720x?xf16, {{.*}}>, [[ARG1:%.+]]: tensor<1x16x720x?xf16, {{.*}}>)
  // CHECK:        [[DIM:%.+]] = tensor.dim [[ARG0]]
  // CHECK:        [[EMPTY:%.+]] = tensor.empty([[DIM]])
  // CHECK:        [[IS_SUB_TILE:%.+]] = arith.cmpi slt, [[DIM]], {{%.+}} : index
  // CHECK:        [[RESULT:%.+]] = scf.if [[IS_SUB_TILE]]

  // Sub-tile path: second fallback loop reads the first fallback loop result
  // CHECK:          [[SUB_A:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]])
  // CHECK:            tensor.extract_slice [[ARG0]]
  // CHECK:            func.call @main_func0_sub_tile(
  // CHECK:          } {sub_tile_fallback}
  // CHECK:          [[SUB_B:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]])
  // CHECK:            tensor.extract_slice [[SUB_A]]
  // CHECK:            func.call @main_func0_sub_tile(
  // CHECK:          } {sub_tile_fallback}
  // CHECK:          scf.yield [[SUB_B]]
  // CHECK:        } else {

  // Adjusted path: second loop reads the first adjusted loop result
  // CHECK:          [[FOR_A:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]])
  // CHECK:            tensor.extract_slice [[ARG0]]
  // CHECK:            func.call @main_func0(
  // CHECK:          }
  // CHECK:          [[FOR_B:%.+]] = scf.for {{.*}} iter_args({{%.+}} = [[EMPTY]])
  // CHECK:            tensor.extract_slice [[FOR_A]]
  // CHECK:            func.call @main_func0(
  // CHECK:          }
  // CHECK:          scf.yield [[FOR_B]]
  // CHECK:        }
  // CHECK:        return [[RESULT]]
}
