//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --scf-compute-ops-outlining  %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>
module @ControlFlowOutliningStaticShape {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x32x64x64xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x256x64x64xf16>
    }

  func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg1)
      %c0_1 = arith.constant 0 : index
      %3 = arith.cmpi eq, %arg1, %c0_1 : index
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
      %4 = scf.if %3 -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
        %5 = VPU.NCE.Convolution(%extracted_slice, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %5 : tensor<1x256x32x64xf16, {order = #NHWC}>
      } else {
        %5 = VPU.NCE.Convolution(%extracted_slice, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %5 : tensor<1x256x32x64xf16, {order = #NHWC}>
      }
      %inserted_slice = tensor.insert_slice %4 into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: @ControlFlowOutliningStaticShape
// CHECK: DataInfo "input1" : tensor<1x32x64x64xf16>
// CHECK: DataInfo "output1" : tensor<1x256x64x64xf16>

// CHECK:  func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
// CHECK:    [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:    [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST]]){{.*}} {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, {{[^}]+}}} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:    return [[OUTPUT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:  }
// CHECK:  func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
// CHECK:    [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:    [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST]]){{.*}} {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, {{[^}]+}}} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:    return [[OUTPUT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:  }
// CHECK:  func.func @main([[ARG_0:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
// CHECK:      [[LOCAL_BUFF:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:      [[CST_1:%.+]] = arith.constant 0 : index
// CHECK:      [[CST_64:%.+]] = arith.constant 64 : index
// CHECK:      [[CST_32:%.+]] = arith.constant 32 : index
// CHECK:      [[SCF_FOR:%.+]] = scf.for [[LOCAL_INPUT0:%.+]] = [[CST_1]] to [[CST_64]] step [[CST_32]] iter_args([[LOCAL_INPUT1:%.+]] = [[LOCAL_BUFF]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
// CHECK:        [[MAX:%.+]] = affine.max #map({{%[^:]+}})
// CHECK:        [[CST_2:%.+]] = arith.constant 0 : index
// CHECK:        [[CMP:%.+]] = arith.cmpi eq, [[LOCAL_INPUT0]], [[CST_2]] : index
// CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[ARG_0]][0, 0, [[MAX]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:        [[SCF_IF:%.+]] = scf.if [[CMP]] -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
// CHECK:          [[CONV_OUTPUT:%.+]] = func.call @main_func0([[SLICE]]) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:          scf.yield [[CONV_OUTPUT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:        } else {
// CHECK:          [[CONV_OUTPUT:%.+]] = func.call @main_func1([[SLICE]]) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:          scf.yield [[CONV_OUTPUT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:        }
// CHECK:        [[MERGED_OUTPUT:%.+]] = tensor.insert_slice [[SCF_IF]] into [[LOCAL_INPUT1]][0, 0, [[LOCAL_INPUT0]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:        scf.yield [[MERGED_OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:      }
// CHECK:      return [[SCF_FOR]] : tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:  }

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 - 1, 0)>
module @ControlFlowOutliningStaticShape1 {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x16x200x200xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x16x200x200xf16>
    }

  func.func @main(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
    %0 = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c200 = arith.constant 200 : index
    %c50 = arith.constant 50 : index
    %1 = scf.for %arg1 = %c0 to %c200 step %c50 iter_args(%arg2 = %0) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg1)
      %c0_0 = arith.constant 0 : index
      %3 = arith.cmpi eq, %arg1, %c0_0 : index
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 16, 51, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x51x200xf16, {order = #NHWC}>
      %4 = scf.if %3 -> (tensor<1x16x50x200xf16, {order = #NHWC}>) {
        %5 = VPU.NCE.MaxPool(%extracted_slice, %cst ) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
        scf.yield %5 : tensor<1x16x50x200xf16, {order = #NHWC}>
      } else {
        %c200_1 = arith.constant 200 : index
        %5 = arith.subi %c200_1, %arg1 : index
        %6 = arith.cmpi eq, %arg1, %5 : index
        %7 = scf.if %6 -> (tensor<1x16x50x200xf16, {order = #NHWC}>) {
          %8 = VPU.NCE.MaxPool(%extracted_slice, %cst ) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
          scf.yield %8 : tensor<1x16x50x200xf16, {order = #NHWC}>
        } else {
          %extracted_slice_2 = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 16, 52, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x52x200xf16, {order = #NHWC}>
          %8 = VPU.NCE.MaxPool(%extracted_slice_2, %cst ) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
          scf.yield %8 : tensor<1x16x50x200xf16, {order = #NHWC}>
        }
        scf.yield %7 : tensor<1x16x50x200xf16, {order = #NHWC}>
      }
      %inserted_slice = tensor.insert_slice %4 into %arg2[0, 0, %arg1, 0] [1, 16, 50, 200] [1, 1, 1, 1] : tensor<1x16x50x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x200x200xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x200x200xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: ControlFlowOutliningStaticShape1
// CHECK: DataInfo "input1" : tensor<1x16x200x200xf16>
// CHECK: DataInfo "output1" : tensor<1x16x200x200xf16>

// CHECK:    func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x16x51x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}> {
// CHECK:      [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:      [[OUTPUT0:%.+]] = VPU.NCE.MaxPool([[INPUT0]], [[CST]] ){{.*}} {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
// CHECK-SAME: strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:      return [[OUTPUT0]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:    }
// CHECK:    func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x16x51x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}> {
// CHECK:      [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:      [[OUTPUT0:%.+]] = VPU.NCE.MaxPool([[INPUT0]], [[CST]] ){{.*}} {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
// CHECK-SAME: strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:      return [[OUTPUT0]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:    }
// CHECK:    func.func nested @main_func2([[INPUT0:%.+]]: tensor<1x16x52x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}> {
// CHECK:      [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>
// CHECK:      [[OUTPUT0:%.+]] = VPU.NCE.MaxPool([[INPUT0]], [[CST]] ){{.*}} {kernel_size = [3, 3], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
// CHECK-SAME: strides = [1, 1]} -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:      return [[OUTPUT0]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:    }
// CHECK:    func.func @main([[INPUT0:%.+]]: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
// CHECK:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x16x200x200xf16, {order = #NHWC}>
// CHECK:      [[CST0:%.+]] = arith.constant 0 : index
// CHECK:      [[CST200:%.+]] = arith.constant 200 : index
// CHECK:      [[CST50:%.+]] = arith.constant 50 : index
// CHECK:      [[SCF_FOR:%.+]] = scf.for [[ARG1:%.+]] = [[CST0]] to [[CST200]] step [[CST50]] iter_args([[ARG2:%.+]] = [[EMPTY]]) -> (tensor<1x16x200x200xf16, {order = #NHWC}>) {
// CHECK:        [[MAX:%.+]] = affine.max #map([[ARG1]])
// CHECK:        [[CST1:%.+]] = arith.constant 0 : index
// CHECK:        [[CMP:%.+]] = arith.cmpi eq, [[ARG1]], [[CST1]] : index
// CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[MAX]], 0] [1, 16, 51, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x51x200xf16, {order = #NHWC}>
// CHECK:        [[SCF_IF0:%.+]] = scf.if [[CMP]] -> (tensor<1x16x50x200xf16, {order = #NHWC}>) {
// CHECK:          [[MAX_POOL:%.+]] = func.call @main_func0([[SLICE]]) : (tensor<1x16x51x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:          scf.yield [[MAX_POOL]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:        } else {
// CHECK:          [[CST200_1:%.+]] = arith.constant 200 : index
// CHECK:          [[SUB:%.+]] = arith.subi [[CST200_1]], [[ARG1]] : index
// CHECK:          [[CMP:%.+]] = arith.cmpi eq, [[ARG1]], [[SUB]] : index
// CHECK:          [[SCF_IF1:%.+]] = scf.if [[CMP]] -> (tensor<1x16x50x200xf16, {order = #NHWC}>) {
// CHECK:            [[MAX_POOL:%.+]] = func.call @main_func1([[SLICE]]) : (tensor<1x16x51x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:            scf.yield [[MAX_POOL]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:          } else {
// CHECK:            [[SLICE1:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[MAX]], 0] [1, 16, 52, 200] [1, 1, 1, 1] : tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x52x200xf16, {order = #NHWC}>
// CHECK:            [[MAX_POOL:%.+]] = func.call @main_func2([[SLICE1]]) : (tensor<1x16x52x200xf16, {order = #NHWC}>) -> tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:            scf.yield [[MAX_POOL]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:          }
// CHECK:          scf.yield [[SCF_IF1]] : tensor<1x16x50x200xf16, {order = #NHWC}>
// CHECK:        }
// CHECK:        [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[SCF_IF0]] into [[ARG2]][0, 0, [[ARG1]], 0] [1, 16, 50, 200] [1, 1, 1, 1] : tensor<1x16x50x200xf16, {order = #NHWC}> into tensor<1x16x200x200xf16, {order = #NHWC}>
// CHECK:        scf.yield [[INSERTED_SLICE]] : tensor<1x16x200x200xf16, {order = #NHWC}>
// CHECK:      }
// CHECK:      return [[SCF_FOR]] : tensor<1x16x200x200xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (240, -d0 + s0)>
module @ControlFlowOutliningDynamicShape1 {
  net.NetworkInfo entryPoint : @main
  inputsInfo : {
      DataInfo "input1" : tensor<1x16x256x?xf16>
      DataInfo "input2" : tensor<1x16x256x?xf16>
  }
  outputsInfo : {
      DataInfo "output1" : tensor<1x16x256x?xf16>
  }

  func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c3_0 = arith.constant 3 : index
    %dim_1 = tensor.dim %arg0, %c3_0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    %c240 = arith.constant 240 : index
    %1 = scf.for %arg2 = %c0 to %dim_1 step %c240 iter_args(%arg3 = %0) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_1]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%extracted_slice, %extracted_slice_2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
  }
}

// CHECK-LABEL: ControlFlowOutliningDynamicShape1
// CHECK: DataInfo "input1" : tensor<1x16x256x?xf16>
// CHECK: DataInfo "input2" : tensor<1x16x256x?xf16>
// CHECK: DataInfo "output1" : tensor<1x16x256x?xf16>

// CHECK:    func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
// CHECK:      [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[INPUT1]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:      return [[OUTPUT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:    }
// CHECK:    func.func @main([[INPUT0:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
// CHECK:        [[CST0:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM0:%.+]] = tensor.dim [[INPUT0]], [[CST0]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[BUFFER:%.+]] = tensor.empty([[DIM0]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[CST1:%.+]] = arith.constant 0 : index
// CHECK:        [[CST3:%.+]] = arith.constant 3 : index
// CHECK:        [[DIM1:%.+]] = tensor.dim [[INPUT0]], [[CST3]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        [[CST240:%.+]] = arith.constant 240 : index
// CHECK:        [[SCF_FOR0:%.+]] = scf.for [[ARG2:%.+]] = [[CST1:%.+]] to [[DIM1]] step [[CST240]] iter_args([[ARG3:%.+]] = [[BUFFER]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {
// CHECK:          [[MIN:%.+]] = affine.min #map([[ARG2]])[[[DIM1]]]
// CHECK:          [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[ARG2]]] [1, 16, 256, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:          [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[ARG2]]] [1, 16, 256, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:          [[ELTWISE:%.+]] = func.call @main_func0([[SLICE0]], [[SLICE1]]) : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:          [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[ELTWISE]] into [[ARG3]][0, 0, 0, [[ARG2]]] [1, 16, 256, [[MIN]]] [1, 1, 1, 1] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:          scf.yield [[INSERTED_SLICE]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:        }
// CHECK:        return [[SCF_FOR0]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>
module @ControlFlowOutliningMultipleOutput {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x32x64x64xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x256x64x64xf16>
    }

  func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg1)
      %c0_1 = arith.constant 0 : index
      %3 = arith.cmpi eq, %arg1, %c0_1 : index
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%extracted_slice, %cst_1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%extracted_slice, %cst_2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
      %6 = scf.if %3 -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
        %7 = VPU.NCE.Convolution(%4, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %7 : tensor<1x256x32x64xf16, {order = #NHWC}>
      } else {
        %7 = VPU.NCE.Convolution(%5, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %7 : tensor<1x256x32x64xf16, {order = #NHWC}>
      }
      %inserted_slice = tensor.insert_slice %6 into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: ControlFlowOutliningMultipleOutput
// CHECK: DataInfo "input1" : tensor<1x32x64x64xf16>
// CHECK: DataInfo "output1" : tensor<1x256x64x64xf16>

// CHECK:   func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> (tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>) {
// CHECK:     [[CST:%.+]] = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
// CHECK:     [[CST_0:%.+]] = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
// CHECK:     [[OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[CST]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:     [[OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[CST_0]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:     return [[OUT0]], [[OUT1]] : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:   }
// CHECK:   func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
// CHECK:     [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:     [[OUT0:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, 
// CHECK-SAME: strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:     return [[OUT0]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:   }
// CHECK:   func.func nested @main_func2([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
// CHECK:     [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
// CHECK:     [[OUT0:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, 
// CHECK-SAME: strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:     return [[OUT0]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:   }
// CHECK:   func.func @main([[INPUT0:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
// CHECK:     [[BUFFER:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:     [[FROM:%.+]] = arith.constant 0 : index
// CHECK:     [[TILL:%.+]] = arith.constant 64 : index
// CHECK:     [[STEP:%.+]] = arith.constant 32 : index
// CHECK:     [[RESULT:%.+]] = scf.for [[ARG0:%.+]] = [[FROM]] to [[TILL]] step [[STEP]] iter_args([[ARG1:%.+]] = [[BUFFER]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
// CHECK:       [[MAX:%.+]] = affine.max #map([[ARG0]])
// CHECK:       [[CST3:%.+]] = arith.constant 0 : index
// CHECK:       [[CMP:%.+]] = arith.cmpi eq, [[ARG0]], [[CST3]] : index
// CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[MAX]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK:       [[OUT0:%.+]]:2 = func.call @main_func0([[SLICE]]) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> (tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>)
// CHECK:       [[TMP_RESULT:%.+]] = scf.if [[CMP]] -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
// CHECK:         [[TMP:%.+]] = func.call @main_func1([[OUT0]]#0) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:         scf.yield [[TMP]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:       } else {
// CHECK:         [[TMP:%.+]] = func.call @main_func2([[OUT0]]#1) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:         scf.yield [[TMP]] : tensor<1x256x32x64xf16, {order = #NHWC}>
// CHECK:       }
// CHECK:       [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[TMP_RESULT]] into [[ARG1]][0, 0, [[ARG0]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:       scf.yield [[INSERTED_SLICE]] : tensor<1x256x64x64xf16, {order = #NHWC}>
// CHECK:     }
// CHECK:     return [[RESULT]] : tensor<1x256x64x64xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Add {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x1x100x100xf32>
    DataInfo "input2" : tensor<1x1x100x100xf32>
  } outputsInfo : {
    DataInfo "Add_3" friendlyName = "output" : tensor<1x1x100x100xf32>
  }
  func.func @main(%arg0: tensor<1x1x100x100xf32>, %arg1: tensor<1x1x100x100xf32>) -> tensor<1x1x100x100xf32>{
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x100x100xf32> -> tensor<1x1x100x100xf32, {order = #NHWC}>
    %1 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x100x100xf32> -> tensor<1x1x100x100xf32, {order = #NHWC}>
    %2 = VPU.ShapeCast {shape = [1, 16, 25, 25]} inputs(%0 : tensor<1x1x100x100xf32, {order = #NHWC}>) -> tensor<1x16x25x25xf32, {order = #NHWC}>
    %3 = VPU.Convert(%2) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x25x25xf32, {order = #NHWC}> -> tensor<1x16x25x25xf16, {order = #NHWC}>
    %4 = VPU.ShapeCast {shape = [1, 16, 25, 25]} inputs(%1 : tensor<1x1x100x100xf32, {order = #NHWC}>) -> tensor<1x16x25x25xf32, {order = #NHWC}>
    %5 = VPU.Convert(%4) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x25x25xf32, {order = #NHWC}> -> tensor<1x16x25x25xf16, {order = #NHWC}>
    %6 = VPU.NCE.Eltwise(%3, %5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00]
, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x25x25xf32, {order = #NHWC}>
    %7 = VPU.ShapeCast {shape = [1, 1, 100, 100]} inputs(%6 : tensor<1x16x25x25xf32, {order = #NHWC}>) -> tensor<1x1x100x100xf32, {order = #NHWC}>
    %8 = VPU.PermuteCast(%7) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x100x100xf32, {order = #NHWC}> -> tensor<1x1x100x100xf32>
    return %8 : tensor<1x1x100x100xf32>
  }

  // CHECK:   func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x1x100x100xf32>, [[INPUT1:%.+]]: tensor<1x1x100x100xf32>) -> tensor<1x1x100x100xf32> {
  // CHECK:   [[CAST0:%.+]] = VPU.PermuteCast([[INPUT0]])
  // CHECK:   [[CAST1:%.+]] = VPU.PermuteCast([[INPUT1]])
  // CHECK:   [[SHAPECAST0:%.+]] = VPU.ShapeCast {shape = [1, 16, 25, 25]} inputs([[CAST0]]
  // CHECK:   [[CONVERT0:%.+]] = VPU.Convert([[SHAPECAST0]])
  // CHECK:   [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 16, 25, 25]} inputs([[CAST1]]
  // CHECK:   [[CONVERT1:%.+]] = VPU.Convert([[SHAPECAST1]])
  // CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONVERT0]], [[CONVERT1]])
  // CHECK:   [[SHAPECAST2:%.+]] = VPU.ShapeCast
  // CHECK:   [[RESULT:%.+]] = VPU.PermuteCast
  // CHECK:   return [[RESULT]] : tensor<1x1x100x100xf32>
  // CHECK:   }
  // CHECK:   func.func @main([[ARGS0:%.+]]: tensor<1x1x100x100xf32>, [[ARGS1:%.+]]: tensor<1x1x100x100xf32>) -> tensor<1x1x100x100xf32> attributes {[[ANY_ATTR:.+]]} {
  // CHECK:   [[RESULTS:%.+]] = call @main_func0([[ARGS0]], [[ARGS1]]) : (tensor<1x1x100x100xf32>, tensor<1x1x100x100xf32>) -> tensor<1x1x100x100xf32>
  // CHECK:   return [[RESULTS]] : tensor<1x1x100x100xf32>
  // CHECK:   }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>
// CHECK: #map = affine_map<(d0) -> (d0 - 1, 0)>

// CHECK-LABEL: @SparseTensorWithCstInputs
module @SparseTensorWithCstInputs {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x32x64x64xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x256x64x64xf16>
    }

  func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<256x1x1x384xi1> = dense<0.0> : tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, [#const.GetSparsityMap]
    %sparse = VPU.GroupSparseTensor(%cst, %cst_3) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>>
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg1)
      %c0_1 = arith.constant 0 : index
      %3 = arith.cmpi eq, %arg1, %c0_1 : index
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%extracted_slice, %cst_1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%extracted_slice, %cst_2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
      %6 = scf.if %3 -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
        %7 = VPU.NCE.Convolution(%4, %sparse) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %7 : tensor<1x256x32x64xf16, {order = #NHWC}>
      } else {
        %7 = VPU.NCE.Convolution(%5, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
        scf.yield %7 : tensor<1x256x32x64xf16, {order = #NHWC}>
      }
      %inserted_slice = tensor.insert_slice %6 into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }

  // CHECK:   func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> (tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>) {
  // CHECK:     [[CST:%.+]] = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
  // CHECK:     [[CST_0:%.+]] = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
  // CHECK:     [[OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[CST]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, {{[^}]+}}} -> tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:     [[OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[CST_0]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, {{[^}]+}}} -> tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:     return [[OUT0]], [[OUT1]] : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:   }

  // CHECK:   func.func nested @main_func1([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  // CHECK:     [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:     [[CST_0:%.+]] = const.Declare tensor<256x1x1x384xi1> = dense<0.000000e+00> : tensor<256x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  // CHECK:     [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST]], [[CST_0]]) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>>
  // CHECK:     [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[SPARSE]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, 
  // CHECK-SAME: strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:     return [[CONV]] : tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:   }

  // CHECK:   func.func nested @main_func2([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  // CHECK:     [[CST:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK:     [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, 
  // CHECK-SAME: strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:     return [[CONV]] : tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:   }

  // CHECK:   func.func @main([[INPUT0:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
  // CHECK:     [[BUFFER:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:     [[FROM:%.+]] = arith.constant 0 : index
  // CHECK:     [[TILL:%.+]] = arith.constant 64 : index
  // CHECK:     [[STEP:%.+]] = arith.constant 32 : index
  // CHECK:     [[RESULT:%.+]] = scf.for [[ARG0:%.+]] = [[FROM]] to [[TILL]] step [[STEP]] iter_args([[ARG1:%.+]] = [[BUFFER]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
  // CHECK:       [[MAX:%.+]] = affine.max #map([[ARG0]])
  // CHECK:       [[CST3:%.+]] = arith.constant 0 : index
  // CHECK:       [[CMP:%.+]] = arith.cmpi eq, [[ARG0]], [[CST3]] : index
  // CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[MAX]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:       [[RESULT_4:%.+]]:2 = func.call @main_func0([[SLICE]]) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> (tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<1x32x33x64xf16, {order = #NHWC}>)
  // CHECK:       [[TMP_RESULT:%.+]] = scf.if [[CMP]] -> (tensor<1x256x32x64xf16, {order = #NHWC}>) {
  // CHECK:         [[TMP:%.+]] = func.call @main_func1([[RESULT_4]]#0) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:         scf.yield [[TMP]] : tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:       } else {
  // CHECK:         [[TMP:%.+]] = func.call @main_func2([[RESULT_4]]#1) : (tensor<1x32x33x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:         scf.yield [[TMP]] : tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:       }
  // CHECK:       [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[TMP_RESULT]] into [[ARG1]][0, 0, [[ARG0]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:       scf.yield [[INSERTED_SLICE]] : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:     }
  // CHECK:     return [[RESULT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:   }
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (d0 - 1, 0)>

// CHECK: #map = affine_map<(d0) -> (d0 - 1, 0)>

// CHECK-LABEL: @SparseTensorWithBlockInput
module @SparseTensorWithBlockInput {
  net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<1x32x64x64xf16>
        DataInfo "input2" : tensor<256x32x3x3xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x256x64x64xf16>
    }

  func.func @main(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>, %arg1: tensor<256x32x3x3xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<2.0> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<256x1x1x384xi1> = dense<0.0> : tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, [#const.GetSparsityMap]
    %sparse = VPU.GroupSparseTensor(%arg1, %cst_3) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>>
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %1 = scf.for %arg3 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.max #map(%arg3)
      %c0_1 = arith.constant 0 : index
      %3 = arith.cmpi eq, %arg3, %c0_1 : index
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%extracted_slice, %cst_1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
      %5 = VPU.NCE.Convolution(%4, %sparse) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>> -> tensor<1x256x32x64xf16, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %5 into %arg2[0, 0, %arg3, 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }

  // CHECK:  func.func nested @main_func0([[INPUT0:%.+]]: tensor<1x32x33x64xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<256x32x3x3xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  // CHECK:    [[CST:%.+]] = const.Declare tensor<1x32x33x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x33x64xf16>, [#const.Reorder<#NHWC>]
  // CHECK:    [[CST_0:%.+]] = const.Declare tensor<256x1x1x384xi1> = dense<0.000000e+00> : tensor<256x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  // CHECK:    [[SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT1]], [[CST_0]]) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>>
  // CHECK:    [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT0]], [[CST]]){{.*}} {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[ELTWISE]], [[SPARSE]]) rawFilterShape [256, 32, 3, 3] {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, 
  // CHECK-SAME: strides = [1, 1]} : tensor<1x32x33x64xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<256x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<32xi64>, alignment = 16 : i64>> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:    return [[CONV]] : tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:  }

  // CHECK:  func.func @main([[INPUT0:%.+]]: tensor<1x32x64x64xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<256x32x3x3xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> attributes {[[ANY_ATTR:.+]]} {
  // CHECK:    [[CST:%.+]] = const.Declare tensor<256x1x1x384xi1> = dense<0.000000e+00> : tensor<256x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
  // CHECK:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK:    [[C64:%.+]] = arith.constant 64 : index
  // CHECK:    [[C32:%.+]] = arith.constant 32 : index
  // CHECK:    [[SCF_FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C64]] step [[C32]] iter_args([[ARG3:%.+]] = [[EMPTY]]) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
  // CHECK:      [[MAX:%.+]] = affine.max #map([[ARG2]])
  // CHECK:      [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK:      [[CMP:%.+]] = arith.cmpi eq, [[ARG2]], [[C0_0]] : index
  // CHECK:      [[SLICE:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, [[MAX]], 0] [1, 32, 33, 64] [1, 1, 1, 1] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
  // CHECK:      [[CALL:%.+]] = func.call @main_func0([[SLICE]], [[INPUT1]]) : (tensor<1x32x33x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:      [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CALL]] into [[ARG3]][0, 0, [[ARG2]], 0] [1, 256, 32, 64] [1, 1, 1, 1] : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:      scf.yield [[INSERTED_SLICE]] : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:    }
  // CHECK:    return [[SCF_FOR]] : tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @ExtractSliceMoveToTop {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" tensorNames = ["input1"] : tensor<1x16x1600x1600xf32>
    DataInfo "input2" tensorNames = ["input2"] : tensor<1x16x1600x1600xf32>
  } outputsInfo : {
    DataInfo "Add_3.0" friendlyName = "output" : tensor<1x16x1600x1600xf32>
  }
  func.func @main(%arg0: tensor<1x16x1600x1600xf32>, %arg1: tensor<1x16x1600x1600xf32>) -> tensor<1x16x1600x1600xf32> {
    %c10 = arith.constant 10 : index
    %c1600 = arith.constant 1600 : index
    %c0 = arith.constant 0 : index
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1600x1600xf32> -> tensor<1x1600x16x1600xf32, {order = #NHWC}>
    %1 = VPU.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x1600x1600xf32> -> tensor<1x1600x16x1600xf32, {order = #NHWC}>
    %2 = tensor.empty() : tensor<1x1600x16x1600xf32, {order = #NHWC}>
    %3 = scf.for %arg2 = %c0 to %c1600 step %c10 iter_args(%arg3 = %2) -> (tensor<1x1600x16x1600xf32, {order = #NHWC}>) {
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, %arg2] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x1600xf32, {order = #NHWC}> to tensor<1x1600x16x10xf32, {order = #NHWC}>
      %5 = VPU.Convert(%extracted_slice) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1600x16x10xf32, {order = #NHWC}> -> tensor<1x1600x16x10xf16, {order = #NHWC}>
      %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0, %arg2] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x1600xf32, {order = #NHWC}> to tensor<1x1600x16x10xf32, {order = #NHWC}>
      %6 = VPU.Convert(%extracted_slice_0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1600x16x10xf32, {order = #NHWC}> -> tensor<1x1600x16x10xf16, {order = #NHWC}>
      %7 = VPU.NCE.Eltwise(%5, %6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x1600x16x10xf32, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %7 into %arg3[0, 0, 0, %arg2] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x10xf32, {order = #NHWC}> into tensor<1x1600x16x1600xf32, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x1600x16x1600xf32, {order = #NHWC}>
    }
    %4 = VPU.PermuteCast(%3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1600x16x1600xf32, {order = #NHWC}> -> tensor<1x16x1600x1600xf32>
    return %4 : tensor<1x16x1600x1600xf32>
  }

  // CHECK:  func.func nested @main_func2([[INPUT0:%.+]]: tensor<1x1600x16x10xf32, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x1600x16x10xf32, {order = #NHWC}>) -> tensor<1x1600x16x10xf32, {order = #NHWC}> {
  // CHECK:    [[CONVERT_0:%.+]] = VPU.Convert([[INPUT0]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1600x16x10xf32, {order = #NHWC}> -> tensor<1x1600x16x10xf16, {order = #NHWC}>
  // CHECK:    [[CONVERT_1:%.+]] = VPU.Convert([[INPUT1]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1600x16x10xf32, {order = #NHWC}> -> tensor<1x1600x16x10xf16, {order = #NHWC}>
  // CHECK:    [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONVERT_0]], [[CONVERT_1]]){{.*}} {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x1600x16x10xf32, {order = #NHWC}>
  // CHECK:    return [[ELTWISE]] : tensor<1x1600x16x10xf32, {order = #NHWC}>
  // CHECK:  }

  // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x16x1600x1600xf32>, [[ARG1:%.+]]: tensor<1x16x1600x1600xf32>) -> tensor<1x16x1600x1600xf32> attributes {[[ANY_ATTR:.+]]} {
  // CHECK:    [[C10:%.+]] = arith.constant 10 : index
  // CHECK:    [[C1600:%.+]] = arith.constant 1600 : index
  // CHECK:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK:    [[CONVERTED_ARG:%.+]]:2 = call @main_func0([[ARG0]], [[ARG1]]) : (tensor<1x16x1600x1600xf32>, tensor<1x16x1600x1600xf32>) -> (tensor<1x1600x16x1600xf32, {order = #NHWC}>, tensor<1x1600x16x1600xf32, {order = #NHWC}>)
  // CHECK:    [[OUTPUT:%.+]] = tensor.empty() : tensor<1x1600x16x1600xf32, {order = #NHWC}>
  // CHECK:    [[RESULT:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C1600]] step [[C10]] iter_args([[ARG3:%.+]] = [[OUTPUT]]) -> (tensor<1x1600x16x1600xf32, {order = #NHWC}>) {
  // CHECK:      [[EXTRACTED_SLICE_0:%.+]] = tensor.extract_slice [[CONVERTED_ARG]]#0[0, 0, 0, [[ARG2]]] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x1600xf32, {order = #NHWC}> to tensor<1x1600x16x10xf32, {order = #NHWC}>
  // CHECK:      [[EXTRACTED_SLICE_1:%.+]] = tensor.extract_slice [[CONVERTED_ARG]]#1[0, 0, 0, [[ARG2]]] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x1600xf32, {order = #NHWC}> to tensor<1x1600x16x10xf32, {order = #NHWC}>
  // CHECK:      [[FUNC_RESULT:%.+]] = func.call @main_func2([[EXTRACTED_SLICE_0]], [[EXTRACTED_SLICE_1]]) : (tensor<1x1600x16x10xf32, {order = #NHWC}>, tensor<1x1600x16x10xf32, {order = #NHWC}>) -> tensor<1x1600x16x10xf32, {order = #NHWC}>
  // CHECK:      [[INSERT_SLICE:%.+]] = tensor.insert_slice [[FUNC_RESULT]] into [[ARG3]][0, 0, 0, [[ARG2]]] [1, 1600, 16, 10] [1, 1, 1, 1] : tensor<1x1600x16x10xf32, {order = #NHWC}> into tensor<1x1600x16x1600xf32, {order = #NHWC}>
  // CHECK:      scf.yield [[INSERT_SLICE]] : tensor<1x1600x16x1600xf32, {order = #NHWC}>
  // CHECK:    }
  // CHECK:    [[CONVERTED_RESULT:%.+]] = call @main_func1([[RESULT]]) : (tensor<1x1600x16x1600xf32, {order = #NHWC}>) -> tensor<1x16x1600x1600xf32>
  // CHECK:    return [[CONVERTED_RESULT]] : tensor<1x16x1600x1600xf32>
  // CHECK:  }
}
