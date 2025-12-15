//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile" --optimize-concat="disable-pass-on-entry-function=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: SkipMainFunc
module @SkipMainFunc {
  net.NetworkInfo entryPoint : @SameSiblingConcat
  inputsInfo : {
    DataInfo "input0" : tensor<100x1x1x1xf16>
    DataInfo "input1" : tensor<112x15x1x1xf16>
  } outputsInfo : {
    DataInfo "output0" : tensor<112x16x1x1xf16>
    DataInfo "output1" : tensor<112x16x1x1xf16>
    DataInfo "output2" : tensor<112x16x1x1xf16>
  }

  // CHECK: func.func [[FUNC:@.+]]([[INPUT_0:%.+]]: tensor<100x1x1x1xf16>, [[INPUT_1:%.+]]: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
  func.func @SameSiblingConcat(%arg0: tensor<100x1x1x1xf16>, %arg1: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_0 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    %1 = VPU.Concat(%0, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %2 = VPU.Concat(%0, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %3 = VPU.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %1, %2, %3 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    // CHECK:     [[CST_0:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT_0:%.+]] = VPU.Concat([[EXPAND]], [[CST]])
    // CHECK:     [[CONCAT_1:%.+]] = VPU.Concat([[EXPAND]], [[CST_0]])
    // CHECK:     [[CONCAT_2:%.+]] = VPU.Concat([[EXPAND]], [[INPUT_1]])
    // CHECK:     return [[CONCAT_0]], [[CONCAT_1]], [[CONCAT_2]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
 }
}

// -----

// CHECK-LABEL: OptimizeConcatPassOnNpuFunction
module @OptimizeConcatPassOnNpuFunction {
  net.NetworkInfo entryPoint : @main
  inputsInfo : {
    DataInfo "input0" : tensor<100x1x1x1xf16>
    DataInfo "input1" : tensor<112x15x1x1xf16>
  } outputsInfo : {
    DataInfo "output0" : tensor<112x16x1x1xf16>
    DataInfo "output1" : tensor<112x16x1x1xf16>
    DataInfo "output2" : tensor<112x16x1x1xf16>
  }

  // CHECK: func.func [[FUNC:@.+]]([[INPUT_0:%.+]]: tensor<100x1x1x1xf16>, [[INPUT_1:%.+]]: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>)
  func.func @SameSiblingConcat(%arg0: tensor<100x1x1x1xf16>, %arg1: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %cst = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %cst_0 = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>
    %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    %1 = VPU.Concat(%0, %cst) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %2 = VPU.Concat(%0, %cst_0) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>
    %3 = VPU.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<112x1x1x1xf16>, tensor<112x15x1x1xf16> -> tensor<112x16x1x1xf16>

    return %1, %2, %3 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<112x15x1x1xf16> = dense<0.000000e+00> : tensor<112x15x1x1xf16>

    // CHECK:     [[EXPAND:%.+]] = VPU.Expand([[INPUT_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<100x1x1x1xf16> -> tensor<112x1x1x1xf16>
    // CHECK:     [[CONCAT_0:%.+]] = VPU.Concat([[EXPAND]], [[CST]])
    // CHECK:     [[CONCAT_1:%.+]] = VPU.Concat([[EXPAND]], [[INPUT_1]])
    // CHECK:     return [[CONCAT_0]], [[CONCAT_0]], [[CONCAT_1]] : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
 }

 // CHECK: func.func [[MAIN:@.+]]([[INPUT_0:%.+]]: tensor<100x1x1x1xf16>, [[INPUT_1:%.+]]: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>)
 func.func @main(%arg0: tensor<100x1x1x1xf16>, %arg1: tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>) {
    %0, %1, %2 = func.call @SameSiblingConcat(%arg0, %arg1) : (tensor<100x1x1x1xf16>, tensor<112x15x1x1xf16>) -> (tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>)
    return %0, %1, %2 : tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>, tensor<112x16x1x1xf16>
 }
}
