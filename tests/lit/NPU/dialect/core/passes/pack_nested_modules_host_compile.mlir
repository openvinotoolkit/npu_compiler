//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile" --pack-nested-modules %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @DynamicStridesCopyInputOutput {
// CHECK-COUNT-1:  net.NetworkInfo entryPoint : @main inputsInfo : {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "in_0" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "out_0" : tensor<1x3x60x60xf16>
  }

// CHECK-LABEL:  module @Module0 attributes {{.+}} {
// CHECK:  net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
// CHECK:    DataInfo "in_0"  tensorNames = ["in_0"] : tensor<1x3x60x60xf16> {dynamicStrides}
// CHECK:    DataInfo "in_1"  tensorNames = ["in_1"] : tensor<1x3x60x60xf16> {dynamicStrides}
// CHECK:  } outputsInfo : {
// CHECK:    DataInfo "out_0" tensorNames = ["out_0"] : tensor<1x3x60x60xf16> {dynamicStrides}


  func.func nested @main_part1(%arg0: memref<1x3x60x60xf16> {func.dynamicStrides = true}, %arg1: memref<1x3x60x60xf16> {func.dynamicStrides = true}) -> (memref<1x3x60x60xf16> {func.dynamicStrides = true}){
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %0 : memref<1x3x60x60xf16>
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %alloc = memref.alloc() : memref<1x3x60x60xf16>
    %0 = func.call @main_part1(%arg0, %alloc) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    %1 = VPUIP.Copy inputs(%0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %arg1 : memref<1x3x60x60xf16>
  }
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @HostCompilePreserveDynamicInfo {
// CHECK-COUNT-1: net.NetworkInfo entryPoint : @main inputsInfo : {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_13" : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>}>
  } outputsInfo : {
    DataInfo "Softmax_14" friendlyName = "Result_15" : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>}>
  }

// CHECK-LABEL: module @Module0 attributes {{.+}} {
// CHECK: net.NetworkInfo entryPoint : @main_func0 inputsInfo : {
// CHECK:   DataInfo "in_0" tensorNames = ["in_0"] : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>
// CHECK: } outputsInfo : {
// CHECK:   DataInfo "out_0" tensorNames = ["out_0"] : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>
// CHECK: }
  func.func nested @main_func0(%main: tensor<?x1x548xf16>) -> tensor<?x1x548xf16> {
    %main_1 = Core.ReinterpretCast(%main) : tensor<?x1x548xf16> -> tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>, order = #CHW}>
    %Softmax_14 = VPU.SoftMax(%main_1) {axisInd = 2 : i64} : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>, order = #CHW}> -> tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>, order = #CHW}>
    %main_2 = Core.ReinterpretCast(%Softmax_14) : tensor<?x1x548xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548]> : tensor<3xsi64>, order = #CHW}> -> tensor<?x1x548xf16>
    return %main_2 : tensor<?x1x548xf16>
  }

// CHECK-LABEL: func.func @main
  func.func @main(%Parameter_13: tensor<?x1x548xf16>) -> tensor<?x1x548xf16> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %Softmax_14 = call @main_func0(%Parameter_13) : (tensor<?x1x548xf16>) -> tensor<?x1x548xf16>
    return %Softmax_14 : tensor<?x1x548xf16>

    // CHECK: %[[RES:.*]] = Core.NestedCall @Module0::@main_func0
    // CHECK: return %[[RES]]
  }
}
