//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler=platform=%platform% --split-input-file --pack-nested-modules=enable-profiling=true %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @AddingProfilingOutput {
  net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "in_0" : tensor<1x3x60x60xf16>
  } outputsInfo : {
      DataInfo "out_0" : tensor<1x3x60x60xf16>
  }

    // CHECK-LABEL: module @Module0 attributes
    // CHECK:  config.PipelineOptions @Options {
    // CHECK:  config.Resources {{.+}} of @NCE at {{.+}} MHz
    // CHECK:  config.Resources {{.+}} of @global
    // CHECK:  net.NetworkInfo entryPoint : @main_part1 inputsInfo : {
    // CHECK-NEXT:  DataInfo "in_0" tensorNames = ["in_0"] : tensor<1x3x60x60xf16>
    // CHECK-NEXT:  DataInfo "in_1" tensorNames = ["in_1"] : tensor<1x3x60x60xf16>
    // CHECK-NEXT: } outputsInfo : {
    // CHECK-NEXT:  DataInfo "out_0" tensorNames = ["out_0"] : tensor<1x3x60x60xf16>
    // CHECK: profilingOutputsInfo : {
  func.func nested @main_part1(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x60x60xf16>) outputs(%arg1 : memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %0 : memref<1x3x60x60xf16>
  }

  func.func @main(%arg0: memref<1x3x60x60xf16>, %arg1: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
    %call = func.call @main_part1(%arg0, %arg1) : (memref<1x3x60x60xf16>, memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16>
    return %call : memref<1x3x60x60xf16>
  }
}
