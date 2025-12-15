//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --pass-pipeline="builtin.module(builtin.module(set-memory-space{memory-space=DDR set-memory-space-for-function-boundaries=false}))" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

//CHECK-LABEL: @NestedFunction
module @NestedFunction {
  //CHECK-LABEL: @Module0
  module @Module0  {
    //CHECK-LABEL: func.func @main_func0
    //CHECK-SAME: ([[SUBMAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
    func.func @main_func0(%arg0: memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16> {
      %alloc = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC0:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %alloc2 = memref.alloc() : memref<1x89x1000x16xf16>
      //CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x89x1000x16xf16, @DDR>
      %0 = VPUIP.Copy inputs(%alloc2 : memref<1x89x1000x16xf16>) outputs(%alloc : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ALLOC1]] : memref<1x89x1000x16xf16, @DDR>) outputs([[ALLOC0]] : memref<1x89x1000x16xf16, @DDR>) -> memref<1x89x1000x16xf16, @DDR>
      %1 = VPUIP.Copy inputs(%0 : memref<1x89x1000x16xf16>) outputs(%arg0 : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      //CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x89x1000x16xf16, @DDR>) outputs([[SUBMAIN_ARG0]] : memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
      return %1 : memref<1x89x1000x16xf16>
      //CHECK: return [[COPY1]] : memref<1x89x1000x16xf16>
    }
  }

  //CHECK-LABEL: func.func @main
  //CHECK-SAME: ([[MAIN_ARG0:%.+]]: memref<1x89x1000x16xf16>) {
  func.func @main(%arg0 : memref<1x89x1000x16xf16>) {    
    %0 = Core.NestedCall @Module0::@main_func0(%arg0) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    //CHECK: Core.NestedCall @Module0::@main_func0([[MAIN_ARG0]]) : (memref<1x89x1000x16xf16>) -> memref<1x89x1000x16xf16>
    return
  }
}
