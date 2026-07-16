//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --serialize-elf-to-binary %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @UsingScfFor

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @UsingScfFor {
  module @Module0 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_fn1 inputsInfo : {
      DataInfo "in_0" : tensor<1x64x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x64x512x512xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_fn1() {
      ELF.Main {}
      return
    }
  }

  module @Module1 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_outline1 inputsInfo : {
      DataInfo "in_0" : tensor<1x3x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x64x512x512xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x3x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_outline1() {
      ELF.Main {}
      return
    }
  }

  module @Module2 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_outline2 inputsInfo : {
      DataInfo "in_0" : tensor<1x64x512x512xf16>
      DataInfo "in_1" : tensor<1x64x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x3x2048x2048xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @in_0_buffDecl_0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
      VPUASM.DeclareBuffer @in_1_buffDecl_1 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @out_0_buffDecl_0 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x2048x2048xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_outline2() {
      ELF.Main {}
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x3x512x512xf16>
    DataInfo "input2" : tensor<1x3x2048x2048xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x2048x2048xf16>
  }

  func.func @main(%input.1: memref<1x3x512x512xf16>, %main: memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(23 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %main_0 = llvm.mlir.constant(33554432 : index) : i64
    %main_1 = builtin.unrealized_conversion_cast %main_0 : i64 to index
    %main_2 = llvm.mlir.constant(58720256 : index) : i64
    %main_3 = builtin.unrealized_conversion_cast %main_2 : i64 to index
    %main_4 = memref.alloc() {alignment = 64 : i64} : memref<92274688xi8>
    %main_5 = memref.view %main_4[%5][] : memref<92274688xi8> to memref<1x64x512x512xf16>
    %main_6 = memref.view %main_4[%main_1][] : memref<92274688xi8> to memref<1x64x512x512xf16>
    %main_7 = memref.view %main_4[%main_3][] : memref<92274688xi8> to memref<1x3x2048x2048xf16>
    %main_8, %main_9 = async.execute -> !async.value<memref<1x64x512x512xf16>> {
      %main_16 = Core.NestedCall @Module1::@main_outline1(%input.1, %main_5) : (memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
      async.yield %main_16 : memref<1x64x512x512xf16>
    }
    %main_10 = async.await %main_9 : !async.value<memref<1x64x512x512xf16>>
    %main_11 = async.create_group %3 : !async.group
    %main_12 = scf.for %main_16 = %5 to %3 step %1 iter_args(%main_17 = %main_10) -> (memref<1x64x512x512xf16>) {
      %main_18, %main_19 = async.execute -> !async.value<memref<1x64x512x512xf16>> {
        %main_22 = Core.NestedCall @Module0::@main_fn1(%main_17, %main_6) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
        async.yield %main_22 : memref<1x64x512x512xf16>
      }
      %main_20 = async.add_to_group %main_18, %main_11 : !async.token
      %main_21 = async.await %main_19 : !async.value<memref<1x64x512x512xf16>>
      scf.yield %main_21 : memref<1x64x512x512xf16>
    }
    async.await_all %main_11
    %main_13, %main_14 = async.execute -> !async.value<memref<1x3x2048x2048xf16>> {
      %main_16 = Core.NestedCall @Module2::@main_outline2(%main_12, %main_10, %main_7) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
      async.yield %main_16 : memref<1x3x2048x2048xf16>
    }
    %main_15 = async.await %main_14 : !async.value<memref<1x3x2048x2048xf16>>
    memref.copy %main_15, %main : memref<1x3x2048x2048xf16> to memref<1x3x2048x2048xf16>
    return %main : memref<1x3x2048x2048xf16>
  }

  // CHECK: HostExec.Binary @Module0 {
  // CHECK:   HostExec.BinaryData @serialized_main_fn1 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_fn1(memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK: }
  // CHECK: HostExec.Binary @Module1 {
  // CHECK:   HostExec.BinaryData @serialized_main_outline1 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_outline1(memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK: }
  // CHECK: HostExec.Binary @Module2 {
  // CHECK:   HostExec.BinaryData @serialized_main_outline2 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_outline2(memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
  // CHECK: }
  // CHECK: func.func @main([[INPUT_1:%[^:]+]]: memref<1x3x512x512xf16>, [[MAIN:%[^:]+]]: memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16> {
  // CHECK:   [[TMP_0:%[^=]+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK:   [[TMP_1:%[^=]+]] = builtin.unrealized_conversion_cast [[TMP_0]] : i64 to index
  // CHECK:   [[TMP_2:%[^=]+]] = llvm.mlir.constant(23 : index) : i64
  // CHECK:   [[TMP_3:%[^=]+]] = builtin.unrealized_conversion_cast [[TMP_2]] : i64 to index
  // CHECK:   [[TMP_4:%[^=]+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK:   [[TMP_5:%[^=]+]] = builtin.unrealized_conversion_cast [[TMP_4]] : i64 to index
  // CHECK:   [[MAIN_0:%[^=]+]] = llvm.mlir.constant(33554432 : index) : i64
  // CHECK:   [[MAIN_1:%[^=]+]] = builtin.unrealized_conversion_cast [[MAIN_0]] : i64 to index
  // CHECK:   [[MAIN_2:%[^=]+]] = llvm.mlir.constant(58720256 : index) : i64
  // CHECK:   [[MAIN_3:%[^=]+]] = builtin.unrealized_conversion_cast [[MAIN_2]] : i64 to index
  // CHECK:   [[MAIN_4:%[^=]+]] = memref.alloc() {alignment = 64 : i64} : memref<92274688xi8>
  // CHECK:   [[MAIN_5:%[^=]+]] = memref.view [[MAIN_4]][[[TMP_5]]][] : memref<92274688xi8> to memref<1x64x512x512xf16>
  // CHECK:   [[MAIN_6:%[^=]+]] = memref.view [[MAIN_4]][[[MAIN_1]]][] : memref<92274688xi8> to memref<1x64x512x512xf16>
  // CHECK:   [[MAIN_7:%[^=]+]] = memref.view [[MAIN_4]][[[MAIN_3]]][] : memref<92274688xi8> to memref<1x3x2048x2048xf16>
  // CHECK:   [[MAIN_8:%[^,]+]], [[MAIN_9:%[^=]+]] = async.execute -> !async.value<memref<1x64x512x512xf16>> {
  // CHECK:     [[MAIN_16:%[^=]+]] = Core.NestedCall @Module1::@main_outline1([[INPUT_1]], [[MAIN_5]]) : (memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:     async.yield [[MAIN_5]] : memref<1x64x512x512xf16>
  // CHECK:   }
  // CHECK:   [[MAIN_10:%[^=]+]] = async.await [[MAIN_9]] : !async.value<memref<1x64x512x512xf16>>
  // CHECK:   [[MAIN_11:%[^=]+]] = async.create_group [[TMP_3]] : !async.group
  // CHECK:   [[MAIN_12:%[^=]+]] = scf.for [[MAIN_16:%[^=]+]] = [[TMP_5]] to [[TMP_3]] step [[TMP_1]] iter_args([[MAIN_17:%[^=]+]] = [[MAIN_10]]) -> (memref<1x64x512x512xf16>) {
  // CHECK:     [[MAIN_18:%[^,]+]], [[MAIN_19:%[^=]+]] = async.execute -> !async.value<memref<1x64x512x512xf16>> {
  // CHECK:       [[MAIN_22:%[^=]+]] = Core.NestedCall @Module0::@main_fn1([[MAIN_17]], [[MAIN_6]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:       async.yield [[MAIN_6]] : memref<1x64x512x512xf16>
  // CHECK:     }
  // CHECK:     [[MAIN_20:%[^=]+]] = async.add_to_group [[MAIN_18]], [[MAIN_11]] : !async.token
  // CHECK:     [[MAIN_21:%[^=]+]] = async.await [[MAIN_19]] : !async.value<memref<1x64x512x512xf16>>
  // CHECK:     scf.yield [[MAIN_21]] : memref<1x64x512x512xf16>
  // CHECK:   }
  // CHECK:   async.await_all [[MAIN_11]]
  // CHECK:   [[MAIN_13:%[^,]+]], [[MAIN_14:%[^=]+]] = async.execute -> !async.value<memref<1x3x2048x2048xf16>> {
  // CHECK:     [[MAIN_16:%[^=]+]] = Core.NestedCall @Module2::@main_outline2([[MAIN_12]], [[MAIN_10]], [[MAIN_7]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
  // CHECK:     async.yield [[MAIN_7]] : memref<1x3x2048x2048xf16>
  // CHECK:   }
  // CHECK:   [[MAIN_15:%[^=]+]] = async.await [[MAIN_14]] : !async.value<memref<1x3x2048x2048xf16>>
  // CHECK:   memref.copy [[MAIN_15]], [[MAIN]] : memref<1x3x2048x2048xf16> to memref<1x3x2048x2048xf16>
  // CHECK:   return [[MAIN]] : memref<1x3x2048x2048xf16>
  // CHECK: }
}


// -----

// CHECK-LABEL: @MultipleCalls

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @MultipleCalls {
  module @Module0 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_fn1 inputsInfo : {
      DataInfo "in_0" : tensor<1x64x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x64x512x512xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_fn1() {
      ELF.Main {}
      return
    }
  }

  module @Module1 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_outline1 inputsInfo : {
      DataInfo "in_0" : tensor<1x3x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x64x512x512xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @input_0_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x3x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @output_0_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_outline1() {
      ELF.Main {}
      return
    }
  }

  module @Module2 {
    net.NetworkInfo {inferenceTiming = 45511766 : i64} entryPoint : @main_outline2 inputsInfo : {
      DataInfo "in_0" : tensor<1x64x512x512xf16>
      DataInfo "in_1" : tensor<1x64x512x512xf16>
    } outputsInfo : {
      DataInfo "out_0" : tensor<1x3x2048x2048xf16>
    }
    VPUASM.InputBindings inputDeclarations : {
      VPUASM.DeclareBuffer @in_0_buffDecl_0 !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
      VPUASM.DeclareBuffer @in_1_buffDecl_1 !VPUASM.Buffer< "NetworkInput"[1] <0> : memref<1x64x512x512xf16> :  swizzling(0)>
    }
    VPUASM.OutputBindings outputDeclarations : {
      VPUASM.DeclareBuffer @out_0_buffDecl_0 !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x3x2048x2048xf16> :  swizzling(0)>
    }
    VPUASM.ProfilingBindings profilingDeclarations : {
    }
    func.func @main_outline2() {
      ELF.Main {}
      return
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x3x512x512xf16>
    DataInfo "input2" : tensor<1x3x2048x2048xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x2048x2048xf16>
  }

  func.func @main(%arg1: memref<1x3x512x512xf16>, %arg2: memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16> {
    %0 = memref.alloc() : memref<1x64x512x512xf16>
    %1 = memref.alloc() : memref<1x64x512x512xf16>
    %2 = memref.alloc() : memref<1x3x2048x2048xf16>
    %3 = Core.NestedCall @Module1::@main_outline1(%arg1, %0) : (memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
    %4 = Core.NestedCall @Module0::@main_fn1(%3, %1) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
    %5 = Core.NestedCall @Module0::@main_fn1(%4, %1) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
    %6 = Core.NestedCall @Module0::@main_fn1(%5, %1) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
    %7 = Core.NestedCall @Module0::@main_fn1(%6, %1) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
    %8 = Core.NestedCall @Module2::@main_outline2(%7, %3, %2) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
    memref.copy %8, %arg2 : memref<1x3x2048x2048xf16> to memref<1x3x2048x2048xf16>
    return %arg2 : memref<1x3x2048x2048xf16>
  }

  // CHECK: HostExec.Binary @Module0 {
  // CHECK:   HostExec.BinaryData @serialized_main_fn1 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_fn1(memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK: }
  // CHECK: HostExec.Binary @Module1 {
  // CHECK:   HostExec.BinaryData @serialized_main_outline1 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_outline1(memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK: }
  // CHECK: HostExec.Binary @Module2 {
  // CHECK:   HostExec.BinaryData @serialized_main_outline2 <object = "\7FELF\02\01\00\00\00\{{.+}}">
  // CHECK:   func.func nested @main_outline2(memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
  // CHECK: }
  // CHECK: func.func @main([[ARG_0:%[^:]+]]: memref<1x3x512x512xf16>, [[ARG_1:%[^:]+]]: memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16> {
  // CHECK:   [[TMP_0:%[^=]+]] = memref.alloc() : memref<1x64x512x512xf16>
  // CHECK:   [[TMP_1:%[^=]+]] = memref.alloc() : memref<1x64x512x512xf16>
  // CHECK:   [[TMP_2:%[^=]+]] = memref.alloc() : memref<1x3x2048x2048xf16>
  // CHECK:   [[TMP_3:%[^=]+]] = Core.NestedCall @Module1::@main_outline1([[ARG_0]], [[TMP_0]]) : (memref<1x3x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:   [[TMP_4:%[^=]+]] = Core.NestedCall @Module0::@main_fn1([[TMP_0]], [[TMP_1]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:   [[TMP_5:%[^=]+]] = Core.NestedCall @Module0::@main_fn1([[TMP_1]], [[TMP_1]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:   [[TMP_6:%[^=]+]] = Core.NestedCall @Module0::@main_fn1([[TMP_1]], [[TMP_1]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:   [[TMP_7:%[^=]+]] = Core.NestedCall @Module0::@main_fn1([[TMP_1]], [[TMP_1]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>) -> memref<1x64x512x512xf16>
  // CHECK:   [[TMP_8:%[^=]+]] = Core.NestedCall @Module2::@main_outline2([[TMP_1]], [[TMP_0]], [[TMP_2]]) : (memref<1x64x512x512xf16>, memref<1x64x512x512xf16>, memref<1x3x2048x2048xf16>) -> memref<1x3x2048x2048xf16>
  // CHECK:   memref.copy [[TMP_2]], [[ARG_1]] : memref<1x3x2048x2048xf16> to memref<1x3x2048x2048xf16>
  // CHECK:   return [[ARG_1]] : memref<1x3x2048x2048xf16>
  // CHECK: }
}
