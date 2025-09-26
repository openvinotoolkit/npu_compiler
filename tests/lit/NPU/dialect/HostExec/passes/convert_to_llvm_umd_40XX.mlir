//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --mlir-elide-elementsattrs-if-larger 8 --convert-to-llvm-umd-calls  %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @StaticEltwiseNHWC attributes {config.arch = #config.arch_kind<NPU40XX>, config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  //CHECK: llvm.mlir.global internal constant @main1_kernel
  config.PipelineOptions @Options {
    config.Option @VPU.EnableExtraStaticShapeOps : true
    config.Option @VPU.EnableAdaptiveStripping : false
    config.Option @VPU.EnableSEPtrsOperations : false
    config.Option @VPU.EnableExperimentalSEPtrsOperations : false
    config.Option @VPU.EnableVPUNNPreSplit : false
    config.Option @VPU.FP16CompressedConv : false
    config.Option @VPU.EnableDCIM : true
    config.Option @VPU.ReduceSupported : false
    config.Option @VPU.AutoPaddingODU : false
    config.Option @VPU.AutoPaddingIDU : false
    config.Option @VPU.SprLUTEnabled : false
    config.Option @VPU.FragmentationAvoidRatioPipeliningLargeWeights : 4.500000e-01 : f32
    config.Option @VPU.UseDedicatedFifoPerShaveEngine : false
    config.Option @VPU.BarrierMaxVariantSum : 64 : ui64
    config.Option @VPU.BarrierMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxVariantCount : 128 : ui64
    config.Option @VPU.MetadataMaxInvariantCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelInvocationCount : 64 : ui64
    config.Option @VPU.MetadataMaxKernelRangeCount : 64 : ui64
    config.Option @VPU.MetadataMaxMediaCount : 4 : ui64
    config.Option @VPU.MaxKernelSize : 11 : si64
  }
  config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x720x1000xf16>
  }
  HostExec.Binary @OneDMAWithoutAttributes {
    HostExec.BinaryData @serialized_main1 <object = "\7FELF\02\01\00\00\10\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00">
    func.func private @main1(memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(720 : index) : i64
    %2 = llvm.mlir.constant(90 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = builtin.unrealized_conversion_cast %1 : i64 to index
    %5 = builtin.unrealized_conversion_cast %0 : i64 to index
    %6 = llvm.sdiv %1, %2  : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = async.create_group %7 : !async.group
    scf.for %arg2 = %5 to %4 step %3 {
      %subview = memref.subview %arg0[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %9 = builtin.unrealized_conversion_cast %subview : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %subview_0 = memref.subview %arg1[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %10 = builtin.unrealized_conversion_cast %subview_0 : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x90x1000x16xf16>> {
        %13 = Core.NestedCall @OneDMAWithoutAttributes::@main1(%9, %10) : (memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
        async.yield %10 : memref<1x90x1000x16xf16>
      }
      %11 = async.add_to_group %token, %8 : !async.token
      %12 = async.await %bodyResults : !async.value<memref<1x90x1000x16xf16>>
    }
    async.await_all %8
    return %arg1 : memref<1x720x1000x16xf16>
  }
  //CHECK-NOT: async.create_group
  //CHECK-NOT: async.add_to_group
  //CHECK-NOT: async.await_all
  //CHECK-NOT: async.await
  //CHECK-NOT: async.execute
  //CHECK: llvm.call @npu_level_zero_reset_commandlist
  //CHECK: llvm.call @npu_level_zero_execute_graph
  //CHECK: llvm.call @npu_level_zero_submit_commandlist
}

// -----

module @Add attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input1" : tensor<1x16x720x1000xf16>
    DataInfo "input2" : tensor<1x16x720x1000xf16>
  } outputsInfo : {
    DataInfo "Add_3" friendlyName = "output" : tensor<1x16x720x1000xf16>
  }
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0 <object = "\7FELF\02\01\00\00">
    func.func private @main_func0(memref<1x16x720x1000xf16, @DDR>, memref<1x16x720x1000xf16, @DDR>, memref<1x720x1000x16xf16, @DDR>, memref<1x720x1000x16xf16, @DDR>) -> (memref<1x720x1000x16xf16, @DDR>, memref<1x720x1000x16xf16, @DDR>)
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_main_func1 <object = "\7FELF\02\01\00\00">
    func.func private @main_func1(memref<1x720x1000x16xf16, @DDR>, memref<1x16x720x1000xf16, @DDR>) -> memref<1x16x720x1000xf16, @DDR>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func2 <object = "\7FELF\02\01\00\00">
    func.func private @main_func2(memref<1x80x1000x16xf16, @DDR>, memref<1x80x1000x16xf16, @DDR>, memref<1x80x1000x16xf16, @DDR>) -> memref<1x80x1000x16xf16, @DDR>
  }
  func.func @main(%arg0: memref<1x16x720x1000xf16>, %arg1: memref<1x16x720x1000xf16>, %arg2: memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16> {
    %0 = llvm.mlir.constant(80 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(720 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %alloc = memref.alloc() : memref<1x720x1000x16xf16>
    %alloc_0 = memref.alloc() : memref<1x720x1000x16xf16>
    %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x720x1000x16xf16>>, !async.value<memref<1x720x1000x16xf16>>) {
      %12:2 = Core.NestedCall @Module0::@main_func0(%arg0, %arg1, %alloc, %alloc_0) : (memref<1x16x720x1000xf16>, memref<1x16x720x1000xf16>, memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>) -> (memref<1x720x1000x16xf16, @DDR>, memref<1x720x1000x16xf16, @DDR>)
      async.yield %alloc, %alloc_0 : memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>
    }
    %6 = async.await %bodyResults#0 : !async.value<memref<1x720x1000x16xf16>>
    %7 = async.await %bodyResults#1 : !async.value<memref<1x720x1000x16xf16>>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x720x1000x16xf16>
    %8 = llvm.sdiv %2, %0  : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = async.create_group %9 : !async.group
    scf.for %arg3 = %5 to %3 step %1 {
      %subview = memref.subview %6[0, %arg3, 0, 0] [1, 80, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %subview_5 = memref.subview %7[0, %arg3, 0, 0] [1, 80, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %12 = builtin.unrealized_conversion_cast %subview : memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x80x1000x16xf16>
      %13 = builtin.unrealized_conversion_cast %subview_5 : memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x80x1000x16xf16>
      %subview_6 = memref.subview %alloc_1[0, %arg3, 0, 0] [1, 80, 1000, 16] [1, 1, 1, 1] : memref<1x720x1000x16xf16> to memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
      %14 = builtin.unrealized_conversion_cast %subview_6 : memref<1x80x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x80x1000x16xf16>
      %token_7, %bodyResults_8 = async.execute -> !async.value<memref<1x80x1000x16xf16>> {
        %17 = Core.NestedCall @Module2::@main_func2(%12, %13, %14) : (memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>) -> memref<1x80x1000x16xf16, @DDR>
        async.yield %14 : memref<1x80x1000x16xf16>
      }
      %15 = async.add_to_group %token_7, %10 : !async.token
      %16 = async.await %bodyResults_8 : !async.value<memref<1x80x1000x16xf16>>
    }
    async.await_all %10
    %alloc_2 = memref.alloc() : memref<1x16x720x1000xf16>
    %token_3, %bodyResults_4 = async.execute -> !async.value<memref<1x16x720x1000xf16>> {
      %12 = Core.NestedCall @Module1::@main_func1(%alloc_1, %alloc_2) : (memref<1x720x1000x16xf16>, memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16, @DDR>
      async.yield %alloc_2 : memref<1x16x720x1000xf16>
    }
    %11 = async.await %bodyResults_4 : !async.value<memref<1x16x720x1000xf16>>
    memref.copy %11, %arg2 : memref<1x16x720x1000xf16> to memref<1x16x720x1000xf16>
    return %arg2 : memref<1x16x720x1000xf16>
  }
  //CHECK-NOT: async.create_group
  //CHECK-NOT: async.add_to_group
  //CHECK-NOT: async.await_all
  //CHECK-NOT: async.await
  //CHECK-NOT: async.execute
  //CHECK: llvm.call @npu_level_zero_reset_commandlist
  //CHECK: llvm.call @npu_level_zero_execute_graph
  //CHECK: llvm.call @npu_level_zero_submit_commandlist
}
