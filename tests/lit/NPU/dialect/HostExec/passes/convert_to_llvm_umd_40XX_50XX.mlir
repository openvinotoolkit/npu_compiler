//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --mlir-elide-elementsattrs-if-larger 8 --convert-to-llvm-umd-calls %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 47)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0 - 47)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0 - 256)>
module @ConvChain attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_1" tensorNames = ["input"] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "add1.0" friendlyName = "output" tensorNames = ["output"] : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_merged_vpu_func_0 <object = "\7FELF\02\01\00\00\00\00\00">
    func.func nested @merged_vpu_func_0(memref<1x3x188x256xf16>, memref<1x188x256x3xf16>) -> memref<1x188x256x3xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func0_static <object = "\7FELF\02\01\00\00\00\00\00\00">
    func.func nested @main_func0_static(memref<1x3x47x256xf16>, memref<1x47x256x3xf16>) -> memref<1x47x256x3xf16>
  }
  func.func @main(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(6 : index) : i64
    %3 = llvm.mlir.constant(188 : index) : i64
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = llvm.mlir.constant(46 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(256 : index) : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    %9 = llvm.mlir.constant(47 : index) : i64
    %10 = builtin.unrealized_conversion_cast %9 : i64 to index
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = builtin.unrealized_conversion_cast %11 : i64 to index
    %13 = llvm.mlir.constant(3 : index) : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %15 = llvm.mlir.constant(2 : index) : i64
    %16 = builtin.unrealized_conversion_cast %15 : i64 to index
    %dim = memref.dim %arg0, %16 : memref<1x3x?x?xf16>
    %17 = builtin.unrealized_conversion_cast %dim : index to i64
    %dim_0 = memref.dim %arg0, %14 : memref<1x3x?x?xf16>
    %18 = builtin.unrealized_conversion_cast %dim_0 : index to i64
    %19 = llvm.mul %17, %18 : i64
    %20 = llvm.mul %19, %0 : i64
    %21 = builtin.unrealized_conversion_cast %20 : i64 to index
    %alloc = memref.alloc(%21) {alignment = 64 : i64} : memref<?xi8>
    %view = memref.view %alloc[%12][%dim, %dim_0] : memref<?xi8> to memref<1x?x?x3xf16>
    %22 = llvm.urem %17, %9 : i64
    %23 = builtin.unrealized_conversion_cast %22 : i64 to index
    %24 = llvm.urem %18, %7 : i64
    %25 = builtin.unrealized_conversion_cast %24 : i64 to index
    %26 = llvm.add %17, %5 : i64
    %27 = llvm.udiv %26, %9 : i64
    %28 = llvm.srem %27, %6 : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.mul %29, %9 : i64
    %31 = builtin.unrealized_conversion_cast %30 : i64 to index
    %32 = llvm.sdiv %30, %3 : i64
    %33 = builtin.unrealized_conversion_cast %32 : i64 to index
    %34 = async.create_group %33 : !async.group
    scf.for %arg2 = %12 to %dim step %4 {
      scf.for %arg3 = %12 to %dim_0 step %8 {
        %subview = memref.subview %arg0[0, 0, %arg2, %arg3] [1, 3, 188, 256] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x188x256xf16, strided<[?, ?, ?, 1], offset: ?>>
        %71 = builtin.unrealized_conversion_cast %subview : memref<1x3x188x256xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x188x256xf16>
        %subview_1 = memref.subview %view[0, %arg2, %arg3, 0] [1, 188, 256, 3] [1, 1, 1, 1] : memref<1x?x?x3xf16> to memref<1x188x256x3xf16, strided<[?, ?, 3, 1], offset: ?>>
        %72 = builtin.unrealized_conversion_cast %subview_1 : memref<1x188x256x3xf16, strided<[?, ?, 3, 1], offset: ?>> to memref<1x188x256x3xf16>
        %token, %bodyResults = async.execute -> !async.value<memref<1x188x256x3xf16>> {
          %75 = Core.NestedCall @Module1::@merged_vpu_func_0(%71, %72) : (memref<1x3x188x256xf16>, memref<1x188x256x3xf16>) -> memref<1x188x256x3xf16>
          async.yield %72 : memref<1x188x256x3xf16>
        }
        %73 = async.add_to_group %token, %34 : !async.token
        %74 = async.await %bodyResults : !async.value<memref<1x188x256x3xf16>>
      }
    } {no_await_all = true}
    %35 = llvm.sub %17, %30 : i64
    %36 = llvm.sdiv %35, %9 : i64
    %37 = builtin.unrealized_conversion_cast %36 : i64 to index
    %38 = async.create_group %37 : !async.group {no_reset_cmdlist = true}
    scf.for %arg2 = %31 to %dim step %10 {
      scf.for %arg3 = %12 to %dim_0 step %8 {
        %subview = memref.subview %arg0[0, 0, %arg2, %arg3] [1, 3, 47, 256] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x47x256xf16, strided<[?, ?, ?, 1], offset: ?>>
        %56 = builtin.unrealized_conversion_cast %subview : memref<1x3x47x256xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x47x256xf16>
        %subview_1 = memref.subview %view[0, %arg2, %arg3, 0] [1, 47, 256, 3] [1, 1, 1, 1] : memref<1x?x?x3xf16> to memref<1x47x256x3xf16, strided<[?, ?, 3, 1], offset: ?>>
        %57 = builtin.unrealized_conversion_cast %subview_1 : memref<1x47x256x3xf16, strided<[?, ?, 3, 1], offset: ?>> to memref<1x47x256x3xf16>
        %token, %bodyResults = async.execute -> !async.value<memref<1x47x256x3xf16>> {
          %60 = Core.NestedCall @Module2::@main_func0_static(%56, %57) : (memref<1x3x47x256xf16>, memref<1x47x256x3xf16>) -> memref<1x47x256x3xf16>
          async.yield %57 : memref<1x47x256x3xf16>
        }
        %58 = async.add_to_group %token, %38 : !async.token
        %59 = async.await %bodyResults : !async.value<memref<1x47x256x3xf16>>
      }
    } {no_reset_cmdlist = true}
    async.await_all %38
    return %arg1 : memref<1x3x?x?xf16>

    //CHECK-COUNT-1: llvm.call @npu_level_zero_submit_commandlist
    //CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
  }
}

// -----

module @StaticEltwiseNHWC attributes {config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  //CHECK: llvm.mlir.global internal constant @main1_kernel
  config.PipelineOptions @Options {
    config.Option @config.EnableExtraStaticShapeOps : true
    config.Option @config.EnableAdaptiveStripping : false
    config.Option @config.EnableSEPtrsOperations : false
    config.Option @config.EnableExperimentalSEPtrsOperations : false
    config.Option @config.EnableVPUNNPreSplit : false
    config.Option @config.FP16CompressedConv : false
    config.Option @config.EnableDCIM : true
    config.Option @config.ReduceSupported : false
    config.Option @config.AutoPaddingODU : false
    config.Option @config.AutoPaddingIDU : false
    config.Option @config.SprLUTEnabled : false
    config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 4.500000e-01 : f32
    config.Option @config.UseDedicatedFifoPerShaveEngine : false
    config.Option @config.BarrierMaxSlotCount : 256 : ui64
    config.Option @config.MetadataMaxVariantCount : 128 : ui64
    config.Option @config.MetadataMaxInvariantCount : 64 : ui64
    config.Option @config.MetadataMaxKernelInvocationCount : 64 : ui64
    config.Option @config.MetadataMaxKernelRangeCount : 64 : ui64
    config.Option @config.MetadataMaxMediaCount : 4 : ui64
  }
  config.Resources 6 of @NCE at 1.850000e+03 MHz {
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
    func.func nested @main1(memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> attributes {config.pureHostCompileFunc} {
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
    func.func nested @main_func0(memref<1x16x720x1000xf16>, memref<1x16x720x1000xf16>, memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>) -> (memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>)
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_main_func1 <object = "\7FELF\02\01\00\00">
    func.func nested @main_func1(memref<1x720x1000x16xf16>, memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func2 <object = "\7FELF\02\01\00\00">
    func.func nested @main_func2(memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>) -> memref<1x80x1000x16xf16>
  }
  func.func @main(%arg0: memref<1x16x720x1000xf16>, %arg1: memref<1x16x720x1000xf16>, %arg2: memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(80 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(720 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %alloc = memref.alloc() : memref<1x720x1000x16xf16>
    %alloc_0 = memref.alloc() : memref<1x720x1000x16xf16>
    %token, %bodyResults:2 = async.execute -> (!async.value<memref<1x720x1000x16xf16>>, !async.value<memref<1x720x1000x16xf16>>) {
      %12:2 = Core.NestedCall @Module0::@main_func0(%arg0, %arg1, %alloc, %alloc_0) : (memref<1x16x720x1000xf16>, memref<1x16x720x1000xf16>, memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>) -> (memref<1x720x1000x16xf16>, memref<1x720x1000x16xf16>)
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
        %17 = Core.NestedCall @Module2::@main_func2(%12, %13, %14) : (memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>, memref<1x80x1000x16xf16>) -> memref<1x80x1000x16xf16>
        async.yield %14 : memref<1x80x1000x16xf16>
      }
      %15 = async.add_to_group %token_7, %10 : !async.token
      %16 = async.await %bodyResults_8 : !async.value<memref<1x80x1000x16xf16>>
    }
    async.await_all %10
    %alloc_2 = memref.alloc() : memref<1x16x720x1000xf16>
    %token_3, %bodyResults_4 = async.execute -> !async.value<memref<1x16x720x1000xf16>> {
      %12 = Core.NestedCall @Module1::@main_func1(%alloc_1, %alloc_2) : (memref<1x720x1000x16xf16>, memref<1x16x720x1000xf16>) -> memref<1x16x720x1000xf16>
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
  //CHECK-NOT: llvm.call @npu_level_zero_reset_commandlist
  //CHECK: llvm.call @npu_level_zero_execute_graph
  //CHECK: llvm.call @npu_level_zero_submit_commandlist
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<()[s0] -> (s0 - 15)>
module @EltwiseWithOutputShape attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  config.PipelineOptions @Options {
    config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 3.600000e-01 : f32
    config.Option @config.UseDedicatedFifoPerShaveEngine : true
    config.Option @config.BarrierMaxSlotCount : 256 : ui64
    config.Option @config.DpuFIFOAddrs : [788529152, 788529184, 788529216, 788529248]
    config.Option @config.ShvFIFOAddrs : [788578304, 788578336, 788578368, 788578400, 788578432, 788578464, 788578496, 788578528]
    config.Option @config.BarrierFIFOAddr : 788594688 : ui64
    config.Option @config.BarrierFIFODepth : 4 : ui64
    config.Option @config.MetadataMaxVariantCount : 128 : ui64
    config.Option @config.MetadataMaxInvariantCount : 64 : ui64
    config.Option @config.MetadataMaxKernelInvocationCount : 32 : ui64
    config.Option @config.MetadataMaxKernelRangeCount : 32 : ui64
    config.Option @config.MetadataMaxMediaCount : 4 : ui64
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
    config.Option @config.AsymmetricPerTensorZP : false
    config.Option @config.AsymmetricPerChannelZP : false
    config.Option @config.ReduceSupported : false
    config.Option @config.FP16CompressedConv : false
    config.Option @config.EnableVPUNNPreSplit : true
    config.Option @config.EnableODULocalRegion : false
    config.Option @config.EnableSEPtrsOperations : true
    config.Option @config.EnableExperimentalSEPtrsOperations : false
    config.Option @config.EnableQDQOptimizationAggressive : false
    config.Option @config.EnableAdaptiveStripping : false
    config.Option @config.EnableExtraStaticShapeOps : true
    config.Option @config.WeightsTableReuseMode : 1 : ui64
    config.Option @config.EnableProfiling : false
    config.Option @config.EnableWeightsDynamicDequantization : false
    config.Option @config.SprLUTEnabled : true
    config.Option @config.EnableDCIM : true
  }
  config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "param0" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    DataInfo "param1" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "Add_15" friendlyName = "Result_16" : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }
  // CHECK: func.func @output_shape({{%[^:]+}}: memref<1x16x?x1280xf16>, {{%[^:]+}}: memref<1x16x?x1280xf16>, {{%[^:]+}}: memref<4xi64>) attributes {config.pureHostCompileFunc} {
  func.func @output_shape(%arg0: memref<1x16x?x1280xf16>, %arg1: memref<1x16x?x1280xf16>, %arg2: memref<4xi64>) -> memref<4xi64> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(3 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(1280 : i64) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = builtin.unrealized_conversion_cast %7 : i64 to index
    %9 = llvm.mlir.constant(16 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %dim = memref.dim %arg0, %8 : memref<1x16x?x1280xf16>
    %11 = builtin.unrealized_conversion_cast %dim : index to i64
    memref.store %10, %arg2[%5] : memref<4xi64>
    // CHECK-NOT: memref.store
    // CHECK:     llvm.store
    memref.store %9, %arg2[%3] : memref<4xi64>
    // CHECK-NOT: memref.store
    // CHECK:     llvm.store
    memref.store %11, %arg2[%8] : memref<4xi64>
    // CHECK-NOT: memref.store
    // CHECK:     llvm.store
    memref.store %6, %arg2[%1] : memref<4xi64>
    // CHECK-NOT: memref.store
    // CHECK:     llvm.store
    return %arg2 : memref<4xi64>
    // CHECK:     return
  }
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0_static <object = "\7FELF\02\01\00\00\00\00\00">
    func.func nested @main_func0_static(memref<1x16x15x1280xf16>, memref<1x16x15x1280xf16>, memref<1x16x15x1280xf16>) -> memref<1x16x15x1280xf16>
  }
  // CHECK: func.func @main({{%[^:]+}}: memref<1x16x?x1280xf16>, {{%[^:]+}}: memref<1x16x?x1280xf16>, {{%[^:]+}}: memref<1x16x?x1280xf16>, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: i64, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr) attributes {config.pureHostCompileFunc} {
  func.func @main(%arg0: memref<1x16x?x1280xf16>, %arg1: memref<1x16x?x1280xf16>, %arg2: memref<1x16x?x1280xf16>) -> memref<1x16x?x1280xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(15 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %dim = memref.dim %arg0, %5 : memref<1x16x?x1280xf16>
    %6 = builtin.unrealized_conversion_cast %dim : index to i64
    %7 = llvm.icmp "sge" %6, %0 : i64
    cf.assert %7, "Not enough elements to backtrack in scf.for loop for Output tensor"
    %8 = llvm.sdiv %6, %0 : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = async.create_group %9 : !async.group
    scf.for %arg3 = %3 to %dim step %1 {
      %11 = builtin.unrealized_conversion_cast %arg3 : index to i64
      %12 = llvm.add %11, %0 : i64
      %13 = llvm.icmp "sgt" %12, %6 : i64
      %14 = scf.if %13 -> (index) {
        %20 = affine.apply #map()[%dim]
        scf.yield %20 : index
      } else {
        scf.yield %arg3 : index
      }
      %subview = memref.subview %arg0[0, 0, %14, 0] [1, 16, 15, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, 0, %14, 0] [1, 16, 15, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %15 = builtin.unrealized_conversion_cast %subview : memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x15x1280xf16>
      %16 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x15x1280xf16>
      %subview_1 = memref.subview %arg2[0, 0, %14, 0] [1, 16, 15, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %17 = builtin.unrealized_conversion_cast %subview_1 : memref<1x16x15x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x15x1280xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x16x15x1280xf16>> {
        %20 = Core.NestedCall @Module0::@main_func0_static(%15, %16, %17) : (memref<1x16x15x1280xf16>, memref<1x16x15x1280xf16>, memref<1x16x15x1280xf16>) -> memref<1x16x15x1280xf16>
        async.yield %17 : memref<1x16x15x1280xf16>
      }
      %18 = async.add_to_group %token, %10 : !async.token
      %19 = async.await %bodyResults : !async.value<memref<1x16x15x1280xf16>>
    }
    async.await_all %10
    return %arg2 : memref<1x16x?x1280xf16>
  }
  // CHECK-NOT: async.create_group
  // CHECK-NOT: async.execute
  // CHECK-NOT: async.add_to_group
  // CHECK-NOT: async.await
  // CHECK-NOT: async.await_all
  // CHECK:     llvm.call @npu_level_zero_execute_graph
  // CHECK:     llvm.call @npu_level_zero_submit_commandlist
}

// -----

module @StaticEltwiseWithOutputShape attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  memref.global "nested" constant @__constant_4xi64 : memref<4xi64> = dense<[1, 16, 1280, 1280]> {alignment = 64 : i64}
  config.PipelineOptions @Options {
    config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 3.600000e-01 : f32
    config.Option @config.UseDedicatedFifoPerShaveEngine : true
    config.Option @config.BarrierMaxSlotCount : 256 : ui64
    config.Option @config.DpuFIFOAddrs : [788529152, 788529184, 788529216, 788529248]
    config.Option @config.ShvFIFOAddrs : [788578304, 788578336, 788578368, 788578400, 788578432, 788578464, 788578496, 788578528]
    config.Option @config.BarrierFIFOAddr : 788594688 : ui64
    config.Option @config.BarrierFIFODepth : 4 : ui64
    config.Option @config.MetadataMaxVariantCount : 128 : ui64
    config.Option @config.MetadataMaxInvariantCount : 64 : ui64
    config.Option @config.MetadataMaxKernelInvocationCount : 32 : ui64
    config.Option @config.MetadataMaxKernelRangeCount : 32 : ui64
    config.Option @config.MetadataMaxMediaCount : 4 : ui64
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
    config.Option @config.AsymmetricPerTensorZP : false
    config.Option @config.AsymmetricPerChannelZP : false
    config.Option @config.ReduceSupported : false
    config.Option @config.FP16CompressedConv : false
    config.Option @config.EnableVPUNNPreSplit : true
    config.Option @config.EnableODULocalRegion : false
    config.Option @config.EnableSEPtrsOperations : true
    config.Option @config.EnableExperimentalSEPtrsOperations : false
    config.Option @config.EnableQDQOptimizationAggressive : false
    config.Option @config.EnableAdaptiveStripping : false
    config.Option @config.EnableExtraStaticShapeOps : true
    config.Option @config.WeightsTableReuseMode : 1 : ui64
    config.Option @config.EnableProfiling : false
    config.Option @config.EnableWeightsDynamicDequantization : false
    config.Option @config.SprLUTEnabled : true
    config.Option @config.EnableDCIM : true
  }
  config.Resources 3 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "param0" : tensor<1x16x1280x1280xf16>
    DataInfo "param1" : tensor<1x16x1280x1280xf16>
  } outputsInfo : {
    DataInfo "Add_15" friendlyName = "Result_16" : tensor<1x16x1280x1280xf16>
  }
  // CHECK: func.func @output_shape({{%[^:]+}}: memref<1x16x1280x1280xf16>, {{%[^:]+}}: memref<1x16x1280x1280xf16>, {{%[^:]+}}: memref<4xi64>) attributes {config.pureHostCompileFunc} {
  func.func @output_shape(%arg0: memref<1x16x1280x1280xf16>, %arg1: memref<1x16x1280x1280xf16>, %arg2: memref<4xi64>) -> memref<4xi64> attributes {config.pureHostCompileFunc} {
    %0 = memref.get_global @__constant_4xi64 : memref<4xi64>
    // CHECK-NOT: memref.get_global @__constant_4xi64
    // CHECK:     llvm.mlir.addressof @__constant_4xi64
    memref.copy %0, %arg2 : memref<4xi64> to memref<4xi64>
    // CHECK-NOT: memref.copy
    // CHECK:     "llvm.intr.memcpy"
    return %arg2 : memref<4xi64>
  }
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0_static <object = "\7FELF\02\01\00\00\00\00\00">
    func.func nested @main_func0_static(memref<1x16x50x1280xf16>, memref<1x50x1280x16xf16>) -> memref<1x50x1280x16xf16>
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_main_func1_static <object = "\7FELF\02\01\00\00\00\00\00">
    func.func nested @main_func1_static(memref<1x16x50x1280xf16>, memref<1x50x1280x16xf16>) -> memref<1x50x1280x16xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func2_static <object = "\7FELF\02\01\00\00\00\00\00">
    func.func nested @main_func2_static(memref<1x33x1280x16xf16>, memref<1x33x1280x16xf16>, memref<1x16x33x1280xf16>) -> memref<1x16x33x1280xf16>
  }
  // CHECK: @main({{%[^:]+}}: memref<1x16x1280x1280xf16>, {{%[^:]+}}: memref<1x16x1280x1280xf16>, {{%[^:]+}}: memref<1x16x1280x1280xf16>, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: i64, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr, {{%[^:]+}}: !llvm.ptr) attributes {config.pureHostCompileFunc} {
  func.func @main(%arg0: memref<1x16x1280x1280xf16>, %arg1: memref<1x16x1280x1280xf16>, %arg2: memref<1x16x1280x1280xf16>) -> memref<1x16x1280x1280xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(1247 : index) : i64
    %1 = llvm.mlir.constant(1230 : index) : i64
    %2 = llvm.mlir.constant(33 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(50 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(1280 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<52428800xi8>
    %view = memref.view %alloc[%9][] : memref<52428800xi8> to memref<1x1280x1280x16xf16>
    %10 = llvm.sdiv %6, %4 : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = async.create_group %11 : !async.group
    scf.for %arg3 = %9 to %7 step %5 {
      %19 = builtin.unrealized_conversion_cast %arg3 : index to i64
      %20 = llvm.add %19, %4 : i64
      %21 = llvm.icmp "sgt" %20, %6 : i64
      %22 = llvm.select %21, %1, %19 : i1, i64
      %23 = builtin.unrealized_conversion_cast %22 : i64 to index
      %subview = memref.subview %arg1[0, 0, %23, 0] [1, 16, 50, 1280] [1, 1, 1, 1] : memref<1x16x1280x1280xf16> to memref<1x16x50x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>>
      %24 = builtin.unrealized_conversion_cast %subview : memref<1x16x50x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>> to memref<1x16x50x1280xf16>
      %subview_0 = memref.subview %view[0, %23, 0, 0] [1, 50, 1280, 16] [1, 1, 1, 1] : memref<1x1280x1280x16xf16> to memref<1x50x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>>
      %25 = builtin.unrealized_conversion_cast %subview_0 : memref<1x50x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>> to memref<1x50x1280x16xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x50x1280x16xf16>> {
        %28 = Core.NestedCall @Module0::@main_func0_static(%24, %25) : (memref<1x16x50x1280xf16>, memref<1x50x1280x16xf16>) -> memref<1x50x1280x16xf16>
        async.yield %25 : memref<1x50x1280x16xf16>
      }
      %26 = async.add_to_group %token, %12 : !async.token
      %27 = async.await %bodyResults : !async.value<memref<1x50x1280x16xf16>>
    }
    async.await_all %12
    %13 = llvm.sdiv %6, %4 : i64
    %14 = builtin.unrealized_conversion_cast %13 : i64 to index
    %15 = async.create_group %14 : !async.group
    scf.for %arg3 = %9 to %7 step %5 {
      %19 = builtin.unrealized_conversion_cast %arg3 : index to i64
      %20 = llvm.add %19, %4 : i64
      %21 = llvm.icmp "sgt" %20, %6 : i64
      %22 = llvm.select %21, %1, %19 : i1, i64
      %23 = builtin.unrealized_conversion_cast %22 : i64 to index
      %subview = memref.subview %arg0[0, 0, %23, 0] [1, 16, 50, 1280] [1, 1, 1, 1] : memref<1x16x1280x1280xf16> to memref<1x16x50x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>>
      %24 = builtin.unrealized_conversion_cast %subview : memref<1x16x50x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>> to memref<1x16x50x1280xf16>
      %subview_0 = memref.subview %view[0, %23, 0, 0] [1, 50, 1280, 16] [1, 1, 1, 1] : memref<1x1280x1280x16xf16> to memref<1x50x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>>
      %25 = builtin.unrealized_conversion_cast %subview_0 : memref<1x50x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>> to memref<1x50x1280x16xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x50x1280x16xf16>> {
        %28 = Core.NestedCall @Module1::@main_func1_static(%24, %25) : (memref<1x16x50x1280xf16>, memref<1x50x1280x16xf16>) -> memref<1x50x1280x16xf16>
        async.yield %25 : memref<1x50x1280x16xf16>
      }
      %26 = async.add_to_group %token, %15 : !async.token
      %27 = async.await %bodyResults : !async.value<memref<1x50x1280x16xf16>>
    }
    async.await_all %15
    %16 = llvm.sdiv %6, %2 : i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %18 = async.create_group %17 : !async.group
    scf.for %arg3 = %9 to %7 step %3 {
      %19 = builtin.unrealized_conversion_cast %arg3 : index to i64
      %20 = llvm.add %19, %2 : i64
      %21 = llvm.icmp "sgt" %20, %6 : i64
      %22 = llvm.select %21, %0, %19 : i1, i64
      %23 = builtin.unrealized_conversion_cast %22 : i64 to index
      %subview = memref.subview %view[0, %23, 0, 0] [1, 33, 1280, 16] [1, 1, 1, 1] : memref<1x1280x1280x16xf16> to memref<1x33x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>>
      %24 = builtin.unrealized_conversion_cast %subview : memref<1x33x1280x16xf16, strided<[26214400, 20480, 16, 1], offset: ?>> to memref<1x33x1280x16xf16>
      %subview_0 = memref.subview %arg2[0, 0, %23, 0] [1, 16, 33, 1280] [1, 1, 1, 1] : memref<1x16x1280x1280xf16> to memref<1x16x33x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>>
      %25 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x33x1280xf16, strided<[26214400, 1638400, 1280, 1], offset: ?>> to memref<1x16x33x1280xf16>
      %token, %bodyResults = async.execute -> !async.value<memref<1x16x33x1280xf16>> {
        %28 = Core.NestedCall @Module2::@main_func2_static(%24, %24, %25) : (memref<1x33x1280x16xf16>, memref<1x33x1280x16xf16>, memref<1x16x33x1280xf16>) -> memref<1x16x33x1280xf16>
        async.yield %25 : memref<1x16x33x1280xf16>
      }
      %26 = async.add_to_group %token, %18 : !async.token
      %27 = async.await %bodyResults : !async.value<memref<1x16x33x1280xf16>>
    }
    async.await_all %18
    return %arg2 : memref<1x16x1280x1280xf16>
  }
  // CHECK-NOT: async.create_group
  // CHECK-NOT: async.add_to_group
  // CHECK-NOT: async.await
  // CHECK-NOT: async.await_all
  // CHECK:     llvm.call @npu_level_zero_execute_graph
  // CHECK:     llvm.call @npu_level_zero_submit_commandlist
}

// -----

module @ExecuteGraphViewInputElementByteSize attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 64, 3]> : tensor<4xsi64>}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 64, 3]> : tensor<4xsi64>}>
  }

  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0 <object = "\7FELF\02\01\00\00">
    func.func nested @main_func0(memref<1x?x?x3xf16>, memref<1x?x?x3xf16>) -> memref<1x?x?x3xf16>
  }

  func.func @main(%arg0: memref<1x?x?x3xf16>) -> memref<1x?x?x3xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index

    %dim = memref.dim %arg0, %3 : memref<1x?x?x3xf16>
    %dim_0 = memref.dim %arg0, %5 : memref<1x?x?x3xf16>
    %dim_i64 = builtin.unrealized_conversion_cast %dim : index to i64
    %dim_0_i64 = builtin.unrealized_conversion_cast %dim_0 : index to i64
    %bytes_per_row = llvm.mlir.constant(6 : i64) : i64
    %bytes_i64_0 = llvm.mul %dim_i64, %dim_0_i64 : i64
    %bytes_i64 = llvm.mul %bytes_i64_0, %bytes_per_row : i64
    %bytes = builtin.unrealized_conversion_cast %bytes_i64 : i64 to index

    %alloc = memref.alloc(%bytes) : memref<?xi8>
    %view = memref.view %alloc[%1][%dim, %dim_0] : memref<?xi8> to memref<1x?x?x3xf16>
    %alloc_out = memref.alloc(%dim, %dim_0) : memref<1x?x?x3xf16>

    %token, %bodyResults = async.execute -> !async.value<memref<1x?x?x3xf16>> {
      %6 = Core.NestedCall @Module0::@main_func0(%view, %alloc_out) : (memref<1x?x?x3xf16>, memref<1x?x?x3xf16>) -> memref<1x?x?x3xf16>
      async.yield %alloc_out : memref<1x?x?x3xf16>
    }
    %7 = async.await %bodyResults : !async.value<memref<1x?x?x3xf16>>
    return %7 : memref<1x?x?x3xf16>

    // CHECK-LABEL: module @ExecuteGraphViewInputElementByteSize
    // CHECK: [[IN_DESC:%.*]] = llvm.alloca {{.*}}!llvm.struct<(ptr, i64, i64, i64, i64, array<5 x i64>, array<5 x i64>)>
    // CHECK: [[OUT_DESC:%.*]] = llvm.alloca {{.*}}!llvm.struct<(ptr, i64, i64, i64, i64, array<5 x i64>, array<5 x i64>)>
    // CHECK: [[IN_ELEM_SIZE_PTR:%.*]] = llvm.getelementptr [[IN_DESC]][0, 2]
    // CHECK-NOT: [[C1:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: [[C1:%.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.store [[C1]], [[IN_ELEM_SIZE_PTR]] : i64, !llvm.ptr
    // CHECK: [[OUT_ELEM_SIZE_PTR:%.*]] = llvm.getelementptr [[OUT_DESC]][0, 2]
    // CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.store [[C2]], [[OUT_ELEM_SIZE_PTR]] : i64, !llvm.ptr
    // CHECK: llvm.call @npu_level_zero_execute_graph([[IN_DESC]], {{.*}}, [[OUT_DESC]],
  }
}

// -----

module @AccumulateViewOffset attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0_static <object = "\7FELF\02\01\00">
    func.func nested @main_func0_static(memref<1x16x720x1280xf32>, memref<1x16x720x1280xf16>) -> memref<1x16x720x1280xf16>
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_main_func1_static <object = "\7FELF\02\01\00">
    func.func nested @main_func1_static(memref<1x16x720x1280xf16>, memref<1x720x1280x16xf16>) -> memref<1x720x1280x16xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func2_static <object = "\7FELF\02\01\00">
    func.func nested @main_func2_static(memref<1x720x1280x16xf16>, memref<1x16x720x1280xf32>) -> memref<1x16x720x1280xf32>
  }

  // Test: Accumulation of subview and view offsets in main function
  func.func @main(%input: memref<1x16x720x1280xf32>, %output: memref<1x16x720x1280xf32>) -> memref<1x16x720x1280xf32> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(58982400 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(29491200 : index) : i64
    %3 = builtin.unrealized_conversion_cast %0 : i64 to index
    %4 = builtin.unrealized_conversion_cast %1 : i64 to index
    %5 = builtin.unrealized_conversion_cast %2 : i64 to index
    %temp = memref.alloc(%3) {alignment = 64 : i64} : memref<?xi8>

    // View chain
    %temp_view1 = memref.view %temp[%4][] : memref<?xi8> to memref<1x16x720x1280xf16>
    %temp_view2 = memref.view %temp[%5][] : memref<?xi8> to memref<1x720x1280x16xf16>

    // Subview to get correct type
    %temp_sub1 = memref.subview %temp_view1[0,0,0,0] [1,16,720,1280] [1,1,1,1] : memref<1x16x720x1280xf16> to memref<1x16x720x1280xf16>
    %temp_sub2 = memref.subview %temp_view2[0,0,0,0] [1,720,1280,16] [1,1,1,1] : memref<1x720x1280x16xf16> to memref<1x720x1280x16xf16>

    %sub1 = memref.subview %input[0,0,0,0] [1,16,720,1280] [1,1,1,1] : memref<1x16x720x1280xf32> to memref<1x16x720x1280xf32>
    async.execute -> !async.value<memref<1x16x720x1280xf16>> {
      %res = Core.NestedCall @Module0::@main_func0_static(%sub1, %temp_sub1) : (memref<1x16x720x1280xf32>, memref<1x16x720x1280xf16>) -> memref<1x16x720x1280xf16>
      async.yield %res : memref<1x16x720x1280xf16>
    }

    async.execute -> !async.value<memref<1x720x1280x16xf16>> {
      %res2 = Core.NestedCall @Module1::@main_func1_static(%temp_sub1, %temp_sub2) : (memref<1x16x720x1280xf16>, memref<1x720x1280x16xf16>) -> memref<1x720x1280x16xf16>
      async.yield %res2 : memref<1x720x1280x16xf16>
    }

    async.execute -> !async.value<memref<1x16x720x1280xf32>> {
      %res2 = Core.NestedCall @Module2::@main_func2_static(%temp_sub2, %output) : (memref<1x720x1280x16xf16>, memref<1x16x720x1280xf32>) -> memref<1x16x720x1280xf32>
      async.yield %res2 : memref<1x16x720x1280xf32>
    }

    // CHECK-LABEL: module @AccumulateViewOffset
    // CHECK-LABEL: func @main
    // CHECK: llvm.mlir.addressof @main_func1_static_kernel
    // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue {{.*}}[2]
    // CHECK: %[[CAST:.*]] = arith.index_cast {{.*}} : index to i64
    // CHECK: %[[CONST:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[SDIV:.*]] = llvm.sdiv %[[CAST]], %[[CONST]] : i64
    // CHECK: %[[ADD:.*]] = llvm.add %[[EXTRACT]], %[[SDIV]] : i64
    // CHECK: llvm.store %[[ADD]], {{.*}} : i64, !llvm.ptr

    return %output : memref<1x16x720x1280xf32>
  }
}

// -----

#map = affine_map<(d0)[s0] -> (s0 - 31, d0)>
#map1 = affine_map<(d0)[s0] -> (s0 - 48, d0)>

module @AccumulateViewOffset2 attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_main_func0_static <object = "\7FELF\02\01\00">
    func.func nested @main_func0_static(memref<1x16x31x1280xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_main_func1_static <object = "\7FELF\02\01\00">
    func.func nested @main_func1_static(memref<1x16x48x1280xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>
  }

  // Test: Accumulation of subview and view offsets in main function
  func.func @main(%Parameter_10: memref<1x16x?x1280xf32>, %main: memref<1x16x?x1280xf32>) -> memref<1x16x?x1280xf32> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(40960 : index) : i64
    %1 = llvm.mlir.constant(false) : i1
    %2 = llvm.mlir.constant(3 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(48 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(31 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %Parameter_10_0 = memref.dim %Parameter_10, %13 : memref<1x16x?x1280xf32>
    %main_1 = builtin.unrealized_conversion_cast %Parameter_10_0 : index to i64
    %main_2 = llvm.mul %main_1, %0 : i64
    %main_3 = builtin.unrealized_conversion_cast %main_2 : i64 to index
    %main_4 = llvm.add %main_2, %main_2 : i64
    %main_5 = builtin.unrealized_conversion_cast %main_4 : i64 to index
    %main_6 = memref.alloc(%main_5) {alignment = 64 : i64} : memref<?xi8>
    %main_7 = memref.view %main_6[%11][%Parameter_10_0] : memref<?xi8> to memref<1x16x?x1280xf16>
    %main_8 = memref.view %main_6[%main_3][%Parameter_10_0] : memref<?xi8> to memref<1x?x1280x16xf16>
    %Parameter_10_9 = llvm.icmp "sge" %main_1, %8 : i64
    cf.assert %Parameter_10_9, "Not enough elements to backtrack in scf.for loop for Output tensor"
    %Parameter_10_10 = llvm.sdiv %main_1, %8 : i64
    %Parameter_10_11 = builtin.unrealized_conversion_cast %Parameter_10_10 : i64 to index
    %Parameter_10_12 = async.create_group %Parameter_10_11 : !async.group
    scf.for %Parameter_10_20 = %11 to %Parameter_10_0 step %9 {
      %Parameter_10_21 = affine.min #map(%Parameter_10_20)[%Parameter_10_0]
      %Parameter_10_22 = memref.subview %Parameter_10[0, 0, %Parameter_10_21, 0] [1, 16, 31, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf32> to memref<1x16x31x1280xf32, strided<[?, ?, 1280, 1], offset: ?>>
      %Parameter_10_23 = memref.cast %Parameter_10_22 : memref<1x16x31x1280xf32, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x31x1280xf32, strided<[?, ?, ?, ?], offset: ?>>
      %Parameter_10_24 = memref.subview %main_7[0, 0, %Parameter_10_21, 0] [1, 16, 31, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x31x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %Parameter_10_25 = memref.cast %Parameter_10_24 : memref<1x16x31x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
      %Parameter_10_26, %Parameter_10_27 = async.execute -> !async.value<memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>> {
        %Parameter_10_30 = Core.NestedCall @Module0::@main_func0_static(%Parameter_10_23, %Parameter_10_25) : (memref<1x16x31x1280xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
        async.yield %Parameter_10_25 : memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
      }
      %Parameter_10_28 = async.add_to_group %Parameter_10_26, %Parameter_10_12 : !async.token
      %Parameter_10_29 = async.await %Parameter_10_27 : !async.value<memref<1x16x31x1280xf16, strided<[?, ?, ?, ?], offset: ?>>>
    }
    async.await_all %Parameter_10_12
    %Convolution_12_13 = llvm.sdiv %main_1, %6 : i64
    %Convolution_12_14 = builtin.unrealized_conversion_cast %Convolution_12_13 : i64 to index
    %Convolution_12_15 = async.create_group %Convolution_12_14 : !async.group
    scf.for %Convolution_12_20 = %11 to %Parameter_10_0 step %7 {
      %Convolution_12_21 = affine.min #map1(%Convolution_12_20)[%Parameter_10_0]
      %Convolution_12_22 = memref.subview %main_7[0, 0, %Convolution_12_21, 0] [1, 16, 48, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x48x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
      %Convolution_12_23 = memref.cast %Convolution_12_22 : memref<1x16x48x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x48x1280xf16, strided<[?, ?, ?, ?], offset: ?>>
      %Convolution_12_24 = memref.subview %main_8[0, %Convolution_12_21, 0, 0] [1, 48, 1280, 16] [1, 1, 1, 1] : memref<1x?x1280x16xf16> to memref<1x48x1280x16xf16, strided<[?, 20480, 16, 1], offset: ?>>
      %Convolution_12_25 = memref.cast %Convolution_12_24 : memref<1x48x1280x16xf16, strided<[?, 20480, 16, 1], offset: ?>> to memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>
      %Convolution_12_26, %Convolution_12_27 = async.execute -> !async.value<memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>> {
        %Convolution_12_30 = Core.NestedCall @Module1::@main_func1_static(%Convolution_12_23, %Convolution_12_25) : (memref<1x16x48x1280xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>
        async.yield %Convolution_12_25 : memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>
      }
      %Convolution_12_28 = async.add_to_group %Convolution_12_26, %Convolution_12_15 : !async.token
      %Convolution_12_29 = async.await %Convolution_12_27 : !async.value<memref<1x48x1280x16xf16, strided<[?, ?, ?, ?], offset: ?>>>
    }
    async.await_all %Convolution_12_15

    // CHECK-LABEL: module @AccumulateViewOffset2
    // CHECK-LABEL: func @main
    // CHECK: llvm.mlir.addressof @main_func1_static_kernel
    // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue {{.*}}[2]
    // CHECK: %[[CAST:.*]] = arith.index_cast {{.*}} : index to i64
    // CHECK: %[[CONST:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[SDIV:.*]] = llvm.sdiv %[[CAST]], %[[CONST]] : i64
    // CHECK: %[[ADD:.*]] = llvm.add %[[EXTRACT]], %[[SDIV]] : i64
    // CHECK: llvm.store %[[ADD]], {{.*}} : i64, !llvm.ptr
    // CHECK: %[[INDEX:.*]] = llvm.getelementptr {{.*}}[0, 4]
    // CHECK: %[[CONST_MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i64) : i64
    // CHECK: llvm.store %[[CONST_MINUS_ONE]], %[[INDEX]] : i64, !llvm.ptr

    return %main : memref<1x16x?x1280xf32>
  }
}

// -----

// Test: Two static processing kernels in scf.for loops followed by a dynamic interpolation kernel.
// Module0 and Module1 preprocess the input in 32x128 static tiles; Module2 performs dynamic-shape interpolation.
module @InterpolateSAPDynInputLayerTest attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
  HostExec.Binary @Module0 {
    HostExec.BinaryData @serialized_preprocess0 <object = "\7FELF\02\01\00">
    func.func nested @preprocess0(memref<1x3x32x128xf16>, memref<1x3x32x128xf16>) -> memref<1x3x32x128xf16>
  }
  HostExec.Binary @Module1 {
    HostExec.BinaryData @serialized_preprocess1 <object = "\7FELF\02\01\00">
    func.func nested @preprocess1(memref<1x3x32x128xf16>, memref<1x3x32x128xf16>) -> memref<1x3x32x128xf16>
  }
  HostExec.Binary @Module2 {
    HostExec.BinaryData @serialized_main_func0 <object = "\7FELF\02\01\00">
    func.func nested @main_func0(memref<2xf32, strided<[?], offset: ?>>, memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>, memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>) -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>)
  }
  func.func @main(%interp_input: memref<1x3x?x?xf16>, %scales: memref<2xf32>, %main: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(960016 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(16 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to index
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
    %10 = llvm.mlir.constant(3 : index) : i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to index
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = builtin.unrealized_conversion_cast %12 : i64 to index
    %14 = llvm.mlir.constant(32 : index) : i64
    %15 = builtin.unrealized_conversion_cast %14 : i64 to index
    %16 = llvm.mlir.constant(128 : index) : i64
    %17 = builtin.unrealized_conversion_cast %16 : i64 to index
    %dim_h = memref.dim %interp_input, %13 : memref<1x3x?x?xf16>
    %dim_h_i64 = builtin.unrealized_conversion_cast %dim_h : index to i64
    %dim_w = memref.dim %interp_input, %11 : memref<1x3x?x?xf16>
    %dim_w_i64 = builtin.unrealized_conversion_cast %dim_w : index to i64
    %alloc_pre = memref.alloc(%dim_h, %dim_w) : memref<1x3x?x?xf16>

    // Static kernel 0: preprocess input tiles (32x128) into intermediate buffer
    %tiles0_h = llvm.sdiv %dim_h_i64, %14 : i64
    %tiles0_w = llvm.sdiv %dim_w_i64, %16 : i64
    %tiles0 = llvm.mul %tiles0_h, %tiles0_w : i64
    %tiles0_idx = builtin.unrealized_conversion_cast %tiles0 : i64 to index
    %group0 = async.create_group %tiles0_idx : !async.group
    scf.for %h0 = %9 to %dim_h step %15 {
      scf.for %w0 = %9 to %dim_w step %17 {
        %sv_in0 = memref.subview %interp_input[0, 0, %h0, %w0] [1, 3, 32, 128] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>>
        %cast_in0 = builtin.unrealized_conversion_cast %sv_in0 : memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x32x128xf16>
        %sv_pre0 = memref.subview %alloc_pre[0, 0, %h0, %w0] [1, 3, 32, 128] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>>
        %cast_pre0 = builtin.unrealized_conversion_cast %sv_pre0 : memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x32x128xf16>
        %token0, %res0 = async.execute -> !async.value<memref<1x3x32x128xf16>> {
          %tmp0 = Core.NestedCall @Module0::@preprocess0(%cast_in0, %cast_pre0) : (memref<1x3x32x128xf16>, memref<1x3x32x128xf16>) -> memref<1x3x32x128xf16>
          async.yield %cast_pre0 : memref<1x3x32x128xf16>
        }
        %add0 = async.add_to_group %token0, %group0 : !async.token
        %await0 = async.await %res0 : !async.value<memref<1x3x32x128xf16>>
      }
    }
    async.await_all %group0

    %alloc_post = memref.alloc(%dim_h, %dim_w) : memref<1x3x?x?xf16>

    // Static kernel 1: process intermediate buffer tiles (32x128) into post-processed buffer
    %tiles1_h = llvm.sdiv %dim_h_i64, %14 : i64
    %tiles1_w = llvm.sdiv %dim_w_i64, %16 : i64
    %tiles1 = llvm.mul %tiles1_h, %tiles1_w : i64
    %tiles1_idx = builtin.unrealized_conversion_cast %tiles1 : i64 to index
    %group1 = async.create_group %tiles1_idx : !async.group
    scf.for %h1 = %9 to %dim_h step %15 {
      scf.for %w1 = %9 to %dim_w step %17 {
        %sv_pre1 = memref.subview %alloc_pre[0, 0, %h1, %w1] [1, 3, 32, 128] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>>
        %cast_pre1 = builtin.unrealized_conversion_cast %sv_pre1 : memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x32x128xf16>
        %sv_in1 = memref.subview %alloc_post[0, 0, %h1, %w1] [1, 3, 32, 128] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>>
        %cast_in1 = builtin.unrealized_conversion_cast %sv_in1 : memref<1x3x32x128xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x32x128xf16>
        %token1, %res1 = async.execute -> !async.value<memref<1x3x32x128xf16>> {
          %tmp1 = Core.NestedCall @Module1::@preprocess1(%cast_pre1, %cast_in1) : (memref<1x3x32x128xf16>, memref<1x3x32x128xf16>) -> memref<1x3x32x128xf16>
          async.yield %cast_in1 : memref<1x3x32x128xf16>
        }
        %add1 = async.add_to_group %token1, %group1 : !async.token
        %await1 = async.await %res1 : !async.value<memref<1x3x32x128xf16>>
      }
    }
    async.await_all %group1

    // Dynamic interpolation kernel takes the post-processed buffer as input
    %scales_0 = memref.cast %scales : memref<2xf32> to memref<2xf32, strided<[?], offset: ?>>
    %alloc_post_cast = memref.cast %alloc_post : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    %main_2 = memref.alloc() {alignment = 64 : i64} : memref<960032xi8>
    %main_3 = memref.view %main_2[%9][] : memref<960032xi8> to memref<4xsi32>
    %main_4 = memref.view %main_2[%3][] : memref<960032xi8> to memref<1x3x400x400xf16>
    %main_5 = memref.view %main_2[%1][] : memref<960032xi8> to memref<4xsi32>
    %scales_6 = memref.cast %main_4 : memref<1x3x400x400xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    %scales_7 = memref.dim %alloc_post, %11 : memref<1x3x?x?xf16>
    %scales_8 = builtin.unrealized_conversion_cast %scales_7 : index to i64
    %scales_9 = llvm.trunc %scales_8 : i64 to i32
    %scales_10 = builtin.unrealized_conversion_cast %scales_9 : i32 to si32
    memref.store %scales_10, %main_3[%9] : memref<4xsi32>
    %scales_11 = memref.dim %alloc_post, %13 : memref<1x3x?x?xf16>
    %scales_12 = builtin.unrealized_conversion_cast %scales_11 : index to i64
    %scales_13 = llvm.trunc %scales_12 : i64 to i32
    %scales_14 = builtin.unrealized_conversion_cast %scales_13 : i32 to si32
    memref.store %scales_14, %main_3[%7] : memref<4xsi32>
    %scales_15 = builtin.unrealized_conversion_cast %5 : i32 to si32
    memref.store %scales_15, %main_3[%13] : memref<4xsi32>
    %scales_16 = builtin.unrealized_conversion_cast %4 : i32 to si32
    memref.store %scales_16, %main_3[%11] : memref<4xsi32>
    %scales_17, %scales_18:2 = async.execute -> (!async.value<memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>>, !async.value<memref<4xsi32>>) {
      %scales_21:2 = Core.NestedCall @Module2::@main_func0(%scales_0, %alloc_post_cast, %main_3, %scales_6, %main_5) : (memref<2xf32, strided<[?], offset: ?>>, memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>, memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>) -> (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>)
      async.yield %scales_6, %main_5 : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, memref<4xsi32>
    }
    %scales_19 = async.await %scales_18#0 : !async.value<memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>>
    %scales_20 = async.await %scales_18#1 : !async.value<memref<4xsi32>>
    memref.copy %scales_19, %main : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x3x?x?xf16>
    return %main : memref<1x3x?x?xf16>
  }
  // CHECK-LABEL: module @InterpolateSAPDynInputLayerTest
  // CHECK-DAG: llvm.mlir.global internal constant @preprocess0_kernel
  // CHECK-DAG: llvm.mlir.global internal constant @preprocess1_kernel
  // CHECK-DAG: llvm.mlir.global internal constant @main_func0_kernel
  // CHECK-NOT: async.create_group
  // CHECK-NOT: async.add_to_group
  // CHECK-NOT: async.await_all
  // CHECK-NOT: async.await
  // CHECK-NOT: async.execute
  // CHECK: llvm.mlir.addressof @preprocess0
  // CHECK: llvm.call @npu_level_zero_execute_graph
  // CHECK: llvm.call @npu_level_zero_submit_commandlist
  // CHECK: llvm.mlir.addressof @preprocess1
  // CHECK: llvm.call @npu_level_zero_execute_graph
  // CHECK: llvm.call @npu_level_zero_submit_commandlist
  // CHECK: llvm.mlir.addressof @main_func0
  // CHECK: llvm.call @npu_level_zero_execute_graph
  // CHECK: llvm.call @npu_level_zero_submit_commandlist
  // CHECK: llvm.call @npu_level_zero_append_memory_copy
  // CHECK: llvm.call @npu_level_zero_submit_commandlist
}
