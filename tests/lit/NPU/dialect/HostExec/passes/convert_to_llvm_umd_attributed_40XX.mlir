//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --mlir-elide-elementsattrs-if-larger 8 --convert-to-llvm-umd-calls  %s | FileCheck %s
// REQUIRES: platform-NPU4000

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
  // CHECK-LABEL:  func.func @main(
  // CHECK-SAME:    [[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    scf.for %arg3 = %5 to %3 step %1 {
        %11 = func.call @not_main_attributed(%arg0, %arg1) : (memref<1x3x?x?xf16>,  memref<1x3x?x?xf16>) -> (memref<1x3x?x?xf16>)
    }
    return %arg1 : memref<1x3x?x?xf16>

    // CHECK: [[STEP_CNST:%.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[STEP:%.+]] = builtin.unrealized_conversion_cast [[STEP_CNST]] : i64 to index
    // CHECK: [[END_CNST:%.+]] = llvm.mlir.constant(4 : index) : i64
    // CHECK: [[END:%.+]] = builtin.unrealized_conversion_cast [[END_CNST]] : i64 to index
    // CHECK: scf.for [[IND_VAR:%.+]] = [[ANY_MATCH:.+]] to [[END]] step [[STEP]] [[ANY_MATCH:.+]]
    // CHECK:   [[ANY_MATCH:.+]] @not_main_attributed[[ANY_MATCH:.+]]
    // CHECK: llvm.call @npu_level_zero_submit_commandlist[[ANY_MATCH:.+]]
    // CHECK: return
  }

  func.func @not_main_attributed(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec, HostExec.InferenceExpectedCommandListsNumber = 1} {
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

    //CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @ConvCallChain attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
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
  // CHECK-LABEL:  func.func @main(
  // CHECK-SAME:    [[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    scf.for %arg3 = %5 to %3 step %1 {
        %11 = func.call @main_delegate(%arg0, %arg1) : (memref<1x3x?x?xf16>,  memref<1x3x?x?xf16>) -> (memref<1x3x?x?xf16>)
    }
    return %arg1 : memref<1x3x?x?xf16>

    // CHECK: [[STEP_CNST:%.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[STEP:%.+]] = builtin.unrealized_conversion_cast [[STEP_CNST]] : i64 to index
    // CHECK: [[END_CNST:%.+]] = llvm.mlir.constant(4 : index) : i64
    // CHECK: [[END:%.+]] = builtin.unrealized_conversion_cast [[END_CNST]] : i64 to index
    // CHECK: scf.for [[IND_VAR:%.+]] = [[ANY_MATCH:.+]] to [[END]] step [[STEP]] [[ANY_MATCH:.+]]
    // CHECK:   [[ANY_MATCH:.+]] @main_delegate[[ANY_MATCH:.+]]
    // CHECK: llvm.call @npu_level_zero_submit_commandlist[[ANY_MATCH:.+]]
    // CHECK: return
  }

  // CHECK-LABEL:  func.func @main_delegate(
  // CHECK-SAME:    [[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main_delegate(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %11 = func.call @not_main_attributed(%arg0, %arg1) : (memref<1x3x?x?xf16>,  memref<1x3x?x?xf16>) -> (memref<1x3x?x?xf16>)
    return %11: memref<1x3x?x?xf16>

    // CHECK:  call @not_main_attributed([[ARG0]], [[ARG1]], [[ANY_MATCH:.+]]) : (memref<1x3x?x?xf16>, memref<1x3x?x?xf16>, [[ANY_MATCH:.+]]) -> ()
    // CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
    // CHECK:  return
  }

  func.func @not_main_attributed(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec, HostExec.InferenceExpectedCommandListsNumber = 1} {
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
  net.NetworkInfo entryPoint : @main_n inputsInfo : {
    DataInfo "input1" : tensor<1x720x1000x16xf16>
    DataInfo "input2" : tensor<1x720x1000x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x720x1000x16xf16>
  }
  HostExec.Binary @OneDMAWithoutAttributes {
    HostExec.BinaryData @serialized_main1 <object = "\7FELF\02\01\00\00\10\00\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00">
    func.func nested @main1(memref<1x90x1000x16xf16>, memref<1x90x1000x16xf16>) -> memref<1x90x1000x16xf16>
  }
  // CHECK-LABEL:  func.func @main_n(
  // CHECK-SAME:   [[ARG0:%.+]]: memref<1x720x1000x16xf16>, [[ARG1:%.+]]: memref<1x720x1000x16xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main_n(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    scf.for %arg3 = %5 to %3 step %1 {
        func.call @not_main_attributed(%arg0, %arg1) : (memref<1x720x1000x16xf16>,  memref<1x720x1000x16xf16>) -> (memref<1x720x1000x16xf16>)
    }
    return %arg1 : memref<1x720x1000x16xf16>

    // CHECK: [[STEP_CNST:%.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[STEP:%.+]] = builtin.unrealized_conversion_cast [[STEP_CNST]] : i64 to index
    // CHECK: [[END_CNST:%.+]] = llvm.mlir.constant(4 : index) : i64
    // CHECK: [[END:%.+]] = builtin.unrealized_conversion_cast [[END_CNST]] : i64 to index
    // CHECK: scf.for [[IND_VAR:%.+]] = [[ANY_MATCH:.+]] to [[END]] step [[STEP]] [[ANY_MATCH:.+]]
    // CHECK:   [[ANY_MATCH:.+]] @not_main_attributed[[ANY_MATCH:.+]]
    // CHECK: llvm.call @npu_level_zero_submit_commandlist[[ANY_MATCH:.+]]
    // CHECK: return
  }

  func.func @not_main_attributed(%arg0: memref<1x720x1000x16xf16>, %arg1: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16> attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec, HostExec.InferenceExpectedCommandListsNumber = 1} {
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
  //CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
  //CHECK: llvm.call @npu_level_zero_execute_graph
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @DoNotFinalizeConvCallChainForNoRepeatingBlocks attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
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
  // CHECK-LABEL:  func.func @main(
  // CHECK-SAME:    [[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to index
    %2 = llvm.mlir.constant(4 : index) : i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to index
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %11 = func.call @main_delegate(%arg0, %arg1) : (memref<1x3x?x?xf16>,  memref<1x3x?x?xf16>) -> (memref<1x3x?x?xf16>)
    return %arg1 : memref<1x3x?x?xf16>

    // CHECK: [[STEP_CNST:%.+]] = llvm.mlir.constant(1 : index) : i64
    // CHECK: [[STEP:%.+]] = builtin.unrealized_conversion_cast [[STEP_CNST]] : i64 to index
    // CHECK: [[END_CNST:%.+]] = llvm.mlir.constant(4 : index) : i64
    // CHECK: [[END:%.+]] = builtin.unrealized_conversion_cast [[END_CNST]] : i64 to index
    // CHECK: [[ANY_MATCH:.+]] @main_delegate[[ANY_MATCH:.+]]
    // CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
    // CHECK: return
  }

  // CHECK-LABEL:  func.func @main_delegate(
  // CHECK-SAME:    [[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>, [[ANY_UMD_ARGS:.+]]) [[ANY_MATCH:.+]]
  func.func @main_delegate(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc} {
    %11 = func.call @not_main_attributed(%arg0, %arg1) : (memref<1x3x?x?xf16>,  memref<1x3x?x?xf16>) -> (memref<1x3x?x?xf16>)
    return %11: memref<1x3x?x?xf16>

    // CHECK:  call @not_main_attributed([[ARG0]], [[ARG1]], [[ANY_MATCH:.+]]) : (memref<1x3x?x?xf16>, memref<1x3x?x?xf16>, [[ANY_MATCH:.+]]) -> ()
    // CHECK-NOT: llvm.call @npu_level_zero_submit_commandlist
    // CHECK:  return
  }

  func.func @not_main_attributed(%arg0: memref<1x3x?x?xf16>, %arg1: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> attributes {config.pureHostCompileFunc, HostExec.HostCompileInferenceExec, HostExec.InferenceExpectedCommandListsNumber = 1} {
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

    //CHECK: llvm.call @npu_level_zero_submit_commandlist
  }
}
