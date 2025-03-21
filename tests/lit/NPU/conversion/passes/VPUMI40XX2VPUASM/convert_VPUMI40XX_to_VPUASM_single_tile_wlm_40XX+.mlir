//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUMI40XX-to-VPUASM="workload-management-enable=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module attributes {VPU.arch = #VPU.arch_kind<NPU40XX>} {
  IE.ExecutorResource 1 of @DMA_NN
  IE.TileResource 1 of @NCE at 6.000000e+02 MHz
  IE.CNNNetwork entryPoint : @twoDma inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x16x16xf16>
    DataInfo "output_1" : tensor<1x16x16x16xf16>
  }
  func.func @twoDma() {
    %11 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8} <1, -1> -> !VPURegMapped.Index<0:0:1>
    %19 = VPUMI40XX.Bootstrap inputs(%11 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI40XX.Bootstrap inputs(%12 : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: VPUASM.Bootstrap @Bootstrap_0_0 {barrier_id = 0 : ui32}
//CHECK: VPUASM.Bootstrap @Bootstrap_0_1 {barrier_id = 1 : ui32}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x14x14xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %28 = VPURegMapped.ViewTaskRange(%2 -> %3 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK-NOT: VPURegMapped.ViewTaskRange

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @Convolution attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>} {
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x14x14xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %4 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %5 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %8 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %9 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %18 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%18 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%19 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %21 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%20 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %22 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%21 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%19 : !VPURegMapped.Index<0:0:1>) updates(%20 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%15 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%17 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%16 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%12 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%20 : !VPURegMapped.Index<0:0:2>) updates(%21 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUVariant taskLocation(%4 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %26 = VPUMI40XX.DPUVariant taskLocation(%5 : !VPURegMapped.Index<0:0:1>) previousTask(%25 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%17 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%16 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %27 = VPURegMapped.ViewTaskRange(%23 -> %24 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %28 = VPURegMapped.ViewTaskRange(%2 -> %3 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %29 = VPURegMapped.ViewTaskRange(%25 -> %26 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8>
    %30 = VPURegMapped.ViewTaskRange(%4 -> %5 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8, [@CMX_NN, 0]>
    %31 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%27 : memref<2x352xui8>) outputs(%28 : memref<2x352xui8, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %32 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%29 : memref<2x224xui8>) outputs(%30 : memref<2x224xui8, [@CMX_NN, 0]>) previousDMA(%31 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %33 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64} inputs(%8 : memref<1x1x1x1xi32, @DDR>) outputs(%9 : memref<1x1x1x1xi32, @DDR>) previousDMA(%32 : !VPURegMapped.Index<0:0:1>) updates(%18 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %34 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%6 : memref<1x16x16x16xf16, @DDR>) outputs(%10 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%33 : !VPURegMapped.Index<0:0:2>) waits(%18 : !VPURegMapped.Index<0:0:0>) updates(%19 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %35 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%13 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%34 : !VPURegMapped.Index<0:0:3>) updates(%20 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %36 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%12 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x14x14xf16, @DDR>) waits(%21 : !VPURegMapped.Index<0:0:3>) updates(%22 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %37 = VPURegMapped.Enqueue at(%18 : !VPURegMapped.Index<0:0:0>) (%25 -> %25 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %38 = VPURegMapped.Enqueue previousTaskIdx(%37 : !VPURegMapped.Index<0:0:0>) at(%18 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%37 : !VPURegMapped.Index<0:0:0>) (%26 -> %26 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %39 = VPUMI40XX.Bootstrap inputs(%18 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %40 = VPUMI40XX.Bootstrap inputs(%19 : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %41 = VPUMI40XX.Bootstrap inputs(%20 : <0:0:2>) -> !VPURegMapped.Index<0:0:2>
    %42 = VPUMI40XX.Bootstrap inputs(%21 : <0:0:3>) -> !VPURegMapped.Index<0:0:3>
    %43 = VPUMI40XX.Bootstrap inputs(%22 : <0:0:4>) -> !VPURegMapped.Index<0:0:4>
    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

    %44 = VPUMI40XX.MappedInference dmas((%31, %36) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%25 : !VPURegMapped.Index<0:0:0>) barriers(%18 : !VPURegMapped.Index<0:0:0>) workItemTasks(%37 : !VPURegMapped.Index<0:0:0>) bootstrapTasks(%39 : !VPURegMapped.Index<0:0:0>) dmaCount([[5, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(5) workItemCount(2) bootstrapTasksCount(5) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_0 idx(!VPURegMapped.Index<0:0:0>) workItemIdx(!VPURegMapped.Index<0:0:0>)
//CHECK-SAME: work_item_count = 2 : ui32
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_1 idx(!VPURegMapped.Index<0:0:1>)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_2 idx(!VPURegMapped.Index<0:0:2>)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_3 idx(!VPURegMapped.Index<0:0:3>)
//CHECK: VPUASM.ManagedBarrier @ConfigureBarrier_0_4 idx(!VPURegMapped.Index<0:0:4>)
//CHECK: VPUASM.WorkItem @[[Enqueue0:.*]] idx(!VPURegMapped.Index<0:0:0>) real_task_index(!VPURegMapped.Index<0:0:0>) next_workitem_idx(!VPURegMapped.Index<0:0:1>) task_type(<DPUVariant>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue1:.*]] idx(!VPURegMapped.Index<0:0:1>) real_task_index(!VPURegMapped.Index<0:0:1>) task_type(<DPUVariant>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_1) task_count(1)

//CHECK: VPUASM.Bootstrap @Bootstrap_0_0 {barrier_id = 0 : ui32}
//CHECK: VPUASM.Bootstrap @Bootstrap_0_1 {barrier_id = 1 : ui32}
//CHECK: VPUASM.Bootstrap @Bootstrap_0_2 {barrier_id = 2 : ui32}
//CHECK: VPUASM.Bootstrap @Bootstrap_0_3 {barrier_id = 3 : ui32}
//CHECK: VPUASM.Bootstrap @Bootstrap_0_4 {barrier_id = 4 : ui32}

//CHECK{LITERAL}: VPUASM.MappedInference @MappedInference : dmas([[@task.dma.0.0::@NNDMA_0_0_0, @task.dma.0.1::@NNDMA_0_1_0]])
//CHECK-SAME: managedMappedInference(@program.mapped_inference::@MappedInference_managed)
//CHECK{LITERAL}: VPUASM.ManagedMappedInference @MappedInference_managed
//CHECK-SAME: workItems(@program.workItem::@[[Enqueue0]])
//CHECK-SAME: bootstrapTasks(@program.bootstrap::@Bootstrap_0_0)
//CHECK-SAME: nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged)
//CHECK-SAME: actshv_used = 0
//CHECK-SAME: dma_from_cmx_used = 1
//CHECK-SAME: dma_from_ddr_used = 1
//CHECK-SAME: dpu_used = 1
//CHECK-SAME: media_used = 0
//CHECK-SAME: workItemsCount = 2

//CHECK: VPUASM.nnrtConfig {isActKernelInvocations} @MappedInference_nnrtConfigManaged

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @BarrierProgramming attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<DefaultHW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>} {
  IE.PipelineOptions @Options {
    IE.Option @VPU.FP16CompressedConv : false
    IE.Option @VPU.ReduceSupported : false
    IE.Option @VPU.AutoPaddingODU : false
    IE.Option @VPU.AutoPaddingIDU : false
    IE.Option @VPU.BarrierMaxVariantSum : 64
    IE.Option @VPU.BarrierMaxVariantCount : 128
    IE.Option @VPU.MaxKernelSize : 11
  }
  IE.TileResource 1 of @NCE at 1.700000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        IE.MemoryResource 512 bytes of @CMX_NN offset 0
      }
    }
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    IE.ExecutorResource 2 of @SHAVE_ACT
    IE.ExecutorResource 1 of @DPU
  }
  IE.ExecutorResource 1 of @M2I
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x14x14xf16>
  }
  VPUASM.IOBindings inputDeclarations : {
    VPUASM.DeclareBuffer @input_buffDecl !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x16x16x16xf16, @DDR> :  swizzling(0)>
  } outputDeclarations : {
    VPUASM.DeclareBuffer @output_buffDecl !VPUASM.Buffer< "NetworkOutput"[0] <0> : memref<1x16x14x14xf16, @DDR> :  swizzling(0)>
  } profilingBuffDeclarations : {
  }
  func.func @main() {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x14x14xf16, @DDR>
    %2 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:0>
    %3 = VPURegMapped.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:1>
    %4 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:0>
    %5 = VPURegMapped.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:1>
    %cst = const.Declare memref<1x1x1x4864xui8> = dense<1> : tensor<1x1x1x4864xui8>
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x16x16xf16, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x14x14xf16, @DDR>
    %8 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %9 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xi32, @DDR>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, [@CMX_NN, 0]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x14x14xf16, [@CMX_NN, 0]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<1x1x1x4864xui8, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [0] <512> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <8704> -> memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [0] <16896> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <17152> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %18 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <4, -1> -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%18 : !VPURegMapped.Index<0:0:0>) <0, -1> -> !VPURegMapped.Index<0:0:1>
    %20 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}(%19 : !VPURegMapped.Index<0:0:1>) <1, -1> -> !VPURegMapped.Index<0:0:2>
    %21 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}(%20 : !VPURegMapped.Index<0:0:2>) <2, -1> -> !VPURegMapped.Index<0:0:3>
    %22 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, isFinalBarrier, producer_count = 1 : ui8}(%21 : !VPURegMapped.Index<0:0:3>) <3, -1> -> !VPURegMapped.Index<0:0:4>
    %23 = VPUMI40XX.DPUInvariant {clean_after = 2 : ui64, is_permute_quantize, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 3 : ui64} taskLocation(%2 : !VPURegMapped.Index<0:0:0>) input(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) outputs(%11 : memref<1x16x16x16xf16, #NWCH, [@CMX_NN, 0]>) waits(%19 : !VPURegMapped.Index<0:0:1>) updates(%20 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %24 = VPUMI40XX.DPUInvariant {clean_after = 3 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 4 : ui64} taskLocation(%3 : !VPURegMapped.Index<0:0:1>) previousTask(%23 : !VPURegMapped.Index<0:0:0>) input(%15 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%17 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%16 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%12 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) waits(%20 : !VPURegMapped.Index<0:0:2>) updates(%21 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %25 = VPUMI40XX.DPUVariant taskLocation(%4 : !VPURegMapped.Index<0:0:0>) calls(%23 : <0:0:0>) weights(%14 : memref<1x16x16x16xf16, #NHWC, [@CMX_NN, 0]>) {end = [15, 15, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %26 = VPUMI40XX.DPUVariant taskLocation(%5 : !VPURegMapped.Index<0:0:1>) previousTask(%25 : !VPURegMapped.Index<0:0:0>) calls(%24 : <0:0:1>) weights(%17 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%16 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {end = [13, 13, 15], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %27 = VPURegMapped.ViewTaskRange(%23 -> %24 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8>
    %28 = VPURegMapped.ViewTaskRange(%2 -> %3 : <0:0:0> -> <0:0:1>) -> memref<2x352xui8, [@CMX_NN, 0]>
    %29 = VPURegMapped.ViewTaskRange(%25 -> %26 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8>
    %30 = VPURegMapped.ViewTaskRange(%4 -> %5 : <0:0:0> -> <0:0:1>) -> memref<2x224xui8, [@CMX_NN, 0]>
    %31 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%27 : memref<2x352xui8>) outputs(%28 : memref<2x352xui8, [@CMX_NN, 0]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %32 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%29 : memref<2x224xui8>) outputs(%30 : memref<2x224xui8, [@CMX_NN, 0]>) previousDMA(%31 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %33 = VPUMI40XX.NNDMA {dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i32, len = 0 : i32, srcWidth = 0 : i32, srcStride = 0 : i32, srcPlaneStride = 0 : i32, dstWidth = 0 : i32, dstStride = 0 : i32, dstPlaneStride = 0 : i32>, port = 0 : i64} inputs(%8 : memref<1x1x1x1xi32, @DDR>) outputs(%9 : memref<1x1x1x1xi32, @DDR>) previousDMA(%32 : !VPURegMapped.Index<0:0:1>) updates(%18 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %34 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%6 : memref<1x16x16x16xf16, @DDR>) outputs(%10 : memref<1x16x16x16xf16, [@CMX_NN, 0]>) previousDMA(%33 : !VPURegMapped.Index<0:0:2>) waits(%18 : !VPURegMapped.Index<0:0:0>) updates(%19 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %35 = VPUMI40XX.NNDMA {is_out_of_order, port = 0 : i64} inputs(%cst : memref<1x1x1x4864xui8>) outputs(%13 : memref<1x1x1x4864xui8, [@CMX_NN, 0]>) previousDMA(%34 : !VPURegMapped.Index<0:0:3>) updates(%20 : !VPURegMapped.Index<0:0:2>) start_after(3) clean_after(2) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %36 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%12 : memref<1x16x14x14xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x16x14x14xf16, @DDR>) waits(%21 : !VPURegMapped.Index<0:0:3>) updates(%22 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:1:0>
    %37 = VPURegMapped.Enqueue at(%18 : !VPURegMapped.Index<0:0:0>) (%25 -> %25 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %38 = VPURegMapped.Enqueue previousTaskIdx(%37 : !VPURegMapped.Index<0:0:0>) at(%18 : !VPURegMapped.Index<0:0:0>) (%26 -> %26 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %39 = VPUMI40XX.Bootstrap inputs(%18 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %40 = VPUMI40XX.Bootstrap inputs(%19 : <0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %41 = VPUMI40XX.Bootstrap inputs(%20 : <0:0:2>) -> !VPURegMapped.Index<0:0:2>
    %42 = VPUMI40XX.Bootstrap inputs(%21 : <0:0:3>) -> !VPURegMapped.Index<0:0:3>
    %43 = VPUMI40XX.Bootstrap inputs(%22 : <0:0:4>) -> !VPURegMapped.Index<0:0:4>
    %barStrides = const.Declare memref<16xui32> = dense<[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xui32>
    %barDescs = const.Declare memref<1xui8> = dense<1> : tensor<1xui8>

    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>

    %173 = VPUMI40XX.MappedInference dmas((%31, %36) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>)) invariants(%23 : !VPURegMapped.Index<0:0:0>) variants(%25 : !VPURegMapped.Index<0:0:0>) barriers(%18 : !VPURegMapped.Index<0:0:0>) workItemTasks(%37 : !VPURegMapped.Index<0:0:0>) bootstrapTasks(%39 : !VPURegMapped.Index<0:0:0>) barrierConfigurationTasks(%barDescs : memref<1xui8>) numOfBarrierReprogrammings(%barStrides : memref<16xui32>) dmaCount([[5, 1]]) invariantCount([2]) variantCount([2]) actKernelRangesCount([0]) actKernelInvocationsCount([0]) mediaCount(0) barrierCount(5) workItemCount(2) bootstrapTasksCount(5) barrierConfigurationTasksCount(128) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: barrierConfigurationTasks(@buffer.Constant.0.constant
//CHECK-SAME: numOfBarrierReprogrammings(@buffer.Constant.0.constant
//CHECK-SAME: barrierConfigurationStride = 8
//CHECK-SAME: barrierConfigurationTasksCount = 128
//CHECK-SAME: barriersReprogrammingCount = 16
