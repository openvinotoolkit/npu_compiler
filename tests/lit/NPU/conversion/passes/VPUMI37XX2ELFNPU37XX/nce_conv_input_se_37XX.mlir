//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  net.NetworkInfo entryPoint : @conv_input_se_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16>
    DataInfo "output_1" : tensor<1x64x16x32xf16>
  }
  func.func nested @conv_input_se_soh_f16_f16_f16(%arg0: memref<1x32x32x32xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, %arg2: memref<1x64x16x32xf16, {order = #NHWC}, @DDR>) -> (memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, memref<1x64x16x32xf16, {order = #NHWC}, @DDR>) {
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [1] <69632> -> memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [1] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0] <103424> -> memref<1x32x16x32xi1, {order = #NHWC}, [@CMX_NN, 0]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [1] <103424> -> memref<1x32x16x32xi1, {order = #NHWC}, [@CMX_NN, 1]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [0] <105472> -> memref<1x1x16x32xi32, {order = #NHWC}, [@CMX_NN, 0]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [1] <105472> -> memref<1x1x16x32xi32, {order = #NHWC}, [@CMX_NN, 1]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <103424> -> !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %38 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <105472> -> !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %24 = VPUMI37XX.DPUInvariant <{clean_after = 0 : ui64, input_se_size = 32 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
     kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>}> input(%11 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
     input_sparsity_map(%33 : memref<1x32x16x32xi1, {order = #NHWC}, [@CMX_NN, 0]>) input_storage_element_table(%35 : memref<1x1x16x32xi32, {order = #NHWC}, [@CMX_NN, 0]>)
     weights(%5 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16,
     {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
     parent_input_sparsity_map(%37 : !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
     parent_input_storage_element_table(%38 : !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>)
     parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%8 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
    }
    %25 = "VPUMI37XX.DPUVariant"(%24) {end = [31, 15, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %26 = VPUMI37XX.DPUInvariant <{clean_after = 0 : ui64, input_se_size = 32 : i64, is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
    kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<CONV>}> input(%12 : memref<1x32x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) input_sparsity_map(%34 : memref<1x32x16x32xi1, {order = #NHWC}, [@CMX_NN, 1]>) input_storage_element_table(%36 : memref<1x1x16x32xi32, {order = #NHWC}, [@CMX_NN, 1]>) weights(%6 : memref<64x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) weight_table(%14 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_sparsity_map(%37 : !VPUIP.DistributedBuffer<1x32x32x32xi1, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_input_storage_element_table(%38 : !VPUIP.DistributedBuffer<1x1x32x32xi32, {order = #NHWC, strides = [1024, 1, 32, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%9 : memref<1x64x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> <0:0:1> PPE : {
    }
    %27 = "VPUMI37XX.DPUVariant"(%26) {end = [31, 31, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 16, 0]} : (!VPURegMapped.Index<0:0:1>) -> !VPURegMapped.Index<0:0:1>
    %32 = VPUMI37XX.MappedInference invariants(%24 : !VPURegMapped.Index<0:0:0>) variants(%25 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>

    return %arg1, %arg2 : memref<1x64x16x32xf16, {order = #NHWC}, @DDR>, memref<1x64x16x32xf16, {order = #NHWC}, @DDR>
  }
}

//CHECK-LABEL: @conv_input_se_soh_f16_f16_f16
//CHECK: [[VAL16:%.+]] = VPUMI37XX.DPUInvariant
//CHECK: [[VAL17:%.+]] = "VPUMI37XX.DPUVariant"
//CHECK: [[VAL18:%.+]] = VPUMI37XX.DPUInvariant
//CHECK: [[VAL19:%.+]] = "VPUMI37XX.DPUVariant"

//CHECK-DAG: [[INVSEC:%.+]] = ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUInvariants"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL16]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL18]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: [[VARSEC:%.+]] = ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUVariants"
//CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL17]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL19]] : !VPURegMapped.Index<0:0:1>

//CHECK: [[BUILTIN_SYMTABSEC:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: [[SYMTABSEC:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection([[BUILTIN_SYMTABSEC]])
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32_MULTICAST_BASE>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32_MULTICAST_BASE>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL16]] : !VPURegMapped.Index<0:0:0>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32_MULTICAST_BASE>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32_MULTICAST_BASE>
//CHECK: ELFNPU37XX.Reloc baseOp([[VAL18]] : !VPURegMapped.Index<0:0:1>) {{.+}} <R_VPU_32>


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SMapDistributed = !VPUIP.DistributedBuffer<1x96x37x256xi1, #NHWC, @CMX_NN, {
  mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
  compute_shapes = [[1, 64, 37, 256], [1, 32, 37, 256]], compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
  memory_shapes = [[1, 64, 37, 256], [1, 32, 37, 256]], memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]
}>

!SETableDistributed = !VPUIP.DistributedBuffer<1x2x37x256xi32, #NHWC, @CMX_NN, {
  mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1],
  compute_shapes = [[1, 1, 37, 256], [1, 1, 37, 256]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
  memory_shapes = [[1, 1, 37, 256], [1, 1, 37, 256]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

net.NetworkInfo entryPoint : @sep_multiple_clusters_dpu_sok_f16_f16_f16 inputsInfo : {
  DataInfo "input_0" : tensor<1x96x128x128xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x96x256x256xf16>
}

func.func @sep_multiple_clusters_dpu_sok_f16_f16_f16() {
  %0 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
  %1 = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <1214464> -> memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <1574912> -> memref<1x64x37x256xi1, {order = #NHWC}, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <1650688> -> memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 0]>
  %5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<64x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <1703936> {swizzlingKey = 5 : i64} -> memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  %7 = VPURT.DeclareBuffer <CMX_NN> <1574912> -> !SMapDistributed
  %8 = VPURT.DeclareBuffer <CMX_NN> <1650688> -> !SETableDistributed
  %9 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>
  %10 = VPURT.DeclareBuffer <CMX_NN> [1] <1214464> -> memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>
  %11 = VPURT.DeclareBuffer <CMX_NN> [1] <1574912> -> memref<1x32x37x256xi1, {order = #NHWC}, [@CMX_NN, 1]>
  %12 = VPURT.DeclareBuffer <CMX_NN> [1] <1650688> -> memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 1]>
  %13 = VPURT.DeclareBuffer <CMX_NN> [1] <0> {swizzlingKey = 5 : i64} -> memref<32x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
  %14 = VPURT.DeclareBuffer <CMX_NN> [1] <1703936> {swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>
  %15 = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>

  %16 = VPUMI37XX.DPUInvariant <{clean_after = 8 : ui64,
  input_se_size = 64 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  kernel_size = [1, 1], kernel_strides = [1, 1],
  mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
  nce_task_type = #VPUIP.nce_task_type<DWCONV>, start_after = 9 : ui64}>
    input(%2 : memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    input_sparsity_map(%3 : memref<1x64x37x256xi1, {order = #NHWC}, [@CMX_NN, 0]>)
    input_storage_element_table(%4 : memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 0]>)
    weights(%5 : memref<64x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
    weight_table(%6 : memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
    parent_input(%2 : memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    parent_input_sparsity_map(%7 : !SMapDistributed)
    parent_input_storage_element_table(%8 : !SETableDistributed)
    parent_output(%9 : memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    outputs(%9 : memref<1x64x37x256xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    waits(%0 : !VPURegMapped.Index<0:0:0>) updates(%1 : !VPURegMapped.Index<0:0:1>) -> <0:0:4>
  PPE : {
    VPUMI37XX.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
  }
  %17 = VPUMI37XX.DPUInvariant <{clean_after = 8 : ui64,
  input_se_size = 32 : i64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  kernel_size = [1, 1], kernel_strides = [1, 1],
  mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>,
  nce_task_type = #VPUIP.nce_task_type<DWCONV>, start_after = 9 : ui64}>
    input(%10 : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    input_sparsity_map(%11 : memref<1x32x37x256xi1, {order = #NHWC}, [@CMX_NN, 1]>)
    input_storage_element_table(%12 : memref<1x1x37x256xi32, {order = #NHWC}, [@CMX_NN, 1]>)
    weights(%13 : memref<32x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
    weight_table(%14 : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
    parent_input(%10 : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    parent_input_sparsity_map(%7 : !SMapDistributed)
    parent_input_storage_element_table(%8 : !SETableDistributed)
    parent_output(%15 : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    outputs(%15 : memref<1x32x37x256xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    waits(%0 : !VPURegMapped.Index<0:0:0>) updates(%1 : !VPURegMapped.Index<0:0:1>) -> <0:0:5>
  PPE : {
    VPUMI37XX.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
  }

  %18 = "VPUMI37XX.DPUVariant"(%16) <{
    cluster_id = 0 : i64, end = [255, 36, 63], start = [0, 0, 0],
    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      : (!VPURegMapped.Index<0:0:4>) -> !VPURegMapped.Index<0:0:4>
  %19 = "VPUMI37XX.DPUVariant"(%17) <{
    cluster_id = 1 : i64, end = [255, 36, 31], start = [0, 0, 0],
    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}>
      : (!VPURegMapped.Index<0:0:5>) -> !VPURegMapped.Index<0:0:5>

  %20 = VPUMI37XX.MappedInference
    invariants(%16 : !VPURegMapped.Index<0:0:4>) variants(%18 : !VPURegMapped.Index<0:0:4>)
    dmaCount([0, 0]) invariantCount(2) variantCount(2)
    actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
  return
}

//CHECK: [[INV0:%.+]] = VPUMI37XX.DPUInvariant <{clean_after = 8 : ui64, input_se_size = 64 : i64
//CHECK: [[INV1:%.+]] = VPUMI37XX.DPUInvariant <{clean_after = 8 : ui64, input_se_size = 32 : i64

// Cluster 0 invariant
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(92) <R_VPU_32> {{.+}} 1214464
//CHECK-SAME:    description = "Input (act_offset[0]) in DPU Invariant reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(264) <R_VPU_64_LSHIFT> {{.+}} 0
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(272) <R_VPU_64_LSHIFT> {{.+}} 0
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(96) <R_VPU_32> {{.+}} 3311616
//CHECK-SAME:    description = "Input (act_offset[1]) in DPU invariant registers reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(4) <R_VPU_32> {{.+}} 1574912
//CHECK-SAME:    description = "Input sparsity map (sparsity_addr in se_sp_addr[0]) in DPU invariant registers for input sparsity map reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(0) <R_VPU_32> {{.+}} 1650688
//CHECK-SAME:    description = "Input se table (se_sp_addr[0]) in DPU invariant registers for input storage element table reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(116) <R_VPU_32> {{.+}} 0
//CHECK-SAME:    description = "Weights (wt_offset, for ELTWISE) in DPU invariant registers reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(160) <R_VPU_32_MULTICAST_BASE> {{.+}} 2048
//CHECK-SAME:    description = "Base (base_adr[0]) offset in DPU reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(164) <R_VPU_32_MULTICAST_BASE> {{.+}} 2048
//CHECK-SAME:    description = "Base (base_adr[1]) offset in DPU reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV0]] : !VPURegMapped.Index<0:0:4>) offset(60) <R_VPU_32> {{.+}} 1703936
//CHECK-SAME:    description = "Weights table (weight_start) in DPU invariant registers reloc

// Cluster 1 invariant
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(92) <R_VPU_32> {{.+}} 3311616
//CHECK-SAME:    description = "Input (act_offset[0]) in DPU Invariant reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(264) <R_VPU_64_LSHIFT> {{.+}} 0
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(272) <R_VPU_64_LSHIFT> {{.+}} 0
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(96) <R_VPU_32> {{.+}} 5408768
//CHECK-SAME:    description = "Input (act_offset[1]) in DPU invariant registers reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(4) <R_VPU_32> {{.+}} 3672064
//CHECK-SAME:    description = "Input sparsity map (sparsity_addr in se_sp_addr[0]) in DPU invariant registers for input sparsity map reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(0) <R_VPU_32> {{.+}} 3747840
//CHECK-SAME:    description = "Input se table (se_sp_addr[0]) in DPU invariant registers for input storage element table reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(116) <R_VPU_32> {{.+}} 2097152
//CHECK-SAME:    description = "Weights (wt_offset, for ELTWISE) in DPU invariant registers reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(160) <R_VPU_32_MULTICAST_BASE> {{.+}} 2099200
//CHECK-SAME:    description = "Base (base_adr[0]) offset in DPU reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(164) <R_VPU_32_MULTICAST_BASE> {{.+}} 2099200
//CHECK-SAME:    description = "Base (base_adr[1]) offset in DPU reloc
//CHECK:  ELFNPU37XX.Reloc baseOp([[INV1]] : !VPURegMapped.Index<0:0:5>) offset(60) <R_VPU_32> {{.+}} 3801088
//CHECK-SAME:    description = "Weights table (weight_start) in DPU invariant registers reloc
