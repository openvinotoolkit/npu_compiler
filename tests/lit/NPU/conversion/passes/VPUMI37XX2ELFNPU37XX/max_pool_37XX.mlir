//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// max_pool_1x64x16x16xfp16_2x2_pads_1x0x1x0_strides_2x2_fp16

// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  net.NetworkInfo entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16>
  }
  func.func nested @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x9x8xf16, {order = #NHWC}, @DDR>) -> memref<1x64x9x8xf16, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <9216> -> memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <41984> -> memref<1x1x1x16xui8, {order = #NHWC}, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <42000> -> memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
    %11 = VPUMI37XX.DPUInvariant <{clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
    input(%0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%9 : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    parent_output(%3 : memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
      VPUMI37XX.PPETask { ppe = #VPU.PPEStub<> }
    }
    %12 = "VPUMI37XX.DPUVariant"(%11) {end = [7, 8, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI37XX.MappedInference invariants(%11 : !VPURegMapped.Index<0:0:0>) variants(%12 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x64x9x8xf16, {order = #NHWC}, @DDR>
  }
}

// CHECK: func.func nested @maxpool_f16_f16

// CHECK: [[VAL11:%.+]] = VPUMI37XX.DPUInvariant
// CHECK: [[VAL12:%.+]] = "VPUMI37XX.DPUVariant"

// CHECK-NEXT: [[VAL14:%.+]] = VPUMI37XX.MappedInference

// CHECK: [[VAL23:%.+]] = ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUInvariants"
// CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL11]]

// CHECK: [[VAL24:%.+]] = ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUVariants"
// CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL12]]

// CHECK: [[VAL33:%.+]] = ELFNPU37XX.Symbol  {{.+}} name("sym_inVariantsSection")
// CHECK: [[VAL34:%.+]] = ELFNPU37XX.Symbol  {{.+}} name("sym_variantsSection")

// CHECK: [[VAL44:%.+]] = ELFNPU37XX.Symbol {{%.+}} name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR")
// CHECK: [[VAL45:%.+]] = ELFNPU37XX.Symbol {{%.+}} name("VPU_NNRD_SYM_RTM_IVAR")

// CHECK: [[VAL51:%.+]] = ELFNPU37XX.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
// CHECK: ELFNPU37XX.PutOpInSection [[VAL44]]
// CHECK: ELFNPU37XX.PutOpInSection [[VAL45]]

// CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection([[VAL51]]) targetSection([[VAL23]])
// CHECK-COUNT-6: ELFNPU37XX.Reloc baseOp([[VAL11]] {{.+}} [[VAL44]]

// CHECK: ELFNPU37XX.CreateRelocationSection secName(".rlt.text.DPUVariants") sourceSymbolTableSection([[VAL51]]) targetSection([[VAL24]])
// CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL12]] {{.+}} [[VAL45]]
// CHECK-NEXT: ELFNPU37XX.Reloc baseOp([[VAL12]] {{.+}} [[VAL44]]
