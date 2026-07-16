//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUMI37XX-to-ELF %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:3>:f32, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f32, 1.000000e+00>
module @mainModule {
  net.NetworkInfo entryPoint : @singleEltwise inputsInfo : {
    DataInfo "input_0" : tensor<1x32x56x56xui8>
    DataInfo "input_1" : tensor<1x32x56x56xui8>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x32x56x56xui8>
  }
  func.func @singleEltwise(%arg0: memref<1x32x56x56x!qElemType, {order = #NHWC}, @DDR>, %arg1: memref<1x32x56x56x!qElemType, {order = #NHWC}, @DDR>, %arg2: memref<1x32x56x56x!qElemType1, {order = #NHWC}, @DDR>) -> memref<1x32x56x56x!qElemType1, {order = #NHWC}, @DDR> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <200704> -> memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <100352> -> memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x56x56x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    %10 = VPUMI37XX.DPUInvariant <{clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, start_after = 0 : ui64, nce_task_type = #VPUIP.nce_task_type<ELTWISE>}>
    input(%0 : memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights(%1 : memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%3 : memref<1x32x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    parent_output(%4 : memref<1x32x56x56x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%2 : memref<1x32x56x56x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) -> <0:0:0> PPE : {
      VPUMI37XX.PPETask { ppe = #VPU.PPEStub<> }
    }
    %11 = "VPUMI37XX.DPUVariant"(%10) {end = [55, 55, 31], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %12 = VPUMI37XX.MappedInference invariants(%10 : !VPURegMapped.Index<0:0:0>) variants(%11 : !VPURegMapped.Index<0:0:0>) dmaCount([0, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(0) -> !VPURegMapped.Index<0:0:0>
    return %arg2 : memref<1x32x56x56x!qElemType1, {order = #NHWC}, @DDR>
  }
}

// CHECK: func.func @singleEltwise
// CHECK: [[VAL10:%.+]] = VPUMI37XX.DPUInvariant
// CHECK: [[VAL11:%.+]] = "VPUMI37XX.DPUVariant"

// CHECK: ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUInvariants"
// CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL10]]

// CHECK: ELFNPU37XX.CreateSection {{.+}} secName = ".text.DPUVariants"
// CHECK-NEXT: ELFNPU37XX.PutOpInSection [[VAL11]]
