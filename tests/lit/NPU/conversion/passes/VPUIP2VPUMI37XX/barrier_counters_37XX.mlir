//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.013495710784313726:128>
module @mainModule {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
  func.func nested @builtin_Convert(memref<*xf16, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
  func.func nested @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func nested @barrier_counters(%arg0: memref<1x32x32x32xf16, {order = #NHWC}, @DDR>, %arg1: memref<1x64x16x32xf16>) -> memref<1x64x16x32xf16> {
    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier

    %cst_0 = const.Declare memref<1x1x1x3088xui8> = dense<0> : tensor<1x1x1x3088xui8>

    %m0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    %m1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x1x1x4xsi32, [@CMX_NN, 0]>
    %m5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x1x1x4xsi32, [@CMX_NN, 0]>
    %m7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %m8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000xf16, [@CMX_NN, 0]>
    %m10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1000xf32, [@CMX_NN, 0]>

    VPURT.Task updates(%b0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst_0 : memref<1x1x1x3088xui8>) outputs(%m0 : memref<1x1x1x3088xui8, [@CMX_NN, 0]>) -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst_0 : memref<1x1x1x3088xui8>) outputs(%m0 : memref<1x1x1x3088xui8, [@CMX_NN, 0]>) -> memref<1x1x1x3088xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 33741 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}> input(%m1 : memref<1x16x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m3 : memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%m4 : memref<96x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%m1 : memref<1x16x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m2 : memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m2 : memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [111, 111, 95], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : { PPETask {ppe = #VPU.PPEStub<>} }
    }
    VPURT.Task waits(%b2 : !VPURT.Barrier) updates(%b3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 27694 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<DWCONV>}> input(%m2 : memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m5 : memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%m6 : memref<96x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%m2 : memref<1x96x112x112x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m8 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m8 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {outEnd = [55, 55, 63], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {outEnd = [55, 55, 95], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : { PPETask {ppe = #VPU.PPEStub<>} }
    }
    VPURT.Task waits(%b3 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%m9 as %arg2: memref<1x1000xf16, [@CMX_NN, 0]>) outputs(%m10 as %arg3: memref<1x1000xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x1000xf32, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1000xf16, [@CMX_NN, 0]>, memref<1x1000xf32, [@CMX_NN, 0]>
      }
    }
  return %arg1 : memref<1x64x16x32xf16>
}
}

// CHECK: [[BAR_0:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
// CHECK: [[BAR_1:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
// CHECK: [[BAR_2:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPURegMapped.Index<0:0:2>
// CHECK: [[BAR_3:%.+]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 2 : ui8}<3, -1> -> !VPURegMapped.Index<0:0:3>
