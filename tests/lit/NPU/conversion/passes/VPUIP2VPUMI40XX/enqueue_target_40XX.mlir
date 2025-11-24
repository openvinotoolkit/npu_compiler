//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.013495710784313726:128>

module @mainModule {

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
  func.func private @builtin_Convert(memref<*xf16, [@CMX_NN, 0]>, memref<*xf32, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
  func.func private @builtin_MemPermute(memref<*x!qElemType, [@CMX_NN, 0]>, memref<*x!qElemType, [@CMX_NN, 0]>, none) attributes {VPU.kernel_code = "reorder.cpp", VPU.kernel_entry = "reorder"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

net.NetworkInfo entryPoint : @barrier_counters inputsInfo : {
  DataInfo "input_0" : tensor<1x32x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<1x64x8x16xf16>
}

func.func private @barrier_counters(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x8x16xf16>) -> memref<1x64x8x16xf16> {
    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %b2 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
    %b3 = VPURT.ConfigureBarrier<3> -> !VPURT.Barrier

    %cst_0 = const.Declare memref<1x1x1x1544xui8> = dense<0> : tensor<1x1x1x1544xui8>

    %m0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1544xui8, [@CMX_NN, 0]>
    %m1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m3 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x1x1x4xsi32, {order = #NCHW}, [@CMX_NN, 0]>
    %m5 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %m6 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<96x1x1x4xsi32, [@CMX_NN, 0]>
    %m7 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x16xui8, [@CMX_NN, 0]>
    %m8 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x28x28x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %m9 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x500xf16, [@CMX_NN, 0]>
    %m10 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x500xf32, [@CMX_NN, 0]>

    VPURT.Task updates(%b0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x1544xui8>) outputs(%m0 : memref<1x1x1x1544xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1544xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b0 : !VPURT.Barrier) updates(%b1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x1544xui8>) outputs(%m0 : memref<1x1x1x1544xui8, [@CMX_NN, 0]>) -> memref<1x1x1x1544xui8, [@CMX_NN, 0]>
    }
    VPURT.Task waits(%b1 : !VPURT.Barrier) updates(%b2 : !VPURT.Barrier) enqueueTarget(%b0 : !VPURT.Barrier)  attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 33741 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%m1 : memref<1x16x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m3 : memref<96x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weight_table(%m4 : memref<96x1x1x4xsi32, {order = #NCHW}, [@CMX_NN, 0]>) parent_input(%m1 : memref<1x16x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m2 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m2 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [7, 7, 7], outEnd = [55, 55, 47], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }
    }
    VPURT.Task waits(%b2 : !VPURT.Barrier) updates(%b3 : !VPURT.Barrier) enqueueTarget(%b0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %t0 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 27694 : i64, task_type = #VPUIP.nce_task_type<DWCONV>} input(%m2 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights(%m5 : memref<96x16x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%m6 : memref<96x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%m2 : memref<1x96x56x56x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%m8 : memref<1x96x28x28x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%m8 : memref<1x96x28x28x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x96x28x28x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {inStart = [0, 0, 0], inEnd = [7, 7, 7], outEnd = [27, 27, 31], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {inStart = [0, 0, 0], inEnd = [7, 7, 7], outEnd = [27, 27, 47], outStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }
    }
    VPURT.Task waits(%b3 : !VPURT.Barrier) enqueueTarget(%b1 : !VPURT.Barrier)  attributes {isTrailingSWLayer = false} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%m9 as %arg2: memref<1x500xf16, [@CMX_NN, 0]>) outputs(%m10 as %arg3: memref<1x500xf32, [@CMX_NN, 0]>) on tile 0 -> memref<1x500xf32, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x500xf16, [@CMX_NN, 0]>, memref<1x500xf32, [@CMX_NN, 0]>
      }
    }
  return %arg1 : memref<1x64x8x16xf16>
}
}

// CHECK: [[BAR0:%.*]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: VPURegMapped.Index<0:0:0>
// CHECK: [[BAR1:%.*]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: VPURegMapped.Index<0:0:1>
// CHECK: [[BAR2:%.*]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: VPURegMapped.Index<0:0:2>
// CHECK: [[BAR3:%.*]] = VPUMI40XX.ConfigureBarrier
// CHECK-SAME: VPURegMapped.Index<0:0:3>
// CHECK: [[D0:%.*]] = VPUMI40XX.NNDMA

// CHECK: [[D1:%.*]] = VPUMI40XX.NNDMA

// CHECK: [[DPU0:%.*]] = VPUMI40XX.DPUInvariant
// CHECK-SAME: enqueueBarrier([[BAR0]]

// CHECK: [[DPU1:%.*]] = VPUMI40XX.DPUInvariant
// CHECK-SAME: enqueueBarrier([[BAR0]]

// CHECK: [[SHV0:%.*]] = VPUMI40XX.ActKernelInvocation
// CHECK-SAME: enqueueBarrier([[BAR1]]
