//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s
// REQUIRES: platform-NPU3720

module {
  net.NetworkInfo entryPoint : @SWKernelDynamicInputs inputsInfo : {
    DataInfo "input_bound" : tensor<1x3x10x10xf16>
    DataInfo "input_shape" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "output_bound" : tensor<1x3x10x10xf16>
    DataInfo "output_shape" : tensor<4xsi32>
  }
  module @VPU.SW {
    func.func nested @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
  // CHECK-LABEL: @SWKernelDynamicInputs
  func.func @SWKernelDynamicInputs(%arg0: memref<1x3x10x10xf16>, %arg1: memref<4xsi32>) -> (memref<1x3x10x10xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>) {
    %alloc_0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x10x10xf16, [@CMX_NN, 0]>
    %alloc_1 = VPURT.DeclareBuffer <CMX_NN> [0] <600> -> memref<4xsi32, [@CMX_NN, 0]>

    %alloc_2 = VPURT.DeclareBuffer <CMX_NN> [0] <616> -> memref<1x3x10x10xf16, [@CMX_NN, 0]>
    %alloc_3 = VPURT.DeclareBuffer <CMX_NN> [0] <1216> -> memref<4xsi32, [@CMX_NN, 0]>

    VPURT.Task {
        VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0>,
            dynamicOutputShapesMap = array<i32: 0>,
            resultSegmentSizes = array<i32: 2, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%alloc_0 as %arg2: memref<1x3x10x10xf16, [@CMX_NN, 0]>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32, [@CMX_NN, 0]>)
            outputs(%alloc_2 as %arg3: memref<1x3x10x10xf16, [@CMX_NN, 0]>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32, [@CMX_NN, 0]>)
                -> (memref<1x3x10x10xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>) {
            VPUIP.SW.Kernel.run(%arg2, %arg3) :
                memref<1x3x10x10xf16, [@CMX_NN, 0]>,
                memref<1x3x10x10xf16, [@CMX_NN, 0]>
            }
    }
    return %alloc_2, %alloc_3 : memref<1x3x10x10xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
  }
  //CHECK:        [[IN_BOUND:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[IN_DYN_SHAPE:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_BOUND:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_DYN_SHAPE:%.+]] = VPURT.DeclareBuffer

  //CHECK:        VPUMI37XX.KernelParams
  //CHECK-SAME:  inputs([[IN_BOUND]]
  //CHECK-SAME:  outputs([[OUT_BOUND]]
  //CHECK-SAME:  dynamicInputShapes(([[IN_DYN_SHAPE]])
  //CHECK-SAME:  dynamicOutputShapes(([[OUT_DYN_SHAPE]])
}
