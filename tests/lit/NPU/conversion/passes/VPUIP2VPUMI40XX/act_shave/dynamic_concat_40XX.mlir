//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

module {
  net.NetworkInfo entryPoint : @DynamicConcat inputsInfo : {
    DataInfo "input_bound" : tensor<1x3x5x16xf16>
    DataInfo "input_shape" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "output_bound" : tensor<2x3x5x16xf16>
    DataInfo "output_shape" : tensor<4xsi32>
  }
  module @VPU.SW {
    func.func nested @builtin_Concat(memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>, memref<*xsi32, [@CMX_NN, 0]>, none, none) attributes {VPU.kernel_code = "concat.cpp", VPU.kernel_entry = "concat"}
  }
  // CHECK-LABEL: @DynamicConcat
  func.func @DynamicConcat(%arg0: memref<1x3x5x16xf16, [@CMX_NN, 0]>, %arg1: memref<4xsi32, [@CMX_NN, 0]>) -> (memref<2x3x5x16xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>) {
    %alloc_0 = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<1x3x5x16xf16, [@CMX_NN, 0]>
    %alloc_1 = VPURT.DeclareBuffer <CMX_NN> [0] <1536> -> memref<4xsi32, [@CMX_NN, 0]>
    %alloc_2 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x3x5x16xf16, [@CMX_NN, 0]>
    %alloc_3 = VPURT.DeclareBuffer <CMX_NN> [0] <960> -> memref<4xsi32, [@CMX_NN, 0]>

    VPURT.Task {
        VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0, 1>,
            dynamicOutputShapesMap = array<i32: 0>,
            resultSegmentSizes = array<i32: 1, 1, 0>
        }
        @VPU.SW::@builtin_Concat
            inputs(%alloc_0 as %arg2: memref<1x3x5x16xf16, [@CMX_NN, 0]>, %alloc_0 as %arg3: memref<1x3x5x16xf16, [@CMX_NN, 0]>)
            dynamicInputShapes(%alloc_1 : memref<4xsi32, [@CMX_NN, 0]>, %alloc_1 : memref<4xsi32, [@CMX_NN, 0]>)
            outputs(%alloc_2 as %arg4 : memref<2x3x5x16xf16, [@CMX_NN, 0]>)
            dynamicOutputShapes(%alloc_3 : memref<4xsi32, [@CMX_NN, 0]>)
                -> (memref<2x3x5x16xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>) {
            VPUIP.SW.Kernel.run {attrs = [[0, 0, 0, 0], [0, 0, 0, 1]]}(%arg2, %arg3, %arg4) :
                memref<1x3x5x16xf16, [@CMX_NN, 0]>,
                memref<1x3x5x16xf16, [@CMX_NN, 0]>,
                memref<2x3x5x16xf16, [@CMX_NN, 0]>
            }
    }

    return %alloc_2, %alloc_3 : memref<2x3x5x16xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
  }
  //CHECK:        [[IN_BOUND:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[IN_DYN_SHAPE:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_BOUND:%.+]] = VPURT.DeclareBuffer
  //CHECK:        [[OUT_DYN_SHAPE:%.+]] = VPURT.DeclareBuffer

  //CHECK:       VPUMI40XX.KernelParams
  //CHECK-SAME:  inputs([[IN_BOUND]]
  //CHECK-SAME:  outputs([[OUT_BOUND]]
  //CHECK-SAME:  dynamicInputShapes(([[IN_DYN_SHAPE]])
  //CHECK-SAME:  dynamicOutputShapes(([[OUT_DYN_SHAPE]])
}
