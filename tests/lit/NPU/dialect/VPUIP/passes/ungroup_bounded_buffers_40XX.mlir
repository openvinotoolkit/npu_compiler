//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --ungroup-bounded-buffers %s | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK-LABEL: UngroupBoundedBuffersforConvertDMA
func.func @UngroupBoundedBuffersforConvertDMA(%arg0: memref<1x18x15xf32, @DDR>, %arg1: memref<3xsi32, @DDR>, %arg2: memref<1x18x15xf16, @DDR>, %arg3: memref<3xsi32, @DDR>)->(!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>) {
    // CHECK-SAME: [[ARG0:%.+]]: memref<1x18x15xf32, @DDR>,
    // CHECK-SAME: [[ARG1:%.+]]: memref<3xsi32, @DDR>,
    // CHECK-SAME: [[ARG2:%.+]]: memref<1x18x15xf16, @DDR>,
    // CHECK-SAME: [[ARG3:%.+]]: memref<3xsi32, @DDR>

    %0 = VPUIP.GroupBoundedBuffer(%arg0, %arg1): memref<1x18x15xf32, @DDR>, memref<3xsi32, @DDR>->!VPUIP.BoundedBuffer<data = memref<1x18x15xf32, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    %1 = VPUIP.GroupBoundedBuffer(%arg2, %arg3): memref<1x18x15xf16, @DDR>, memref<3xsi32, @DDR>->!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    %2 = VPUIP.ConvertDMA inputs(%0: !VPUIP.BoundedBuffer<data = memref<1x18x15xf32, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>)
                          outputs(%1: !VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>)
        ->!VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>
    return %2 : !VPUIP.BoundedBuffer<data = memref<1x18x15xf16, @DDR>, dynamic_shape = memref<3xsi32, @DDR>>

    // CHECK: [[DATA_CONVERTDMA:%.+]] = VPUIP.ConvertDMA inputs([[ARG0]] : memref<1x18x15xf32, @DDR>) outputs([[ARG2]] : memref<1x18x15xf16, @DDR>) -> memref<1x18x15xf16, @DDR>
    // CHECK: [[SHAPE_COPY:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<3xsi32, @DDR>) outputs([[ARG3]] : memref<3xsi32, @DDR>) -> memref<3xsi32, @DDR>
    // CHECK: [[BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[DATA_CONVERTDMA]], [[SHAPE_COPY]]) : memref<1x18x15xf16, @DDR>, memref<3xsi32, @DDR> -> !VPUIP.BoundedBuffer<data=memref<1x18x15xf16, @DDR>, dynamic_shape=memref<3xsi32, @DDR>>
    // CHECK: return [[BOUNDED_BUFFER]]
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @VPU.SW {
  func.func nested @builtin_LSTMSequence(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64, none, i64) attributes {VPU.kernel_code = "lstm_sequence.cpp", VPU.kernel_entry = "lstm_sequence"}
  func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @ProcessBuffers(%arg0: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, %arg1: memref<1x1x1x128xf16, @DDR>, %arg2: memref<1x1x1x128xf16, @DDR>, %arg3: memref<1x4x128x128xf16, {order = #NWHC}>, %arg4: memref<1x1x1x2xsi32>) -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>) {
    %alloc_0 = memref.alloc() : memref<1x1x35x512xf16, @DDR>
    %alloc_1 = memref.alloc() : memref<4xsi32, @DDR>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x1x35x512xf16, @DDR>, memref<4xsi32, @DDR> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, @DDR>, dynamic_shape=memref<4xsi32, @DDR>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) outputs(%0 : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, @DDR>, dynamic_shape=memref<4xsi32, @DDR>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, @DDR>, dynamic_shape=memref<4xsi32, @DDR>>
    %alloc_2 = memref.alloc() : memref<1x1x35x512xf16, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %2 = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x1x35x512xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %3 = VPUIP.Copy inputs(%1 : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, @DDR>, dynamic_shape=memref<4xsi32, @DDR>>) outputs(%2 : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %alloc_4 = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%arg1 : memref<1x1x1x128xf16, @DDR>) outputs(%alloc_4 : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%arg2 : memref<1x1x1x128xf16, @DDR>) outputs(%alloc_5 : memref<1x1x1x128xf16, [@CMX_NN, 0]>) -> memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %alloc_6 = memref.alloc() : memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>
    %6 = VPUIP.Copy inputs(%arg3 : memref<1x4x128x128xf16, {order = #NWHC}>) outputs(%alloc_6 : memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>) -> memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    %7 = VPUIP.Copy inputs(%arg4 : memref<1x1x1x2xsi32>) outputs(%alloc_7 : memref<1x1x1x2xsi32, [@CMX_NN, 0]>) -> memref<1x1x1x2xsi32, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x1x35x128xf16, [@CMX_NN, 0]>
    %alloc_9 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %8 = VPUIP.GroupBoundedBuffer(%alloc_8, %alloc_9) : memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %alloc_10 = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_LSTMSequence inputs(%3 as %arg8: !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, %4 as %arg9: memref<1x1x1x128xf16, [@CMX_NN, 0]>, %5 as %arg10: memref<1x1x1x128xf16, [@CMX_NN, 0]>, %6 as %arg11: memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>, %7 as %arg12: memref<1x1x1x2xsi32, [@CMX_NN, 0]>) outputs(%8 as %arg13: !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, %alloc_10 as %arg14: memref<1x1x1x128xf16, [@CMX_NN, 0]>, %alloc_11 as %arg15: memref<1x1x1x128xf16, [@CMX_NN, 0]>) on tile 0 -> (!VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>) {
        VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 1023], 1]}(%arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) : !VPUIP.BoundedBuffer<data=memref<1x1x35x512xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
    }

    return %8, %alloc_10, %alloc_11 : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>

    // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer({{%.+}}, {{%.+}}) : memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    // CHECK: [[ALLOC_0:%.+]] = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK: [[DATA:%.+]], [[DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP]]) : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>> -> memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>
    // CHECK: VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1, -1, -1, -1>, dynamicOutputShapesMap = array<i32: 0, -1, -1>, resultSegmentSizes = array<i32: 3, 1, 0>} @VPU.SW::@builtin_LSTMSequence inputs({{%.+}} as [[ARG_5:%[^:]+]]: memref<1x1x35x512xf16, [@CMX_NN, 0]>, {{%.+}} as [[ARG_6:%[^:]+]]: memref<1x1x1x128xf16, [@CMX_NN, 0]>, {{%.+}} as [[ARG_7:%[^:]+]]: memref<1x1x1x128xf16, [@CMX_NN, 0]>, {{%.+}} as [[ARG_8:%[^:]+]]: memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>, {{%.+}} as [[ARG_9:%[^:]+]]: memref<1x1x1x2xsi32, [@CMX_NN, 0]>) dynamicInputShapes({{%.+}} : memref<4xsi32, [@CMX_NN, 0]>) outputs([[DATA]] as [[ARG_10:%[^:]+]]: memref<1x1x35x128xf16, [@CMX_NN, 0]>, [[ALLOC_0]] as [[ARG_11:%[^:]+]]: memref<1x1x1x128xf16, [@CMX_NN, 0]>, [[ALLOC_1]] as [[ARG_12:%[^:]+]]: memref<1x1x1x128xf16, [@CMX_NN, 0]>) dynamicOutputShapes([[DYN_SHAPE]] : memref<4xsi32, [@CMX_NN, 0]>) on tile 0 -> (memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]>){
    // CHECK{LITERAL}: VPUIP.SW.Kernel.run {attrs = [[-9223372036854775808, 1023], 1]}
    // CHECK: ([[ARG_5]], [[ARG_6]], [[ARG_7]], [[ARG_8]], [[ARG_9]], [[ARG_10]], [[ARG_11]], [[ARG_12]]) : memref<1x1x35x512xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x4x128x128xf16, {order = #NWHC}, [@CMX_NN, 0]>, memref<1x1x1x2xsi32, [@CMX_NN, 0]>, memref<1x1x35x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
    // CHECK: }
    // CHECK: return [[GROUP]], [[ALLOC_0]], [[ALLOC_1]] : !VPUIP.BoundedBuffer<data=memref<1x1x35x128xf16, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>, memref<1x1x1x128xf16, [@CMX_NN, 0]>, memref<1x1x1x128xf16, [@CMX_NN, 0]>
}
