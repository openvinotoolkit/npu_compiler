//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ungroup-bounded-buffers-as-func-args --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @TensorsWithBounds inputsInfo : {
// CHECK:     net.NetworkInfo entryPoint : @TensorsWithBounds inputsInfo
    DataInfo "Parameter" : tensor<1x18x3x3xf32>
// CHECK:     DataInfo "Parameter" : tensor<1x18x3x3xf32>
// CHECK:     DataInfo "vpux_ie_shape_Parameter" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Copy_result" : tensor<1x18x3x3xf32>
// CHECK:     DataInfo "Copy_result" : tensor<1x18x3x3xf32>
// CHECK:     DataInfo "vpux_ie_shape_Copy_result" : tensor<4xsi32>
  }
// CHECK:       @TensorsWithBounds([[ARG0:%.*]]: memref<1x18x3x3xf32, #NHWC>, [[ARG1:%.*]]: memref<4xsi32>) -> (memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  func.func @TensorsWithBounds(%arg0: !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>> {
    %alloc = memref.alloc() : memref<1x18x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc, %alloc_0) : memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) outputs(%0 : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>) -> !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
    return %1 : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
  }
}

// CHECK:       [[INPUT_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG1]]) : memref<1x18x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK:       [[OUTPUT_DATA:%.*]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_SHAPE:%.*]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[OUTPUT_DATA]], [[OUTPUT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK:       [[COPY_OP:%.*]] = VPUIP.Copy inputs([[INPUT_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:                               outputs([[OUTPUT_BOUNDED_BUFFER:%.*]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)

// CHECK:       [[OUTPUT_DATA:%.*]], [[OUTPUT_SHAPE:%.*]] = VPUIP.UngroupBoundedBuffer([[COPY_OP]])
// CHECK-SAME:      : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
// CHECK:       return [[OUTPUT_DATA:%.*]], [[OUTPUT_SHAPE:%.*]] : memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @TensorsWithBoundsMultiple inputsInfo : {
// CHECK:     net.NetworkInfo entryPoint : @TensorsWithBoundsMultiple inputsInfo
    DataInfo "Parameter1" : tensor<1x10x3x3xf32>
    DataInfo "Parameter2" : tensor<1x20x3x3xf32>
    DataInfo "Parameter3" : tensor<1x30x3x3xf32>
    DataInfo "Parameter4" : tensor<1x40x3x3xf32>

// CHECK:     DataInfo "Parameter1" : tensor<1x10x3x3xf32>
// CHECK:     DataInfo "Parameter2" : tensor<1x20x3x3xf32>
// CHECK:     DataInfo "Parameter3" : tensor<1x30x3x3xf32>
// CHECK:     DataInfo "Parameter4" : tensor<1x40x3x3xf32>

// CHECK:     DataInfo "vpux_ie_shape_Parameter1" : tensor<4xsi32>
// CHECK:     DataInfo "vpux_ie_shape_Parameter3" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "Result1" : tensor<1x10x3x3xf32>
    DataInfo "Result2" : tensor<1x20x3x3xf32>
    DataInfo "Result3" : tensor<1x30x3x3xf32>
    DataInfo "Result4" : tensor<1x40x3x3xf32>
// CHECK:     DataInfo "Result1" : tensor<1x10x3x3xf32>
// CHECK:     DataInfo "Result2" : tensor<1x20x3x3xf32>
// CHECK:     DataInfo "Result3" : tensor<1x30x3x3xf32>
// CHECK:     DataInfo "Result4" : tensor<1x40x3x3xf32>

// CHECK:     DataInfo "vpux_ie_shape_Result1" : tensor<4xsi32>
// CHECK:     DataInfo "vpux_ie_shape_Result3" : tensor<4xsi32>
  }
  func.func @TensorsWithBoundsMultiple(
                              %arg0: !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>,
                              %arg1: memref<1x20x3x3xf32, #NHWC>,
                              %arg2: !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>,
                              %arg3: memref<1x40x3x3xf32, #NHWC>)
                              -> (
                              !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
                              memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                              !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
                              memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {
// CHECK:       @TensorsWithBoundsMultiple([[ARG0_0:%.*]]: memref<1x10x3x3xf32, #NHWC>, [[ARG1:%.*]]: memref<1x20x3x3xf32, #NHWC>,
// CHECK-SAME:                     [[ARG2_0:%.*]]: memref<1x30x3x3xf32, #NHWC>, [[ARG3:%.*]]: memref<1x40x3x3xf32, #NHWC>,
// CHECK-SAME:                     [[ARG0_1:%.*]]: memref<4xsi32>, [[ARG2_1:%.*]]: memref<4xsi32>) ->
// CHECK-SAME:                     (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:                      memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:                      memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:                      memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:                      memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
    %1 = VPUIP.Copy inputs(%arg0 : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) outputs(%0 : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>) -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
    %4 = VPUIP.Copy inputs(%arg2 : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>) outputs(%3 : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>) -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, #NHWC>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

    return %1, %2, %4, %5 :
      !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
      memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
      !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>,
      memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }
}

// CHECK:       [[INPUT_2_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[ARG2_0]], [[ARG2_1]]) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[INPUT_0_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[ARG0_0]], [[ARG0_1]]) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[OUTPUT_0_DATA:%.*]] = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_0_SHAPE:%.*]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_0_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[OUTPUT_0_DATA]], [[OUTPUT_0_SHAPE]]) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
// CHECK:       [[OUTPUT_0_BOUNDED_BUFFER_COPY:%.*]] = VPUIP.Copy
// CHECK-SAME:    inputs([[INPUT_0_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:    outputs([[OUTPUT_0_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

// CHECK:       [[INPUT_1_STATIC:%.*]] = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_1_STATIC:%.*]] = VPUIP.Copy inputs([[ARG1]] : memref<1x20x3x3xf32, #NHWC>) outputs([[INPUT_1_STATIC]] : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

// CHECK:       [[OUTPUT_2_DATA:%.*]] = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_2_SHAPE:%.*]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_2_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[OUTPUT_2_DATA]], [[OUTPUT_2_SHAPE]]) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
// CHECK:       [[OUTPUT_2_BOUNDED_BUFFER_COPY:%.*]] = VPUIP.Copy
// CHECK-SAME:    inputs([[INPUT_2_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:    outputs([[OUTPUT_2_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

// CHECK:       [[INPUT_3_STATIC:%.*]] = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_3_STATIC:%.*]] = VPUIP.Copy inputs([[ARG3]] : memref<1x40x3x3xf32, #NHWC>) outputs([[INPUT_3_STATIC]] : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

// CHECK:       [[OUTPUT_0_DATA_RESULT:%.*]], [[OUTPUT_0_SHAPE_RESULT:%.*]] = VPUIP.UngroupBoundedBuffer([[OUTPUT_0_BOUNDED_BUFFER_COPY]]) :
// CHECK-SAME:    !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>> -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

// CHECK:       [[OUTPUT_2_DATA_RESULT:%.*]], [[OUTPUT_2_SHAPE_RESULT:%.*]] = VPUIP.UngroupBoundedBuffer([[OUTPUT_2_BOUNDED_BUFFER_COPY]]) :
// CHECK-SAME:    !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>> -> memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

// CHECK:       return [[OUTPUT_0_DATA_RESULT]], [[OUTPUT_1_STATIC]], [[OUTPUT_2_DATA_RESULT]], [[OUTPUT_3_STATIC]], [[OUTPUT_0_SHAPE_RESULT]], [[OUTPUT_2_SHAPE_RESULT]]
// CHECK-SAME:    : memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:      memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:      memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:      memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
// CHECK-SAME:      memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @DynamicScatterNDUpdateCheckInputsOrder {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_57" : tensor<1x3x3xsi32>
    DataInfo "Parameter_58" : tensor<1x2x3x3xsi32>
    DataInfo "Parameter_59" : tensor<1x2x3xsi32>

// CHECK:     DataInfo "Parameter_57" : tensor<1x3x3xsi32>
// CHECK:     DataInfo "Parameter_58" : tensor<1x2x3x3xsi32>
// CHECK:     DataInfo "Parameter_59" : tensor<1x2x3xsi32>

// CHECK:     DataInfo "vpux_ie_shape_Parameter_58" : tensor<4xsi32>
// CHECK:     DataInfo "vpux_ie_shape_Parameter_59" : tensor<3xsi32>
  } outputsInfo : {
    DataInfo "Copy_result" : tensor<1x2x3xsi32>
  }
  func.func @main(%arg0: memref<1x3x3xsi32>, %arg1: !VPUIP.BoundedBuffer<data=memref<1x2x3x3xsi32>, dynamic_shape=memref<4xsi32>>, %arg2: !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32>, dynamic_shape=memref<3xsi32>>) -> !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>> {
    %alloc = memref.alloc() : memref<1x3x3xsi32, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x3xsi32>) outputs(%alloc : memref<1x3x3xsi32, [@CMX_NN, 0]>) -> memref<1x3x3xsi32, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x2x3x3xsi32, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %1 = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x2x3x3xsi32, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x2x3x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %2 = VPUIP.Copy inputs(%arg1 : !VPUIP.BoundedBuffer<data=memref<1x2x3x3xsi32>, dynamic_shape=memref<4xsi32>>) outputs(%1 : !VPUIP.BoundedBuffer<data=memref<1x2x3x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<1x2x3x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<4xsi32, [@CMX_NN, 0]>>
    %alloc_2 = memref.alloc() : memref<1x2x3xsi32, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x2x3xsi32, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    %4 = VPUIP.Copy inputs(%arg2 : !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32>, dynamic_shape=memref<3xsi32>>) outputs(%3 : !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>) -> !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
    return %4 : !VPUIP.BoundedBuffer<data=memref<1x2x3xsi32, [@CMX_NN, 0]>, dynamic_shape=memref<3xsi32, [@CMX_NN, 0]>>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff1CMX = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
!BoundedBuff2 = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff2CMX = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

module @OutlinedMainContentInOneFunc {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter1" : tensor<1x10x3x3xf32>
    DataInfo "Parameter2" : tensor<1x20x3x3xf32>
    DataInfo "Parameter3" : tensor<1x30x3x3xf32>
    DataInfo "Parameter4" : tensor<1x40x3x3xf32>

  } outputsInfo : {
    DataInfo "Result1" : tensor<1x10x3x3xf32>
    DataInfo "Result2" : tensor<1x20x3x3xf32>
    DataInfo "Result3" : tensor<1x30x3x3xf32>
    DataInfo "Result4" : tensor<1x40x3x3xf32>
  }

// CHECK:  net.NetworkInfo entryPoint : @main inputsInfo
// CHECK:   DataInfo "Parameter1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Parameter2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Parameter3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Parameter4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter3" : tensor<4xsi32>

// CHECK:  outputsInfo
// CHECK:   DataInfo "Result1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Result2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Result3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Result4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Result1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Result3" : tensor<4xsi32>

  func.func private @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, #NHWC>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func private @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]]) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[BOUNDED_BUFF0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]]) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY0:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF0]] : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // Original Consumer of %arg2
  // CHECK:      [[ORIG_COPY1:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF1]] : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:      [[UNGROUP_DATA0:%.+]], [[UNGROUP_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK:      [[UNGROUP_DATA1:%.+]], [[UNGROUP_DYN_SHAPE1:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK:      return [[UNGROUP_DATA0]], {{%.+}}, [[UNGROUP_DATA1]], {{%.+}}, [[UNGROUP_DYN_SHAPE0]], [[UNGROUP_DYN_SHAPE1]]

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                  -> (
                  !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {

    %res:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, #NHWC>, !BoundedBuff2, memref<1x40x3x3xf32, #NHWC>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>)

    return %res#0, %res#1, %res#2, %res#3 :
        !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
        !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>

  // CHECK:     [[RES:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, #NHWC>, memref<1x20x3x3xf32, #NHWC>, memref<1x30x3x3xf32, #NHWC>, memref<1x40x3x3xf32, #NHWC>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<1x40x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_OUT0:%.+]] = VPUIP.GroupBoundedBuffer([[RES]]#0, [[RES]]#4)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
  // CHECK:      [[BOUNDED_BUFF_OUT1:%.+]] = VPUIP.GroupBoundedBuffer([[RES]]#2, [[RES]]#5)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:      [[UNGROUP_DATA_OUT0:%.+]], [[UNGROUP_DYN_SHAPE_OUT0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_OUT0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK:      [[UNGROUP_DATA_OUT1:%.+]], [[UNGROUP_DYN_SHAPE_OUT1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_OUT1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     return [[UNGROUP_DATA_OUT0]], [[RES]]#1, [[UNGROUP_DATA_OUT1]], [[RES]]#3, [[UNGROUP_DYN_SHAPE_OUT0]], [[UNGROUP_DYN_SHAPE_OUT1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff1CMX = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
!BoundedBuff2 = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff2CMX = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

module @Outlined2SequentialFunc {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter1" : tensor<1x10x3x3xf32>
    DataInfo "Parameter2" : tensor<1x20x3x3xf32>
    DataInfo "Parameter3" : tensor<1x30x3x3xf32>
    DataInfo "Parameter4" : tensor<1x40x3x3xf32>

  } outputsInfo : {
    DataInfo "Result1" : tensor<1x10x3x3xf32>
    DataInfo "Result2" : tensor<1x20x3x3xf32>
    DataInfo "Result3" : tensor<1x30x3x3xf32>
    DataInfo "Result4" : tensor<1x40x3x3xf32>
  }

// CHECK:  net.NetworkInfo entryPoint : @main inputsInfo
// CHECK:   DataInfo "Parameter1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Parameter2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Parameter3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Parameter4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter3" : tensor<4xsi32>

// CHECK:  outputsInfo
// CHECK:   DataInfo "Result1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Result2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Result3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Result4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Result1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Result3" : tensor<4xsi32>

  func.func private @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, #NHWC>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func private @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Identical to @foo in @OutlinedMainContentInOneFunc test, skipping checks

  func.func private @foo1(%arg0: !BoundedBuff1CMX) -> !BoundedBuff1 {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32> -> !BoundedBuff1
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1CMX) outputs(%0 : !BoundedBuff1) -> !BoundedBuff1

    return %1 : !BoundedBuff1
  }

  // CHECK:        func.func private @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG1]]) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF]] : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA:%.+]], [[UNGROUP_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      return [[UNGROUP_DATA]], [[UNGROUP_DYN_SHAPE]]

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                  -> (
                  !BoundedBuff1, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, #NHWC>, !BoundedBuff2, memref<1x40x3x3xf32, #NHWC>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>)

    %res_foo1 = call @foo1(%res_foo#0)
      : (!BoundedBuff1CMX) -> (!BoundedBuff1)

    return %res_foo1, %res_foo#1, %res_foo#2, %res_foo#3 : !BoundedBuff1, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>

  // CHECK:     [[RES_FOO:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, #NHWC>, memref<1x20x3x3xf32, #NHWC>, memref<1x30x3x3xf32, #NHWC>, memref<1x40x3x3xf32, #NHWC>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<1x40x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:     [[GROUP_FOO_0:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#0, [[RES_FOO]]#4)
  // CHECK-SAME:    memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:     [[GROUP_FOO_1:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#2, [[RES_FOO]]#5)
  // CHECK-SAME:    memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:     [[DATA:%.+]], [[DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_0]])
  // CHECK-SAME:    : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO1:%.+]]:2 = call @foo1([[DATA]], [[DYN_SHAPE]]) : (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO1]]#0, [[RES_FOO1]]#1)
  // CHECK-SAME:    memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA_RET0:%.+]], [[UNGROUP_DYN_SHAPE_RET0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_RET1:%.+]], [[UNGROUP_DYN_SHAPE_RET1:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     return [[UNGROUP_DATA_RET0]], [[RES_FOO]]#1, [[UNGROUP_DATA_RET1]], [[RES_FOO]]#3, [[UNGROUP_DYN_SHAPE_RET0]], [[UNGROUP_DYN_SHAPE_RET1]]
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff1CMX = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
!BoundedBuff2 = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff2CMX = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

module @CallAndNonCallOpConsumersOfBoundedBuff {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter1" : tensor<1x10x3x3xf32>
    DataInfo "Parameter2" : tensor<1x20x3x3xf32>
    DataInfo "Parameter3" : tensor<1x30x3x3xf32>
    DataInfo "Parameter4" : tensor<1x40x3x3xf32>

  } outputsInfo : {
    DataInfo "Result1" : tensor<1x10x3x3xf32>
    DataInfo "Result2" : tensor<1x20x3x3xf32>
    DataInfo "Result3" : tensor<1x30x3x3xf32>
    DataInfo "Result4" : tensor<1x40x3x3xf32>
    DataInfo "Result5" : tensor<1x30x3x3xf32>
  }

// CHECK:  net.NetworkInfo entryPoint : @main inputsInfo
// CHECK:   DataInfo "Parameter1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Parameter2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Parameter3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Parameter4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter3" : tensor<4xsi32>

// CHECK:  outputsInfo
// CHECK:   DataInfo "Result1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Result2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Result3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Result4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "Result5" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Result1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Result3" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Result5" : tensor<4xsi32>

  func.func private @foo(%arg0: !BoundedBuff1,
                         %arg1: memref<1x20x3x3xf32, #NHWC>,
                         %arg2: !BoundedBuff2,
                         %arg3: memref<1x40x3x3xf32, #NHWC>)
                            -> (
                         !BoundedBuff1CMX,
                         memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                         !BoundedBuff2CMX,
                         memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, #NHWC>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX,
      memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
      !BoundedBuff2CMX,
      memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func private @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Identical to @foo in @OutlinedMainContentInOneFunc test, skipping checks

  func.func private @foo1(%arg0: !BoundedBuff1CMX) -> !BoundedBuff1 {
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32> -> !BoundedBuff1
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1CMX) outputs(%0 : !BoundedBuff1) -> !BoundedBuff1

    return %1 : !BoundedBuff1
  }

  // CHECK:        func.func private @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>)

  // Identical to @foo1 in @Outlined2SequentialFunc test, skipping checks

  func.func private @foo2(%arg0: !BoundedBuff2CMX, %arg1: memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> (!BoundedBuff2, memref<1x20x3x3xf32, #NHWC>) {
    %alloc_0_0 = memref.alloc() : memref<1x30x3x3xf32, #NHWC>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32> -> !BoundedBuff2
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff2CMX) outputs(%0 : !BoundedBuff2) -> !BoundedBuff2

    // 1st arg
    %alloc_1 = memref.alloc() : memref<1x20x3x3xf32, #NHWC>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) outputs(%alloc_1 : memref<1x20x3x3xf32, #NHWC>) -> memref<1x20x3x3xf32, #NHWC>

    return %1, %2 : !BoundedBuff2, memref<1x20x3x3xf32, #NHWC>
  }

  // CHECK:        func.func private @foo2(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x30x3x3xf32, #NHWC>, memref<1x20x3x3xf32, #NHWC>, memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG2]]) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF]] : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // Copy other, unbounded, arg
  // CHECK:      [[OTHER_ARG:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x20x3x3xf32, #NHWC, @CMX_NN>)
  // CHECK-SAME:    -> memref<1x20x3x3xf32, #NHWC>

  // CHECK:      [[UNGROUP_DATA:%.+]], [[UNGROUP_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      return [[UNGROUP_DATA]], [[OTHER_ARG]], [[UNGROUP_DYN_SHAPE]]

  func.func @main(%arg0: !BoundedBuff1,
                  %arg1: memref<1x20x3x3xf32, #NHWC>,
                  %arg2: !BoundedBuff2,
                  %arg3: memref<1x40x3x3xf32, #NHWC>)
                  -> (
                  !BoundedBuff1,
                  memref<1x20x3x3xf32, #NHWC>,
                  !BoundedBuff2,
                  memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
                  !BoundedBuff2) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, #NHWC>, !BoundedBuff2, memref<1x40x3x3xf32, #NHWC>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>)

    %alloc_0_0 = memref.alloc() : memref<1x30x3x3xf32, #NHWC>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32> -> !BoundedBuff2
    %1 = VPUIP.Copy inputs(%res_foo#2 : !BoundedBuff2CMX) outputs(%0 : !BoundedBuff2) -> !BoundedBuff2

    %res_foo1 = call @foo1(%res_foo#0)
      : (!BoundedBuff1CMX) -> (!BoundedBuff1)

    %res_foo2:2 = call @foo2(%res_foo#2, %res_foo#1)
      : (!BoundedBuff2CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> (!BoundedBuff2, memref<1x20x3x3xf32, #NHWC>)

    return %res_foo1, %res_foo2#1, %res_foo2#0, %res_foo#3, %1 :
        !BoundedBuff1, memref<1x20x3x3xf32, #NHWC>,
        !BoundedBuff2,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
        !BoundedBuff2
  }

    // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>

  // CHECK:     [[RES_FOO:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, #NHWC>, memref<1x20x3x3xf32, #NHWC>, memref<1x30x3x3xf32, #NHWC>, memref<1x40x3x3xf32, #NHWC>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<1x40x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:     [[GROUP_FOO_0:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#0, [[RES_FOO]]#4)
  // CHECK-SAME:    memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:     [[GROUP_FOO_1:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#2, [[RES_FOO]]#5)
  // CHECK-SAME:    memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // Non Call Op with func @foo's 2nd Bounded Buff result as input
  // CHECK:     [[NON_CALL_OP:%.+]] = VPUIP.Copy
  // CHECK-SAME:   inputs([[GROUP_FOO_1]] : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
  // CHECK-SAME:   outputs({{%.+}} : !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
  // CHECK-SAME:     -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // @foo1 CallOp
  // CHECK:     [[DATA:%.+]], [[DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_0]])
  // CHECK-SAME:    : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO1:%.+]]:2 = call @foo1([[DATA]], [[DYN_SHAPE]]) : (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO1]]#0, [[RES_FOO1]]#1)
  // CHECK-SAME:    memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // @foo2 CallOp
  // CHECK:      [[UNGROUP_DATA_FOO2_DATA0:%.+]], [[UNGROUP_DATA_FOO2_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO2:%.+]]:3 = call @foo2([[UNGROUP_DATA_FOO2_DATA0]], [[RES_FOO]]#1, [[UNGROUP_DATA_FOO2_DYN_SHAPE0]])
  // CHECK-SAME:    : (memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x30x3x3xf32, #NHWC>, memref<1x20x3x3xf32, #NHWC>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO2_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO2]]#0, [[RES_FOO2]]#2)
  // CHECK-SAME:    memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // Ungroup return values
  // CHECK:      [[UNGROUP_DATA_RET0:%.+]], [[UNGROUP_DYN_SHAPE_RET0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_RET1:%.+]], [[UNGROUP_DYN_SHAPE_RET1:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO2_RES]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_RET2:%.+]], [[UNGROUP_DYN_SHAPE_RET2:%.+]] = VPUIP.UngroupBoundedBuffer([[NON_CALL_OP]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>

  // CHECK:     return [[UNGROUP_DATA_RET0]], [[RES_FOO2]]#1, [[UNGROUP_DATA_RET1]], [[RES_FOO]]#3, [[UNGROUP_DATA_RET2]], [[UNGROUP_DYN_SHAPE_RET0]], [[UNGROUP_DYN_SHAPE_RET1]], [[UNGROUP_DYN_SHAPE_RET2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff1CMX = !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
!BoundedBuff2 = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>
!BoundedBuff2CMX = !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

module @NestedCallOps {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter1" : tensor<1x10x3x3xf32>
    DataInfo "Parameter2" : tensor<1x20x3x3xf32>
    DataInfo "Parameter3" : tensor<1x30x3x3xf32>
    DataInfo "Parameter4" : tensor<1x40x3x3xf32>

  } outputsInfo : {
    DataInfo "Result1" : tensor<1x10x3x3xf32>
    DataInfo "Result2" : tensor<1x20x3x3xf32>
    DataInfo "Result3" : tensor<1x30x3x3xf32>
    DataInfo "Result4" : tensor<1x40x3x3xf32>
  }

// CHECK:  net.NetworkInfo entryPoint : @main inputsInfo
// CHECK:   DataInfo "Parameter1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Parameter2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Parameter3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Parameter4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Parameter3" : tensor<4xsi32>

// CHECK:  outputsInfo
// CHECK:   DataInfo "Result1" : tensor<1x10x3x3xf32>
// CHECK:   DataInfo "Result2" : tensor<1x20x3x3xf32>
// CHECK:   DataInfo "Result3" : tensor<1x30x3x3xf32>
// CHECK:   DataInfo "Result4" : tensor<1x40x3x3xf32>
// CHECK:   DataInfo "vpux_ie_shape_Result1" : tensor<4xsi32>
// CHECK:   DataInfo "vpux_ie_shape_Result3" : tensor<4xsi32>

  func.func private @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {
    // 0th arg
    %res_foo1 = call @foo1(%arg0) : (!BoundedBuff1) -> (!BoundedBuff1CMX)

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, #NHWC, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, #NHWC>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x20x3x3xf32, #NHWC, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, #NHWC, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, #NHWC, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, #NHWC>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, #NHWC, @CMX_NN>) -> memref<1x40x3x3xf32, #NHWC, @CMX_NN>

    return %res_foo1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func private @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]]) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[BOUNDED_BUFF0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]]) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

  // CHECK:      [[UNGROUP_DATA0:%.+]], [[UNGROUP_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>

  // CHECK:      [[FOO1_CALL:%.+]]:2 = call @foo1([[UNGROUP_DATA0]], [[UNGROUP_DYN_SHAPE0]])
  // CHECK-SAME:     : (memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>) -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:      [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[FOO1_CALL]]#0, [[FOO1_CALL]]#1)
  // CHECK-SAME:    : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>

  // CHECK:      [[UNGROUP_FOO1_RES_DATA:%.+]], [[UNGROUP_FOO1_RES_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    : !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:      return [[UNGROUP_FOO1_RES_DATA]], {{%.+}}, {{%.+}}, {{%.+}}, [[UNGROUP_FOO1_RES_DYN_SHAPE]], {{%.+}}

  func.func private @foo1(%arg0: !BoundedBuff1) -> !BoundedBuff1CMX {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, #NHWC, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    return %1 : !BoundedBuff1CMX
  }

  // CHECK:        func.func private @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Almost identical to @foo1 in @Outlined2SequentialFunc test, skipping checks

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, #NHWC>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, #NHWC>)
                  -> (
                  !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, #NHWC, @CMX_NN>) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, #NHWC>, !BoundedBuff2, memref<1x40x3x3xf32, #NHWC>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>)

    return %res_foo#0, %res_foo#1, %res_foo#2, %res_foo#3 : !BoundedBuff1CMX, memref<1x20x3x3xf32, #NHWC, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, #NHWC, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, #NHWC>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, #NHWC, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // main func almost identical to the one in @OutlinedMainContentInOneFunc, skip checks
}
