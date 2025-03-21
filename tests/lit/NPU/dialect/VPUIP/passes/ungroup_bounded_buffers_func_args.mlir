//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ungroup-bounded-buffers-as-func-args --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @TensorsWithBounds inputsInfo : {
// CHECK:     IE.CNNNetwork entryPoint : @TensorsWithBounds inputsInfo
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
// CHECK:       [[INPUT_DATA:%.*]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC>
// CHECK:       [[INPUT_SHAPE:%.*]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[INPUT_DATA_COPY:%.*]] = VPUIP.Copy inputs([[ARG0]] : memref<1x18x3x3xf32, #NHWC>) outputs([[INPUT_DATA]] : memref<1x18x3x3xf32, #NHWC>)
// CHECK:       [[INPUT_SHAPE_COPY:%.*]] = VPUIP.Copy inputs([[ARG1]] : memref<4xsi32>) outputs([[INPUT_SHAPE]] : memref<4xsi32>) -> memref<4xsi32>
// CHECK:       [[INPUT_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[INPUT_DATA_COPY]], [[INPUT_SHAPE_COPY]]) : memref<1x18x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK:       [[OUTPUT_DATA:%.*]] = memref.alloc() : memref<1x18x3x3xf32, #NHWC, @CMX_NN>
// CHECK:       [[OUTPUT_SHAPE:%.*]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[OUTPUT_DATA]], [[OUTPUT_SHAPE]]) : memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK:       [[COPY_OP:%.*]] = VPUIP.Copy inputs([[INPUT_BOUNDED_BUFFER]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>)
// CHECK-SAME:  outputs([[OUTPUT_BOUNDED_BUFFER:%.*]] : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>)
// CHECK:       [[OUTPUT_DATA:%.*]], [[OUTPUT_SHAPE:%.*]] = VPUIP.UngroupBoundedBuffer([[COPY_OP]]) : !VPUIP.BoundedBuffer<data=memref<1x18x3x3xf32, #NHWC, @CMX_NN>, dynamic_shape=memref<4xsi32, @CMX_NN>>
// CHECK:       return [[OUTPUT_DATA:%.*]], [[OUTPUT_SHAPE:%.*]] : memref<1x18x3x3xf32, #NHWC, @CMX_NN>, memref<4xsi32, @CMX_NN>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  IE.CNNNetwork entryPoint : @TensorsWithBoundsMultiple inputsInfo : {
// CHECK:     IE.CNNNetwork entryPoint : @TensorsWithBoundsMultiple inputsInfo
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

// CHECK:       [[INPUT_0_DATA:%.*]] = memref.alloc() : memref<1x10x3x3xf32, #NHWC>
// CHECK:       [[INPUT_0_SHAPE:%.*]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[INPUT_0_DATA_COPY:%.*]] = VPUIP.Copy inputs([[ARG0_0]] : memref<1x10x3x3xf32, #NHWC>) outputs([[INPUT_0_DATA]] : memref<1x10x3x3xf32, #NHWC>) -> memref<1x10x3x3xf32, #NHWC>
// CHECK:       [[INPUT_0_SHAPE_COPY:%.*]] = VPUIP.Copy inputs([[ARG0_1]] : memref<4xsi32>) outputs([[INPUT_0_SHAPE]] : memref<4xsi32>) -> memref<4xsi32>
// CHECK:       [[INPUT_0_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[INPUT_0_DATA_COPY]], [[INPUT_0_SHAPE_COPY]]) : memref<1x10x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x10x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

// CHECK:       [[INPUT_2_DATA:%.*]] = memref.alloc() : memref<1x30x3x3xf32, #NHWC>
// CHECK:       [[INPUT_2_SHAPE:%.*]] = memref.alloc() : memref<4xsi32>
// CHECK:       [[INPUT_2_DATA_COPY:%.*]] = VPUIP.Copy inputs([[ARG2_0]] : memref<1x30x3x3xf32, #NHWC>) outputs([[INPUT_2_DATA]] : memref<1x30x3x3xf32, #NHWC>) -> memref<1x30x3x3xf32, #NHWC>
// CHECK:       [[INPUT_2_SHAPE_COPY:%.*]] = VPUIP.Copy inputs([[ARG2_1]] : memref<4xsi32>) outputs([[INPUT_2_SHAPE]] : memref<4xsi32>) -> memref<4xsi32>
// CHECK:       [[INPUT_2_BOUNDED_BUFFER:%.*]] = VPUIP.GroupBoundedBuffer([[INPUT_2_DATA_COPY]], [[INPUT_2_SHAPE_COPY]]) : memref<1x30x3x3xf32, #NHWC>, memref<4xsi32>
// CHECK-SAME:    -> !VPUIP.BoundedBuffer<data=memref<1x30x3x3xf32, #NHWC>, dynamic_shape=memref<4xsi32>>

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
  IE.CNNNetwork entryPoint : @main inputsInfo : {
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
