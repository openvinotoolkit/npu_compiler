//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ungroup-bounded-buffers-as-func-args --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: module @TensorWithBounds
module @TensorWithBounds {
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
// CHECK:       @TensorsWithBounds([[ARG0:%.+]]: memref<1x18x3x3xf32, {order = #NHWC}>, [[ARG1:%.+]]: memref<4xsi32>)
// CHECK-SAME:  -> (memref<1x18x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  func.func @TensorsWithBounds(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
      -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> {
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc(%c3) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
    %1 = VPUIP.Copy
      inputs(%arg0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
      outputs(%alloc : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
      -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
    return %1 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
  }
}

// CHECK:       [[INPUT_DYNAMIC_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG1]])
// CHECK-SAME:    -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
// CHECK:       [[OUTPUT_DATA:%.+]] = memref.alloc()
// CHECK-SAME:    : memref<1x18x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_CAST:%.+]] = memref.cast [[OUTPUT_DATA]]
// CHECK-SAME:    to memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
// CHECK:       [[COPY_OP:%.+]] = VPUIP.Copy inputs([[INPUT_DYNAMIC_BUFFER]]
// CHECK-SAME:                               outputs([[OUTPUT_CAST:%.+]]

// CHECK:       [[OUTPUT_DATA:%.+]], [[OUTPUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[COPY_OP]])
// CHECK-SAME:      : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
// CHECK:       return [[OUTPUT_DATA:%.+]], [[OUTPUT_SHAPE:%.+]]

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: module @TensorWithBoundsMultiple
module @TensorWithBoundsMultiple {
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
                              %arg0: memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>,
                              %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                              %arg2: memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>,
                              %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                              -> (
                              memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
                              memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                              memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
                              memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {
// CHECK:       @TensorsWithBoundsMultiple([[ARG0_0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>, [[ARG1:%.+]]: memref<1x20x3x3xf32, {order = #NHWC}>,
// CHECK-SAME:                     [[ARG2_0:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>, [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
// CHECK-SAME:                     [[ARG0_1:%.+]]: memref<4xsi32>, [[ARG2_1:%.+]]: memref<4xsi32>) ->
// CHECK-SAME:                     (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                      memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                      memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                      memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                      memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) outputs(%0 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>) -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
    %4 = VPUIP.Copy inputs(%arg2 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) outputs(%3 : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>) -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, {order = #NHWC}>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

    return %1, %2, %4, %5 :
      memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
      memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
      memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>,
      memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }
}

// CHECK:       [[INPUT_2_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2_0]], [[ARG2_1]]) : memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[INPUT_0_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0_0]], [[ARG0_1]]) : memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

// CHECK:       [[OUTPUT_0_DATA:%.+]] = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_0_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_0_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUTPUT_0_DATA]], [[OUTPUT_0_SHAPE]]) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_0_BOUNDED_BUFFER_COPY:%.+]] = VPUIP.Copy
// CHECK-SAME:    inputs([[INPUT_0_BOUNDED_BUFFER]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:    outputs([[OUTPUT_0_BOUNDED_BUFFER]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK:       [[OUTPUT_0_DATA_RESULT:%.+]], [[OUTPUT_0_SHAPE_RESULT:%.+]] = VPUIP.UngroupBoundedBuffer([[OUTPUT_0_BOUNDED_BUFFER_COPY]]) :
// CHECK-SAME:    memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

// CHECK:       [[INPUT_1_STATIC:%.+]] = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_1_STATIC:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x20x3x3xf32, {order = #NHWC}>) outputs([[INPUT_1_STATIC]] : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

// CHECK:       [[OUTPUT_2_DATA:%.+]] = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_2_SHAPE:%.+]] = memref.alloc() : memref<4xsi32, @CMX_NN>
// CHECK:       [[OUTPUT_2_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUTPUT_2_DATA]], [[OUTPUT_2_SHAPE]]) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_2_BOUNDED_BUFFER_COPY:%.+]] = VPUIP.Copy
// CHECK-SAME:    inputs([[INPUT_2_BOUNDED_BUFFER]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
// CHECK-SAME:    outputs([[OUTPUT_2_BOUNDED_BUFFER]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
// CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK:       [[OUTPUT_2_DATA_RESULT:%.+]], [[OUTPUT_2_SHAPE_RESULT:%.+]] = VPUIP.UngroupBoundedBuffer([[OUTPUT_2_BOUNDED_BUFFER_COPY]]) :
// CHECK-SAME:    memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN> -> memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

// CHECK:       [[INPUT_3_STATIC:%.+]] = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUTPUT_3_STATIC:%.+]] = VPUIP.Copy inputs([[ARG3]] : memref<1x40x3x3xf32, {order = #NHWC}>) outputs([[INPUT_3_STATIC]] : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

// CHECK:       return [[OUTPUT_0_DATA_RESULT]], [[OUTPUT_1_STATIC]], [[OUTPUT_2_DATA_RESULT]], [[OUTPUT_3_STATIC]], [[OUTPUT_0_SHAPE_RESULT]], [[OUTPUT_2_SHAPE_RESULT]]
// CHECK-SAME:    : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: module @DynamicScatterNDUpdateCheckInputsOrder
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
  func.func @main(%arg0: memref<1x3x3xsi32>, %arg1: memref<1x?x3x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 3]> : tensor<4xsi64>, order = #NCHW}>, %arg2: memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}>) -> memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]> {
    %alloc = memref.alloc() : memref<1x3x3xsi32, [@CMX_NN, 0]>
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x3x3xsi32>) outputs(%alloc : memref<1x3x3xsi32, [@CMX_NN, 0]>) -> memref<1x3x3xsi32, [@CMX_NN, 0]>
    %alloc_0 = memref.alloc() : memref<1x2x3x3xsi32, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<4xsi32, [@CMX_NN, 0]>
    %1 = VPUIP.GroupBoundedBuffer(%alloc_0, %alloc_1) : memref<1x2x3x3xsi32, [@CMX_NN, 0]>, memref<4xsi32, [@CMX_NN, 0]> -> memref<1x?x3x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 3]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x?x3x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 3]> : tensor<4xsi64>, order = #NCHW}>) outputs(%1 : memref<1x?x3x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 3]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>) -> memref<1x?x3x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3, 3]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x2x3xsi32, [@CMX_NN, 0]>
    %alloc_3 = memref.alloc() : memref<3xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2, %alloc_3) : memref<1x2x3xsi32, [@CMX_NN, 0]>, memref<3xsi32, [@CMX_NN, 0]> -> memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%arg2 : memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}>) outputs(%3 : memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>) -> memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
    return %4 : memref<1x?x3xsi32, {bounds = #const.OpaqueI64Elements<[1, 2, 3]> : tensor<3xsi64>, order = #CHW}, [@CMX_NN, 0]>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff1CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
!BoundedBuff2 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff2CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK-LABEL: module @OutlinedMainContentInOneFunc
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

  func.func nested @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, {order = #NHWC}>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func nested @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]]) : memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[BOUNDED_BUFF0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]]) : memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY0:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF0]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:      [[UNGROUP_DATA0:%.+]], [[UNGROUP_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // Original Consumer of %arg2
  // CHECK:      [[ORIG_COPY1:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF1]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:      [[UNGROUP_DATA1:%.+]], [[UNGROUP_DYN_SHAPE1:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK:      return [[UNGROUP_DATA0]], {{%.+}}, [[UNGROUP_DATA1]], {{%.+}}, [[UNGROUP_DYN_SHAPE0]], [[UNGROUP_DYN_SHAPE1]]

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                  -> (
                  !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {

    %res:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}>, !BoundedBuff2, memref<1x40x3x3xf32, {order = #NHWC}>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>)

    return %res#0, %res#1, %res#2, %res#3 :
        !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
        !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:     [[RES:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, {order = #NHWC}>, memref<1x20x3x3xf32, {order = #NHWC}>, memref<1x30x3x3xf32, {order = #NHWC}>, memref<1x40x3x3xf32, {order = #NHWC}>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_OUT0:%.+]] = VPUIP.GroupBoundedBuffer([[RES]]#0, [[RES]]#4)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
  // CHECK:      [[BOUNDED_BUFF_OUT1:%.+]] = VPUIP.GroupBoundedBuffer([[RES]]#2, [[RES]]#5)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:      [[UNGROUP_DATA_OUT1:%.+]], [[UNGROUP_DYN_SHAPE_OUT1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_OUT1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK:      [[UNGROUP_DATA_OUT0:%.+]], [[UNGROUP_DYN_SHAPE_OUT0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_OUT0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     return [[UNGROUP_DATA_OUT0]], [[RES]]#1, [[UNGROUP_DATA_OUT1]], [[RES]]#3, [[UNGROUP_DYN_SHAPE_OUT0]], [[UNGROUP_DYN_SHAPE_OUT1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff1CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
!BoundedBuff2 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff2CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK-LABEL: module @Outlined2SequentialFunc
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

  func.func nested @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, {order = #NHWC}>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func nested @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Identical to @foo in @OutlinedMainContentInOneFunc test, skipping checks

  func.func nested @foo1(%arg0: !BoundedBuff1CMX) -> !BoundedBuff1 {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32> -> !BoundedBuff1
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1CMX) outputs(%0 : !BoundedBuff1) -> !BoundedBuff1

    return %1 : !BoundedBuff1
  }

  // CHECK:        func.func nested @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG1]]) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA:%.+]], [[UNGROUP_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK:      return [[UNGROUP_DATA]], [[UNGROUP_DYN_SHAPE]]

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                  -> (
                  !BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}>, !BoundedBuff2, memref<1x40x3x3xf32, {order = #NHWC}>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>)

    %res_foo1 = call @foo1(%res_foo#0)
      : (!BoundedBuff1CMX) -> (!BoundedBuff1)

    return %res_foo1, %res_foo#1, %res_foo#2, %res_foo#3 : !BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:     [[RES_FOO:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, {order = #NHWC}>, memref<1x20x3x3xf32, {order = #NHWC}>, memref<1x30x3x3xf32, {order = #NHWC}>, memref<1x40x3x3xf32, {order = #NHWC}>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:     [[GROUP_FOO_0:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#0, [[RES_FOO]]#4)
  // CHECK-SAME:    memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:     [[GROUP_FOO_1:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#2, [[RES_FOO]]#5)
  // CHECK-SAME:    memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:      [[UNGROUP_DATA_RET1:%.+]], [[UNGROUP_DYN_SHAPE_RET1:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[DATA:%.+]], [[DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_0]])
  // CHECK-SAME:    : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO1:%.+]]:2 = call @foo1([[DATA]], [[DYN_SHAPE]]) : (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO1]]#0, [[RES_FOO1]]#1)
  // CHECK-SAME:    memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_RET0:%.+]], [[UNGROUP_DYN_SHAPE_RET0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:     return [[UNGROUP_DATA_RET0]], [[RES_FOO]]#1, [[UNGROUP_DATA_RET1]], [[RES_FOO]]#3, [[UNGROUP_DYN_SHAPE_RET0]], [[UNGROUP_DYN_SHAPE_RET1]]
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff1CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
!BoundedBuff2 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff2CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK-LABEL: module @CallAndNonCallOpConsumersOfBoundedBuff
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

  func.func nested @foo(%arg0: !BoundedBuff1,
                         %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                         %arg2: !BoundedBuff2,
                         %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                            -> (
                         !BoundedBuff1CMX,
                         memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                         !BoundedBuff2CMX,
                         memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, {order = #NHWC}>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

    return %1, %2, %4, %5 :
      !BoundedBuff1CMX,
      memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
      !BoundedBuff2CMX,
      memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func nested @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Identical to @foo in @OutlinedMainContentInOneFunc test, skipping checks

  func.func nested @foo1(%arg0: !BoundedBuff1CMX) -> !BoundedBuff1 {
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32> -> !BoundedBuff1
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1CMX) outputs(%0 : !BoundedBuff1) -> !BoundedBuff1

    return %1 : !BoundedBuff1
  }

  // CHECK:        func.func nested @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>)

  // Identical to @foo1 in @Outlined2SequentialFunc test, skipping checks

  func.func nested @foo2(%arg0: !BoundedBuff2CMX, %arg1: memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> (!BoundedBuff2, memref<1x20x3x3xf32, {order = #NHWC}>) {
    %alloc_0_0 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32> -> !BoundedBuff2
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff2CMX) outputs(%0 : !BoundedBuff2) -> !BoundedBuff2

    // 1st arg
    %alloc_1 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) outputs(%alloc_1 : memref<1x20x3x3xf32, {order = #NHWC}>) -> memref<1x20x3x3xf32, {order = #NHWC}>

    return %1, %2 : !BoundedBuff2, memref<1x20x3x3xf32, {order = #NHWC}>
  }

  // CHECK:        func.func nested @foo2(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:   -> (memref<1x30x3x3xf32, {order = #NHWC}>, memref<1x20x3x3xf32, {order = #NHWC}>, memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG2]]) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // Original Consumer of %arg0
  // CHECK:      [[ORIG_COPY:%.+]] = VPUIP.Copy inputs([[BOUNDED_BUFF]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA:%.+]], [[UNGROUP_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[ORIG_COPY]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // Copy other, unbounded, arg
  // CHECK:      [[OTHER_ARG:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:    -> memref<1x20x3x3xf32, {order = #NHWC}>

  // CHECK:      return [[UNGROUP_DATA]], [[OTHER_ARG]], [[UNGROUP_DYN_SHAPE]]

  func.func @main(%arg0: !BoundedBuff1,
                  %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                  %arg2: !BoundedBuff2,
                  %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                  -> (
                  !BoundedBuff1,
                  memref<1x20x3x3xf32, {order = #NHWC}>,
                  !BoundedBuff2,
                  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
                  !BoundedBuff2) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}>, !BoundedBuff2, memref<1x40x3x3xf32, {order = #NHWC}>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>)

    %alloc_0_0 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}>
    %alloc_0_1 = memref.alloc() : memref<4xsi32>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32> -> !BoundedBuff2
    %1 = VPUIP.Copy inputs(%res_foo#2 : !BoundedBuff2CMX) outputs(%0 : !BoundedBuff2) -> !BoundedBuff2

    %res_foo1 = call @foo1(%res_foo#0)
      : (!BoundedBuff1CMX) -> (!BoundedBuff1)

    %res_foo2:2 = call @foo2(%res_foo#2, %res_foo#1)
      : (!BoundedBuff2CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> (!BoundedBuff2, memref<1x20x3x3xf32, {order = #NHWC}>)

    return %res_foo1, %res_foo2#1, %res_foo2#0, %res_foo#3, %1 :
        !BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}>,
        !BoundedBuff2,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
        !BoundedBuff2
  }

    // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32>,
  // CHECK-SAME:       memref<4xsi32>)

  // CHECK:      [[BOUNDED_BUFF_IN1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:      [[BOUNDED_BUFF_IN0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]])
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_IN0:%.+]], [[UNGROUP_DYN_SHAPE_IN0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK:      [[UNGROUP_DATA_IN1:%.+]], [[UNGROUP_DYN_SHAPE_IN1:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF_IN1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:     [[RES_FOO:%.+]]:6 = call @foo([[UNGROUP_DATA_IN0]], [[ARG1]], [[UNGROUP_DATA_IN1]], [[ARG3]], [[UNGROUP_DYN_SHAPE_IN0]], [[UNGROUP_DYN_SHAPE_IN1]])
  // CHECK-SAME:   : (memref<1x10x3x3xf32, {order = #NHWC}>, memref<1x20x3x3xf32, {order = #NHWC}>, memref<1x30x3x3xf32, {order = #NHWC}>, memref<1x40x3x3xf32, {order = #NHWC}>, memref<4xsi32>, memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:     [[GROUP_FOO_0:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#0, [[RES_FOO]]#4)
  // CHECK-SAME:    memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:     [[GROUP_FOO_1:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO]]#2, [[RES_FOO]]#5)
  // CHECK-SAME:    memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // Non Call Op with func @foo's 2nd Bounded Buff result as input
  // CHECK:     [[NON_CALL_OP:%.+]] = VPUIP.Copy
  // CHECK-SAME:   inputs([[GROUP_FOO_1]] : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>)
  // CHECK-SAME:   outputs({{%.+}} : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
  // CHECK-SAME:     -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_RET2:%.+]], [[UNGROUP_DYN_SHAPE_RET2:%.+]] = VPUIP.UngroupBoundedBuffer([[NON_CALL_OP]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // @foo1 CallOp
  // CHECK:     [[DATA:%.+]], [[DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_0]])
  // CHECK-SAME:    : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO1:%.+]]:2 = call @foo1([[DATA]], [[DYN_SHAPE]]) : (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO1]]#0, [[RES_FOO1]]#1)
  // CHECK-SAME:    memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_RET0:%.+]], [[UNGROUP_DYN_SHAPE_RET0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // @foo2 CallOp
  // CHECK:      [[UNGROUP_DATA_FOO2_DATA0:%.+]], [[UNGROUP_DATA_FOO2_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO_1]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:     [[RES_FOO2:%.+]]:3 = call @foo2([[UNGROUP_DATA_FOO2_DATA0]], [[RES_FOO]]#1, [[UNGROUP_DATA_FOO2_DYN_SHAPE0]])
  // CHECK-SAME:    : (memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>)
  // CHECK-SAME:    -> (memref<1x30x3x3xf32, {order = #NHWC}>, memref<1x20x3x3xf32, {order = #NHWC}>, memref<4xsi32>)

  // CHECK:     [[GROUP_FOO2_RES:%.+]] = VPUIP.GroupBoundedBuffer([[RES_FOO2]]#0, [[RES_FOO2]]#2)
  // CHECK-SAME:    memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA_RET1:%.+]], [[UNGROUP_DYN_SHAPE_RET1:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO2_RES]])
  // CHECK-SAME:    -> memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:     return [[UNGROUP_DATA_RET0]], [[RES_FOO2]]#1, [[UNGROUP_DATA_RET1]], [[RES_FOO]]#3, [[UNGROUP_DATA_RET2]], [[UNGROUP_DYN_SHAPE_RET0]], [[UNGROUP_DYN_SHAPE_RET1]], [[UNGROUP_DYN_SHAPE_RET2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!BoundedBuff1 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff1CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
!BoundedBuff2 = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!BoundedBuff2CMX = memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

// CHECK-LABEL: module @NestedCallOps
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

  func.func nested @foo(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                         %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                            -> (
                         !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                         !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {
    // 0th arg
    %res_foo1 = call @foo1(%arg0) : (!BoundedBuff1) -> (!BoundedBuff1CMX)

    // 1st arg
    %alloc_1_0 = memref.alloc() : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.Copy inputs(%arg1 : memref<1x20x3x3xf32, {order = #NHWC}>) outputs(%alloc_1_0 : memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>

    // 2nd arg
    %alloc_2_1 = memref.alloc() : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_2_2 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %3 = VPUIP.GroupBoundedBuffer(%alloc_2_1, %alloc_2_2) : memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff2CMX
    %4 = VPUIP.Copy inputs(%arg2 : !BoundedBuff2) outputs(%3 : !BoundedBuff2CMX) -> !BoundedBuff2CMX

    // 3rd arg
    %alloc_3_0 = memref.alloc() : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.Copy inputs(%arg3 : memref<1x40x3x3xf32, {order = #NHWC}>) outputs(%alloc_3_0 : memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) -> memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>

    return %res_foo1, %2, %4, %5 :
      !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func nested @foo(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // CHECK:      [[BOUNDED_BUFF1:%.+]] = VPUIP.GroupBoundedBuffer([[ARG2]], [[ARG5]]) : memref<1x30x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 30, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[BOUNDED_BUFF0:%.+]] = VPUIP.GroupBoundedBuffer([[ARG0]], [[ARG4]]) : memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

  // CHECK:      [[UNGROUP_DATA0:%.+]], [[UNGROUP_DYN_SHAPE0:%.+]] = VPUIP.UngroupBoundedBuffer([[BOUNDED_BUFF0]])
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>

  // CHECK:      [[FOO1_CALL:%.+]]:2 = call @foo1([[UNGROUP_DATA0]], [[UNGROUP_DYN_SHAPE0]])
  // CHECK-SAME:     : (memref<1x10x3x3xf32, {order = #NHWC}>, memref<4xsi32>) -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>)

  // CHECK:      [[GROUP_FOO1_RES:%.+]] = VPUIP.GroupBoundedBuffer([[FOO1_CALL]]#0, [[FOO1_CALL]]#1)
  // CHECK-SAME:    : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>
  // CHECK-SAME:    -> memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>

  // CHECK:      [[UNGROUP_FOO1_RES_DATA:%.+]], [[UNGROUP_FOO1_RES_DYN_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[GROUP_FOO1_RES]])
  // CHECK-SAME:    : memref<1x?x3x3xf32, {bounds = #const.OpaqueI64Elements<[1, 10, 3, 3]> : tensor<4xsi64>, order = #NHWC}, @CMX_NN>
  // CHECK-SAME:    -> memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN>

  // CHECK:      return [[UNGROUP_FOO1_RES_DATA]], {{%.+}}, {{%.+}}, {{%.+}}, [[UNGROUP_FOO1_RES_DYN_SHAPE]], {{%.+}}

  func.func nested @foo1(%arg0: !BoundedBuff1) -> !BoundedBuff1CMX {
    // 0th arg
    %alloc_0_0 = memref.alloc() : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>
    %alloc_0_1 = memref.alloc() : memref<4xsi32, @CMX_NN>
    %0 = VPUIP.GroupBoundedBuffer(%alloc_0_0, %alloc_0_1) : memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>, memref<4xsi32, @CMX_NN> -> !BoundedBuff1CMX
    %1 = VPUIP.Copy inputs(%arg0 : !BoundedBuff1) outputs(%0 : !BoundedBuff1CMX) -> !BoundedBuff1CMX

    return %1 : !BoundedBuff1CMX
  }

  // CHECK:        func.func nested @foo1(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // Almost identical to @foo1 in @Outlined2SequentialFunc test, skipping checks

  func.func @main(%arg0: !BoundedBuff1, %arg1: memref<1x20x3x3xf32, {order = #NHWC}>,
                  %arg2: !BoundedBuff2, %arg3: memref<1x40x3x3xf32, {order = #NHWC}>)
                  -> (
                  !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
                  !BoundedBuff2CMX, memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>) {

    %res_foo:4 = call @foo(%arg0, %arg1, %arg2, %arg3)
      : (!BoundedBuff1, memref<1x20x3x3xf32, {order = #NHWC}>, !BoundedBuff2, memref<1x40x3x3xf32, {order = #NHWC}>)
      -> (!BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>)

    return %res_foo#0, %res_foo#1, %res_foo#2, %res_foo#3 : !BoundedBuff1CMX, memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>, !BoundedBuff2CMX,  memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>
  }

  // CHECK:        func.func @main(
  // CHECK-SAME:     [[ARG0:%.+]]: memref<1x10x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG1:%.+]]:  memref<1x20x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG2:%.+]]: memref<1x30x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG3:%.+]]: memref<1x40x3x3xf32, {order = #NHWC}>,
  // CHECK-SAME:     [[ARG4:%.+]]: memref<4xsi32>,
  // CHECK-SAME:     [[ARG5:%.+]]: memref<4xsi32>)
  // CHECK-SAME:   -> (memref<1x10x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x20x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x30x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<1x40x3x3xf32, {order = #NHWC}, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>,
  // CHECK-SAME:       memref<4xsi32, @CMX_NN>)

  // main func almost identical to the one in @OutlinedMainContentInOneFunc, skip checks
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: module @NonTrivialOutputUsers
module @NonTrivialOutputUsers {
    net.NetworkInfo entryPoint : @MultipleSimpleUsers inputsInfo : {
        // CHECK:     net.NetworkInfo entryPoint : @MultipleSimpleUsers inputsInfo
        DataInfo "in0" : tensor<1x18x3x3xf32>
        // CHECK:     DataInfo "in0" : tensor<1x18x3x3xf32>
        // CHECK:     DataInfo "vpux_ie_shape_in0" : tensor<4xsi32>
    } outputsInfo : {
        DataInfo "out0" : tensor<1x18x3x3xf32>
        DataInfo "out1" : tensor<1x18x3x3xf32>
        // CHECK:     DataInfo "out0" : tensor<1x18x3x3xf32>
        // CHECK:     DataInfo "out1" : tensor<1x18x3x3xf32>
        // CHECK:     DataInfo "vpux_ie_shape_out0" : tensor<4xsi32>
    }

    // CHECK:   @MultipleSimpleUsers
    // CHECK-SAME: ([[IN:%.+]]: memref<1x18x3x3xf32>, [[SHAPE:%.+]]: memref<4xsi32>)
    // CHECK-SAME: -> (memref<1x18x3x3xf32>, memref<1x18x3x3xf32>, memref<4xsi32>)
    func.func @MultipleSimpleUsers(%arg0: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            -> (memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>, memref<1x18x3x3xf32>) {
        %c3 = arith.constant 3 : index
        %alloc0 = memref.alloc(%c3) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>
        %internal0 = VPUIP.Copy
            inputs(%arg0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%alloc0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>
        // CHECK: [[GROUP:%.+]] = VPUIP.GroupBoundedBuffer([[IN]], [[SHAPE]])
        // CHECK: [[INTERNAL0:%.+]] = VPUIP.Copy inputs([[GROUP]]

        %alloc1 = memref.alloc(%c3) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
        %first_out = VPUIP.Copy
            inputs(%internal0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>)
            outputs(%alloc1 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[FIRST_OUT:%.+]] = VPUIP.Copy inputs([[INTERNAL0]]
        // CHECK: [[FIRST_OUT_DATA:%.+]], [[FIRST_OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[FIRST_OUT]])


        // This copy is the extra user of the return operand. It should continue
        // to use the original value of the VPUIP.Copy above, not the ungrouped
        // one since otherwise the IR validity is questionable (e.g. with
        // ungroup are we going to copy a data memref into a bounded buffer?)
        %internal1 = VPUIP.Copy
            inputs(%first_out : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%alloc0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>
        %normal_cast = Core.ReinterpretCast(%internal1)
            : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}, @CMX_NN>
            -> memref<1x18x3x3xf32>
        // CHECK: [[INTERNAL1:%.+]] = VPUIP.Copy inputs([[FIRST_OUT]]
        // CHECK: [[NORMAL_CAST:%.+]] = Core.ReinterpretCast([[INTERNAL1]]) {{.*}} -> memref<1x18x3x3xf32>


        return %first_out, %normal_cast : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>, memref<1x18x3x3xf32>
        // CHECK: return [[FIRST_OUT_DATA]], [[NORMAL_CAST]], [[FIRST_OUT_SHAPE]]
    }

    // CHECK: @MultipleUsersWithCasts
    // CHECK-SAME: ([[IN:%.+]]: memref<1x18x3x3xf32>)
    // CHECK-SAME: -> (memref<1x18x3x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x18x3x3xf32, @CMX_NN>, memref<4xsi32>)
    func.func nested @MultipleUsersWithCasts(%arg: memref<1x18x3x3xf32>)
            -> (memref<1x18x3x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x18x3x3xf32, @CMX_NN>) {
        %bounded = Core.ReinterpretCast(%arg) : memref<1x18x3x3xf32>
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[BOUNDED:%.+]] = Core.ReinterpretCast([[IN]])
        // CHECK-SAME: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>

        %c3 = arith.constant 3 : index
        %alloc0 = memref.alloc(%c3) : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
        %first_out = VPUIP.Copy
            inputs(%bounded: memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            outputs(%alloc0 : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>)
            -> memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
        // CHECK: [[FIRST_OUT:%.+]] = VPUIP.Copy inputs([[BOUNDED]]
        // CHECK: [[FIRST_OUT_DATA:%.+]], [[FIRST_OUT_SHAPE:%.+]] = VPUIP.UngroupBoundedBuffer([[FIRST_OUT]])

        %first_out_strided = Core.ReinterpretCast(%first_out)
            : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
            -> memref<1x18x3x?xf32, strided<[?, ?, ?, ?], offset: ?>>
        // CHECK: [[FIRST_OUT_STRIDED:%.+]] = Core.ReinterpretCast([[FIRST_OUT_DATA]])
        // CHECK-SAME:  -> memref<1x18x3x?xf32, strided<[?, ?, ?, ?], offset: ?>>

        %internal_cast = Core.ReinterpretCast(%first_out)
            : memref<1x18x3x?xf32, {bounds = #const.OpaqueI64Elements<[1, 18, 3, 3]> : tensor<4xsi64>, order = #NCHW}>
            -> memref<1x18x3x3xf32>
        // Note: after canonicalization, internal cast is removed as a no-op
        %alloc1 = memref.alloc() : memref<1x18x3x3xf32, @CMX_NN>
        %second_out = VPUIP.Copy
            inputs(%internal_cast : memref<1x18x3x3xf32>)
            outputs(%alloc1 : memref<1x18x3x3xf32, @CMX_NN>)
            -> memref<1x18x3x3xf32, @CMX_NN>
        // CHECK: [[SECOND_OUT:%.+]] = VPUIP.Copy inputs([[FIRST_OUT_DATA]]

        return %first_out_strided, %second_out
            : memref<1x18x3x?xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x18x3x3xf32, @CMX_NN>
        // CHECK: return [[FIRST_OUT_STRIDED]], [[SECOND_OUT]], [[FIRST_OUT_SHAPE]]
    }
}
