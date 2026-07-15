//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// To build an output_shape func these passes should be called in a certain order:
// - extract-return-shapes: extracts shapes info from return op operands of the main func and creates tensor.dim ops for each shape dimension
// - resolve-shaped-type-result-dims: resolves tensor.dim of result of operations that implement the `InferShapedTypeOpInterface` or
//  `ReifyRankedShapedTypeOpInterface` in terms of shapes of its operands
// - outline-dim-operations: outlines these tensor and arith operations to the new output_shape func

// RUN: vpux-opt --split-input-file --platform=%platform% --output-shape-predict %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @outputShapeChanged
module @outputShapeChanged {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 64]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK-LABEL: @output_shape
// CHECK:   [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:   [[CST_16_i64:%.+]] = arith.constant 16 : i64
// CHECK:   [[CST_1_i64:%.+]] = arith.constant 1 : i64

// CHECK:   [[DIM:%.+]] = tensor.dim [[ARG]], [[CST_3]] : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[DIV_0:%.+]] = arith.divsi [[DIM]], [[CST_2]] : index
// CHECK:   [[DIV_1:%.+]] = arith.divsi [[DIV_0]], [[CST_2]] : index
// CHECK:   [[IND_CAST:%.+]] = arith.index_cast [[DIV_1]] : index to i64

// CHECK:   [[FROM_ELEM:%.+]] = tensor.from_elements [[CST_1_i64]], [[CST_16_i64]], [[CST_16_i64]], [[IND_CAST]] : tensor<4xi64>
// CHECK:   return [[FROM_ELEM]] : tensor<4xi64>

// CHECK-LABEL: @main
func.func @main(%arg: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>) {
    %cst_1 = const.Declare tensor<32x16x3x3xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    %cst_2 = const.Declare tensor<16x32x1x1xf16, {order = #NCHW}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>

    // output shape for IE.MaxPool operation is calculated as:
    // H_out = floor((H + pads_begin[0] + pads_end[0] - kernel[0]) / strides[0] + 1)
    // W_out = floor((W + pads_begin[1] + pads_end[1] - kernel[1]) / strides[1] + 1)
    // so in this example:
    // H_out = floor((64 + 0 + 0 - 2) / 2 + 1) = 32
    // W_out = floor((x + 0 + 0 - 2) / 2 + 1) = x / 2, where `x` is unknown dynamic dimension
    %maxpool_1 = IE.MaxPool(%arg) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>

    // output shape for IE.Convolution operation is calculated as:
    // C_out = OC, where [OC, IC, KH, KW] - shape of the weights
    // H_out = floor((H - (1 + (KH - 1) * (dilations[0] + 1)) + pads_begin[0] + pads_end[0]) / strides[0] + 1)
    // W_out = floor((W - (1 + (KW - 1) * (dilations[1] + 1)) + pads_begin[1] + pads_end[1]) / strides[1] + 1)
    // so in this example:
    // H_out = floor((32 - (1 + (3 - 1) * 1) + 1 + 1) / 1 + 1) = floor((32 - 3 + 2) / 1) + 1 = 32
    // W_out = floor((x - (1 + (3 - 1) * 1) + 1 + 1) / 1 + 1) = floor(x) = x, where `x` is unknown dynamic dimension
    %conv_1 = IE.Convolution(%maxpool_1, %cst_1) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
                tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<32x16x3x3xf16, {order = #NCHW}> -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>

    // output shape is the same as input
    %relu = IE.ReLU(%conv_1) : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
                           -> tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>

    // H_out = floor((32 + 0 + 0 - 2) / 2 + 1) = 16
    // W_out = floor((x + 0 + 0 - 2) / 2 + 1) = x / 2, where `x` is unknown dynamic dimension
    %maxpool_2 = IE.MaxPool(%relu) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x32x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x32x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

    // H_out = floor((16 - (1 + (1 - 1) * 1) + 0 + 0) / 1 + 1) = floor((16 - 1) / 1) + 1 = 16
    // W_out = floor((x - (1 + (1 - 1) * 1) + 0 + 0) / 1 + 1) = floor(x) = x, where `x` is unknown dynamic dimension
    %conv_2 = IE.Convolution(%maxpool_2, %cst_2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
                tensor<1x32x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 16, 16]> : tensor<4xsi64>, order = #NCHW}>, tensor<16x32x1x1xf16, {order = #NCHW}>
                -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

    // so if input W dim = dyn_w, then output dyn_w' = (dyn_w / 2) / 2
    return %conv_2 : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @sameOutputShape
module @sameOutputShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK-LABEL: @output_shape
// CHECK:   [[ARG:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:   [[CST_64_i64:%.+]] = arith.constant 64 : i64
// CHECK:   [[CST_16_i64:%.+]] = arith.constant 16 : i64
// CHECK:   [[CST_1_i64:%.+]] = arith.constant 1 : i64

// CHECK:   [[DIM:%.+]] = tensor.dim [[ARG]], [[CST_3]] : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[IND_CAST:%.+]] = arith.index_cast [[DIM]] : index to i64

// CHECK:   [[FROM_ELEM:%.+]] = tensor.from_elements [[CST_1_i64]], [[CST_16_i64]], [[CST_64_i64]], [[IND_CAST]] : tensor<4xi64>
// CHECK:   return [[FROM_ELEM]] : tensor<4xi64>

// CHECK-LABEL: @main
func.func @main(%arg: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>) {

    %relu = IE.ReLU(%arg) : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
                           -> tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>

    return %relu : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @sameOutputShapeAllDimsDynamic
module @sameOutputShapeAllDimsDynamic {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK-LABEL: @output_shape
// CHECK:   [[ARG:%.+]]: tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>

// CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:   [[CST_1:%.+]] = arith.constant 1 : index
// CHECK:   [[CST_0:%.+]] = arith.constant 0 : index

// CHECK:   [[DIM_0:%.+]] = tensor.dim [[ARG]], [[CST_0]] : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[IND_CAST_0:%.+]] = arith.index_cast [[DIM_0]] : index to i64
// CHECK:   [[DIM_1:%.+]] = tensor.dim [[ARG]], [[CST_1]] : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[IND_CAST_1:%.+]] = arith.index_cast [[DIM_1]] : index to i64
// CHECK:   [[DIM_2:%.+]] = tensor.dim [[ARG]], [[CST_2]] : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[IND_CAST_2:%.+]] = arith.index_cast [[DIM_2]] : index to i64
// CHECK:   [[DIM_3:%.+]] = tensor.dim [[ARG]], [[CST_3]] : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[IND_CAST_3:%.+]] = arith.index_cast [[DIM_3]] : index to i64

// CHECK:   [[FROM_ELEM:%.+]] = tensor.from_elements [[IND_CAST_0]], [[IND_CAST_1]], [[IND_CAST_2]], [[IND_CAST_3]] : tensor<4xi64>
// CHECK:   return [[FROM_ELEM]] : tensor<4xi64>

// CHECK-LABEL: @main
func.func @main(%arg: tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>) {

    %relu = IE.ReLU(%arg) : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
                           -> tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>

    return %relu : tensor<?x?x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @staticShape
module @staticShape {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x16x64x64xf16, {order = #NCHW}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x16x64x64xf16, {order = #NCHW}>
  }

// CHECK-LABEL: @output_shape
// CHECK:   [[ARG:%.+]]: tensor<1x16x64x64xf16, {order = #NCHW}>

// CHECK:   [[CST:%.+]] = arith.constant dense<[1, 16, 64, 64]> : tensor<4xi64>
// CHECK:   return [[CST]] : tensor<4xi64>

// CHECK-LABEL: @main
func.func @main(%arg: tensor<1x16x64x64xf16, {order = #NCHW}>)
    -> (tensor<1x16x64x64xf16, {order = #NCHW}>) {

    %relu = IE.ReLU(%arg) : tensor<1x16x64x64xf16, {order = #NCHW}>
                           -> tensor<1x16x64x64xf16, {order = #NCHW}>

    return %relu : tensor<1x16x64x64xf16, {order = #NCHW}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @staticShapeMultipleInputs
module @staticShapeMultipleInputs {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x64x64xf16, {order = #NCHW}>
    DataInfo "input_1" : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16, {order = #NCHW}>
    DataInfo "output_1" : tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
  }

// CHECK-LABEL: @output_shape
// CHECK:   [[ARG_0:%.+]]: tensor<1x16x64x64xf16, {order = #NCHW}>
// CHECK:   [[ARG_1:%.+]]: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>) -> (tensor<4xi64>, tensor<4xi64>

// CHECK:   [[CST_2:%.+]] = arith.constant 2 : index
// CHECK:   [[CST_3:%.+]] = arith.constant 3 : index
// CHECK:   [[CST_32_i64:%.+]] = arith.constant 32 : i64
// CHECK:   [[CST_16_i64:%.+]] = arith.constant 16 : i64
// CHECK:   [[CST_1_i64:%.+]] = arith.constant 1 : i64
// CHECK:   [[CST:%.+]] = arith.constant dense<[1, 16, 64, 64]> : tensor<4xi64>

// CHECK:   [[DIM:%.+]] = tensor.dim [[ARG_1]], [[CST_3]] : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
// CHECK:   [[DIV:%.+]] = arith.divsi [[DIM]], [[CST_2]] : index
// CHECK:   [[IND_CAST:%.+]] = arith.index_cast [[DIV]] : index to i64

// CHECK:   [[FROM_ELEM:%.+]] = tensor.from_elements [[CST_1_i64]], [[CST_16_i64]], [[CST_32_i64]], [[IND_CAST]] : tensor<4xi64>
// CHECK:   return [[CST]], [[FROM_ELEM]] : tensor<4xi64>, tensor<4xi64>

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<1x16x64x64xf16, {order = #NCHW}>,
                %arg1: tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x64x64xf16, {order = #NCHW}>, tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>) {

    %relu = IE.ReLU(%arg0) : tensor<1x16x64x64xf16, {order = #NCHW}>
                           -> tensor<1x16x64x64xf16, {order = #NCHW}>

    %maxpool = IE.MaxPool(%arg1) {
            kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x16x64x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 64, 64]> : tensor<4xsi64>, order = #NCHW}>
      -> tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>

    return %relu, %maxpool : tensor<1x16x64x64xf16, {order = #NCHW}>,
                             tensor<1x16x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 32, 32]> : tensor<4xsi64>, order = #NCHW}>
}

}
