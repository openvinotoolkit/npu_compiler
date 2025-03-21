//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-reify-result-shapes-residuals %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LegalizeTensorDim
// CHECK-SAME:  [[IN:%.+]]: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
func.func @LegalizeTensorDim(
    %arg0: tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
) -> tensor<1xsi64> {
    %0 = IE.ReLU(%arg0) :
        tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
        -> tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>

    %c3 = arith.constant 3 : index
    %dim = tensor.dim %0, %c3 : tensor<1x3x16x?xf32, {bounds = [1, 3, 16, 32], order = #NCHW}>
    %1 = arith.index_cast %dim : index to i64
    %from_elements = tensor.from_elements %1 : tensor<1xi64>
    %2 = tensor.bitcast %from_elements : tensor<1xi64> to tensor<1xsi64>

    return %2 : tensor<1xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[IN]])

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[RELU]]) {dstElemType = si64}
    // CHECK-SAME:      -> tensor<4xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]
    // CHECK-SAME:      to tensor<1xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|tensor.dim|tensor.bitcast)}}
    // CHECK:       return [[SLICE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LegalizeTensorDimAndDivSI
// CHECK-SAME:  [[IN:%.+]]: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
func.func @LegalizeTensorDimAndDivSI(
    %arg0: tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}>
) -> tensor<1xsi64> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = IE.MaxPool(%arg0) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x16x32x?xf16, {bounds = [1, 16, 32, 64], order = #NCHW}> -> tensor<1x16x16x?xf16, {bounds = [1, 16, 16, 64], order = #NCHW}>
    %dim = tensor.dim %0, %c3 : tensor<1x16x16x?xf16, {bounds = [1, 16, 16, 64], order = #NCHW}>
    %1 = arith.divsi %dim, %c2 : index
    %2 = arith.index_cast %1 : index to i64
    %from_elements = tensor.from_elements %2 : tensor<1xi64>
    %3 = tensor.bitcast %from_elements : tensor<1xi64> to tensor<1xsi64>

    return %3 : tensor<1xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[CONST:%.+]] = const.Declare
    // CHECK-SAME:      tensor<1xsi64> = dense<2> : tensor<1xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[IN]])

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[SHAPE_OF:%.+]] = IE.ShapeOf([[MAXPOOL]]) {dstElemType = si64}
    // CHECK-SAME:      -> tensor<4xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[SHAPE_OF]] [3] [1]
    // CHECK-SAME:      to tensor<1xsi64>

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       [[DIVIDE:%.+]] = IE.Divide([[SLICE]], [[CONST]])

    // CHECK-NOT:   {{(arith.constant|arith.index_cast|arith.divsi|tensor.dim|tensor.bitcast)}}
    // CHECK:       return [[DIVIDE]]
}

// -----

// CHECK-LABEL: @LegalizeDivSI
// CHECK-SAME:  [[IN:%.+]]: index
func.func @LegalizeDivSI(
    %arg0: index
) -> index {
    %c2 = arith.constant 2 : index
    %1 = arith.divsi %arg0, %c2 : index
    return %1 : index

    // CHECK:   [[CONST0:%.+]] = arith.constant 0 : index
    // CHECK:   [[CONST2:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>

    // CHECK:   [[IND_CAST0:%.+]] = arith.index_cast [[IN]] : index to i64
    // CHECK:   [[FROM_ELEMS:%.+]] = tensor.from_elements [[IND_CAST0]] : tensor<1xi64>
    // CHECK:   [[BIT_CAST0:%.+]] = tensor.bitcast [[FROM_ELEMS]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK:   [[DIVIDE:%.+]] = IE.Divide([[BIT_CAST0]], [[CONST2]])
    // CHECK:   [[BIT_CAST1:%.+]] = tensor.bitcast [[DIVIDE]] : tensor<1xsi64> to tensor<1xi64>
    // CHECK:   [[EXTRACT:%.+]] = tensor.extract [[BIT_CAST1]][[[CONST0]]] : tensor<1xi64
    // CHECK:   [[IND_CAST1:%.+]] = arith.index_cast [[EXTRACT]] : i64 to index

    // CHECK:   return [[IND_CAST1]] : index
}

// -----

// CHECK-LABEL: @LegalizeAddI
// CHECK-SAME:  [[IN:%.+]]: index
func.func @LegalizeAddI(
    %arg0: index
) -> index {
    %c2 = arith.constant 2 : index
    %1 = arith.addi %arg0, %c2 : index
    return %1 : index

    // CHECK:   [[CONST0:%.+]] = arith.constant 0 : index
    // CHECK:   [[CONST2:%.+]] = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>

    // CHECK:   [[IND_CAST0:%.+]] = arith.index_cast [[IN]] : index to i64
    // CHECK:   [[FROM_ELEMS:%.+]] = tensor.from_elements [[IND_CAST0]] : tensor<1xi64>
    // CHECK:   [[BIT_CAST0:%.+]] = tensor.bitcast [[FROM_ELEMS]] : tensor<1xi64> to tensor<1xsi64>
    // CHECK:   [[ADD:%.+]] = IE.Add([[BIT_CAST0]], [[CONST2]])
    // CHECK:   [[BIT_CAST1:%.+]] = tensor.bitcast [[ADD]] : tensor<1xsi64> to tensor<1xi64>
    // CHECK:   [[EXTRACT:%.+]] = tensor.extract [[BIT_CAST1]][[[CONST0]]] : tensor<1xi64
    // CHECK:   [[IND_CAST1:%.+]] = arith.index_cast [[EXTRACT]] : i64 to index

    // CHECK:   return [[IND_CAST1]] : index
}

// -----

// CHECK-LABEL: @LegalizeAddIWithTwoIndexes
// CHECK-SAME:  [[IN0:%.+]]: index, [[IN1:%.+]]: index
func.func @LegalizeAddIWithTwoIndexes(
    %arg0: index,
    %arg1: index
) -> index {
    %1 = arith.addi %arg0, %arg1 : index
    // CHECK:   [[CONST0:%.+]] = arith.constant 0 : index
    // CHECK:   [[IND_CAST0:%.+]] = arith.index_cast [[IN0]] : index to i64
    // CHECK:   [[FROM_ELEMS0:%.+]] = tensor.from_elements [[IND_CAST0]] : tensor<1xi64>
    // CHECK:   [[BIT_CAST0:%.+]] = tensor.bitcast [[FROM_ELEMS0]] : tensor<1xi64> to tensor<1xsi64>

    // CHECK:   [[IND_CAST1:%.+]] = arith.index_cast [[IN1]] : index to i64
    // CHECK:   [[FROM_ELEMS1:%.+]] = tensor.from_elements [[IND_CAST1]] : tensor<1xi64>
    // CHECK:   [[BIT_CAST1:%.+]] = tensor.bitcast [[FROM_ELEMS1]] : tensor<1xi64> to tensor<1xsi64>

    return %1 : index
    // CHECK:   [[ADD:%.+]] = IE.Add([[BIT_CAST0]], [[BIT_CAST1]])
    // CHECK:   [[BIT_CAST_OUT:%.+]] = tensor.bitcast [[ADD]] : tensor<1xsi64> to tensor<1xi64>
    // CHECK:   [[EXTRACT:%.+]] = tensor.extract [[BIT_CAST_OUT]][[[CONST0]]] : tensor<1xi64
    // CHECK:   [[IND_CAST_OUT:%.+]] = arith.index_cast [[EXTRACT]] : i64 to index

    // CHECK:   return [[IND_CAST_OUT]] : index
}
