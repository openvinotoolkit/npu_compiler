//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-strided-slice %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ResolveStridedSliceWithStride
func.func @ResolveStridedSliceWithStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x5x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, 2, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x5x5xf16>

    return %0 : tensor<1x5x5x5xf16>
    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)

    // Only attributes with name *_attr could have values != 0
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0]
    // CHECK-SAME:  begins_attr = [0, 0, 0, 15]
    // CHECK-SAME:  ellipsis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  end_mask = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 10, 20]
    // CHECK-SAME:  new_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  shrink_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  strides_attr = [1, 1, 2, 1]
}

// -----

// CHECK-LABEL: @ResolveStridedSliceWoutStride
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x10x20x30xf16>
func.func @ResolveStridedSliceWoutStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x10x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, 1, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x10x5xf16>

    return %0 : tensor<1x5x10x5xf16>

    // CHECK:       [[VAL0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 15] [1, 5, 10, 5]
    // CHECK-NOT:   IE.StridedSlice
    // CHECK:       return [[VAL0]]  : tensor<1x5x10x5xf16>
}

// -----

// CHECK-LABEL: @ResolveStridedSliceNegStride
func.func @ResolveStridedSliceNegStride(%arg0: tensor<1x10x20x30xf16>) -> tensor<1x5x5x5xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 15],
        ends_attr = [1, 5, 10, 20],
        strides_attr = [1, 1, -2, 1],
        begin_mask = [0, 1, 1, 0],
        end_mask = [1, 0, 0, 0],
        new_axis_mask = [0, 0, 0, 0],
        shrink_axis_mask = [0, 0, 0, 0],
        ellipsis_mask = [0, 0, 0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x10x20x30xf16> -> tensor<1x5x5x5xf16>

    return %0 : tensor<1x5x5x5xf16>

    // CHECK:       %[[VAL0:.*]] = IE.StridedSlice(%arg0)

    // Only attributes with name *_attr could have values != 0
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0]
    // CHECK-SAME:  begins_attr = [0, 0, 11, 15]
    // CHECK-SAME:  ellipsis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  end_mask = [0, 0, 0, 0]
    // CHECK-SAME:  ends_attr = [1, 5, 20, 20]
    // CHECK-SAME:  new_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  shrink_axis_mask = [0, 0, 0, 0]
    // CHECK-SAME:  strides_attr = [1, 1, 2, 1]
}

// -----

// CHECK-LABEL: @ResolveStridedSliceWoutStrideMergeAdjacentFirstTwo1
func.func @ResolveStridedSliceWoutStrideMergeAdjacentFirstTwo1(%arg0: tensor<1x1x9x16x10xf16>) -> tensor<1x1x9x16x1xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 0, 0],
        ends_attr = [1, 1, 9, 16, 1],
        strides_attr = [1, 1, 1, 1, 1],
        begin_mask = [1, 1, 1, 1, 0],
        end_mask = [1, 1, 1, 1, 0],
        new_axis_mask = [],
        shrink_axis_mask = [],
        ellipsis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x1x9x16x10xf16> -> tensor<1x1x9x16x1xf16>

    return %0 : tensor<1x1x9x16x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 9, 16, 10]} : tensor<1x1x9x16x10xf16> -> tensor<1x9x16x10xf16>
    // CHECK:       %[[VAL1:.*]] = IE.Slice %[[VAL0]] [0, 0, 0, 0] [1, 9, 16, 1] : tensor<1x9x16x10xf16> to tensor<1x9x16x1xf16>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[VAL1]]) {shape_value = [1, 1, 9, 16, 1]} : tensor<1x9x16x1xf16> -> tensor<1x1x9x16x1xf16>
}

// -----

// CHECK-LABEL: @ResolveStridedSliceWoutStrideMergeAdjacentLastTwo1
func.func @ResolveStridedSliceWoutStrideMergeAdjacentLastTwo1(%arg0: tensor<1x9x16x1x1xf16>) -> tensor<1x9x8x1x1xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 0, 0],
        ends_attr = [1, 9, 8, 1, 1],
        strides_attr = [1, 1, 1, 1, 1],
        begin_mask = [1, 1, 0, 1, 1],
        end_mask = [1, 1, 0, 1, 1],
        new_axis_mask = [],
        shrink_axis_mask = [],
        ellipsis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x9x16x1x1xf16> -> tensor<1x9x8x1x1xf16>

    return %0 : tensor<1x9x8x1x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 9, 16, 1]} : tensor<1x9x16x1x1xf16> -> tensor<1x9x16x1xf16>
    // CHECK:       %[[VAL1:.*]] = IE.Slice %[[VAL0]] [0, 0, 0, 0] [1, 9, 8, 1] : tensor<1x9x16x1xf16> to tensor<1x9x8x1xf16>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[VAL1]]) {shape_value = [1, 9, 8, 1, 1]} : tensor<1x9x8x1xf16> -> tensor<1x9x8x1x1xf16>
}

// -----

// CHECK-LABEL: @ResolveStridedSliceWoutStrideNotMergeAdjacentDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x16x16xf16>
func.func @ResolveStridedSliceWoutStrideNotMergeAdjacentDim(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0, 0, 0, 0],
        ends_attr = [1, 1, 16, 16],
        strides_attr = [1, 1, 1, 1],
        begin_mask = [1, 0, 1, 1],
        end_mask = [1, 0, 1, 1],
        new_axis_mask = [],
        shrink_axis_mask = [],
        ellipsis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x8x16x16xf16> -> tensor<1x1x16x16xf16>

    return %0 : tensor<1x1x16x16xf16>
    // CHECK:       [[VAL0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 16, 16] : tensor<1x8x16x16xf16> to tensor<1x1x16x16xf16>
    // CHECK-NOT:   IE.StridedSlice
    // CHECK:       return [[VAL0]]  : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: @AdjustStridedSliceAttrsRank
func.func @AdjustStridedSliceAttrsRank(%arg0: tensor<157x1x2048xf16>) -> tensor<79x1x2048xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begins_attr = [0],
        ends_attr = [200],
        strides_attr = [2],
        begin_mask = [0],
        end_mask = [0],
        new_axis_mask = [],
        shrink_axis_mask = [],
        ellipsis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>
        } : tensor<157x1x2048xf16> -> tensor<79x1x2048xf16>

    return %0 : tensor<79x1x2048xf16>
    // CHECK:       IE.StridedSlice(%arg0)
    // CHECK-SAME:  begin_mask = [0, 0, 0]
    // CHECK-SAME:  begins_attr = [0, 0, 0]
    // CHECK-SAME:  ellipsis_mask = [0, 0, 0]
    // CHECK-SAME:  end_mask = [0, 0, 0]
    // CHECK-SAME:  ends_attr = [157, 1, 2048]
    // CHECK-SAME:  new_axis_mask = [0, 0, 0]
    // CHECK-SAME:  shrink_axis_mask = [0, 0, 0]
    // CHECK-SAME:  strides_attr = [2, 1, 1]
}

// -----

// CHECK-LABEL: @StridedSliceWith0dOutput
// CHECK-SAME:      [[INPUT:%.+]]: tensor<4xsi32>
func.func @StridedSliceWith0dOutput(%arg0: tensor<4xsi32>) -> tensor<1xsi32> {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0], begins_attr = [3], ellipsis_mask = [0], end_mask = [0], ends_attr = [4], new_axis_mask = [0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [1], strides_attr = [1]} : tensor<4xsi32> -> tensor<1xsi32>
    return %0 : tensor<1xsi32>

    // CHECK:       [[VAL:%.+]] = IE.Slice [[INPUT]] [3] [1] : tensor<4xsi32> to tensor<1xsi32>
    // CHECK:       return [[VAL]]  : tensor<1xsi32>
}

// -----

// CHECK-LABEL: @StridedSliceWithEllipsisAttr
func.func @StridedSliceWithEllipsisAttr(%arg0: tensor<1x1x10x2xf16>) -> tensor<1x1x10x2xf16> {
    %0 = IE.StridedSlice(%arg0) {
        begin_mask = [0, 1],
        begins_attr = [0, 0],
        ellipsis_mask = [1, 0],
        end_mask = [0, 1],
        ends_attr = [0, 0],
        new_axis_mask = [0, 0],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [0, 0],
        strides_attr = [1, -1]
        } : tensor<1x1x10x2xf16> -> tensor<1x1x10x2xf16>

    return %0 : tensor<1x1x10x2xf16>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0] [1, 1, 10, 1] : tensor<1x1x10x2xf16> to tensor<1x1x10x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 1] [1, 1, 10, 1] : tensor<1x1x10x2xf16> to tensor<1x1x10x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE1]], [[SLICE0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x10x1xf16>, tensor<1x1x10x1xf16> -> tensor<1x1x10x2xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 1, 10, 2] : tensor<1x1x10x2xf16> to tensor<1x1x10x2xf16>

    // CHECK:       return [[SLICE2]] : tensor<1x1x10x2xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipDynamicStridedSlice
func.func @SkipDynamicStridedSlice(%IN: tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = #NCHW}> {
    // CHECK:   [[IN:%.+]]: tensor<1x3x16x?xf16
    %SLICE = IE.StridedSlice(%IN) {
        begin_mask = [],
        begins_attr = [0, 0, 0, 1],
        ellipsis_mask = [],
        end_mask = [],
        ends_attr = [1, 3, 16, 30],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [],
        strides_attr = [1, 1, 1, 1]
    } : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 32]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE:%.+]] = IE.StridedSlice([[IN]])

    return %SLICE : tensor<1x3x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 16, 29]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[SLICE]]
}

// -----

// CHECK-LABEL:  @StridedSliceNegBeginNegEndNegStrides
func.func @StridedSliceNegBeginNegEndNegStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, -1], 
      ends_attr = [0, -3], 
      strides_attr = [1, -1],
      begin_mask = [1, 0], 
      end_mask = [1, 0], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK-DAG:  [[VAL0:%.*]] = IE.Slice [[IN]] [0, 0, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL1:%.*]] = IE.Slice [[IN]] [0, 1, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL2:%.*]] = IE.Slice [[IN]] [0, 2, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL3:%.*]] = IE.Slice [[IN]] [0, 3, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL4:%.*]] = IE.Slice [[IN]] [0, 4, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL5:%.*]] = IE.Slice [[IN]] [0, 5, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL6:%.*]] = IE.Slice [[IN]] [0, 6, 0] [1, 1, 1536]
    // CHECK:  [[VALCAT:%.*]] = IE.Concat([[VAL6]], [[VAL5]], [[VAL4]], [[VAL3]], [[VAL2]], [[VAL1]], [[VAL0]])
    // CHECK-SAME:  axis = 1 :
    // CHECK:  [[OUT:%.*]] = IE.Slice [[VALCAT]] [0, 0, 0] [1, 2, 1536]
    // CHECK:  return [[OUT]]

    return %0 : tensor<1x2x1536xf32>
}

// -----

// CHECK-LABEL:  @StridedSliceBeginGreaterThanEndNegStrides
func.func @StridedSliceBeginGreaterThanEndNegStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, 2], 
      ends_attr = [0, 0], 
      strides_attr = [1, -1],
      begin_mask = [1, 0], 
      end_mask = [1, 0], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK-DAG:  [[VAL0:%.*]] = IE.Slice [[IN]] [0, 0, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL1:%.*]] = IE.Slice [[IN]] [0, 1, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL2:%.*]] = IE.Slice [[IN]] [0, 2, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL3:%.*]] = IE.Slice [[IN]] [0, 3, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL4:%.*]] = IE.Slice [[IN]] [0, 4, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL5:%.*]] = IE.Slice [[IN]] [0, 5, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL6:%.*]] = IE.Slice [[IN]] [0, 6, 0] [1, 1, 1536]
    // CHECK:  [[VALCAT:%.*]] = IE.Concat([[VAL6]], [[VAL5]], [[VAL4]], [[VAL3]], [[VAL2]], [[VAL1]], [[VAL0]])
    // CHECK-SAME:  axis = 1 :
    // CHECK:  [[OUT:%.*]] = IE.Slice [[VALCAT]] [0, 4, 0] [1, 2, 1536]
    // CHECK:  return [[OUT]]

    return %0 : tensor<1x2x1536xf32>
}

// -----

// CHECK-LABEL:  @StridedSliceMaskedEndNegStrides
func.func @StridedSliceMaskedEndNegStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, 1], 
      ends_attr = [0, 100], 
      strides_attr = [1, -1],
      begin_mask = [1, 0], 
      end_mask = [1, 1], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK-DAG:  [[VAL0:%.*]] = IE.Slice [[IN]] [0, 0, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL1:%.*]] = IE.Slice [[IN]] [0, 1, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL2:%.*]] = IE.Slice [[IN]] [0, 2, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL3:%.*]] = IE.Slice [[IN]] [0, 3, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL4:%.*]] = IE.Slice [[IN]] [0, 4, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL5:%.*]] = IE.Slice [[IN]] [0, 5, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL6:%.*]] = IE.Slice [[IN]] [0, 6, 0] [1, 1, 1536]
    // CHECK:  [[VALCAT:%.*]] = IE.Concat([[VAL6]], [[VAL5]], [[VAL4]], [[VAL3]], [[VAL2]], [[VAL1]], [[VAL0]])
    // CHECK-SAME:  axis = 1 :
    // CHECK:  [[OUT:%.*]] = IE.Slice [[VALCAT]] [0, 5, 0] [1, 2, 1536]
    // CHECK:  return [[OUT]]

    return %0 : tensor<1x2x1536xf32>
}

// -----

// CHECK-LABEL:  @StridedSliceMaskedBeginNegEndNegStrides
func.func @StridedSliceMaskedBeginNegEndNegStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, 0, 0], 
      ends_attr = [0, -3, 0], 
      strides_attr = [1, -1, 1],
      begin_mask = [1, 1, 1], 
      end_mask = [1, 0, 1], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK-DAG:  [[VAL0:%.*]] = IE.Slice [[IN]] [0, 0, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL1:%.*]] = IE.Slice [[IN]] [0, 1, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL2:%.*]] = IE.Slice [[IN]] [0, 2, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL3:%.*]] = IE.Slice [[IN]] [0, 3, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL4:%.*]] = IE.Slice [[IN]] [0, 4, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL5:%.*]] = IE.Slice [[IN]] [0, 5, 0] [1, 1, 1536]
    // CHECK-DAG:  [[VAL6:%.*]] = IE.Slice [[IN]] [0, 6, 0] [1, 1, 1536]
    // CHECK:  [[VALCAT:%.*]] = IE.Concat([[VAL6]], [[VAL5]], [[VAL4]], [[VAL3]], [[VAL2]], [[VAL1]], [[VAL0]])
    // CHECK-SAME:  axis = 1 :
    // CHECK:  [[OUT:%.*]] = IE.Slice [[VALCAT]] [0, 0, 0] [1, 2, 1536]
    // CHECK:  return [[OUT]]

    return %0 : tensor<1x2x1536xf32>
}

// -----

// CHECK-LABEL:  @SkipStridedSliceNonUnitaryNegStrides
func.func @SkipStridedSliceNonUnitaryNegStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, 3], 
      ends_attr = [0, 0], 
      strides_attr = [1, -2],
      begin_mask = [1, 0], 
      end_mask = [1, 0], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    return %0 : tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK:   [[OUT:%.+]] = IE.StridedSlice([[IN]])
    // CHECK:   return [[OUT]]
}

// -----

// CHECK-LABEL:  @SkipStridedSliceNonUnitaryPosStrides
func.func @SkipStridedSliceNonUnitaryPosStrides(%arg0: tensor<1x7x1536xf32>) -> tensor<1x2x1536xf32> {
    %0 = IE.StridedSlice(%arg0) {
      begins_attr = [0, 0], 
      ends_attr = [0, 3], 
      strides_attr = [1, 2],
      begin_mask = [1, 0], 
      end_mask = [1, 0], 
      new_axis_mask = [], 
      ellipsis_mask = [], 
      shrink_axis_mask = [], 
      operandSegmentSizes = array<i32: 1, 0, 0, 0>
    } : tensor<1x7x1536xf32> -> tensor<1x2x1536xf32>

    return %0 : tensor<1x2x1536xf32>

    // CHECK:   [[IN:%.+]]: tensor<1x7x1536xf32>
    // CHECK:   [[OUT:%.+]] = IE.StridedSlice([[IN]])
    // CHECK:   return [[OUT]]
}

// -----

// CHECK-LABEL: @ResolveStridedSliceWithTwoAxisReversed
func.func @ResolveStridedSliceWithTwoAxisReversed(%arg0: tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32> {
    %0 = IE.StridedSlice(%arg0) {
        begin_mask = [0, 0],
        begins_attr = [-1, -1],
        ellipsis_mask = [],
        end_mask = [0, 0],
        ends_attr = [-9223372036854775807, -9223372036854775807],
        new_axis_mask = [],
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        shrink_axis_mask = [],
        strides_attr = [-1, -1]
        } : tensor<3x3x256x256xf32> -> tensor<3x3x256x256xf32>

    return %0 : tensor<3x3x256x256xf32>

    // CHECK:   [[IN:%.+]]: tensor<3x3x256x256xf32>
    // CHECK-DAG:  [[VAL0:%.+]] = IE.Slice [[IN]] [0, 0, 0, 0] [1, 3, 256, 256]
    // CHECK-DAG:  [[VAL1:%.+]] = IE.Slice [[IN]] [1, 0, 0, 0] [1, 3, 256, 256]
    // CHECK-DAG:  [[VAL2:%.+]] = IE.Slice [[IN]] [2, 0, 0, 0] [1, 3, 256, 256]
    // CHECK:  [[VALCAT:%.+]] = IE.Concat([[VAL2]], [[VAL1]], [[VAL0]])
    // CHECK-SAME:  axis = 0 :
    // CHECK-DAG:  [[VAL3:%.+]] = IE.Slice [[VALCAT]] [0, 0, 0, 0] [3, 1, 256, 256]
    // CHECK-DAG:  [[VAL4:%.+]] = IE.Slice [[VALCAT]] [0, 1, 0, 0] [3, 1, 256, 256]
    // CHECK-DAG:  [[VAL5:%.+]] = IE.Slice [[VALCAT]] [0, 2, 0, 0] [3, 1, 256, 256]
    // CHECK:  [[VALCAT2:%.+]] = IE.Concat([[VAL5]], [[VAL4]], [[VAL3]])
    // CHECK-SAME:  axis = 1 :

    // CHECK-DAG:  [[OUT:%.+]] = IE.Slice [[VALCAT2]] [0, 0, 0, 0] [3, 3, 256, 256] : tensor<3x3x256x256xf32> to tensor<3x3x256x256xf32>
    // CHECK:   return [[OUT]]
}
