//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-eltwise-layers-to-math %s -o - | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

// CHECK: func.func @PaddedFlat(
func.func @PaddedFlat(%arg0: tensor<1x16x4000x200xf32>) -> tensor<16x4000x200xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], output_padding = [15, 0, 0]} : tensor<1x16x4000x200xf32> -> tensor<16x4000x200xf32>
    IE.CGCYield %1 : tensor<16x4000x200xf32>
  } -> tensor<16x4000x200xf32>
  return %0 : tensor<16x4000x200xf32>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x16x4000x200xf32>)
// CHECK-NEXT:      [[F32_ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[PAD:%.+]] = tensor.empty() : tensor<16x4000x200xf32>
// CHECK-NEXT:      [[PAD_FILL:%.+]] = linalg.fill ins([[F32_ZERO]] : f32) outs([[PAD]] : tensor<16x4000x200xf32>) -> tensor<16x4000x200xf32>
// CHECK-NEXT:      [[EXTRACT_PAD:%.+]] = tensor.extract_slice [[PAD_FILL]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<16x4000x200xf32> to tensor<1x4000x200xf32>
// CHECK-NEXT:      [[OUT_SLICE:%.+]] = linalg.fill ins([[F32_ZERO]] : f32) outs([[EXTRACT_PAD]] : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[IN_SLICE:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 12, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x12x4000x200xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins([[IN_SLICE]] : tensor<1x12x4000x200xf32>) outs([[OUT_SLICE]] : tensor<1x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[OUT:%.+]] = tensor.insert_slice [[REDUCE]] into [[PAD_FILL]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<1x4000x200xf32> into tensor<16x4000x200xf32>
// CHECK-NEXT:      IE.CGCYield [[OUT]] : tensor<16x4000x200xf32>
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

// CHECK: func.func @PaddedFlatKeepDims
func.func @PaddedFlatKeepDims(%arg0: tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x4000x200xf32> -> tensor<1x16x4000x200xf32>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32>
  } -> tensor<1x16x4000x200xf32>
  return %0 : tensor<1x16x4000x200xf32>

// CHECK:    IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x16x4000x200xf32>) {
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 12, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x12x4000x200xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins([[REDUCE_IN]] : tensor<1x12x4000x200xf32>) outs([[REDUCE_INIT]] : tensor<1x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[PAD:%.+]] = tensor.empty() : tensor<1x16x4000x200xf32>
// CHECK-NEXT:      [[PAD_FILL:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[PAD]] : tensor<1x16x4000x200xf32>) -> tensor<1x16x4000x200xf32>
// CHECK-NEXT:      [[PAD_SLICE:%.+]] = tensor.extract_slice [[PAD_FILL]][0, 0, 0, 0] [1, 1, 4000, 200] [1, 1, 1, 1] : tensor<1x16x4000x200xf32> to tensor<1x1x4000x200xf32>
// CHECK-NEXT:      [[RESHAPE:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x4000x200xf32>) outs([[PAD_SLICE]] : tensor<1x1x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, {{.+}}: f32):
// CHECK-NEXT:        linalg.yield [[IN]] : f32
// CHECK-NEXT:      } -> tensor<1x1x4000x200xf32>
// CHECK-NEXT:      [[OUT:%.+]] = tensor.insert_slice [[RESHAPE]] into [[PAD_FILL]][0, 0, 0, 0] [1, 1, 4000, 200] [1, 1, 1, 1] : tensor<1x1x4000x200xf32> into tensor<1x16x4000x200xf32>
// CHECK-NEXT:      IE.CGCYield [[OUT]] : tensor<1x16x4000x200xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK: func.func @PaddedNHWCKeepDims(
func.func @PaddedNHWCKeepDims(%arg0: tensor<1x16x4000x200xf32, {order = #NHWC}>) -> tensor<1x16x4000x200xf32, {order = #NHWC}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order = #NHWC}>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x4000x200xf32, {order = #NHWC}> -> tensor<1x16x4000x200xf32, {order = #NHWC}>
    IE.CGCYield %1 : tensor<1x16x4000x200xf32, {order = #NHWC}>
  } -> tensor<1x16x4000x200xf32, {order = #NHWC}>
  return %0 : tensor<1x16x4000x200xf32, {order = #NHWC}>

// CHECK:    IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[REDUCE_IN]] : tensor<1x4000x200x12xf32>) outs([[REDUCE_INIT]] : tensor<1x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[PAD:%.+]] = tensor.empty() : tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[PAD_FILL:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[PAD]] : tensor<1x4000x200x16xf32>) -> tensor<1x4000x200x16xf32>
// CHECK-NEXT:      [[RESHAPE_OUT_SLICE:%.+]] = tensor.extract_slice [[PAD_FILL]][0, 0, 0, 0] [1, 4000, 200, 1] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x1xf32>
// CHECK-NEXT:      [[RESHAPE:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x4000x200xf32>) outs([[RESHAPE_OUT_SLICE]] : tensor<1x4000x200x1xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, {{.+}}: f32):
// CHECK-NEXT:        linalg.yield [[IN]] : f32
// CHECK-NEXT:      } -> tensor<1x4000x200x1xf32>
// CHECK-NEXT:      [[OUT:%.+]] = tensor.insert_slice [[RESHAPE]] into [[PAD_FILL]][0, 0, 0, 0] [1, 4000, 200, 1] [1, 1, 1, 1] : tensor<1x4000x200x1xf32> into tensor<1x4000x200x16xf32>
// CHECK-NEXT:      IE.CGCYield [[OUT]] : tensor<1x4000x200x16xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK: func.func @PaddedNHWCToFlatKeepDims(
func.func @PaddedNHWCToFlatKeepDims(%arg0: tensor<1x16x4000x200xf32, {order = #NHWC}>) -> tensor<16x4000x200xf32> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order = #NHWC}>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], output_padding = [15, 0, 0]} : tensor<1x16x4000x200xf32, {order = #NHWC}> -> tensor<16x4000x200xf32>
    IE.CGCYield %1 : tensor<16x4000x200xf32>
  } -> tensor<16x4000x200xf32>
  return %0 : tensor<16x4000x200xf32>

// CHECK: IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:      [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPT:%.+]] = tensor.empty() : tensor<16x4000x200xf32>
// CHECK-NEXT:      [[PAD:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<16x4000x200xf32>) -> tensor<16x4000x200xf32>
// CHECK-NEXT:      [[PAD_SLICE:%.+]] = tensor.extract_slice [[PAD]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<16x4000x200xf32> to tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_OUT_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[PAD_SLICE]] : tensor<1x4000x200xf32>) -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[REDUCE_IN]] : tensor<1x4000x200x12xf32>) outs([[REDUCE_OUT_INIT]] : tensor<1x4000x200xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x4000x200xf32>
// CHECK-NEXT:      [[OUT:%.+]] = tensor.insert_slice [[REDUCE]] into [[PAD]][0, 0, 0] [1, 4000, 200] [1, 1, 1] : tensor<1x4000x200xf32> into tensor<16x4000x200xf32>
// CHECK-NEXT:      IE.CGCYield [[OUT]] : tensor<16x4000x200xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK: func.func @PaddedNHWCToCWH(
func.func @PaddedNHWCToCWH(%arg0: tensor<1x16x4000x200xf32, {order = #NHWC}>) -> tensor<16x4000x200xf32, {order = #map}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf32, {order = #NHWC}>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], output_padding = [15, 0, 0]} : tensor<1x16x4000x200xf32, {order = #NHWC}> -> tensor<16x4000x200xf32, {order = #map}>
    IE.CGCYield %1 : tensor<16x4000x200xf32, {order = #map}>
  } -> tensor<16x4000x200xf32, {order = #map}>
  return %0 : tensor<16x4000x200xf32, {order = #map}>

// CHECK:    IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf32>) {
// CHECK-NEXT:     [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     [[EMPT:%.+]] = tensor.empty() : tensor<16x200x4000xf32>
// CHECK-NEXT:     [[PAD:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<16x200x4000xf32>) -> tensor<16x200x4000xf32>
// CHECK-NEXT:     [[PAD_SLICE:%.+]] = tensor.extract_slice [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<16x200x4000xf32> to tensor<1x200x4000xf32>
// CHECK-NEXT:     [[REDUCE_OUT_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[PAD_SLICE]] : tensor<1x200x4000xf32>) -> tensor<1x200x4000xf32>
// CHECK-NEXT:     [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf32> to tensor<1x4000x200x12xf32>
// CHECK-NEXT:     [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[REDUCE_IN]] : tensor<1x4000x200x12xf32>) outs([[REDUCE_OUT_INIT]] : tensor<1x200x4000xf32>) {
// CHECK-NEXT:     ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
// CHECK-NEXT:       [[ADD:%.+]] = arith.addf [[OUT]], [[IN]] fastmath<reassoc> : f32
// CHECK-NEXT:       linalg.yield [[ADD]] : f32
// CHECK-NEXT:     } -> tensor<1x200x4000xf32>
// CHECK-NEXT:     [[RES:%.+]] = tensor.insert_slice [[REDUCE]] into [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<1x200x4000xf32> into tensor<16x200x4000xf32>
// CHECK-NEXT:     IE.CGCYield [[RES]] : tensor<16x200x4000xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK: func.func @PaddedNHWCToCWHF16(
func.func @PaddedNHWCToCWHF16(%arg0: tensor<1x16x4000x200xf16, {order = #NHWC}>) -> tensor<16x4000x200xf16, {order = #map}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xf16, {order = #NHWC}>) {
    %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], output_padding = [15, 0, 0]} : tensor<1x16x4000x200xf16, {order = #NHWC}> -> tensor<16x4000x200xf16, {order = #map}>
    IE.CGCYield %1 : tensor<16x4000x200xf16, {order = #map}>
  } -> tensor<16x4000x200xf16, {order = #map}>
  return %0 : tensor<16x4000x200xf16, {order = #map}>

// CHECK:    IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xf16>) {
// CHECK-NEXT:     [[ZERO1:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:     [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     [[EMPT:%.+]] = tensor.empty() : tensor<1x200x4000xf32>
// CHECK-NEXT:     [[REDUCE_OUT_INIT:%.+]] = linalg.fill ins([[ZERO]] : f32) outs([[EMPT]] : tensor<1x200x4000xf32>) -> tensor<1x200x4000xf32>
// CHECK-NEXT:     [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xf16> to tensor<1x4000x200x12xf16>
// CHECK-NEXT:     [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[REDUCE_IN]] : tensor<1x4000x200x12xf16>) outs([[REDUCE_OUT_INIT]] : tensor<1x200x4000xf32>) {
// CHECK-NEXT:     ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:       [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:       [[ADD:%.+]] = arith.addf [[OUT]], [[EXT]] fastmath<reassoc> : f32
// CHECK-NEXT:       linalg.yield [[ADD]] : f32
// CHECK-NEXT:     } -> tensor<1x200x4000xf32>
// CHECK-NEXT:     [[PAD_EMPT:%.+]] = tensor.empty() : tensor<16x200x4000xf16>
// CHECK-NEXT:     [[PAD:%.+]] = linalg.fill ins([[ZERO1]] : f16) outs([[PAD_EMPT]] : tensor<16x200x4000xf16>) -> tensor<16x200x4000xf16>
// CHECK-NEXT:     [[PAD_SLICE:%.+]] = tensor.extract_slice [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<16x200x4000xf16> to tensor<1x200x4000xf16>
// CHECK-NEXT:     [[RESHAPE:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x200x4000xf32>) outs([[PAD_SLICE]] : tensor<1x200x4000xf16>) {
// CHECK-NEXT:     ^bb0([[IN:%.+]]: f32, {{.+}}: f16):
// CHECK-NEXT:       [[TRUNC:%.+]] = arith.truncf [[IN]] : f32 to f16
// CHECK-NEXT:       linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:     } -> tensor<1x200x4000xf16>
// CHECK-NEXT:     [[OUT:%.+]] = tensor.insert_slice [[RESHAPE]] into [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<1x200x4000xf16> into tensor<16x200x4000xf16>
// CHECK-NEXT:     IE.CGCYield [[OUT]] : tensor<16x200x4000xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWH = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[map:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK: func.func @PaddedNHWCToCWHUI32(
func.func @PaddedNHWCToCWHUI32(%arg0: tensor<1x16x4000x200xui32, {order=#NHWC}>) -> tensor<16x4000x200xui32, {order=#NWH}> {
  %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x16x4000x200xui32, {order=#NHWC}>) {
     %1 = IE.ReduceSum(%arg1) {axes_value = [1], input_padding = [0, 4, 0, 0], output_padding = [15, 0, 0]} : tensor<1x16x4000x200xui32, {order=#NHWC}> -> tensor<16x4000x200xui32, {order=#NWH}>
    IE.CGCYield %1 : tensor<16x4000x200xui32, {order=#NWH}>
  } -> tensor<16x4000x200xui32, {order=#NWH}>
  return %0 : tensor<16x4000x200xui32, {order=#NWH}>

// CHECK:    IE.CodeGenCapsule inputs({{.+}} as [[ARG1:%.+]]: tensor<1x4000x200x16xi32>) {
// CHECK-NEXT:     [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:     [[EMPT:%.+]] = tensor.empty() : tensor<16x200x4000xi32>
// CHECK-NEXT:     [[PAD:%.+]] = linalg.fill ins([[ZERO]] : i32) outs([[EMPT]] : tensor<16x200x4000xi32>) -> tensor<16x200x4000xi32>
// CHECK-NEXT:     [[REDUCE_OUT:%.+]] = tensor.extract_slice [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<16x200x4000xi32> to tensor<1x200x4000xi32>
// CHECK-NEXT:     [[REDUCE_OUT_INIT:%.+]] = linalg.fill ins([[ZERO]] : i32) outs([[REDUCE_OUT]] : tensor<1x200x4000xi32>) -> tensor<1x200x4000xi32>
// CHECK-NEXT:     [[REDUCE_IN:%.+]] = tensor.extract_slice [[ARG1]][0, 0, 0, 0] [1, 4000, 200, 12] [1, 1, 1, 1] : tensor<1x4000x200x16xi32> to tensor<1x4000x200x12xi32>
// CHECK-NEXT:     [[OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[REDUCE_IN]] : tensor<1x4000x200x12xi32>) outs([[REDUCE_OUT_INIT]] : tensor<1x200x4000xi32>) {
// CHECK-NEXT:     ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:       [[ADD:%.+]] = arith.addi [[OUT]], [[IN]] : i32
// CHECK-NEXT:       linalg.yield [[ADD]] : i32
// CHECK-NEXT:     } -> tensor<1x200x4000xi32>
// CHECK-NEXT:     [[OUT:%.+]] = tensor.insert_slice [[OP]] into [[PAD]][0, 0, 0] [1, 200, 4000] [1, 1, 1] : tensor<1x200x4000xi32> into tensor<16x200x4000xi32>
// CHECK-NEXT:     IE.CGCYield [[OUT]] : tensor<16x200x4000xi32>
}
