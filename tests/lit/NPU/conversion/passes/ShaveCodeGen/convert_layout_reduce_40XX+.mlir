//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: @KeepDimsSameLayout
module @KeepDimsSameLayout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x28x29xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x1x28x29xf16, {order = #NHWC}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceMax(%arg1) {axes_value = [1], keep_dims} : tensor<1x3x28x29xf16, {order = #NHWC}>  -> tensor<1x1x28x29xf16, {order = #NHWC}>
      IE.CGCYield %1 : tensor<1x1x28x29xf16, {order = #NHWC}>
    } -> tensor<1x1x28x29xf16, {order = #NHWC}>
    return %0 : tensor<1x1x28x29xf16, {order = #NHWC}>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0xFC00 : f16
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x28x29xf16>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f16) outs([[EMPTY]] : tensor<1x28x29xf16>) -> tensor<1x28x29xf16>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[ARG]] : tensor<1x28x29x3xf16>) outs([[FILL]] : tensor<1x28x29xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[MAX:%.+]] = arith.maximumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[MAX]] : f16
// CHECK-NEXT:      } -> tensor<1x28x29xf16>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<1x28x29x1xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x28x29xf16>) outs([[RES_EMPTY]] : tensor<1x28x29x1xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:        linalg.yield [[IN]] : f16
// CHECK-NEXT:      } -> tensor<1x28x29x1xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<1x28x29x1xf16>
// CHECK-NEXT:    } -> tensor<1x1x28x29xf16, {order = [[NHWC]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: @SameLayout
module @SameLayout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x28x29xf16>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x28x29xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1]} : tensor<1x3x28x29xf16, {order = #NHWC}>  -> tensor<1x28x29xf16>
      IE.CGCYield %1 : tensor<1x28x29xf16>
    } -> tensor<1x28x29xf16>
    return %0 : tensor<1x28x29xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x28x29xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x28x29xf32>) -> tensor<1x28x29xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[ARG]] : tensor<1x28x29x3xf16>) outs([[FILL]] : tensor<1x28x29xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[MUL:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[MUL]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x28x29xf32>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<1x28x29xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x28x29xf32>) outs([[RES_EMPTY]] : tensor<1x28x29xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQ:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SQ]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<1x28x29xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<1x28x29xf16>
// CHECK-NEXT:    } -> tensor<1x28x29xf16>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWH = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-NEXT: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
// CHECK: @DifferentLayout
module @DifferentLayout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x28x29xf16, {order = #NWH}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x28x29xf16, {order = #NWH}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1]} : tensor<1x3x28x29xf16, {order = #NHWC}>  -> tensor<1x28x29xf16, {order = #NWH}>
      IE.CGCYield %1 : tensor<1x28x29xf16, {order = #NWH}>
    } -> tensor<1x28x29xf16, {order = #NWH}>
    return %0 : tensor<1x28x29xf16, {order = #NWH}>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x29x28xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x29x28xf32>) -> tensor<1x29x28xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[ARG]] : tensor<1x28x29x3xf16>) outs([[FILL]] : tensor<1x29x28xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[MUL:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[MUL]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x29x28xf32>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<1x29x28xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x29x28xf32>) outs([[RES_EMPTY]] : tensor<1x29x28xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQ:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SQ]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<1x29x28xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<1x29x28xf16>
// CHECK-NEXT:    } -> tensor<1x28x29xf16, {order = [[map]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[NWCH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
// CHECK-NEXT: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @KeepDimsDifferentLayout
module @KeepDimsDifferentLayout {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x28x29xf16, {order = #NWCH}>
  }
  func.func @main(%arg0: tensor<1x3x28x29xf16, {order = #NHWC}>) -> tensor<1x1x28x29xf16, {order = #NWCH}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1], keep_dims} : tensor<1x3x28x29xf16, {order = #NHWC}>  -> tensor<1x1x28x29xf16, {order = #NWCH}>
      IE.CGCYield %1 : tensor<1x1x28x29xf16, {order = #NWCH}>
    } -> tensor<1x1x28x29xf16, {order = #NWCH}>
    return %0 : tensor<1x1x28x29xf16, {order = #NWCH}>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x29x28xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<1x29x28xf32>) -> tensor<1x29x28xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins([[ARG]] : tensor<1x28x29x3xf16>) outs([[FILL]] : tensor<1x29x28xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[MUL:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[MUL]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<1x29x28xf32>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<1x29x1x28xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[map1]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<1x29x28xf32>) outs([[RES_EMPTY]] : tensor<1x29x1x28xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQ:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SQ]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<1x29x1x28xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<1x29x1x28xf16>
// CHECK-NEXT:    } -> tensor<1x1x28x29xf16, {order = [[NWCH]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#WH = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-NEXT: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0)>
// CHECK: @DifferentLayoutMultipleAxis
module @DifferentLayoutMultipleAxis {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<2x28xf16, {order = #WH}>
  }
  func.func @main(%arg0: tensor<2x3x28x29xf16, {order = #NHWC}>) -> tensor<2x28xf16, {order = #WH}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1,3]} : tensor<2x3x28x29xf16, {order = #NHWC}>  -> tensor<2x28xf16, {order = #WH}>
      IE.CGCYield %1 : tensor<2x28xf16, {order = #WH}>
    } -> tensor<2x28xf16, {order = #WH}>
    return %0 : tensor<2x28xf16, {order = #WH}>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<28x2xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<28x2xf32>) -> tensor<28x2xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins([[ARG]] : tensor<2x28x29x3xf16>) outs([[FILL]] : tensor<28x2xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[MUL:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[MUL]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<28x2xf32>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<28x2xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[NC]], [[NC]]], iterator_types = ["parallel", "parallel"]} ins([[REDUCE]] : tensor<28x2xf32>) outs([[RES_EMPTY]] : tensor<28x2xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQ:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SQ]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<28x2xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<28x2xf16>
// CHECK-NEXT:    } -> tensor<2x28xf16, {order = [[CN]]}>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CNWH = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-NEXT: [[map1:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-NEXT: [[map2:#.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK: @KeepDimsDifferentLayoutMultipleAxis
module @KeepDimsDifferentLayoutMultipleAxis {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x28x29xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output" : tensor<2x1x28x1xf16, {order = #CNWH}>
  }
  func.func @main(%arg0: tensor<2x3x28x29xf16, {order = #NHWC}>) -> tensor<2x1x28x1xf16, {order = #CNWH}> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x28x29xf16, {order = #NHWC}>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1,3], keep_dims} : tensor<2x3x28x29xf16, {order = #NHWC}>  -> tensor<2x1x28x1xf16, {order = #CNWH}>
      IE.CGCYield %1 : tensor<2x1x28x1xf16, {order = #CNWH}>
    } -> tensor<2x1x28x1xf16, {order = #CNWH}>
    return %0 : tensor<2x1x28x1xf16, {order = #CNWH}>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x28x29x3xf16>) {
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x28xf32>
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<2x28xf32>) -> tensor<2x28xf32>
// CHECK-NEXT:      [[REDUCE:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map1]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins([[ARG]] : tensor<2x28x29x3xf16>) outs([[FILL]] : tensor<2x28xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[MUL:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[ADD:%.+]] = arith.addf [[OUT]], [[MUL]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[ADD]] : f32
// CHECK-NEXT:      } -> tensor<2x28xf32>
// CHECK-NEXT:      [[RES_EMPTY:%.+]] = tensor.empty() : tensor<1x2x1x28xf16>
// CHECK-NEXT:      [[RES:%.+]] = linalg.generic {indexing_maps = [[[map2]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE]] : tensor<2x28xf32>) outs([[RES_EMPTY]] : tensor<1x2x1x28xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQ:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNC:%.+]] = arith.truncf [[SQ]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNC]] : f16
// CHECK-NEXT:      } -> tensor<1x2x1x28xf16>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<1x2x1x28xf16>
// CHECK-NEXT:    } -> tensor<2x1x28x1xf16, {order = [[map]]}>
  }
}
