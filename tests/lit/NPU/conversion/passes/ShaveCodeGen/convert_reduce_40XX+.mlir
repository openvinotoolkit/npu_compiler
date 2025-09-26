//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// IE.ReduceMax

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMaxI32
module @ReduceMaxI32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xsi32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xsi32>) -> tensor<2x3x5xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xsi32>) {
      %1 = IE.ReduceMax(%arg1) {axes_value = [2]} : tensor<2x3x4x5xsi32> -> tensor<2x3x5xsi32>
      IE.CGCYield %1 : tensor<2x3x5xsi32>
    } -> tensor<2x3x5xsi32>
    return %0 : tensor<2x3x5xsi32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xsi32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant -2147483648 : i32
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xsi32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[FILL]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[MAX:%.+]] = arith.maxsi [[IN]], [[OUT]] : i32
// CHECK-NEXT:        linalg.yield [[MAX]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<2x3x5xi32> to tensor<2x3x5xsi32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xsi32>
// CHECK-NEXT:    } -> tensor<2x3x5xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMaxUI32
module @ReduceMaxUI32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xui32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xui32>) -> tensor<2x3x5xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xui32>) {
      %1 = IE.ReduceMax(%arg1) {axes_value = [2]} : tensor<2x3x4x5xui32> -> tensor<2x3x5xui32>
      IE.CGCYield %1 : tensor<2x3x5xui32>
    } -> tensor<2x3x5xui32>
    return %0 : tensor<2x3x5xui32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xui32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xui32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[FILL]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[MAX:%.+]] = arith.maxui [[IN]], [[OUT]] : i32
// CHECK-NEXT:        linalg.yield [[MAX]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<2x3x5xi32> to tensor<2x3x5xui32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xui32>
// CHECK-NEXT:    } -> tensor<2x3x5xui32>
  }
}

// -----


// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMaxF16
module @ReduceMaxF16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x3x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceMax(%arg1) {axes_value = [2]} : tensor<2x3x4x5xf16> -> tensor<2x3x5xf16>
      IE.CGCYield %1 : tensor<2x3x5xf16>
    } -> tensor<2x3x5xf16>
    return %0 : tensor<2x3x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xf16>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0xFC00 : f16
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f16) outs([[EMPTY]] : tensor<2x3x5xf16>) -> tensor<2x3x5xf16>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[FILL]] : tensor<2x3x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[MAX:%.+]] = arith.maximumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[MAX]] : f16
// CHECK-NEXT:      } -> tensor<2x3x5xf16>
// CHECK-NEXT:      IE.CGCYield [[LINALG_OP]] : tensor<2x3x5xf16>
// CHECK-NEXT:    } -> tensor<2x3x5xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: @KeepDimsReduceMaxF16
module @KeepDimsReduceMaxF16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x1x1x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x1x1x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceMax(%arg1) {axes_value = [1, 2], keep_dims} : tensor<2x3x4x5xf16> -> tensor<2x1x1x5xf16>
      IE.CGCYield %1 : tensor<2x1x1x5xf16>
    } -> tensor<2x1x1x5xf16>
    return %0 : tensor<2x1x1x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x5xf16>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0xFC00 : f16
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f16) outs([[EMPTY]] : tensor<2x5xf16>) -> tensor<2x5xf16>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[FILL]] : tensor<2x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[MAX:%.+]] = arith.maximumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[MAX]] : f16
// CHECK-NEXT:      } -> tensor<2x5xf16>
// CHECK-NEXT:      [[EMPTY_POST:%.+]] = tensor.empty() : tensor<2x1x1x5xf16>
// CHECK-NEXT:      [[LINALG_POST:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LINALG_OP]] : tensor<2x5xf16>) outs([[EMPTY_POST]] : tensor<2x1x1x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        linalg.yield [[IN]] : f16
// CHECK-NEXT:      } -> tensor<2x1x1x5xf16>
// CHECK-NEXT:      IE.CGCYield [[LINALG_POST]] : tensor<2x1x1x5xf16>
// CHECK-NEXT:    } -> tensor<2x1x1x5xf16>
  }
}

// -----

// IE.ReduceMin


// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMinI32
module @ReduceMinI32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xsi32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xsi32>) -> tensor<2x3x5xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xsi32>) {
      %1 = IE.ReduceMin(%arg1) {axes_value = [2]} : tensor<2x3x4x5xsi32> -> tensor<2x3x5xsi32>
      IE.CGCYield %1 : tensor<2x3x5xsi32>
    } -> tensor<2x3x5xsi32>
    return %0 : tensor<2x3x5xsi32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xsi32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 2147483647 : i32
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xsi32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[FILL]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[MAX:%.+]] = arith.minsi [[IN]], [[OUT]] : i32
// CHECK-NEXT:        linalg.yield [[MAX]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<2x3x5xi32> to tensor<2x3x5xsi32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xsi32>
// CHECK-NEXT:    } -> tensor<2x3x5xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMinUI32
module @ReduceMinUI32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xui32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xui32>) -> tensor<2x3x5xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xui32>) {
      %1 = IE.ReduceMin(%arg1) {axes_value = [2]} : tensor<2x3x4x5xui32> -> tensor<2x3x5xui32>
      IE.CGCYield %1 : tensor<2x3x5xui32>
    } -> tensor<2x3x5xui32>
    return %0 : tensor<2x3x5xui32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xui32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant -1 : i32
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xui32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[FILL]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[MAX:%.+]] = arith.minui [[IN]], [[OUT]] : i32
// CHECK-NEXT:        linalg.yield [[MAX]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<2x3x5xi32> to tensor<2x3x5xui32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xui32>
// CHECK-NEXT:    } -> tensor<2x3x5xui32>
  }
}

// -----


// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceMinF16
module @ReduceMinF16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x3x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceMin(%arg1) {axes_value = [2]} : tensor<2x3x4x5xf16> -> tensor<2x3x5xf16>
      IE.CGCYield %1 : tensor<2x3x5xf16>
    } -> tensor<2x3x5xf16>
    return %0 : tensor<2x3x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xf16>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0x7C00 : f16
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f16) outs([[EMPTY]] : tensor<2x3x5xf16>) -> tensor<2x3x5xf16>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[FILL]] : tensor<2x3x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[MAX:%.+]] = arith.minimumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[MAX]] : f16
// CHECK-NEXT:      } -> tensor<2x3x5xf16>
// CHECK-NEXT:      IE.CGCYield [[LINALG_OP]] : tensor<2x3x5xf16>
// CHECK-NEXT:    } -> tensor<2x3x5xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: @KeepDimsReduceMinF16
module @KeepDimsReduceMinF16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x1x1x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x1x1x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceMin(%arg1) {axes_value = [1, 2], keep_dims} : tensor<2x3x4x5xf16> -> tensor<2x1x1x5xf16>
      IE.CGCYield %1 : tensor<2x1x1x5xf16>
    } -> tensor<2x1x1x5xf16>
    return %0 : tensor<2x1x1x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x5xf16>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0x7C00 : f16
// CHECK-NEXT:      [[FILL:%.+]] = linalg.fill ins([[CST]] : f16) outs([[EMPTY]] : tensor<2x5xf16>) -> tensor<2x5xf16>
// CHECK-NEXT:      [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[FILL]] : tensor<2x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[MAX:%.+]] = arith.minimumf [[IN]], [[OUT]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:        linalg.yield [[MAX]] : f16
// CHECK-NEXT:      } -> tensor<2x5xf16>
// CHECK-NEXT:      [[EMPTY_POST:%.+]] = tensor.empty() : tensor<2x1x1x5xf16>
// CHECK-NEXT:      [[LINALG_POST:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LINALG_OP]] : tensor<2x5xf16>) outs([[EMPTY_POST]] : tensor<2x1x1x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
// CHECK-NEXT:        linalg.yield [[IN]] : f16
// CHECK-NEXT:      } -> tensor<2x1x1x5xf16>
// CHECK-NEXT:      IE.CGCYield [[LINALG_POST]] : tensor<2x1x1x5xf16>
// CHECK-NEXT:    } -> tensor<2x1x1x5xf16>
  }
}

// -----

// IE.ReduceL2

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceL2I32
module @ReduceL2I32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xsi32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xsi32>) -> tensor<2x3x5xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xsi32>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [2]} : tensor<2x3x4x5xsi32> -> tensor<2x3x5xsi32>
      IE.CGCYield %1 : tensor<2x3x5xsi32>
    } -> tensor<2x3x5xsi32>
    return %0 : tensor<2x3x5xsi32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xsi32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[SPLAT:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xsi32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[REDUCE_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[SPLAT]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[SQ:%.+]] = arith.muli [[IN]], [[IN]] : i32
// CHECK-NEXT:        [[SUM:%.+]] = arith.addi [[OUT]], [[SQ]] : i32
// CHECK-NEXT:        linalg.yield [[SUM]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[POST:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE_OP]] : tensor<2x3x5xi32>) outs([[REDUCE_OP]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[FPCAST:%.+]] = arith.uitofp [[IN]] : i32 to f32
// CHECK-NEXT:        [[SQRT:%.+]] = math.sqrt [[FPCAST]] fastmath<afn> : f32
// CHECK-NEXT:        [[INTCAST:%.+]] = arith.fptoui [[SQRT]] : f32 to i32
// CHECK-NEXT:        linalg.yield [[INTCAST]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[POST]] : tensor<2x3x5xi32> to tensor<2x3x5xsi32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xsi32>
// CHECK-NEXT:    } -> tensor<2x3x5xsi32>
  }
}

// -----

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceL2UI32
module @ReduceL2UI32 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xui32>
  }

  func.func @main(%arg0: tensor<2x3x4x5xui32>) -> tensor<2x3x5xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xui32>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [2]} : tensor<2x3x4x5xui32> -> tensor<2x3x5xui32>
      IE.CGCYield %1 : tensor<2x3x5xui32>
    } -> tensor<2x3x5xui32>
    return %0 : tensor<2x3x5xui32>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xui32>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xi32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[SPLAT:%.+]] = linalg.fill ins([[CST]] : i32) outs([[EMPTY]] : tensor<2x3x5xi32>) -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[ARG_BC:%.+]] = tensor.bitcast [[ARG]] : tensor<2x3x4x5xui32> to tensor<2x3x4x5xi32>
// CHECK-NEXT:      [[REDUCE_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG_BC]] : tensor<2x3x4x5xi32>) outs([[SPLAT]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[SQ:%.+]] = arith.muli [[IN]], [[IN]] : i32
// CHECK-NEXT:        [[SUM:%.+]] = arith.addi [[OUT]], [[SQ]] : i32
// CHECK-NEXT:        linalg.yield [[SUM]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[POST:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE_OP]] : tensor<2x3x5xi32>) outs([[REDUCE_OP]] : tensor<2x3x5xi32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
// CHECK-NEXT:        [[FPCAST:%.+]] = arith.uitofp [[IN]] : i32 to f32
// CHECK-NEXT:        [[SQRT:%.+]] = math.sqrt [[FPCAST]] fastmath<afn> : f32
// CHECK-NEXT:        [[INTCAST:%.+]] = arith.fptoui [[SQRT]] : f32 to i32
// CHECK-NEXT:        linalg.yield [[INTCAST]] : i32
// CHECK-NEXT:      } -> tensor<2x3x5xi32>
// CHECK-NEXT:      [[RES:%.+]] = tensor.bitcast [[POST]] : tensor<2x3x5xi32> to tensor<2x3x5xui32>
// CHECK-NEXT:      IE.CGCYield [[RES]] : tensor<2x3x5xui32>
// CHECK-NEXT:    } -> tensor<2x3x5xui32>
  }
}

// -----


// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: @ReduceL2F16
module @ReduceL2F16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x3x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [2]} : tensor<2x3x4x5xf16> -> tensor<2x3x5xf16>
      IE.CGCYield %1 : tensor<2x3x5xf16>
    } -> tensor<2x3x5xf16>
    return %0 : tensor<2x3x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xf32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[SPLAT:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:      [[REDUCE_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[SPLAT]] : tensor<2x3x5xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[SQ:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[SUM:%.+]] = arith.addf [[OUT]], [[SQ]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[SUM]] : f32
// CHECK-NEXT:      } -> tensor<2x3x5xf32>
// CHECK-NEXT:      [[POST_EMPTY:%.+]] = tensor.empty() : tensor<2x3x5xf16>
// CHECK-NEXT:      [[POST:%.+]] = linalg.generic {indexing_maps = [[[CHW]], [[CHW]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[REDUCE_OP]] : tensor<2x3x5xf32>) outs([[POST_EMPTY]] : tensor<2x3x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQRT:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNCF:%.+]] = arith.truncf [[SQRT]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNCF]] : f16
// CHECK-NEXT:      } -> tensor<2x3x5xf16>
// CHECK-NEXT:      IE.CGCYield [[POST]] : tensor<2x3x5xf16>
// CHECK-NEXT:    } -> tensor<2x3x5xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: [[map:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK: @KeepDimsReduceL2F16
module @KeepDimsReduceL2F16 {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<2x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x1x1x5xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x1x1x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5xf16>) {
      %1 = IE.ReduceL2(%arg1) {axes_value = [1, 2], keep_dims} : tensor<2x3x4x5xf16> -> tensor<2x1x1x5xf16>
      IE.CGCYield %1 : tensor<2x1x1x5xf16>
    } -> tensor<2x1x1x5xf16>
    return %0 : tensor<2x1x1x5xf16>

// CHECK:    IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<2x3x4x5xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<2x5xf32>
// CHECK-NEXT:      [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      [[SPLAT:%.+]] = linalg.fill ins([[CST]] : f32) outs([[EMPTY]] : tensor<2x5xf32>) -> tensor<2x5xf32>
// CHECK-NEXT:      [[REDUCE_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[map]]], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins([[ARG]] : tensor<2x3x4x5xf16>) outs([[SPLAT]] : tensor<2x5xf32>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f32):
// CHECK-NEXT:        [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK-NEXT:        [[SQ:%.+]] = arith.mulf [[EXT]], [[EXT]] : f32
// CHECK-NEXT:        [[SUM:%.+]] = arith.addf [[OUT]], [[SQ]] fastmath<reassoc> : f32
// CHECK-NEXT:        linalg.yield [[SUM]] : f32
// CHECK-NEXT:      } -> tensor<2x5xf32>
// CHECK-NEXT:      [[POST_EMPTY:%.+]] = tensor.empty() : tensor<2x1x1x5xf16>
// CHECK-NEXT:      [[POST:%.+]] = linalg.generic {indexing_maps = [[[map]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[REDUCE_OP]] : tensor<2x5xf32>) outs([[POST_EMPTY]] : tensor<2x1x1x5xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f16):
// CHECK-NEXT:        [[SQRT:%.+]] = math.sqrt [[IN]] fastmath<afn> : f32
// CHECK-NEXT:        [[TRUNCF:%.+]] = arith.truncf [[SQRT]] : f32 to f16
// CHECK-NEXT:        linalg.yield [[TRUNCF]] : f16
// CHECK-NEXT:      } -> tensor<2x1x1x5xf16>
// CHECK-NEXT:      IE.CGCYield [[POST]] : tensor<2x1x1x5xf16>
// CHECK-NEXT:    } -> tensor<2x1x1x5xf16>
  }
}
