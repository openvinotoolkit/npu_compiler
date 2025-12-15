//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleCosLayer
module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Cos(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
  }
    // CHECK-NOT: IE.Cos
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[COS:%.+]] = math.cos [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[COS]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
}

// -----
// IE.Divide

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivFPLayer
module @SingleDivFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>
  }
    // CHECK-NOT: IE.Divide
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[OP:%.+]] = arith.divf [[LHS]], [[RHS]] fastmath<arcp> : f16
    // CHECK-NEXT:      linalg.yield [[OP]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivSILayer
module @SingleDivSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>
  }
    // CHECK-NOT:     IE.Divide
    // CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
    // CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
    // CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
    // CHECK-NEXT:    ^bb0([[LHS_S:%.+]]: i32, [[RHS_S:%.+]]: i32, {{%.+}}: i32):
    // CHECK-NEXT:      [[OP:%.+]] = arith.divsi [[LHS_S]], [[RHS_S]] : i32
    // CHECK-NEXT:      linalg.yield [[OP]] : i32
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
    // CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
    // CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleDivUILayer
module @SingleDivUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.Divide(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>
  }

    // CHECK-NOT:     IE.Divide
    // CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
    // CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
    // CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
    // CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
    // CHECK-NEXT:    ^bb0([[LHS_S:%.+]]: i32, [[RHS_S:%.+]]: i32, {{%.+}}: i32):
    // CHECK-NEXT:      [[OP:%.+]] = arith.divui [[LHS_S]], [[RHS_S]] : i32
    // CHECK-NEXT:      linalg.yield [[OP]] : i32
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
    // CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
    // CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
}

// -----
// IE.Maximum

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxFPLayer
module @SingleMaxFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Maximum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Maximum
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[OP:%.+]] = arith.maximumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
    // CHECK-NEXT:      linalg.yield [[OP]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxSILayer
module @SingleMaxSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Maximum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Maximum
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.maxsi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMaxUILayer
module @SingleMaxUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.Maximum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.Maximum
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.maxui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----
// IE.Minimum

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinFPLayer
module @SingleMinFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Minimum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Minimum
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.minimumf [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      linalg.yield [[OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinSILayer
module @SingleMinSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Minimum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Minimum
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.minsi [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleMinUILayer
module @SingleMinUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xui32>
    DataInfo "input1" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>, %arg1: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xui32>, %arg1 as %arg3: tensor<1x1x1x1000xui32>) {
      %1 = IE.Minimum(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xui32>, tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT:     IE.Minimum
// CHECK:         [[LHS_BC:%.+]] = tensor.bitcast {{%.+}} : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RHS_BC:%.+]] = tensor.bitcast {{%.+}} : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.minui [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @SingleLogLayer
module @SingleLogLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "log" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Log(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:  IE.Log
// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:        [[LOG:%.+]] = math.log [[IN]] fastmath<afn> : f16
// CHECK-NEXT:        linalg.yield [[LOG]] : f16
// CHECK-NEXT:      } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:      IE.CGCYield [[OP]] : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
	}
}
// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @SingleExpLayer
module @SingleExpLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "exp" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Exp(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:  IE.Exp
// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:        [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
// CHECK-NEXT:        linalg.yield [[EXP]] : f16
// CHECK-NEXT:      } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:      IE.CGCYield [[OP]] : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
	}
}

// -----
// IE.Sin

module @SingleSinLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "sin" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Sin(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:    IE.Sin
    // CHECK:        [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:   {{.*}} = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[SIN:%.+]] = math.sin [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[SIN]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:   } -> tensor<1x1x1x1000xf16>

  }
}

// -----
// IE.Sqrt

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: module @SingleSqrtLayer
module @SingleSqrtLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "sqrt" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Sqrt(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT: IE.Sqrt
// CHECK: IE.CodeGenCapsule inputs({{.*}} as [[ARG:%.+]]: tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:      [[OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:      ^bb0([[IN:%.+]]: f16, {{.*}}: f16):
// CHECK-NEXT:        [[SQRT:%.+]] = math.sqrt [[IN]] fastmath<afn> : f16
// CHECK-NEXT:        linalg.yield [[SQRT]] : f16
// CHECK-NEXT:      } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:      IE.CGCYield [[OP]] : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Round

module @SingleRoundLayerHalfToEven  {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Round" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Round(%arg1) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Round
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    {{.*}} = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[VAR0:%.+]] = math.roundeven [[IN]] : f16
    // CHECK-NEXT:      linalg.yield [[VAR0]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    }
  }

// -----

module @SingleRoundLayerHalfAwayFromZero {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Round" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Round(%arg1) {mode = #IE.round_mode<HALF_AWAY_FROM_ZERO>} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Round
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    {{.*}} = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[VAR0:%.+]] = math.round [[IN]] : f16
    // CHECK-NEXT:      linalg.yield [[VAR0]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
    }
  }

// -----
// IE.Erf

module @SingleErfLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "Erf" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Erf(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT:     IE.Erf
    // CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK-NEXT:    {{.*}} = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK-NEXT:    ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):
    // CHECK-NEXT:      [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32
    // CHECK-NEXT:      [[ERF:%.+]] = math.erf [[EXT]] : f32
    // CHECK-NEXT:      [[TRUNC:%.+]] = arith.truncf [[ERF]] : f32 to f16
    // CHECK-NEXT:      linalg.yield [[TRUNC]] : f16
    // CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Convert

module @SingleConvertFPToSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Convert(%arg1) {dstElemType = si32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT: IE.Convert
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK:     ^bb0([[IN:%.+]]: f16, {{%.+}}: i32):
// CHECK:       [[OP:%.+]] = arith.fptosi [[IN]] : f16 to i32
// CHECK:       linalg.yield [[OP]] : i32
// CHECK:     [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK:     IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>

  }
}

// -----

module @SingleConvertFPToUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Convert(%arg1) {dstElemType = ui32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT: IE.Convert
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK:     ^bb0([[IN:%.+]]: f16, {{%.+}}: i32):
// CHECK:       [[OP:%.+]] = arith.fptoui [[IN]] : f16 to i32
// CHECK:       linalg.yield [[OP]] : i32
// CHECK:     [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK:     IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>

  }
}

// -----

module @SingleConvertSIToFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Convert(%arg1) {dstElemType = f16} : tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT: IE.Convert
// CHECK:     [[RET:%.+]] = tensor.bitcast [[ARG0:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK:     ^bb0([[IN:%.+]]: i32, {{%.+}}: f16):
// CHECK:       [[OP:%.+]] = arith.sitofp [[IN]] : i32 to f16
// CHECK:       linalg.yield [[OP]] : f16
// CHECK:     IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>


  }
}

// -----

module @SingleConvertUIToFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui32>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xui32>) {
      %1 = IE.Convert(%arg1) {dstElemType = f16} : tensor<1x1x1x1000xui32> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT: IE.Convert
// CHECK:     [[RET:%.+]] = tensor.bitcast [[ARG0:%.+]] : tensor<1x1x1x1000xui32> to tensor<1x1x1x1000xi32>
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK:     ^bb0([[IN:%.+]]: i32, {{%.+}}: f16):
// CHECK:       [[OP:%.+]] = arith.uitofp [[IN]] : i32 to f16
// CHECK:       linalg.yield [[OP]] : f16
// CHECK:     IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>

  }
}

// -----

module @SingleConvertExtFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
      IE.CGCYield %1 : tensor<1x1x1x1000xf32>
    } -> tensor<1x1x1x1000xf32>
    return %0 : tensor<1x1x1x1000xf32>

// CHECK-NOT: IE.Convert
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf32>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf32>) {
// CHECK:     ^bb0([[IN:%.+]]: f16, {{%.+}}: f32):
// CHECK:       [[OP:%.+]] = arith.extf [[IN]] : f16 to f32
// CHECK:       linalg.yield [[OP]] : f32
// CHECK:     IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf32>

  }
}

// -----

module @SingleConvertTruncFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
      %1 = IE.Convert(%arg1) {dstElemType = f16} : tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT: IE.Convert
// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf16> {
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf32>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK:     ^bb0([[IN:%.+]]: f32, {{%.+}}: f16):
// CHECK:       [[OP:%.+]] = arith.truncf [[IN]] : f32 to f16
// CHECK:       linalg.yield [[OP]] : f16
// CHECK:     IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>

  }
}

// -----

module @SingleConvertExtSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi16>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi16>) {
      %1 = IE.Convert(%arg1) {dstElemType = si32} : tensor<1x1x1x1000xsi16> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT: IE.Convert
// CHECK:     [[RET:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xsi16> to tensor<1x1x1x1000xi16>
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK:     ^bb0([[IN:%.+]]: i16, {{%.+}}: i32):
// CHECK:       [[OP:%.+]] = arith.extsi [[IN]] : i16 to i32
// CHECK:       linalg.yield [[OP]] : i32
// CHECK:     [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK:     IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>

  }
}

// -----

module @SingleConvertExtUILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xui16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xui32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xui16>) -> tensor<1x1x1x1000xui32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xui16>) {
      %1 = IE.Convert(%arg1) {dstElemType = ui32} : tensor<1x1x1x1000xui16> -> tensor<1x1x1x1000xui32>
      IE.CGCYield %1 : tensor<1x1x1x1000xui32>
    } -> tensor<1x1x1x1000xui32>
    return %0 : tensor<1x1x1x1000xui32>

// CHECK-NOT: IE.Convert
// CHECK:     [[RET:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xui16> to tensor<1x1x1x1000xi16>
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RET]] : tensor<1x1x1x1000xi16>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK:     ^bb0([[ARG0:%.+]]: i16, {{%.+}}: i32):
// CHECK:       [[OP:%.+]] = arith.extui %{{.+}} : i16 to i32
// CHECK:       linalg.yield [[OP]] : i32
// CHECK:     [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xui32>
// CHECK:     IE.CGCYield [[RET]] : tensor<1x1x1x1000xui32>

  }
}

// -----

module @SingleConvertTruncILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xi32>) -> tensor<1x1x1x1000xi16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xi32>) {
      %1 = IE.Convert(%arg1) {dstElemType = i16} : tensor<1x1x1x1000xi32> -> tensor<1x1x1x1000xi16>
      IE.CGCYield %1 : tensor<1x1x1x1000xi16>
    } -> tensor<1x1x1x1000xi16>
    return %0 : tensor<1x1x1x1000xi16>

// CHECK-NOT: IE.Convert
// CHECK:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi16>
// CHECK:     [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG:%.+]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi16>) {
// CHECK:     ^bb0([[IN:%.+]]: i32, {{%.+}}: i16):
// CHECK:       [[OP:%.+]] = arith.trunci [[IN]] : i32 to i16
// CHECK:       linalg.yield [[OP]] : i16
// CHECK:     IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi16>
  }
} 

// -----
// IE.Abs 

module @SingleAbsFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Abs(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Abs
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
    // CHECK: [[ABS:%.+]] = math.absf [[IN]] : f16
    // CHECK: linalg.yield [[ABS]] : f16
    // CHECK: IE.CGCYield [[LINALG_OP]]  : tensor<1x1x1x1000xf16>
	}
}

// -----
// IE.Negative 

// CHECK: module @SingleNegativeFloatLayer
module @SingleNegativeFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Negative(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Negative
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
    // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK: [[NEG:%.+]] = arith.subf [[ZERO]], [[IN]] : f16
    // CHECK: linalg.yield [[NEG]] : f16
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
	}
}

// -----

// CHECK: module @SingleNegativeSI32Layer
module @SingleNegativeSI32Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Negative(%arg1) : tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
    IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

    // CHECK-NOT: IE.Negative
    // CHECK: [[BC_ARG:%.+]] = tensor.bitcast [[ARG:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[BC_ARG]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
    // CHECK: ^bb0([[IN:%.+]]: i32, [[OUT:%.+]]: i32):
    // CHECK: [[ZERO:%.+]] = arith.constant 0 : i32
    // CHECK: [[NEG:%.+]] = arith.subi [[ZERO]], [[IN]] : i32
    // CHECK: linalg.yield [[NEG]] : i32
    // CHECK: [[RES:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
    // CHECK: IE.CGCYield [[RES]] : tensor<1x1x1x1000xsi32>
	}
}

// -----
// IE.Sign  

module @SingleSignFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Sign(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Sign
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[NEGF_ONE:%.+]] = arith.constant -1.000000e+00 : f16
    // CHECK:   [[POSF_ONE:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK:   [[BITCAST:%.+]] = arith.bitcast [[IN]] : f16 to i16
    // CHECK:   [[CST_32768:%.+]] = arith.constant -32768 : i16
    // CHECK:   [[AND:%.+]] = arith.andi [[BITCAST]], [[CST_32768]] : i16
    // CHECK:   [[ONE:%.+]] = arith.constant 1 : i16
    // CHECK:   [[SHL:%.+]] = arith.shli [[BITCAST]], [[ONE]] : i16
    // CHECK:   [[INT_ZERO:%.+]] = arith.constant 0 : i16
    // CHECK:   [[CMP_EQ:%.+]] = arith.cmpi eq,  [[SHL]], [[INT_ZERO]] : i16
    // CHECK:   [[INT_ZERO2:%.+]] = arith.constant 0 : i16
    // CHECK:   [[CMP_NE:%.+]] = arith.cmpi ne, [[AND]], [[INT_ZERO2]] : i16
    // CHECK:   [[SELECT_1:%.+]] = arith.select [[CMP_NE]], [[NEGF_ONE]], [[POSF_ONE]] : f16
    // CHECK:   [[SELECT_2:%.+]] = arith.select [[CMP_EQ]], [[ZERO]], [[SELECT_1]] : f16
    // CHECK:   linalg.yield [[SELECT_2]] : f16
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
	}
}

// -----
// IE.HSwish  

module @SingleHSwishFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.HSwish(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.HSwish
    // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[THREE:%.+]] = arith.constant 3.000000e+00 : f16
    // CHECK:   [[SIX:%.+]] = arith.constant 6.000000e+00 : f16
    // CHECK:   [[DIV_CST:%.+]] = arith.constant 1.666260e-01 : f16
    // CHECK:   [[ADD:%.+]] = arith.addf %{{.+}}, [[THREE]] : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf [[ADD]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf [[MAX]], [[SIX]] fastmath<nnan,nsz> : f16
    // CHECK:   [[DIV:%.+]] = arith.mulf [[MIN]], [[DIV_CST]] : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[IN]], [[DIV]] : f16
    // CHECK:   linalg.yield [[MUL]] : f16
    // CHECK: IE.CGCYield [[LINALG_OP]]  : tensor<1x1x1x1000xf16>
    
	}
}

// -----
// IE.HSigmoid  

module @SingleHSigmoidFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.HSigmoid(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.HSigmoid
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs({{%.+}} : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[THREE:%.+]] = arith.constant 3.000000e+00 : f16
    // CHECK:   [[SIX:%.+]] = arith.constant 6.000000e+00 : f16
    // CHECK:   [[DIV_CST:%.+]] = arith.constant 1.666260e-01 : f16
    // CHECK:   [[ADD:%.+]] = arith.addf %{{.+}}, [[THREE]] : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf [[ADD]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf [[MAX]], [[SIX]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], [[DIV_CST]] : f16
    // CHECK:   linalg.yield [[MUL]] : f16
    // CHECK: IE.CGCYield [[LINALG_OP]]  : tensor<1x1x1x1000xf16>
    
	}
}

// -----
// IE.Gelu

module @SingleGeluFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Gelu(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Gelu
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs([[OUT:%.+]] : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, [[OUT:%.+]]: f16):

    // Constants with precise f16 values
    // CHECK: [[SQRT2_DIV_PI:%.+]] = arith.constant 7.978520e-01 : f16
    // CHECK: [[MUL_FITTING_CONST:%.+]] = arith.constant 3.567500e-02 : f16
    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK: [[HALF:%.+]] = arith.constant 5.000000e-01 : f16

    // Computation steps
    // CHECK: [[SQUARE:%.+]] = arith.mulf [[IN]], [[IN]] : f16
    // CHECK: [[SCALED_SQUARE:%.+]] = arith.mulf [[MUL_FITTING_CONST]], [[SQUARE]] : f16
    // CHECK: [[SCALED_BASE:%.+]] = arith.addf [[SQRT2_DIV_PI]], [[SCALED_SQUARE]] : f16
    // CHECK: [[SCALED_INPUT:%.+]] = arith.mulf [[IN]], [[SCALED_BASE]] : f16
    // CHECK: [[TANH:%.+]] = math.tanh [[SCALED_INPUT]] fastmath<afn> : f16
    // CHECK: [[TANH_PLUS_ONE:%.+]] = arith.addf [[ONE]], [[TANH]] : f16
    // CHECK: [[GELU_INTERMEDIATE:%.+]] = arith.mulf [[IN]], [[TANH_PLUS_ONE]] : f16
    // CHECK: [[RESULT:%.+]] = arith.mulf [[GELU_INTERMEDIATE]], [[HALF]] : f16

    // CHECK: linalg.yield [[RESULT]] : f16
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Elu

// CHECK-LABEL: @SingleEluFloatLayer
module @SingleEluFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      // Attribute 'alpha' is named 'x'
      %1 = IE.Elu(%arg1) {x = 1.0} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Elu
    // CHECK: linalg.generic
    // CHECK: ^bb0(%[[IN:.*]]: f16, %{{.*}}: f16):
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK-DAG: %[[ALPHA:.*]] = arith.constant 1.000000e+00 : f16
    // CHECK: %[[MIN:.*]] = arith.minimumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK: %[[MAX:.*]] = arith.maximumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK: %[[EXP:.*]] = math.exp %[[MIN]] fastmath<afn> : f16
    // CHECK: %[[EXP_MINUS_ONE:.*]] = arith.subf %[[EXP]], %[[ONE]] : f16
    // CHECK: %[[SCALED:.*]] = arith.mulf %[[ALPHA]], %[[EXP_MINUS_ONE]] : f16
    // CHECK: %[[RESULT:.*]] = arith.addf %[[MAX]], %[[SCALED]] : f16
    // CHECK: linalg.yield %[[RESULT]] : f16
  }
}

// -----
// IE.Selu

module @SingleSeluFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
      %1 = IE.Selu(%arg1) {
        alphaValue = 1.67326319 : f64,
        lambdaValue = 1.05084102 : f64,
        operandSegmentSizes = array<i32: 1, 0, 0>
      } : tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>
    IE.CGCYield %1 : tensor<1x1x1x1000xf32>
    } -> tensor<1x1x1x1000xf32>
    return %0 : tensor<1x1x1x1000xf32>

    // CHECK-NOT: IE.Selu
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf32>) outs([[OUT:%.+]] : tensor<1x1x1x1000xf32>) {
    // CHECK: ^bb0([[IN:%.+]]: f32, [[OUT:%.+]]: f32):
    // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32
    // CHECK: [[ALPHA:%.+]] = arith.constant 1.67326319 : f32
    // CHECK: [[LAMBDA:%.+]] = arith.constant 1.05084097 : f32
    // CHECK: [[MIN:%.+]] = arith.minimumf [[IN]], [[ZERO]] fastmath<nnan,nsz> : f32
    // CHECK: [[MAX:%.+]] = arith.maximumf [[IN]], [[ZERO]] fastmath<nnan,nsz> : f32
    // CHECK: [[EXP_NEG:%.+]] = math.exp [[MIN]] fastmath<afn> : f32
    // CHECK: [[EXP_MINUS_ONE:%.+]] = arith.subf [[EXP_NEG]], [[ONE]] : f32
    // CHECK: [[SCALED:%.+]] = arith.mulf [[ALPHA]], [[EXP_MINUS_ONE]] : f32
    // CHECK: [[COMBINED:%.+]] = arith.addf [[MAX]], [[SCALED]] : f32
    // CHECK: [[RESULT:%.+]] = arith.mulf [[LAMBDA]], [[COMBINED]] : f32
    // CHECK: linalg.yield [[RESULT]] : f32
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf32>
  }
}

// -----
// IE.PRelu

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @SinglePReluScalarSlope
module @SinglePReluScalarSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x4x5xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x4x5xf16>
  }

  func.func @main(%arg0: tensor<1x3x4x5xf16>) -> tensor<1x3x4x5xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x4x5xf16>) {
      %slope = arith.constant dense<0.250000e+00> : tensor<f16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x3x4x5xf16>, tensor<f16> -> tensor<1x3x4x5xf16>
    IE.CGCYield %1 : tensor<1x3x4x5xf16>
    } -> tensor<1x3x4x5xf16>
    return %0 : tensor<1x3x4x5xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu with 1D slope that doesn't match channel dimension

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @PRelu1DSlopeNonChannel
module @PRelu1DSlopeNonChannel {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x5x7xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x5x7xf16>
  }

  func.func @main(%arg0: tensor<1x3x5x7xf16>) -> tensor<1x3x5x7xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x5x7xf16>) {
      %slope = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]> : tensor<7xf16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x3x5x7xf16>, tensor<7xf16> -> tensor<1x3x5x7xf16>
    IE.CGCYield %1 : tensor<1x3x5x7xf16>
    } -> tensor<1x3x5x7xf16>
    return %0 : tensor<1x3x5x7xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK-SAME: ins({{.*}}tensor<1x3x5x7xf16>, tensor<7xf16>)
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu with 1D slope that matches channel dimension

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @PRelu1DSlopeChannel
module @PRelu1DSlopeChannel {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x5x7xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x5x7xf16>
  }

  func.func @main(%arg0: tensor<1x3x5x7xf16>) -> tensor<1x3x5x7xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x5x7xf16>) {
      %slope = arith.constant dense<[0.1, 0.2, 0.3]> : tensor<3xf16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x3x5x7xf16>, tensor<3xf16> -> tensor<1x3x5x7xf16>
    IE.CGCYield %1 : tensor<1x3x5x7xf16>
    } -> tensor<1x3x5x7xf16>
    return %0 : tensor<1x3x5x7xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK-SAME: ins({{.*}}tensor<1x3x5x7xf16>, tensor<3xf16>)
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu - 1D input with scalar slope

// CHECK: [[C:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @PRelu1DInputScalarSlope
module @PRelu1DInputScalarSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<128xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<128xf16>
  }

  func.func @main(%arg0: tensor<128xf16>) -> tensor<128xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<128xf16>) {
      %slope = arith.constant dense<0.100000e+00> : tensor<f16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<128xf16>, tensor<f16> -> tensor<128xf16>
    IE.CGCYield %1 : tensor<128xf16>
    } -> tensor<128xf16>
    return %0 : tensor<128xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel"]
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu - 2D input with 1D slope matching second dimension

// CHECK: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @PRelu2DInput1DSlope
module @PRelu2DInput1DSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<20x128xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<20x128xf16>
  }

  func.func @main(%arg0: tensor<20x128xf16>) -> tensor<20x128xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<20x128xf16>) {
      %slope = arith.constant dense<0.200000e+00> : tensor<128xf16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<20x128xf16>, tensor<128xf16> -> tensor<20x128xf16>
    IE.CGCYield %1 : tensor<20x128xf16>
    } -> tensor<20x128xf16>
    return %0 : tensor<20x128xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK-SAME: ins({{.*}}tensor<20x128xf16>, tensor<128xf16>)
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu - 4D input with 1D slope matching channel dimension

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @PRelu4DInput1DSlope
module @PRelu4DInput1DSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x20x128x128xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x20x128x128xf16>
  }

  func.func @main(%arg0: tensor<1x20x128x128xf16>) -> tensor<1x20x128x128xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x20x128x128xf16>) {
      %slope = arith.constant dense<0.300000e+00> : tensor<20xf16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x20x128x128xf16>, tensor<20xf16> -> tensor<1x20x128x128xf16>
    IE.CGCYield %1 : tensor<1x20x128x128xf16>
    } -> tensor<1x20x128x128xf16>
    return %0 : tensor<1x20x128x128xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK-SAME: ins({{.*}}tensor<1x20x128x128xf16>, tensor<20xf16>)
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK:   %[[MIN:.*]] = arith.minimumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   %[[MAX:.*]] = arith.maximumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   %[[MUL:.*]] = arith.mulf %[[MIN]], %[[SLOPE]] : f16
    // CHECK:   %[[ADD:.*]] = arith.addf %[[MAX]], %[[MUL]] : f16
    // CHECK:   linalg.yield %[[ADD]] : f16
  }
}

// -----
// IE.PRelu - 3D input with 1D slope

// CHECK: [[CHW:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @PRelu3DInput1DSlope
module @PRelu3DInput1DSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<4x5x6xf16>  // Dimensions: batch=4, channels=5, width=6
  } outputsInfo : {
    DataInfo "output" : tensor<4x5x6xf16>
  }

  func.func @main(%arg0: tensor<4x5x6xf16>) -> tensor<4x5x6xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<4x5x6xf16>) {
      %slope = arith.constant dense<0.250000e+00> : tensor<5xf16>  // Matches channel dimension (5)
      %1 = IE.PRelu(%arg1, %slope) : tensor<4x5x6xf16>, tensor<5xf16> -> tensor<4x5x6xf16>
    IE.CGCYield %1 : tensor<4x5x6xf16>
    } -> tensor<4x5x6xf16>
    return %0 : tensor<4x5x6xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
    // CHECK-SAME: ins({{.*}}tensor<4x5x6xf16>, tensor<5xf16>)
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK:   %[[MIN:.*]] = arith.minimumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   %[[MAX:.*]] = arith.maximumf %[[IN]], %[[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   %[[MUL:.*]] = arith.mulf %[[MIN]], %[[SLOPE]] : f16
    // CHECK:   %[[ADD:.*]] = arith.addf %[[MAX]], %[[MUL]] : f16
    // CHECK:   linalg.yield %[[ADD]] : f16
  }
}

// -----
// IE.PRelu - Different data types (f32)

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @PReluF32DataType
module @PReluF32DataType {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x4x5xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x4x5xf32>
  }

  func.func @main(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x3x4x5xf32>) {
      %slope = arith.constant dense<0.100000e+00> : tensor<f32>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x3x4x5xf32>, tensor<f32> -> tensor<1x3x4x5xf32>
    IE.CGCYield %1 : tensor<1x3x4x5xf32>
    } -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f32, %[[SLOPE:[a-zA-Z0-9_]*]]: f32, %{{.*}}: f32):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f32
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f32
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f32
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f32
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f32
    // CHECK:   linalg.yield [[ADD]] : f32
  }
}

// -----
// IE.PRelu - Edge case with negative values

// CHECK: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @PReluNegativeValues
module @PReluNegativeValues {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<2x2xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x2xf16>
  }

  func.func @main(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x2xf16>) {
      %slope = arith.constant dense<0.750000e+00> : tensor<f16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<2x2xf16>, tensor<f16> -> tensor<2x2xf16>
    IE.CGCYield %1 : tensor<2x2xf16>
    } -> tensor<2x2xf16>
    return %0 : tensor<2x2xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel"]
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.PRelu - Large slope value

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @PReluLargeSlope
module @PReluLargeSlope {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x3x3xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x3x3xf16>
  }

  func.func @main(%arg0: tensor<1x1x3x3xf16>) -> tensor<1x1x3x3xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x3x3xf16>) {
      %slope = arith.constant dense<2.500000e+00> : tensor<f16>
      %1 = IE.PRelu(%arg1, %slope) : tensor<1x1x3x3xf16>, tensor<f16> -> tensor<1x1x3x3xf16>
    IE.CGCYield %1 : tensor<1x1x3x3xf16>
    } -> tensor<1x1x3x3xf16>
    return %0 : tensor<1x1x3x3xf16>

    // CHECK-NOT: IE.PRelu
    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    // CHECK: ^bb0(%[[IN:[a-zA-Z0-9_]*]]: f16, %[[SLOPE:[a-zA-Z0-9_]*]]: f16, %{{.*}}: f16):
    // CHECK:   [[ZERO:%.+]] = arith.constant 0.000000e+00 : f16
    // CHECK:   [[MIN:%.+]] = arith.minimumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MAX:%.+]] = arith.maximumf %[[IN]], [[ZERO]] fastmath<nnan,nsz> : f16
    // CHECK:   [[MUL:%.+]] = arith.mulf [[MIN]], %[[SLOPE]] : f16
    // CHECK:   [[ADD:%.+]] = arith.addf [[MAX]], [[MUL]] : f16
    // CHECK:   linalg.yield [[ADD]] : f16
  }
}

// -----
// IE.SoftPlus (FP16)

module @SingleSoftPlusHalfLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.SoftPlus(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.SoftPlus
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs({{%.+}} : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):

    // CHECK: [[EXT:%.+]] = arith.extf [[IN]] : f16 to f32

    // CHECK: [[THRESHOLD:%.+]] = arith.constant 1.100000e+01 : f32
    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32

    // SoftPlus computation: log(1 + exp(input)) in FP32
    // CHECK: [[EXP:%.+]] = math.exp [[EXT]] fastmath<{{.*}}afn{{.*}}> : f32
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f32
    // CHECK: [[SOFTPLUS_RESULT:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<{{.*}}afn{{.*}}> : f32

    // CHECK: [[USE_LINEAR:%.+]] = arith.cmpf olt, [[EXT]], [[THRESHOLD]] : f32
    // CHECK: [[RESULT_F32:%.+]] = arith.select [[USE_LINEAR]], [[SOFTPLUS_RESULT]], [[EXT]] : f32

    // CHECK: [[TRUNC:%.+]] = arith.truncf [[RESULT_F32]] : f32 to f16
    // CHECK: linalg.yield [[TRUNC]] : f16
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.SoftPlus (FP32)

module @SingleSoftPlusFloat32Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf32>) -> tensor<1x1x1x1000xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf32>) {
      %1 = IE.SoftPlus(%arg1) : tensor<1x1x1x1000xf32> -> tensor<1x1x1x1000xf32>
    IE.CGCYield %1 : tensor<1x1x1x1000xf32>
    } -> tensor<1x1x1x1000xf32>
    return %0 : tensor<1x1x1x1000xf32>

    // CHECK-NOT: IE.SoftPlus
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf32>) outs({{%.+}} : tensor<1x1x1x1000xf32>) {
    // CHECK: ^bb0([[IN:%.+]]: f32, {{%.+}}: f32):

    // CHECK: [[THRESHOLD:%.+]] = arith.constant 2.000000e+01 : f32
    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32

    // SoftPlus computation: log(1 + exp(input))
    // CHECK: [[EXP:%.+]] = math.exp [[IN]] fastmath<{{.*}}afn{{.*}}> : f32
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f32
    // CHECK: [[SOFTPLUS_RESULT:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<{{.*}}afn{{.*}}> : f32

    // CHECK: [[USE_LINEAR:%.+]] = arith.cmpf olt, [[IN]], [[THRESHOLD]] : f32
    // CHECK: [[RESULT:%.+]] = arith.select [[USE_LINEAR]], [[SOFTPLUS_RESULT]], [[IN]] : f32

    // CHECK: linalg.yield [[RESULT]] : f32
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf32>
  }
}

// -----
// IE.SoftPlus (FP16 big values)

module @SingleSoftPlusBigValuesLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<16x128x256x512xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<16x128x256x512xf16>
  }

  func.func @main(%arg0: tensor<16x128x256x512xf16>) -> tensor<16x128x256x512xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<16x128x256x512xf16>) {
      %1 = IE.SoftPlus(%arg1) : tensor<16x128x256x512xf16> -> tensor<16x128x256x512xf16>
    IE.CGCYield %1 : tensor<16x128x256x512xf16>
    } -> tensor<16x128x256x512xf16>
    return %0 : tensor<16x128x256x512xf16>

    // CHECK-NOT: IE.SoftPlus
    // CHECK: linalg.generic
    // CHECK: [[EXT:%.+]] = arith.extf {{%.+}} : f16 to f32
    // CHECK: [[THRESHOLD:%.+]] = arith.constant 1.100000e+01 : f32
    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32
    // CHECK: [[EXP:%.+]] = math.exp [[EXT]] fastmath<{{.*}}afn{{.*}}> : f32
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f32
    // CHECK: [[SOFTPLUS_RESULT:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<{{.*}}afn{{.*}}> : f32
    // CHECK: [[USE_LINEAR:%.+]] = arith.cmpf olt, [[EXT]], [[THRESHOLD]] : f32
    // CHECK: [[RESULT_F32:%.+]] = arith.select [[USE_LINEAR]], [[SOFTPLUS_RESULT]], [[EXT]] : f32
    // CHECK: [[TRUNC:%.+]] = arith.truncf [[RESULT_F32]] : f32 to f16
  }
}

// -----
// IE.Mish

module @SingleMishFloatLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.Mish(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

    // CHECK-NOT: IE.Mish
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x1x1x1000xf16>) outs({{%.+}} : tensor<1x1x1x1000xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):

    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f16

    // Computation steps for Mish(x) = x * tanh(ln(1 + exp(x)))
    // CHECK: [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f16
    // CHECK: [[LOG:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<afn> : f16
    // CHECK: [[TANH:%.+]] = math.tanh [[LOG]] fastmath<afn> : f16
    // CHECK: [[RESULT:%.+]] = arith.mulf [[IN]], [[TANH]] : f16

    // CHECK: linalg.yield [[RESULT]] : f16
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Mish with f32

module @SingleMishFloat32Layer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x2x3x4xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x2x3x4xf32>
  }

  func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x2x3x4xf32>) {
      %1 = IE.Mish(%arg1) : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32>
    IE.CGCYield %1 : tensor<1x2x3x4xf32>
    } -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>

    // CHECK-NOT: IE.Mish
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<1x2x3x4xf32>) outs({{%.+}} : tensor<1x2x3x4xf32>) {
    // CHECK: ^bb0([[IN:%.+]]: f32, {{%.+}}: f32):

    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32
    // CHECK: [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f32
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f32
    // CHECK: [[LOG:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<afn> : f32
    // CHECK: [[TANH:%.+]] = math.tanh [[LOG]] fastmath<afn> : f32
    // CHECK: [[RESULT:%.+]] = arith.mulf [[IN]], [[TANH]] : f32

    // CHECK: linalg.yield [[RESULT]] : f32
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<1x2x3x4xf32>
  }
}

// -----
// IE.Mish with 2D tensor

module @SingleMish2DLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<10x20xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<10x20xf16>
  }

  func.func @main(%arg0: tensor<10x20xf16>) -> tensor<10x20xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<10x20xf16>) {
      %1 = IE.Mish(%arg1) : tensor<10x20xf16> -> tensor<10x20xf16>
    IE.CGCYield %1 : tensor<10x20xf16>
    } -> tensor<10x20xf16>
    return %0 : tensor<10x20xf16>

    // CHECK-NOT: IE.Mish
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<10x20xf16>) outs({{%.+}} : tensor<10x20xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):

    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK: [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f16
    // CHECK: [[LOG:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<afn> : f16
    // CHECK: [[TANH:%.+]] = math.tanh [[LOG]] fastmath<afn> : f16
    // CHECK: [[RESULT:%.+]] = arith.mulf [[IN]], [[TANH]] : f16

    // CHECK: linalg.yield [[RESULT]] : f16
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<10x20xf16>
  }
}

// -----
// IE.Mish with 5D tensor

module @SingleMish5DLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<2x3x4x5x6xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<2x3x4x5x6xf16>
  }

  func.func @main(%arg0: tensor<2x3x4x5x6xf16>) -> tensor<2x3x4x5x6xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<2x3x4x5x6xf16>) {
      %1 = IE.Mish(%arg1) : tensor<2x3x4x5x6xf16> -> tensor<2x3x4x5x6xf16>
    IE.CGCYield %1 : tensor<2x3x4x5x6xf16>
    } -> tensor<2x3x4x5x6xf16>
    return %0 : tensor<2x3x4x5x6xf16>

    // CHECK-NOT: IE.Mish
    // CHECK: [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [#[[MAP:.+]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins([[ARG0:%.+]] : tensor<2x3x4x5x6xf16>) outs({{%.+}} : tensor<2x3x4x5x6xf16>) {
    // CHECK: ^bb0([[IN:%.+]]: f16, {{%.+}}: f16):

    // CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f16
    // CHECK: [[EXP:%.+]] = math.exp [[IN]] fastmath<afn> : f16
    // CHECK: [[ONE_PLUS_EXP:%.+]] = arith.addf [[ONE]], [[EXP]] : f16
    // CHECK: [[LOG:%.+]] = math.log [[ONE_PLUS_EXP]] fastmath<afn> : f16
    // CHECK: [[TANH:%.+]] = math.tanh [[LOG]] fastmath<afn> : f16
    // CHECK: [[RESULT:%.+]] = arith.mulf [[IN]], [[TANH]] : f16

    // CHECK: linalg.yield [[RESULT]] : f16
    // CHECK: }
    // CHECK: IE.CGCYield [[LINALG_OP]] : tensor<2x3x4x5x6xf16>
  }
}
