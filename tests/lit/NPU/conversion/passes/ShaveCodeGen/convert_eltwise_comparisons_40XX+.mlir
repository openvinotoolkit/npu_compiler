//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// IE.Equal

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleEqualSILayer
module @SingleEqualSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Equal(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Equal
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi eq, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleEqualFPLayer
module @SingleEqualFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Equal(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Equal
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf oeq, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----
// IE.NotEqual

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleNotEqualSILayer
module @SingleNotEqualSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.NotEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.NotEqual
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi ne, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleNotEqualFPLayer
module @SingleNotEqualFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.NotEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.NotEqual
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf one, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----
// IE.Less

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleLessSILayer
module @SingleLessSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Less(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Less
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi slt, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleLessFPLayer
module @SingleLessFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Less(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Less
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf olt, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----
// IE.LessEqual

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleLessEqualSILayer
module @SingleLessEqualSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.LessEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.LessEqual
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi sle, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleLessEqualFPLayer
module @SingleLessEqualFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.LessEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.LessEqual
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf ole, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----
// IE.Greater

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleGreaterSILayer
module @SingleGreaterSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Greater(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Greater
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi sgt, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleGreaterFPLayer
module @SingleGreaterFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xi8>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xi8> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.Greater(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xi8>
      IE.CGCYield %1 : tensor<1x1x1x1000xi8>
    } -> tensor<1x1x1x1000xi8>
    return %0 : tensor<1x1x1x1000xi8>

// CHECK-NOT:     IE.Greater
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi8>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xi8>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: i8):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf ogt, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i8
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i8
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi8>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xi8>
  }
}

// -----
// IE.GreaterEqual

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleGreaterEqualSILayer
module @SingleGreaterEqualSILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.GreaterEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.GreaterEqual
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpi sge, [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.extui [[OP]] : i1 to i32
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: SingleGreaterEqualFPLayer
module @SingleGreaterEqualFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.GreaterEqual(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.GreaterEqual
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[OP:%.+]] = arith.cmpf oge, [[LHS]], [[RHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[EXT_OP:%.+]] = arith.uitofp [[OP]] : i1 to f16
// CHECK-NEXT:      linalg.yield [[EXT_OP]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}
