//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-layers-to-math %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

// IE.AndOp

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @AndILayer
module @AndILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.And(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.And
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpi ne, [[LHS]], [[ZLHS]] : i32
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpi ne, [[RHS]], [[ZRHS]] : i32
// CHECK-NEXT:      [[RES_B:%.+]] = arith.andi [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.extui [[RES_B]] : i1 to i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @AndFPLayer
module @AndFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.And(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.And
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpf one, [[LHS]], [[ZLHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpf one, [[RHS]], [[ZRHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[RES_B:%.+]] = arith.andi [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.uitofp [[RES_B]] : i1 to f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.LogicalXorOp

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalXorILayer
module @LogicalXorILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.LogicalXor(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.LogicalXor
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpi ne, [[LHS]], [[ZLHS]] : i32
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpi ne, [[RHS]], [[ZRHS]] : i32
// CHECK-NEXT:      [[RES_B:%.+]] = arith.xori [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.extui [[RES_B]] : i1 to i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalXorFPLayer
module @LogicalXorFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.LogicalXor(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.LogicalXor
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpf one, [[LHS]], [[ZLHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpf one, [[RHS]], [[ZRHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[RES_B:%.+]] = arith.xori [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.uitofp [[RES_B]] : i1 to f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.LogicalOrOp

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalOrILayer
module @LogicalOrILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xsi32>, %arg1 as %arg3: tensor<1x1x1x1000xsi32>) {
      %1 = IE.LogicalOr(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.LogicalOr
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpi ne, [[LHS]], [[ZLHS]] : i32
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpi ne, [[RHS]], [[ZRHS]] : i32
// CHECK-NEXT:      [[RES_B:%.+]] = arith.ori [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.extui [[RES_B]] : i1 to i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalOrFPLayer
module @LogicalOrFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg2: tensor<1x1x1x1000xf16>, %arg1 as %arg3: tensor<1x1x1x1000xf16>) {
      %1 = IE.LogicalOr(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.LogicalOr
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpf one, [[LHS]], [[ZLHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[ZRHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BRHS:%.+]] = arith.cmpf one, [[RHS]], [[ZRHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[RES_B:%.+]] = arith.ori [[BLHS]], [[BRHS]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.uitofp [[RES_B]] : i1 to f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.LogicalNotOp

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalNotILayer
module @LogicalNotILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xsi32>) {
      %1 = IE.LogicalNot(%arg1) : tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.LogicalNot
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[LHS_BC]] : tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpi ne, [[LHS]], [[ZLHS]] : i32
// CHECK-NEXT:      [[TRUE:%.+]] = arith.constant true
// CHECK-NEXT:      [[RES_B:%.+]] = arith.xori [[BLHS]], [[TRUE]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.extui [[RES_B]] : i1 to i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @LogicalNotFPLayer
module @LogicalNotFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %1 = IE.LogicalNot(%arg1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.LogicalNot
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}} : tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[LHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZLHS:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BLHS:%.+]] = arith.cmpf one, [[LHS]], [[ZLHS]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[TRUE:%.+]] = arith.constant true
// CHECK-NEXT:      [[RES_B:%.+]] = arith.xori [[BLHS]], [[TRUE]] : i1
// CHECK-NEXT:      [[RES:%.+]] = arith.uitofp [[RES_B]] : i1 to f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----
// IE.Select

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @SelectILayer
module @SelectILayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xsi32>
    DataInfo "input2" : tensor<1x1x1x1000xsi32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xsi32>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xsi32>, %arg2: tensor<1x1x1x1000xsi32>) -> tensor<1x1x1x1000xsi32> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg3: tensor<1x1x1x1000xsi32>, %arg1 as %arg4: tensor<1x1x1x1000xsi32>, %arg2 as %arg5: tensor<1x1x1x1000xsi32>) {
      %1 = IE.Select(%arg3, %arg4, %arg5) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xsi32> -> tensor<1x1x1x1000xsi32>
      IE.CGCYield %1 : tensor<1x1x1x1000xsi32>
    } -> tensor<1x1x1x1000xsi32>
    return %0 : tensor<1x1x1x1000xsi32>

// CHECK-NOT:     IE.Select
// CHECK-DAG:     [[COND_BC:%.+]] = tensor.bitcast [[COND:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[LHS_BC:%.+]] = tensor.bitcast [[LHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[RHS_BC:%.+]] = tensor.bitcast [[RHS:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-DAG:     [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xi32>
// CHECK:         [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[COND_BC]], [[LHS_BC]], [[RHS_BC]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xi32>) outs([[EMPTY]] : tensor<1x1x1x1000xi32>) {
// CHECK-NEXT:    ^bb0([[COND:%.+]]: i32, [[LHS:%.+]]: i32, [[RHS:%.+]]: i32, {{%.+}}: i32):
// CHECK-NEXT:      [[ZCOND:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BCOND:%.+]] = arith.cmpi ne, [[COND]], [[ZCOND]] : i32
// CHECK-NEXT:      [[RES:%.+]] = arith.select [[BCOND]], [[LHS]], [[RHS]] : i32
// CHECK-NEXT:      linalg.yield [[RES]] : i32
// CHECK-NEXT:    } -> tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[RET:%.+]] = tensor.bitcast [[LINALG_OP]] : tensor<1x1x1x1000xi32> to tensor<1x1x1x1000xsi32>
// CHECK-NEXT:    IE.CGCYield [[RET]] : tensor<1x1x1x1000xsi32>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @SelectFPLayer
module @SelectFPLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
    DataInfo "input2" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>, %arg2: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg3: tensor<1x1x1x1000xf16>, %arg1 as %arg4: tensor<1x1x1x1000xf16>, %arg2 as %arg5: tensor<1x1x1x1000xf16>) {
      %1 = IE.Select(%arg3, %arg4, %arg5) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Select
// CHECK:         [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{%.+}}, {{%.+}}, {{%.+}} : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[COND:%.+]]: f16, [[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZCOND:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      [[BCOND:%.+]] = arith.cmpf one, [[COND]], [[ZCOND]] fastmath<nnan,nsz> : f16
// CHECK-NEXT:      [[RES:%.+]] = arith.select [[BCOND]], [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}

// -----

// CHECK: [[NCHW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: @MixedSelectLayer
module @MixedSelectLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x1x1000xsi32>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
    DataInfo "input2" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xsi32>, %arg1: tensor<1x1x1x1000xf16>, %arg2: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %0 = IE.CodeGenCapsule inputs(%arg0 as %arg3: tensor<1x1x1x1000xsi32>, %arg1 as %arg4: tensor<1x1x1x1000xf16>, %arg2 as %arg5: tensor<1x1x1x1000xf16>) {
      %1 = IE.Select(%arg3, %arg4, %arg5) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x1x1000xsi32>, tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
      IE.CGCYield %1 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %0 : tensor<1x1x1x1000xf16>

// CHECK-NOT:     IE.Select
// CHECK:         [[COND_BC:%.+]] = tensor.bitcast [[COND:%.+]] : tensor<1x1x1x1000xsi32> to tensor<1x1x1x1000xi32>
// CHECK-NEXT:    [[EMPTY:%.+]] = tensor.empty() : tensor<1x1x1x1000xf16>
// CHECK-NEXT:    [[LINALG_OP:%.+]] = linalg.generic {indexing_maps = [[[NCHW]], [[NCHW]], [[NCHW]], [[NCHW]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[COND_BC]], [[LHS:%.+]], [[RHS:%.+]] : tensor<1x1x1x1000xi32>, tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16>) outs([[EMPTY]] : tensor<1x1x1x1000xf16>) {
// CHECK-NEXT:    ^bb0([[COND:%.+]]: i32, [[LHS:%.+]]: f16, [[RHS:%.+]]: f16, {{%.+}}: f16):
// CHECK-NEXT:      [[ZCOND:%.+]] = arith.constant 0 : i32
// CHECK-NEXT:      [[BCOND:%.+]] = arith.cmpi ne, [[COND]], [[ZCOND]] : i32
// CHECK-NEXT:      [[RES:%.+]] = arith.select [[BCOND]], [[LHS]], [[RHS]] : f16
// CHECK-NEXT:      linalg.yield [[RES]] : f16
// CHECK-NEXT:    } -> tensor<1x1x1x1000xf16>
// CHECK-NEXT:    IE.CGCYield [[LINALG_OP]] : tensor<1x1x1x1000xf16>
  }
}
