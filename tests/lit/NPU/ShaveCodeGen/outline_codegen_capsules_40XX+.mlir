//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize --outline-codegen-capsules %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %capsule = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x1x1000xf16>) {
      %0 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x1x1000xf16>) outs(%arg1 : tensor<1x1x1x1000xf16>) {
      ^bb0(%in: f16, %out: f16):
        %1 = math.cos %in : f16
        linalg.yield %1 : f16
      } -> tensor<1x1x1x1000xf16>
      IE.CGCYield %0 : tensor<1x1x1x1000xf16>
    } -> tensor<1x1x1x1000xf16>
    return %capsule : tensor<1x1x1x1000xf16>
  }
}
  // CHECK: module @SingleCosLayer
  // CHECK: module @VPU.SW
  // CHECK: func.func @generated_0
  // CHECK-SAME: [[VAR0:%.+]]: tensor<1x1x1x1000xf16>,
  // CHECK-SAME: [[VAR1:%.+]]: tensor<1x1x1x1000xf16>
  // CHECK-SAME: -> tensor<1x1x1x1000xf16>

  // CHECK: [[COMPUTATION_RESULT:%.+]] = linalg.generic
  // CHECK-SAME: ins([[VAR0]]
  // CHECK-SAME: outs([[VAR1]]
  // CHECK: [[SCALAR_COS_RES:%.+]] = math.cos
  // CHECK: linalg.yield [[SCALAR_COS_RES]]

  // CHECK: return [[COMPUTATION_RESULT]] : tensor<1x1x1x1000xf16>

  // CHECK: func.func @main
  // CHECK: [[GENERIC_SW_LAYER_RES:%.+]] = VPU.GenericSwLayer
  // CHECK-SAME: callee = @VPU.SW::@generated_0
  // CHECK-SAME: tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
  // CHECK: return [[GENERIC_SW_LAYER_RES]] : tensor<1x1x1x1000xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @TestIsolateFromAbove {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" : tensor<1x1x256x56xsi32>
  } outputsInfo : {
    DataInfo "output0" : tensor<1x1x256x56xsi32>
  }

  func.func @main(%arg0: tensor<1x1x256x56xsi32>) -> tensor<1x1x256x56xsi32> {
    %capsule = IE.CodeGenCapsule inputs(%arg0 as %arg1: tensor<1x1x256x56xsi32>) {
      %ct = arith.constant -1 : i32
      %1 = tensor.bitcast %arg1 : tensor<1x1x256x56xsi32> to tensor<1x1x256x56xi32>
      %2 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x256x56xi32>) outs(%1 : tensor<1x1x256x56xi32>) {
      ^bb0(%in: i32, %out: i32):
        %5 = arith.xori %in, %ct : i32
        linalg.yield %5 : i32
      } -> tensor<1x1x256x56xi32>
      %3 = tensor.bitcast %2 : tensor<1x1x256x56xi32> to tensor<1x1x256x56xsi32>
      IE.CGCYield %3 : tensor<1x1x256x56xsi32>
    } -> tensor<1x1x256x56xsi32>
    return %capsule : tensor<1x1x256x56xsi32>
  }
}

  // CHECK: module @TestIsolateFromAbove
  // CHECK: module @VPU.SW
  // CHECK: func.func @generated_0
  // CHECK-SAME: [[VAR0:%.+]]: tensor<1x1x256x56xi32>,
  // CHECK-SAME: [[VAR1:%.+]]: tensor<1x1x256x56xi32>
  // CHECK-SAME: -> tensor<1x1x256x56xi32>
  // CHECK: [[CONST:%.+]] = arith.constant -1 : i32
  // CHECK: [[COMPUTATION_RESULT:%.+]] = linalg.generic
  // CHECK-SAME: ins([[VAR0]]
  // CHECK-SAME: outs([[VAR1]]
  // CHECK: [[SCALAR_RES:%.+]] = arith.xori {{.*}}, [[CONST]] : i32
  // CHECK: linalg.yield [[SCALAR_RES]]

  // CHECK: func.func @main
  // CHECK-NOT: arith.constant
  // CHECK-NOT: tensor.bitcast
  // CHECK: [[GENERIC_SW_LAYER_RES:%.+]] = VPU.GenericSwLayer
  // CHECK-SAME: callee = @VPU.SW::@generated_0
  // CHECK-SAME: tensor<1x1x256x56xsi32> -> tensor<1x1x256x56xsi32>
  // CHECK-NOT: tensor.bitcast
  // CHECK: return [[GENERIC_SW_LAYER_RES]] : tensor<1x1x256x56xsi32>
