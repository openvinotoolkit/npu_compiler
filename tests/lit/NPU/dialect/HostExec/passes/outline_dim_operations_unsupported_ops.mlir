//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --platform=%platform% --outline-dim-operations %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Verify that OutlineDimOperationsPass fails when IE dialect operations appear in the
// output shape computation chain. This happens when an IE op does not implement
// reifyResultShapes, leaving tensor.dim ops unresolved.

!NonZeroOut = tensor<2x?xsi64, {bounds = #const.OpaqueI64Elements<[2, 16]> : tensor<2xsi64>, order = affine_map<(d0, d1) -> (d0, d1)>}>

module @unresolvedIEOpInShapeComputation {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<4x4xf16>
  } outputsInfo : {
    DataInfo "output" : !NonZeroOut
    DataInfo "out_0" : tensor<2xi64>
  }

  func.func @main(%arg: tensor<4x4xf16>) -> (!NonZeroOut, tensor<2xi64>) {
    %nonzero = IE.NonZero(%arg) {dstElemType = si64} : tensor<4x4xf16> -> !NonZeroOut

    %c1 = arith.constant 1 : index
    %c2_i64 = arith.constant 2 : i64
    %dim = tensor.dim %nonzero, %c1 : !NonZeroOut
    %idx = arith.index_cast %dim : index to i64
    %shape = tensor.from_elements %c2_i64, %idx : tensor<2xi64>

    return %nonzero, %shape : !NonZeroOut, tensor<2xi64>
  }

  // CHECK: Found 1 unreified IE op type(s) in @output_shape chain: IE.NonZero
}
