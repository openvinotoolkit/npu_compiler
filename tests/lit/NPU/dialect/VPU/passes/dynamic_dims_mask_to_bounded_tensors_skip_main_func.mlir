//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile" --dynamic-dims-mask-to-bounded-tensors %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @EmptyFunction
module @EmptyFunction {
  net.NetworkInfo entryPoint : @EmptyFunction
  inputsInfo : {
    DataInfo "input" : tensor<?x?x64xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<?x?x64xf16>
  }

  func.func @EmptyFunction(%arg0: tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>)
        -> tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}> {
      return %arg0 : tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>
  }

  // CHECK: func.func [[EMPTY_FUNC:@.+]]([[_:%.+]]: tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>)
  // CHECK-SAME: -> tensor<32x32x64xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[1, 1, 0]> : tensor<3xsi64>, order = #CHW}>
}
