//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --resolve-shaped-type-result-dims --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!InType = tensor<1x160x?xf32, {bounds = #const.OpaqueI64Elements<[1, 160, 64]> : tensor<3xsi64>, order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>}>
!OutType = tensor<1x160x?x1x1xf32, {bounds = #const.OpaqueI64Elements<[1, 160, 64, 1, 1]> : tensor<5xsi64>, order = #NCHW}>

// CHECK-LABEL: @DuplicateUnsqueezeAxes
func.func @DuplicateUnsqueezeAxes(%input: !InType) -> index {
    %axis = arith.constant 2 : index

    // expected-error@+2 {{Axes must be unique}}
    // expected-error@+1 {{'IE.Unsqueeze' op failed to infer returned types}}
    %unsqueezed = IE.Unsqueeze(%input) {axes_value = [3, 3]} : !InType -> !OutType
    %dim = tensor.dim %unsqueezed, %axis : !OutType

    return %dim : index
}
