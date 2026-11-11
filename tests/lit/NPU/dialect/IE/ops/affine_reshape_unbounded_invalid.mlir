//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// An unbounded dim has no element count, so it cannot absorb a static extent.

func.func @UnboundedMergedWithStatic(%arg0: tensor<?x64xf16>) -> tensor<?xf16> {
    // expected-error@+1 {{AffineReshape of an unbounded shape must keep every dynamic dim and the static volume}}
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0]], shape_value = [-9223372036854775808]} : tensor<?x64xf16> -> tensor<?xf16>
    return %0 : tensor<?xf16>
}
