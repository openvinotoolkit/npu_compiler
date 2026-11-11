//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --construct-ws-analysis --introduce-init-function="ws-extraction-mode=gen-init" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Note: this tests that empty init is rejected by the pass - otherwise - in
// gen-init - a no-input, no-output entry point is generated

{-#
    dialect_resources: {
        builtin: {
            some_other_origin: "0x10000000AABBCCDD"
        }
    }
#-}

// expected-error@+1 {{Cannot generate empty init schedule}}
module @NoConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        // Note: name doesn't start with "ov" and thus constant is ignored
        %cst1 = const.Declare tensor<4xui8> = dense_resource<some_other_origin> : tensor<4xui8>, [#const.Add<1.0>]
        return %arg : tensor<4x16xf16>
    }
}
