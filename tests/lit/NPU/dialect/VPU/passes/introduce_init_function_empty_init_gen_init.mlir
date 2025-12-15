//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" --verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
