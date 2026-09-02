//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --forbid-four-bit-outputs --verify-diagnostics %s
// REQUIRES: platform-NPU3720

module @I4 {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x1x1x1000xi4>
    } outputsInfo : {
        // expected-error@+1 {{Network has 4-bit output, which is not yet supported}}
        DataInfo "output" : tensor<1x1x1x1000xi4>
    }
    func.func @main(%arg0: tensor<1x1x1x1000xi4>) -> tensor<1x1x1x1000xi4> {
        return %arg0 : tensor<1x1x1x1000xi4>
    }
}

// -----

module @SI4 {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x1x1x1000xsi4>
    } outputsInfo : {
        // expected-error@+1 {{Network has 4-bit output, which is not yet supported}}
        DataInfo "output" : tensor<1x1x1x1000xsi4>
    }
    func.func @main(%arg0: tensor<1x1x1x1000xsi4>) -> tensor<1x1x1x1000xsi4> {
        return %arg0 : tensor<1x1x1x1000xsi4>
    }
}

// -----

module @UI4 {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x1x1x1000xui4>
    } outputsInfo : {
        // expected-error@+1 {{Network has 4-bit output, which is not yet supported}}
        DataInfo "output" : tensor<1x1x1x1000xui4>
    }
    func.func @main(%arg0: tensor<1x1x1x1000xui4>) -> tensor<1x1x1x1000xui4> {
        return %arg0 : tensor<1x1x1x1000xui4>
    }
}

// -----

module @F16 {
    net.NetworkInfo entryPoint : @main inputsInfo :  {
        DataInfo "input" : tensor<1x1x1x1000xf16>
    } outputsInfo : {
        // No error expected
        DataInfo "output" : tensor<1x1x1x1000xf16>
    }
    func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
        return %arg0 : tensor<1x1x1x1000xf16>
    }
}
