//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_INFO vpux-opt  --split-input-file --platform=%platform% --setup-pass-tools="enable-outdated-pass-detection=true" --set-memory-space=memory-space=DDR -o /dev/null %s 2>&1 | FileCheck %s
// REQUIRES: dev-build && (platform-NPU3720 || platform-NPU4000 || platform-NPU5010)

module @net {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x4x4xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x4x4xf16>
    }

    func.func @main(%input: memref<1x16x4x4xf16>, %output: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
        return %output : memref<1x16x4x4xf16>
    }

    // CHECK: [pass-usage-observer] SetMemorySpace        CHANGED
}

// -----

module @net {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x4x4xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x4x4xf16>
    }

    func.func @main(%input: memref<1x16x4x4xf16, @DDR>, %output: memref<1x16x4x4xf16, @DDR>) -> memref<1x16x4x4xf16, @DDR> {
        return %output : memref<1x16x4x4xf16, @DDR>
    }

    // CHECK: [pass-usage-observer] SetMemorySpace        SAME
}
