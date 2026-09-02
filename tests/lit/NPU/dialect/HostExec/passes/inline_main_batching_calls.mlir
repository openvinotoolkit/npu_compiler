//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --inline-main-batching-calls %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// -----

// Verify that a call to a main_batching* function is inlined into the caller,
// the caller is marked with HostCompileInferenceExec and disable_pipelined_cmdlist_recording,
// and the now-dead callee is erased.
module @InlineMainBatching {
    func.func @main_batching_func(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16> {
        return %arg1 : memref<1x3x224x224xf16>
    }

    func.func @main(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16> {
        %0 = func.call @main_batching_func(%arg0, %arg1) : (memref<1x3x224x224xf16>, memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16>
        return %0 : memref<1x3x224x224xf16>
    }
}

// CHECK-LABEL: module @InlineMainBatching
// The callee is erased after inlining.
// CHECK-NOT: func.func @main_batching_func
// The caller gains the two attributes.
// CHECK: func.func @main
// CHECK-SAME: HostExec.HostCompileInferenceExec
// CHECK-SAME: disable_pipelined_cmdlist_recording
// The body of main_batching_func is inlined: the return of %arg1 becomes a direct return.
// CHECK:   return %{{.+}} : memref<1x3x224x224xf16>

// -----

// Verify that functions whose names do not start with "main_batching" are left untouched.
module @NoInlineUnrelatedFunc {
    func.func @helper_func(%arg0: memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16> {
        return %arg0 : memref<1x3x224x224xf16>
    }

    func.func @main(%arg0: memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16> {
        %0 = func.call @helper_func(%arg0) : (memref<1x3x224x224xf16>) -> memref<1x3x224x224xf16>
        return %0 : memref<1x3x224x224xf16>
    }
}

// CHECK-LABEL: module @NoInlineUnrelatedFunc
// The unrelated callee is preserved.
// CHECK: func.func @helper_func
// The caller is not attributed.
// CHECK: func.func @main
// CHECK-NOT: HostExec.HostCompileInferenceExec
// CHECK-NOT: disable_pipelined_cmdlist_recording
