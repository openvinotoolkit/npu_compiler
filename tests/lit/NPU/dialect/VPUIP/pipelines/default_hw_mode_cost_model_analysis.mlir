//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_DEBUG vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="function-outlining='naive'" %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!MemRef = memref<1x3x62x62xf16>

// actual checks:
// CHECK: Load preserved cost model
// CHECK-NOT: Create new cost model instance

// simply make sure that log messages are printed:
// CHECK: [dump-statistics-of-task-ops]

module @ChainCalls {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf16>
    }
    func.func nested @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
        %0 = VPUIP.Copy inputs(%in: !MemRef) outputs(%out: !MemRef) -> !MemRef
        return %0 : !MemRef
    }

    func.func @main(%arg0: !MemRef, %arg1: !MemRef) -> !MemRef {
        %alloc = memref.alloc() : !MemRef
        %alloc2 = memref.alloc() : !MemRef
        %0 = func.call @foo(%arg0, %alloc) : (!MemRef, !MemRef) -> !MemRef
        %1 = func.call @foo(%0, %alloc2) : (!MemRef, !MemRef) -> !MemRef
        %out = VPUIP.Copy inputs(%1: !MemRef) outputs(%arg1: !MemRef) -> !MemRef
        return %out : !MemRef
    }
}
