//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --linearize-call-ops %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ParallelCallOps
module @ParallelCallOps {

func.func private @function1(%arg0: memref<10xf16>) -> memref<10xf16> {
    %alloc = memref.alloc() : memref<10xf16>
    %t0, %f0 = async.execute -> (!async.value<memref<10xf16>>) {
        %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<10xf16>) outputs(%alloc : memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }
    return %alloc : memref<10xf16>
}

func.func private @function2(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<20xf16> {
    %alloc = memref.alloc() : memref<20xf16>
    %concat = VPUIP.ConcatView inputs(%arg0, %arg1 : memref<10xf16>, memref<10xf16>) outputs(%alloc : memref<20xf16>) -> memref<20xf16>
    return %concat : memref<20xf16>
}

func.func @main(%input: memref<10xf16>) -> memref<20xf16> {
    %t0, %f0 = async.execute -> (!async.value<memref<10xf16>>) {
        %0 = func.call @function1(%input) : (memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }
    %t1, %f1 = async.execute [%t0] (%f0 as %arg: !async.value<memref<10xf16>>) -> (!async.value<memref<10xf16>>) {
        %0 = func.call @function1(%arg) : (memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }
    %t2, %f2 = async.execute [%t0] (%f0 as %arg: !async.value<memref<10xf16>>) -> (!async.value<memref<10xf16>>) {
        %0 = func.call @function1(%arg) : (memref<10xf16>) -> memref<10xf16>
        async.yield %0 : memref<10xf16>
    }
    %t3, %f3 = async.execute [%t1, %t2] (%f1 as %arg0: !async.value<memref<10xf16>>, %f2 as %arg1: !async.value<memref<10xf16>>) -> (!async.value<memref<20xf16>>) {
        %0 = func.call @function2(%arg0, %arg1) : (memref<10xf16>, memref<10xf16>) -> memref<20xf16>
        async.yield %0 : memref<20xf16>
    }
    %out = async.await %f3 : !async.value<memref<20xf16>>
    return %out : memref<20xf16>

    // CHECK:       func.func @main
    // CHECK:       [[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK-NEXT:      func.call @function1
    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute [[[T0]]]
    // CHECK-NEXT:      func.call @function1
    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute [[[T0]], [[T1]]]
    // CHECK-NEXT:      func.call @function1
    // CHECK:       [[T3:%.+]], [[F3:%.+]] = async.execute [[[T1]], [[T2]]]
    // CHECK-NEXT:      func.call @function2
}

}
