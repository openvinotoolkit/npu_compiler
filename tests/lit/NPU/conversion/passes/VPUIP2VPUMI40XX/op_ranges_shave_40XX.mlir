//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @SingleShaveTile0 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

module @VPU.SW {
    func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 0 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }

    return %2: memref<1x1x1x1000xf16>
    // CHECK-NOT: return
    // CHECK: VPUMI40XX.OpRanges
    // CHECK-SAME: types([
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
    // CHECK-SAME: ])
    // CHECK-SAME: begins([[VAL0:%[0-9]+]], [[VAL1:%[0-9]+]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: ends([[VAL0]], [[VAL1]] : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
}
}

// -----

module @ThreeShaveTile0 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

module @VPU.SW {
    func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 0 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 0 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 0 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }

    return %2: memref<1x1x1x1000xf16>
    // CHECK-NOT: return
    // CHECK: VPUMI40XX.OpRanges
    // CHECK-SAME: types([
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
    // CHECK-SAME: ])
    // CHECK-SAME: begins({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: ends({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<0:0:2>, !VPURegMapped.Index<0:0:2>)
}
}

// -----

module @SingleShaveTile2 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

module @VPU.SW {
    func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 2 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }

    return %2: memref<1x1x1x1000xf16>
    // CHECK-NOT: return
    // CHECK: VPUMI40XX.OpRanges
    // CHECK-SAME: types([
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
    // CHECK-SAME: ])
    // CHECK-SAME: begins([[VAL0:%[0-9]+]], [[VAL1:%[0-9]+]] : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
    // CHECK-SAME: ends([[VAL0]], [[VAL1]] : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
}
}

// -----

module @ThreeShaveTile0 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

module @VPU.SW {
    func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func @main(%1: memref<1x1x1x1000xf16>, %2: memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16> {
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 2 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 2 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }
    VPURT.Task {
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%1 as %arg0: memref<1x1x1x1000xf16>)
            outputs(%2 as %arg1: memref<1x1x1x1000xf16>)
            on tile 2 -> memref<1x1x1x1000xf16> {
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1) : memref<1x1x1x1000xf16>, memref<1x1x1x1000xf16>
        }
    }

    return %2: memref<1x1x1x1000xf16>
    // CHECK-NOT: return
    // CHECK: VPUMI40XX.OpRanges
    // CHECK-SAME: types([
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelRange>
    // CHECK-DAG: #VPURegMapped.task_type<ActKernelInvocation>
    // CHECK-SAME: ])
    // CHECK-SAME: begins({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<2:0:0>, !VPURegMapped.Index<2:0:0>)
    // CHECK-SAME: ends({{%[0-9]+}}, {{%[0-9]+}} : !VPURegMapped.Index<2:0:2>, !VPURegMapped.Index<2:0:2>)
}
}
