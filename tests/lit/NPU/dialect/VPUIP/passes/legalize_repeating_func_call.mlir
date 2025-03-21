//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" --legalize-repeating-func-calls --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @NoRepetitions
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @NoRepetitions(%arg: !MemRef, %out_arg: !MemRef) -> !MemRef {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef
    %out = VPUIP.Copy inputs(%0: !MemRef) outputs(%out_arg: !MemRef) -> !MemRef
    return %out : !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[CALL:%.+]] = call @foo([[ARG]], [[ALLOC]])
    // CHECK: [[RES:%.+]] = VPUIP.Copy inputs([[CALL]]
    // CHECK-SAME:  outputs([[OUT]]
    // CHECK: return [[RES]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @NoOpRepetition
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @NoOpRepetition(%arg: !MemRef, %out_arg1: !MemRef, %out_arg2: !MemRef) -> (!MemRef, !MemRef) {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef

    // Note: this Copy is a bit of a stretch, but generally is allowed to e.g.
    // enable different SubViews on top of the same root
    %fakeAlloc = VPUIP.Copy inputs(%arg: !MemRef) outputs(%alloc: !MemRef) -> !MemRef

    %1 = func.call @foo(%arg, %fakeAlloc) : (!MemRef, !MemRef) -> !MemRef
    %out1 = VPUIP.Copy inputs(%0: !MemRef) outputs(%out_arg1: !MemRef) -> !MemRef
    %out2 = VPUIP.Copy inputs(%1: !MemRef) outputs(%out_arg2: !MemRef) -> !MemRef
    return %out1, %out2 : !MemRef, !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[ALLOC]])
    // CHECK: [[ALLOC_ALIAS:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[ALLOC]]
    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[ALLOC_ALIAS]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME:  outputs([[OUT0]]
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME:  outputs([[OUT1]]
    // CHECK: return [[RES0]], [[RES1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
    func.func private @builtin_sigmoid(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }
}

func.func private @foo(%in: !MemRef, %out0: !MemRef, %out1: !MemRef) -> (!MemRef, !MemRef) {
    %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out0 as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_sigmoid
            inputs(%in as %arg0: !MemRef)
            outputs(%out1 as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %0, %1 : !MemRef, !MemRef
}

// expected-error@+1 {{is used as output at multiple call-sites}}
func.func @SameValueOutputs(%arg: !MemRef, %out_arg: !MemRef) -> !MemRef {
    %alloc0 = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    %alloc1 = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    %0:2 = func.call @foo(%arg, %alloc0, %out_arg) : (!MemRef, !MemRef, !MemRef) -> (!MemRef, !MemRef)
    %1:2 = func.call @foo(%0#0, %out_arg, %alloc1) : (!MemRef, !MemRef, !MemRef) -> (!MemRef, !MemRef)
    return %1#1 : !MemRef
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @SameRootOutputs
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @SameRootOutputs(%arg: !MemRef, %out_arg: !MemRef) -> !MemRef {
    %0 = func.call @foo(%arg, %out_arg) : (!MemRef, !MemRef) -> !MemRef

    // Note: this Copy is a bit of a stretch, but generally is allowed to e.g.
    // enable different SubViews on top of the same root
    %alias = VPUIP.Copy inputs(%0: !MemRef) outputs(%out_arg: !MemRef) -> !MemRef

    %1 = func.call @foo(%arg, %alias) : (!MemRef, !MemRef) -> !MemRef
    return %1 : !MemRef

    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[OUT]])
    // CHECK: [[ALIAS_OUT:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[OUT]]
    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[ALIAS_OUT]])
    // CHECK: return [[CALL1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @InputUpdate
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @InputUpdate(%arg: !MemRef, %out_arg: !MemRef) -> !MemRef {
    %alloc = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    %0 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%0, %out_arg) : (!MemRef, !MemRef) -> !MemRef
    return %1 : !MemRef

    // CHECK: [[IN_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[IN_COPY1:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME:  outputs([[IN_ALLOC]]
    // CHECK: [[CALL0:%.+]] = call @foo([[IN_COPY1]], [[OUT_ALLOC]])

    // CHECK: [[IN_COPY2:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME:  outputs([[IN_ALLOC]]
    // CHECK: [[CALL1:%.+]] = call @foo([[IN_COPY2]], [[OUT_ALLOC]])

    // CHECK: [[RES:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME:  outputs([[OUT]]
    // CHECK: return [[RES]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @OutputUpdate
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @OutputUpdate(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %0 = func.call @foo(%arg, %out_arg0) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%arg, %out_arg1) : (!MemRef, !MemRef) -> !MemRef
    return %0, %1 : !MemRef, !MemRef

    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK: return [[RES0]], [[RES1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
    func.func private @builtin_sigmoid(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @OutputUpdateMultiUsers
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @OutputUpdateMultiUsers(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %out_arg0) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef

    // Note: purposefully overwrite the data in %alloc with Sigmoid result
    %useAfter = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_sigmoid
            inputs(%1 as %arg0: !MemRef)
            outputs(%alloc as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : !MemRef, !MemRef
    }

    %copy = VPUIP.Copy inputs(%useAfter: !MemRef) outputs(%out_arg1: !MemRef) -> !MemRef
    return %0, %copy : !MemRef, !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[ALLOC]]

    // CHECK: [[USE_AFTER:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_sigmoid inputs([[RES1]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)
    // CHECK-SAME: outputs([[ALLOC]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)

    // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[USE_AFTER]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK: return [[RES0]], [[COPY]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
    func.func private @builtin_sigmoid(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @ResultMultiUsers
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @ResultMultiUsers(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef
    %useBetween = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_sigmoid
            inputs(%0 as %arg0: !MemRef)
            outputs(%out_arg0 as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : !MemRef, !MemRef
    }
    %1 = func.call @foo(%0, %out_arg1) : (!MemRef, !MemRef) -> !MemRef
    return %useBetween, %1 : !MemRef, !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[IN_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL0:%.+]] = call @foo([[COPY0]], [[OUT_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[ALLOC]]

    // CHECK: [[USE_BETWEEN:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_sigmoid inputs([[RES0]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)
    // CHECK-SAME: outputs([[OUT0]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)

    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL1:%.+]] = call @foo([[COPY1]], [[OUT_ALLOC]])
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK: return [[USE_BETWEEN]], [[RES1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @OutputUpdateImmediateTrailingCopy
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @OutputUpdateImmediateTrailingCopy(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %out_arg0) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef
    %copy = VPUIP.Copy inputs(%1: !MemRef) outputs(%out_arg1: !MemRef) -> !MemRef
    return %0, %copy : !MemRef, !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])

    // Note: an extra copy is going to be inserted here (for the result of
    // 2nd foo call); in real networks, the existing copy is likely
    // optimized by --fuse-last-copy pass.

    // CHECK: [[EXTRA_COPY:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[ALLOC]]

    // CHECK-NEXT: [[EXISTING_COPY:%.+]] = VPUIP.Copy inputs([[EXTRA_COPY]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK-NEXT: return [[RES0]], [[EXISTING_COPY]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @OutputUpdateDelayedTrailingCopy
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @OutputUpdateDelayedTrailingCopy(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %alloc = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %alloc) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%arg, %out_arg1) : (!MemRef, !MemRef) -> !MemRef
    %copy = VPUIP.Copy inputs(%0: !MemRef) outputs(%out_arg0: !MemRef) -> !MemRef
    return %copy, %1 : !MemRef, !MemRef

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[CALL0:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])

    // Note: an extra copy is going to be inserted here (for the result of
    // 1st foo call); in real networks, the existing copy is likely
    // optimized by --fuse-last-copy pass.

    // CHECK: [[EXTRA_COPY:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[ALLOC]]

    // CHECK: [[CALL1:%.+]] = call @foo([[ARG]], [[OUT_ALLOC]])
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK-NEXT: [[EXISTING_COPY:%.+]] = VPUIP.Copy inputs([[EXTRA_COPY]]
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK-NEXT: return [[EXISTING_COPY]], [[RES1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @ChainCalls
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @ChainCalls(%arg: !MemRef, %out_arg: !MemRef) -> !MemRef {
    %alloc0 = memref.alloc() : !MemRef
    %alloc1 = memref.alloc() : !MemRef
    %0 = func.call @foo(%arg, %alloc0) : (!MemRef, !MemRef) -> !MemRef
    %1 = func.call @foo(%0, %alloc1) : (!MemRef, !MemRef) -> !MemRef
    %2 = func.call @foo(%1, %out_arg) : (!MemRef, !MemRef) -> !MemRef

    return %2 : !MemRef

    // CHECK: [[IN_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL0:%.+]] = call @foo([[COPY0]], [[OUT_ALLOC]])

    // CHECK-NEXT: [[COPY1:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL1:%.+]] = call @foo([[COPY1]], [[OUT_ALLOC]])

    // CHECK-NEXT: [[COPY2:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL2:%.+]] = call @foo([[COPY2]], [[OUT_ALLOC]])

    // CHECK-NEXT: [[RES:%.+]] = VPUIP.Copy inputs([[CALL2]]
    // CHECK-SAME: outputs([[OUT]]

    // CHECK: return [[RES]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
    func.func private @builtin_sigmoid(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
}

// CHECK-LABEL: @InplaceArg
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>)
func.func @InplaceArg(%arg: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef)
        -> (!MemRef, !MemRef) {
    %res0 = func.call @foo(%arg, %out_arg0) : (!MemRef, !MemRef) -> !MemRef
    %alloc = memref.alloc() : !MemRef
    %inOut = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_sigmoid
            inputs(%arg as %arg0: !MemRef)
            outputs(%alloc as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : !MemRef, !MemRef
    }
    %inOutCall = func.call @foo(%inOut, %inOut) : (!MemRef, !MemRef) -> !MemRef
    %res1 = VPUIP.Copy inputs(%inOutCall: !MemRef) outputs(%out_arg1: !MemRef) -> !MemRef
    return %res0, %res1 : !MemRef, !MemRef

    // CHECK: [[IN_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL0:%.+]] = call @foo([[COPY0]], [[OUT_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[SIGMOID:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_sigmoid inputs([[ARG]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)
    // CHECK-SAME: outputs([[ALLOC]] as {{[^:]+}}: memref<1x32x4x4xf16, @DDR>)

    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[SIGMOID]]
    // CHECK-SAME: outputs([[IN_ALLOC]]
    // CHECK: [[CALL1:%.+]] = call @foo([[COPY1]], [[OUT_ALLOC]])
    // CHECK: [[CALL1_RES:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[ALLOC]]

    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[CALL1_RES]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK: return [[RES0]], [[RES1]]
}

// -----

!MemRef = memref<1x32x4x4xf16, @DDR>
!MemRef2 = memref<1x64x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
    func.func private @builtin_sigmoid(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>)
        attributes {
            VPU.kernel_code = "activation_sigmoid.cpp",
            VPU.kernel_entry = "activation_sigmoid"
        }
}

func.func private @foo(%in: !MemRef, %out: !MemRef, %out2: !MemRef) -> (!MemRef, !MemRef) {
    %res0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    %res1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_sigmoid
            inputs(%res0 as %arg0: !MemRef)
            outputs(%out2 as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res0, %res1 : !MemRef, !MemRef
}

func.func private @bar(%in: !MemRef, %in2: !MemRef, %out: !MemRef2) -> !MemRef2 {
    %res = VPUIP.ConcatView inputs(%in, %in2: !MemRef, !MemRef) outputs(%out: !MemRef2) -> !MemRef2
    return %res : !MemRef2
}

// CHECK-LABEL: @DifferentRepetitions
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT0:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT1:%.+]]: memref<1x64x4x4xf16, @DDR>)
// CHECK-SAME: -> (memref<1x32x4x4xf16, @DDR>, memref<1x64x4x4xf16, @DDR>)
func.func @DifferentRepetitions(%arg0: !MemRef, %out_arg0: !MemRef, %out_arg1: !MemRef2)
        -> (!MemRef, !MemRef2) {
    %foo_call0_alloc0 = memref.alloc() : !MemRef
    %foo_call0_alloc1 = memref.alloc() : !MemRef
    %foo_call0:2 = func.call @foo(%arg0, %foo_call0_alloc0, %foo_call0_alloc1)
        : (!MemRef, !MemRef, !MemRef) -> (!MemRef, !MemRef)

    %foo_call1_alloc0 = memref.alloc() : !MemRef
    %foo_call1_alloc1 = memref.alloc() : !MemRef
    %foo_call1:2 = func.call @foo(%foo_call0#1, %foo_call1_alloc0, %foo_call1_alloc1)
        : (!MemRef, !MemRef, !MemRef) -> (!MemRef, !MemRef)

    %bar_alloc = memref.alloc() : !MemRef2
    %bar_call0 = func.call @bar(%arg0, %foo_call0#0, %bar_alloc)
        : (!MemRef, !MemRef, !MemRef2) -> !MemRef2

    %foo_call2_alloc1 = memref.alloc() : !MemRef
    %foo_call2:2 = func.call @foo(%foo_call1#1, %out_arg0, %foo_call2_alloc1)
        : (!MemRef, !MemRef, !MemRef) -> (!MemRef, !MemRef)

    %bar_call1 = func.call @bar(%foo_call1#1, %arg0, %out_arg1)
        : (!MemRef, !MemRef, !MemRef2) -> !MemRef2

    return %foo_call2#0, %bar_call1 : !MemRef, !MemRef2

    // CHECK: [[FOO_CALL0_ALLOC0:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[FOO_IN_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[FOO_OUT0_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[FOO_OUT1_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[FOO_COPY0:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[FOO_IN_ALLOC]]
    // CHECK: [[FOO_CALL0:%.+]]:2 = call @foo([[FOO_COPY0]], [[FOO_OUT0_ALLOC]], [[FOO_OUT1_ALLOC]])
    // CHECK: [[FOO_CALL0_RES0_COPY:%.+]] = VPUIP.Copy inputs([[FOO_CALL0]]#0
    // CHECK-SAME: outputs([[FOO_CALL0_ALLOC0]]

    // CHECK: [[FOO_CALL1_ALLOC1:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[FOO_COPY1:%.+]] = VPUIP.Copy inputs([[FOO_CALL0]]#1
    // CHECK-SAME: outputs([[FOO_IN_ALLOC]]
    // CHECK: [[FOO_CALL1:%.+]]:2 = call @foo([[FOO_COPY1]], [[FOO_OUT0_ALLOC]], [[FOO_OUT1_ALLOC]])
    // CHECK: [[FOO_CALL1_RES1_COPY:%.+]] = VPUIP.Copy inputs([[FOO_CALL1]]#1
    // CHECK-SAME: outputs([[FOO_CALL1_ALLOC1]]

    // CHECK: [[BAR_IN0_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[BAR_IN1_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[BAR_OUT_ALLOC:%.+]] = memref.alloc() : memref<1x64x4x4xf16, @DDR>

    // CHECK: [[BAR_IN0_COPY0:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[BAR_IN0_ALLOC]]
    // CHECK: [[BAR_IN1_COPY0:%.+]] = VPUIP.Copy inputs([[FOO_CALL0_RES0_COPY]]
    // CHECK-SAME: outputs([[BAR_IN1_ALLOC]]
    // CHECK: {{%.+}} = call @bar([[BAR_IN0_COPY0]], [[BAR_IN1_COPY0]], [[BAR_OUT_ALLOC]])

    // CHECK: [[FOO_COPY2:%.+]] = VPUIP.Copy inputs([[FOO_CALL1]]#1
    // CHECK-SAME: outputs([[FOO_IN_ALLOC]]
    // CHECK: [[FOO_CALL2:%.+]]:2 = call @foo([[FOO_COPY2]], [[FOO_OUT0_ALLOC]], [[FOO_OUT1_ALLOC]])
    // CHECK: [[RES0:%.+]] = VPUIP.Copy inputs([[FOO_CALL2]]#0
    // CHECK-SAME: outputs([[OUT0]]

    // CHECK: [[BAR_IN0_COPY1:%.+]] = VPUIP.Copy inputs([[FOO_CALL1_RES1_COPY]]
    // CHECK-SAME: outputs([[BAR_IN0_ALLOC]]
    // CHECK: [[BAR_IN1_COPY1:%.+]] = VPUIP.Copy inputs([[ARG]]
    // CHECK-SAME: outputs([[BAR_IN1_ALLOC]]
    // CHECK: [[BAR_CALL1:%.+]] = call @bar([[BAR_IN0_COPY1]], [[BAR_IN1_COPY1]], [[BAR_OUT_ALLOC]])
    // CHECK: [[RES1:%.+]] = VPUIP.Copy inputs([[BAR_CALL1]]
    // CHECK-SAME: outputs([[OUT1]]

    // CHECK: return [[RES0]], [[RES1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!MemRef = memref<1x32x4x4xf16, @DDR>
!MemRefStrided = memref<1x32x4x4xf16, {order = #NCHW, strides = [1024, 16, 4, 1]}, @DDR>

func.func private @foo(%in: !MemRefStrided, %out: !MemRef) -> !MemRef {
    %res = VPUIP.Copy inputs(%in : !MemRefStrided) outputs(%out : !MemRef) -> !MemRef
    return %res : !MemRef
}

// CHECK-LABEL: @SubView
// CHECK-SAME: ([[ARG:%.+]]: memref<1x64x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x64x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x64x4x4xf16, @DDR>
func.func @SubView(%arg: memref<1x64x4x4xf16, @DDR>, %out: memref<1x64x4x4xf16, @DDR>)
        -> memref<1x64x4x4xf16, @DDR> {
    %alloc_part0 = memref.alloc() : !MemRef
    %alloc_part1 = memref.alloc() : !MemRef

    %part0 = VPUIP.SubView %arg [0, 0, 0, 0] [1, 32, 4, 4] : memref<1x64x4x4xf16, @DDR> to !MemRefStrided
    %call0 = func.call @foo(%part0, %alloc_part0) : (!MemRefStrided, !MemRef) -> !MemRef
    %part1 = VPUIP.SubView %arg [0, 32, 0, 0] [1, 32, 4, 4] : memref<1x64x4x4xf16, @DDR> to !MemRefStrided
    %call1 = func.call @foo(%part1, %alloc_part1) : (!MemRefStrided, !MemRef) -> !MemRef

    %res = VPUIP.ConcatView inputs(%call0, %call1 : !MemRef, !MemRef) outputs(%out : memref<1x64x4x4xf16, @DDR>)
        -> memref<1x64x4x4xf16, @DDR>

    return %res : memref<1x64x4x4xf16, @DDR>

    // CHECK: [[ALLOC_PART0:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>
    // CHECK: [[ALLOC_PART1:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[PART0:%.+]] = VPUIP.SubView [[ARG]] [0, 0, 0, 0]

    // CHECK: [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x32x4x4xf16, @DDR>

    // CHECK: [[CALL0:%.+]] = call @foo([[PART0]], [[OUT_ALLOC]])
    // CHECK: [[CALL0_COPY:%.+]] = VPUIP.Copy inputs([[CALL0]]
    // CHECK-SAME: outputs([[ALLOC_PART0]]

    // CHECK: [[PART1:%.+]] = VPUIP.SubView [[ARG]] [0, 32, 0, 0]

    // CHECK: [[CALL1:%.+]] = call @foo([[PART1]], [[OUT_ALLOC]])
    // CHECK: [[CALL1_COPY:%.+]] = VPUIP.Copy inputs([[CALL1]]
    // CHECK-SAME: outputs([[ALLOC_PART1]]

    // CHECK: [[RES:%.+]] = VPUIP.ConcatView inputs([[CALL0_COPY]], [[CALL1_COPY]]
    // CHECK-SAME: outputs([[OUT]]

    // CHECK: return [[RES]]
}
