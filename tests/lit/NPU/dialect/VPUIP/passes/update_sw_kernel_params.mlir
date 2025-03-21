//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --update-sw-kernel-params  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!MemRef = memref<1x32x4x4xf16, @DDR>

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16, @DDR>, %output : memref<*xf16, @DDR>, %axis : i64)
        attributes {
            VPU.kernel_code = "softmax.cpp",
            VPU.kernel_entry = "softmax"
        }
}

// CHECK-LABEL: @applyParamsUpdate
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @applyParamsUpdate(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [1, 0]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
    // CHECK: [[RES:%.+]] =  VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_softmax inputs([[ARG]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x32x4x4xf16, @DDR>) outputs([[OUT]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x32x4x4xf16, @DDR>) on tile 0 -> memref<1x32x4x4xf16, @DDR>{
    // CHECK: VPUIP.SW.Kernel.run {attrs = [
    // CHECK: [1, 12884901891, 17179869188, 4294967297, 17179869188, 17179869188, 137438953504, 68719476752]
    // CHECK: ]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>
    // CHECK: return [[RES]]
}

// -----

// CHECK-LABEL: @applyParamsNotUpdate
// CHECK-SAME: ([[ARG:%.+]]: memref<1x32x4x4xf16, @DDR>, [[OUT:%.+]]: memref<1x32x4x4xf16, @DDR>)
// CHECK-SAME: -> memref<1x32x4x4xf16, @DDR>
func.func @applyParamsNotUpdate(%in: !MemRef, %out: !MemRef) -> !MemRef {
    %res = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
            @VPU.SW::@builtin_softmax
            inputs(%in as %arg0: !MemRef)
            outputs(%out as %arg1: !MemRef)
            on tile 0
    -> !MemRef {
            VPUIP.SW.Kernel.run {attrs = [[1, 12884901891, 17179869188, 4294967297, 17179869188, 17179869188, 137438953504, 68719476752]]}(%arg0, %arg1)
                : !MemRef, !MemRef
    }
    return %res : !MemRef
    // CHECK: [[RES:%.+]] =  VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_softmax inputs([[ARG]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x32x4x4xf16, @DDR>) outputs([[OUT]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x32x4x4xf16, @DDR>) on tile 0 -> memref<1x32x4x4xf16, @DDR>{
    // CHECK: VPUIP.SW.Kernel.run {attrs = [
    // CHECK: [1, 12884901891, 17179869188, 4294967297, 17179869188, 17179869188, 137438953504, 68719476752]
    // CHECK: ]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x32x4x4xf16, @DDR>, memref<1x32x4x4xf16, @DDR>
    // CHECK: return [[RES]]
}
