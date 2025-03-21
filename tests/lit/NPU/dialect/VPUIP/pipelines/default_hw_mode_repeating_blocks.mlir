//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="function-outlining='naive'" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!MemRef = memref<1x3x62x62xf16>

module @ChainCalls {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf16>
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
        %0 = VPUIP.Copy inputs(%in: !MemRef) outputs(%out: !MemRef) -> !MemRef
        return %0 : !MemRef
    }

    // CHECK: func.func @main(
    // CHECK-SAME: {{%.+}}: memref<1x3x62x62xf16, @DDR>,
    // CHECK-SAME: [[OUT:%.+]]: memref<1x3x62x62xf16, @DDR>) -> memref<1x3x62x62xf16, @DDR>
    func.func @main(%arg0: !MemRef, %arg1: !MemRef) -> !MemRef {
        %alloc = memref.alloc() : !MemRef
        %alloc2 = memref.alloc() : !MemRef
        %0 = func.call @foo(%arg0, %alloc) : (!MemRef, !MemRef) -> !MemRef
        %1 = func.call @foo(%0, %alloc2) : (!MemRef, !MemRef) -> !MemRef
        %out = VPUIP.Copy inputs(%1: !MemRef) outputs(%arg1: !MemRef) -> !MemRef
        return %out : !MemRef

        // CHECK:      [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[NET_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[MAIN_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[MAIN_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1:[0-9]+]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL0_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL1_ALLOC0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]> -> memref<1x3x62x62xf16, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_ALLOC1:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]> -> memref<1x3x62x62xf16, @DDR>

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[NET_IN]] {{.*}} outputs([[MAIN_ALLOC0]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[FOO_CALL0_ALLOC0]] {{.*}} outputs([[FOO_CALL0_ALLOC1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_ALLOC1]] {{.*}} outputs([[MAIN_ALLOC0]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[FOO_CALL1_ALLOC0]] {{.*}} outputs([[FOO_CALL1_ALLOC1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_ALLOC1]] {{.*}} outputs([[NET_OUT]]

        // CHECK: return [[OUT]]
    }
}

// -----

!MemRef = memref<1x1x2x64xf16>
module @SwKernelsChainCalls {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1x2x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x2x64xf16>
    }

    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64)
            attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
        %alloc = memref.alloc() : !MemRef
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
            inputs(%in as %0: !MemRef) outputs(%alloc as %1: !MemRef) on tile 0
            -> !MemRef
        {
            VPUIP.SW.Kernel.run {attrs = [0, 0]} (%0, %1) : !MemRef, !MemRef
        }
        %res = VPUIP.Copy inputs(%0: !MemRef) outputs(%out: !MemRef) -> !MemRef
        return %res : !MemRef
    }

    // CHECK: func.func @main(
    // CHECK-SAME: {{%.+}}: memref<1x1x2x64xf16, @DDR>,
    // CHECK-SAME: [[OUT:%.+]]: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    func.func @main(%arg0: !MemRef, %arg1: !MemRef) -> !MemRef {
        %alloc0 = memref.alloc() : !MemRef
        %alloc1 = memref.alloc() : !MemRef
        %0 = func.call @foo(%arg0, %alloc0) : (!MemRef, !MemRef) -> !MemRef
        %1 = func.call @foo(%0, %alloc1) : (!MemRef, !MemRef) -> !MemRef
        %res = VPUIP.Copy inputs(%1: !MemRef) outputs(%arg1: !MemRef) -> !MemRef
        return %res : !MemRef

        // CHECK:      [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[NET_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[DDR_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]>
        // CHECK-SAME: -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[DDR_IN:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]>
        // CHECK-SAME: -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[MAIN_IN:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0:[0-9]+]]>
        // CHECK-SAME: -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[MAIN_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1:[0-9]+]]>
        // CHECK-SAME: -> memref<1x1x2x64xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL0_IN_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_OUT_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_IN_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_OUT_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>

        // CHECK-NEXT: [[FOO_CALL1_IN_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR0]]>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_OUT_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <[[ADDR1]]>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_IN_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_OUT_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[NET_IN]] {{.*}} outputs([[DDR_IN]]

        // CHECK: @builtin_SoftMax inputs([[FOO_CALL0_IN_SLICE0]] {{.*}} outputs([[FOO_CALL0_OUT_SLICE0]]
        // CHECK: @builtin_SoftMax inputs([[FOO_CALL0_IN_SLICE1]] {{.*}} outputs([[FOO_CALL0_OUT_SLICE1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_OUT]] {{.*}} outputs([[MAIN_IN]]

        // CHECK: @builtin_SoftMax inputs([[FOO_CALL1_IN_SLICE0]] {{.*}} outputs([[FOO_CALL1_OUT_SLICE0]]
        // CHECK: @builtin_SoftMax inputs([[FOO_CALL1_IN_SLICE1]] {{.*}} outputs([[FOO_CALL1_OUT_SLICE1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_OUT]] {{.*}} outputs([[DDR_OUT]]

        // CHECK: return [[OUT]]
    }
}

// -----

!MemRef = memref<1x1x2x64xf16>
module @SwKernelsIndependentCalls {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1x2x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x2x64xf16>
        DataInfo "output" : tensor<1x1x2x64xf16>
    }

    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64)
            attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    // CHECK-NOT: func.func private @foo
    func.func private @foo(%in: !MemRef, %out: !MemRef) -> !MemRef {
        %alloc = memref.alloc() : !MemRef
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
            inputs(%in as %0: !MemRef) outputs(%alloc as %1: !MemRef) on tile 0
            -> !MemRef
        {
            VPUIP.SW.Kernel.run {attrs = [0, 0]} (%0, %1) : !MemRef, !MemRef
        }
        %res = VPUIP.Copy inputs(%0: !MemRef) outputs(%out: !MemRef) -> !MemRef
        return %res : !MemRef
    }

    // CHECK: func.func @main(
    // CHECK-SAME: {{%.+}}: memref<1x1x2x64xf16, @DDR>, [[OUT0:%.+]]: memref<1x1x2x64xf16, @DDR>, [[OUT1:%.+]]: memref<1x1x2x64xf16, @DDR>)
    // CHECK-SAME: -> (memref<1x1x2x64xf16, @DDR>, memref<1x1x2x64xf16, @DDR>)
    func.func @main(%arg0: !MemRef, %arg1: !MemRef, %arg2: !MemRef) -> (!MemRef, !MemRef) {
        %alloc0 = memref.alloc() : !MemRef
        %alloc1 = memref.alloc() : !MemRef
        %0 = func.call @foo(%arg0, %alloc0) : (!MemRef, !MemRef) -> !MemRef
        %res0 = VPUIP.Copy inputs(%0: !MemRef) outputs(%arg1: !MemRef) -> !MemRef
        %1 = func.call @foo(%arg0, %alloc1) : (!MemRef, !MemRef) -> !MemRef
        %res1 = VPUIP.Copy inputs(%1: !MemRef) outputs(%arg2: !MemRef) -> !MemRef
        return %res0, %res1 : !MemRef, !MemRef

        // CHECK:      [[NET_IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[NET_OUT0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[NET_OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[DDR_OUT0:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[DDR_OUT1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[MAIN_IN:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}> -> memref<1x1x2x64xf16, @DDR>
        // CHECK-NEXT: [[MAIN_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}> -> memref<1x1x2x64xf16, @DDR>

        // CHECK-NEXT: [[FOO_CALL0_IN_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_OUT_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_IN_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL0_OUT_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>

        // CHECK-NEXT: [[FOO_CALL1_IN_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_OUT_SLICE0:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_IN_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
        // CHECK-NEXT: [[FOO_CALL1_OUT_SLICE1:%.+]] = VPURT.DeclareBuffer <DDR> <{{[0-9]+}}>
        // CHECK-SAME: -> memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[NET_IN]] {{.*}} outputs([[MAIN_IN]]

        // CHECK: @builtin_SoftMax inputs([[FOO_CALL0_IN_SLICE0]] {{.*}} outputs([[FOO_CALL0_OUT_SLICE0]]
        // CHECK: @builtin_SoftMax inputs([[FOO_CALL0_IN_SLICE1]] {{.*}} outputs([[FOO_CALL0_OUT_SLICE1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_OUT]] {{.*}} outputs([[DDR_OUT1]]

        // CHECK: @builtin_SoftMax inputs([[FOO_CALL1_IN_SLICE0]] {{.*}} outputs([[FOO_CALL1_OUT_SLICE0]]
        // CHECK: @builtin_SoftMax inputs([[FOO_CALL1_IN_SLICE1]] {{.*}} outputs([[FOO_CALL1_OUT_SLICE1]]

        // CHECK: VPUIP.NNDMA {{.*}} inputs([[MAIN_OUT]] {{.*}} outputs([[DDR_OUT0]]

        // CHECK: return [[OUT0]], [[OUT1]]
    }
}
