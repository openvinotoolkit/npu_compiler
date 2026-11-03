//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --add-copy-between-swkernels-and-network-io %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// The copy of a network IO becomes a DMA, so a SW kernel that does not move its own data cannot access an IO
// without an upper bound.

// expected-error-re@+1 {{Cannot copy network IO of type 'memref<1x1x1x?xf16, @DDR>' used by a SW kernel at {{.+}}: a dimension has no upper bound, and only a SwIoDma kernel can access such an IO}}
module @UnboundedInput {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func nested @builtin_relu(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1x1x?xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x1x1000xf16>
    }

    func.func @main(%arg0: memref<1x1x1x?xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %arg2: memref<1x1x1x?xf16, @DDR>) outputs(%arg1 as %arg3: memref<1x1x1x1000xf16, @DDR>) on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x?xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
        }
        return %0 : memref<1x1x1x1000xf16, @DDR>
    }
}

// -----

// expected-error-re@+1 {{Cannot copy network IO of type 'memref<1x1x1x?xf16, @DDR>' used by a SW kernel at {{.+}}: a dimension has no upper bound, and only a SwIoDma kernel can access such an IO}}
module @UnboundedOutputThroughCall {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func nested @builtin_relu(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1x1x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x1x?xf16>
    }

    func.func private @relu(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x?xf16, @DDR>) -> memref<1x1x1x?xf16, @DDR> {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %arg2: memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 as %arg3: memref<1x1x1x?xf16, @DDR>) on tile 0 -> memref<1x1x1x?xf16, @DDR> {
            VPUIP.SW.Kernel.run(%arg2, %arg3) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x?xf16, @DDR>
        }
        return %0 : memref<1x1x1x?xf16, @DDR>
    }

    func.func @main(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x?xf16, @DDR>) -> memref<1x1x1x?xf16, @DDR> {
        %0 = func.call @relu(%arg0, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x?xf16, @DDR>) -> memref<1x1x1x?xf16, @DDR>
        return %0 : memref<1x1x1x?xf16, @DDR>
    }
}
