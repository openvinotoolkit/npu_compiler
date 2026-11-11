//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values" --static-allocation="memory-space=DDR" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// The placeholder between two SwIoDma kernels is not a network output, so it has no size to allocate.

module @VPU.SW {
    func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, i64)
            attributes {VPU.kernel_code = "activation_atan_dma.cpp", VPU.kernel_entry = "activation_atan_dma", VPU.kernel_name = "activation_atan_dma", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @UnboundedIntermediatePlaceholder(%in: memref<1x1x1x?xf16>, %out: memref<1x1x1x?xf16>) -> memref<1x1x1x?xf16> {
    %aux = memref.alloc() : memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    // expected-error@+1 {{Cannot allocate a buffer of type 'memref<1x1x1x?xf16>': its shape has a dimension without an upper bound and it was not replaced by a network output}}
    %buf = VPURT.Alloc -> memref<1x1x1x?xf16>
    %first:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_AtanDma
            inputs(%in as %arg0: memref<1x1x1x?xf16>, %aux as %arg1: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            outputs(%buf as %arg2: memref<1x1x1x?xf16>, %aux as %arg3: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            on tile 0 -> (memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>) {
        VPUIP.SW.Kernel.run {attrs = [8589934593]}(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>, memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    }
    %second:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_AtanDma
            inputs(%first#0 as %arg0: memref<1x1x1x?xf16>, %aux as %arg1: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            outputs(%out as %arg2: memref<1x1x1x?xf16>, %aux as %arg3: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            on tile 0 -> (memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>) {
        VPUIP.SW.Kernel.run {attrs = [8589934593]}(%arg0, %arg1, %arg2, %arg3) : memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>, memref<1x1x1x?xf16>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>
    }
    return %second#0 : memref<1x1x1x?xf16>
}
