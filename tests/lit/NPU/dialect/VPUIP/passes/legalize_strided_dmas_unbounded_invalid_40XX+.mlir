//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --verify-diagnostics --init-compiler="platform=%platform%" --legalize-strided-dmas %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// The SwIoDma kernel that accesses an unbounded IO derives compact strides from the runtime extents, so it cannot
// honour strides supplied at run time.

// expected-error@+1 {{Network IO 0 of type 'memref<1x1x1x?xf16, @DDR>' cannot have dynamic strides: a dimension has no upper bound}}
module @DynamicStridesUnboundedIO {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Parameter" : tensor<1x1x1x?xf16> {dynamicStrides}
        DataInfo "ParameterShape" : tensor<4xsi32>
    } outputsInfo : {
        DataInfo "Result" : tensor<1x1x1x?xf16>
        DataInfo "ResultShape" : tensor<4xsi32>
    }

    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func nested @builtin_AtanDma(memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, memref<*xf16>, memref<*xsi32>, memref<*xui8, @CMX_NN>, i64)
                attributes {VPU.kernel_code = "activation_atan_dma.cpp", VPU.kernel_entry = "activation_atan_dma", VPU.kernel_name = "activation_atan_dma", VPU.task_type = @COMPUTE}
        func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    func.func @main(%arg0: memref<1x1x1x?xf16, @DDR>, %arg1: memref<4xsi32, @DDR>, %arg2: memref<1x1x1x?xf16, @DDR>, %arg3: memref<4xsi32, @DDR>) -> (memref<1x1x1x?xf16, @DDR>, memref<4xsi32, @DDR>) {
        %aux_buf = memref.alloc() : memref<1x1x1x524288xui8, [@CMX_NN, 0]>
        %results:2, %dynamicOutputShapes = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0, -1>, resultSegmentSizes = array<i32: 2, 1, 0>}
            @VPU.SW::@builtin_AtanDma
            inputs(%arg0 as %arg4: memref<1x1x1x?xf16, @DDR>, %aux_buf as %arg5: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            dynamicInputShapes(%arg1 : memref<4xsi32, @DDR>)
            outputs(%arg2 as %arg6: memref<1x1x1x?xf16, @DDR>, %aux_buf as %arg7: memref<1x1x1x524288xui8, [@CMX_NN, 0]>)
            dynamicOutputShapes(%arg3 : memref<4xsi32, @DDR>)
            on tile 0 -> (memref<1x1x1x?xf16, @DDR>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>, memref<4xsi32, @DDR>) {
                VPUIP.SW.Kernel.run {attrs = [8589934593]}(%arg4, %arg5, %arg6, %arg7) : memref<1x1x1x?xf16, @DDR>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>, memref<1x1x1x?xf16, @DDR>, memref<1x1x1x524288xui8, [@CMX_NN, 0]>
            }
        return %results#0, %dynamicOutputShapes : memref<1x1x1x?xf16, @DDR>, memref<4xsi32, @DDR>
    }
}
