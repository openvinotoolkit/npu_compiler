//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_TRACE env IE_NPU_LOG_FILTER="vpux-compiler|memory-usage-info" vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values" --static-allocation="memory-space=DDR" -o /dev/null %s 2>&1 | FileCheck %s
// REQUIRES: dev-build && (platform-NPU3720 || platform-NPU4000 || platform-NPU5010)

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @LinearGraph {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "data" : tensor<1x1x1x1024xf16>
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x1x1x1024xf16>
    }

module @VPU.SW {
func.func nested @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func nested @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

func.func @main(%in: memref<1x1x1x1024xf16>, %out: memref<1x1x1x1024xf16>) -> memref<1x1x1x1024xf16> {
    %buf = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x512xf16, @DDR>

    %buf0 = memref.alloc() : memref<1x1x1x1024xf16, @DDR>
    %buf1 = memref.alloc() : memref<1x1x1x512xf16, @DDR>
    %buf2 = memref.alloc() : memref<1x1x1x1024xf16, @DDR>
    %buf3 = memref.alloc() : memref<1x1x1x512xf16, @DDR>
    %buf4 = memref.alloc() : memref<1x1x1x1024xf16, @DDR>

    %t0, %f0 = async.execute -> !async.value<memref<1x1x1x1024xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_ACT, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%in as %input_0: memref<1x1x1x1024xf16>)
                    outputs(%buf0 as %output_0: memref<1x1x1x1024xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1024xf16, @DDR> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x1024xf16>
                    , memref<1x1x1x1024xf16, @DDR>
        }
        async.yield %0 : memref<1x1x1x1024xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Buffer size to allocate      2048 B
    // CHECK:  [memory-usage-info]   main: DDR free memory before alloc 0 B
    // CHECK:  [memory-usage-info]   main: Max allocated size 2048 B
    // CHECK:  [memory-usage-info]   main: DDR used memory    2048 100.00%
    // CHECK:  [memory-usage-info]   main: DDR free memory    0 0.00%

    %t1, %f1 = async.execute -> !async.value<memref<1x1x1x512xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_ACT, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SubView %in [0, 0, 0, 512] [1, 1, 1, 512] : memref<1x1x1x1024xf16> to memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 1024, 1]}>
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%0 as %input_0: memref<1x1x1x1024xf16>)
                    outputs(%buf1 as %output_0: memref<1x1x1x512xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x512xf16, @DDR> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x1024xf16>
                    , memref<1x1x1x512xf16, @DDR>
        }
        async.yield %1 : memref<1x1x1x512xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Buffer size to allocate      1024 B
    // CHECK:  [memory-usage-info]   main: DDR free memory before alloc 0 B
    // CHECK:  [memory-usage-info]   main: Max allocated size 3072 B
    // CHECK:  [memory-usage-info]   main: DDR used memory    3072 100.00%
    // CHECK:  [memory-usage-info]   main: DDR free memory    0 0.00%

    %t2, %f2 = async.execute -> !async.value<memref<1x1x1x1024xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_ACT, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%in as %input_0: memref<1x1x1x1024xf16>)
                    outputs(%buf2 as %output_0: memref<1x1x1x1024xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1024xf16, @DDR> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x1024xf16>
                    , memref<1x1x1x1024xf16, @DDR>
        }
        async.yield %0 : memref<1x1x1x1024xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Buffer size to allocate      2048 B
    // CHECK:  [memory-usage-info]   main: DDR free memory before alloc 0 B
    // CHECK:  [memory-usage-info]   main: Max allocated size 5120 B
    // CHECK:  [memory-usage-info]   main: DDR used memory    5120 100.00%
    // CHECK:  [memory-usage-info]   main: DDR free memory    0 0.00%

    %t3, %f3 = async.execute -> !async.value<memref<1x1x1x512xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_ACT, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SubView %in [0, 0, 0, 512] [1, 1, 1, 512] : memref<1x1x1x1024xf16> to memref<1x1x1x512xf16, {order = #NCHW, strides = [1024, 1024, 1024, 1]}>
        %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%0 as %input_0: memref<1x1x1x1024xf16>)
                    outputs(%buf3 as %output_0: memref<1x1x1x512xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x512xf16, @DDR> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x1024xf16>
                    , memref<1x1x1x512xf16, @DDR>
        }
        async.yield %1 : memref<1x1x1x512xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Buffer size to allocate      1024 B
    // CHECK:  [memory-usage-info]   main: DDR free memory before alloc 0 B
    // CHECK:  [memory-usage-info]   main: Max allocated size 6144 B
    // CHECK:  [memory-usage-info]   main: DDR used memory    6144 100.00%
    // CHECK:  [memory-usage-info]   main: DDR free memory    0 0.00%

    %t_out0, %f_out0 = async.execute [%t1, %t3] (
            %f1 as %0 : !async.value<memref<1x1x1x512xf16, @DDR>>,
            %f3 as %1 : !async.value<memref<1x1x1x512xf16, @DDR>>
        ) -> !async.value<memref<1x1x1x512xf16, @DDR>>
        attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {

        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x512xf16, @DDR>) outputs(%buf : memref<1x1x1x512xf16, @DDR>) -> memref<1x1x1x512xf16, @DDR>
        %3 = VPUIP.Copy inputs(%1 : memref<1x1x1x512xf16, @DDR>) outputs(%buf : memref<1x1x1x512xf16, @DDR>) -> memref<1x1x1x512xf16, @DDR>
        async.yield %3 : memref<1x1x1x512xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Free buffer of size 1024 B
    // CHECK:  [memory-usage-info]   main: Free buffer of size 1024 B

    %t4, %f4 = async.execute -> !async.value<memref<1x1x1x1024xf16, @DDR>>
        attributes {VPUIP.executor = @SHAVE_ACT, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%in as %input_0: memref<1x1x1x1024xf16>)
                    outputs(%buf4 as %output_0: memref<1x1x1x1024xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1024xf16, @DDR> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x1024xf16>
                    , memref<1x1x1x1024xf16, @DDR>
        }
        async.yield %0 : memref<1x1x1x1024xf16, @DDR>
    }
    // CHECK:  [memory-usage-info]   main: Buffer size to allocate      2048 B
    // CHECK:  [memory-usage-info]   main: DDR free memory before alloc 2048 B
    // CHECK:  [memory-usage-info]   main: Max allocated size 7168 B
    // CHECK:  [memory-usage-info]   main: Increased allocation size to 7168 B due to fragmentation!
    // CHECK:  [memory-usage-info]   main: DDR used memory    6144 85.71%
    // CHECK:  [memory-usage-info]   main: DDR free memory    1024 14.29%

    %t_out1, %f_out1 = async.execute [%t0, %t2] (
            %f0 as %0 : !async.value<memref<1x1x1x1024xf16, @DDR>>,
            %f2 as %1 : !async.value<memref<1x1x1x1024xf16, @DDR>>
        ) -> !async.value<memref<1x1x1x1024xf16>>
        attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 3 : i64} {

        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x1024xf16, @DDR>) outputs(%out : memref<1x1x1x1024xf16>) -> memref<1x1x1x1024xf16>
        %3 = VPUIP.Copy inputs(%1 : memref<1x1x1x1024xf16, @DDR>) outputs(%out : memref<1x1x1x1024xf16>) -> memref<1x1x1x1024xf16>
        async.yield %3 : memref<1x1x1x1024xf16>
    }

    %ret = async.await %f_out1 : !async.value<memref<1x1x1x1024xf16>>
    return %ret : memref<1x1x1x1024xf16>
    // CHECK:  [memory-usage-info]   main: Free buffer of size 2048 B
    // CHECK:  [memory-usage-info]   main: Free buffer of size 2048 B
    // CHECK:  [memory-usage-info]   main: Free buffer of size 2048 B
}

}
