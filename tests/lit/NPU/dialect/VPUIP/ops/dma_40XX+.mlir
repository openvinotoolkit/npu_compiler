//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>
!OutputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x16x16xf32, #NHWC, @DDR>
!Output_DDR = memref<1x32x16x16xf16, #NHWC, @DDR>

VPURT.SW.Runtime entryPoint: @VPU.SW::@runtime stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW  {
    func.func private @builtin_Convert(!InputDistributed, !OutputDistributed) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @ParsePrintDistributedBufferConvertDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {

            %5 = VPUIP.ConvertDMA inputs(%input : !Input_DDR) outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output_cmx = VPURT.AllocDistributed -> !OutputDistributed

    %t2 = async.execute attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
        %5 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert
            inputs(%input_cmx as %arg3: !InputDistributed)
            outputs(%output_cmx as %arg4: !OutputDistributed) on tile 0 -> !OutputDistributed{
                VPUIP.SW.Kernel.run(%arg3, %arg4) : !InputDistributed, !OutputDistributed
        }
        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.ConvertDMA inputs(%output_cmx : !OutputDistributed) outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.ConvertDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x32x16x16xf32, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf32, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64} {
    //CHECK:              %results = VPUIP.SW.Kernel
    //CHECK-SAME:                          inputs([[INPUT_CMX]] as %arg1: !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT_CMX]] as %arg2: !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>
    //CHECK:        %token_1 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.ConvertDMA
    //CHECK-SAME:                          inputs([[OUTPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x32x16x16xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x32x16x16xf16, #NHWC, @DDR>
}

// -----

!InputGatherDMA_DDR = memref<50257x768xf32, @DDR>
!IndicesGatherDMA_CMX = memref<1024x1xi64, [@CMX_NN, 0]>
!OutputGatherDMA_CMX_Large = memref<1024x768xf32, [@CMX_NN, 0]>

module @ParseGatherDMAOpModuleDDR {
    func.func @ParseGatherDMAOp(%input: !InputGatherDMA_DDR, %indices: !IndicesGatherDMA_CMX) -> !OutputGatherDMA_CMX_Large {
        %output = VPURT.DeclareBuffer <CMX_NN> [0] <3145728> -> !OutputGatherDMA_CMX_Large

        %barrier0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barrier1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task
            waits(%barrier0 : !VPURT.Barrier)
            updates(%barrier1 : !VPURT.Barrier)
        {
            %gather = VPUIP.GatherDMA {
                elementSize = 16,
                padding = 0,
                port = 0: i64
            }
                inputs(%input: !InputGatherDMA_DDR)
                indices(%indices: !IndicesGatherDMA_CMX)
                outputs(%output: !OutputGatherDMA_CMX_Large) -> !OutputGatherDMA_CMX_Large
        }

        return %output: !OutputGatherDMA_CMX_Large
    }
}

// -----

!InputGatherDMA_CMX = memref<500xi64,  [@CMX_NN, 0]>
!IndicesGatherDMA_CMX_Small = memref<100xi64, [@CMX_NN, 0]>
!OutputGatherDMA_CMX = memref<100xi64,  [@CMX_NN, 0]>

module @ParseGatherDMAOpModuleCMX {
    func.func @ParseGatherDMAOp(%input: !InputGatherDMA_CMX, %indices: !IndicesGatherDMA_CMX_Small) -> !OutputGatherDMA_CMX {
        %output = VPURT.DeclareBuffer <CMX_NN> [0] <800> -> !OutputGatherDMA_CMX

        %barrier0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barrier1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task
            waits(%barrier0 : !VPURT.Barrier)
            updates(%barrier1 : !VPURT.Barrier)
        {
            %gather = VPUIP.GatherDMA {
                elementSize = 16,
                padding = 0,
                port = 0: i64
            }
                inputs(%input: !InputGatherDMA_CMX)
                indices(%indices: !IndicesGatherDMA_CMX_Small)
                outputs(%output: !OutputGatherDMA_CMX) -> !OutputGatherDMA_CMX
        }

        return %output: !OutputGatherDMA_CMX
    }
}
