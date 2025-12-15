//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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

!Input_DDR = memref<1x32x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x32x16x16xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferNNDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        VPUIP.NNDMA inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        VPUIP.NNDMA inputs(%input_cmx: !InputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.NNDMA inputs(%arg0 : memref<1x32x16x16xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.NNDMA inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x32x16x16xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x32x16x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x16x16xf16, @DDR>
!Output_DDR = memref<1x32x16x16xf16, @DDR>

func.func @ParsePrintDistributedBufferPermuteDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {

            %5 = VPUIP.PermuteDMA {mem_perm = #NHWC} inputs(%input : !Input_DDR) outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.PermuteDMA {mem_perm = #NCHW} inputs(%input_cmx : !InputDistributed) outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.PermuteDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x32x16x16xf16, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x32x16x16xf16, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.PermuteDMA
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x32x16x16xf16, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x32x16x16xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x8x2x3xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x2x4x6xf16, #NHWC, @DDR>
!Output_DDR = memref<1x2x4x6xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferSpaceToDepthDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6}
            inputs(%input : !Input_DDR)
            outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
            inputs(%input_cmx : !InputDistributed)
            outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x8x2x3xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.SpaceToDepthDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.DepthToSpaceDMA
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x2x4x6xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x2x4x6xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x2x4x6xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x8x2x3xf16, #NHWC, @DDR>
!Output_DDR = memref<1x8x2x3xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferDepthToSpaceDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
            inputs(%input : !Input_DDR)
            outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.SpaceToDepthDMA {block_size = 2 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, output_channel = 2 : i64, output_width = 6}
            inputs(%input_cmx : !InputDistributed)
            outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x4x6xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.DepthToSpaceDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x8x2x3xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x8x2x3xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.SpaceToDepthDMA
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x8x2x3xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x8x2x3xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x35x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x2x35x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x32x35x16xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferPerAxisTileDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 8 : i64}
            inputs(%input : !Input_DDR)
            outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 2 : i64}
            inputs(%input_cmx : !InputDistributed)
            outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x35x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.PerAxisTileDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x2x35x16xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x32x35x16xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.PerAxisTileDMA
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x32x35x16xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x32x35x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x35x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x34x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x33x35x16xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferExpandDMA(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]}
            inputs(%input : !Input_DDR)
            outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.ExpandDMA {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            inputs(%input_cmx : !InputDistributed)
            outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x35x16xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.ExpandDMA
    //CHECK-SAME:                          inputs(%arg0 : memref<1x32x34x16xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x33x35x16xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.ExpandDMA
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x33x35x16xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x33x35x16xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x64x64xf16, #NHWC, @DDR>

func.func @ParsePrintDistributedBufferUpsamplingDMAOp(%input: !Input_DDR) -> !Output_DDR {
    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
            inputs(%input : !Input_DDR)
            outputs(%input_cmx : !InputDistributed) -> !InputDistributed

        async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %5 = VPUIP.UpsamplingDMAOp {port = 0 : i64, upsampling_factor = [1, 1, 2, 2]}
            inputs(%input_cmx : !InputDistributed)
            outputs(%output : !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    //CHECK:        [[INPUT_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN,
    //CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    //CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    //CHECK:        %token = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.UpsamplingDMAOp
    //CHECK-SAME:                          inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>
    //CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK:              async.yield
    //CHECK:        }

    //CHECK:        [[OUTPUT:%.*]] = memref.alloc() : memref<1x16x64x64xf16, #NHWC, @DDR>
    //CHECK:        %token_0 = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    //CHECK:              VPUIP.UpsamplingDMAOp
    //CHECK-SAME:                          inputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    //CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x16x64x64xf16, #NHWC, @DDR>
    //CHECK:              async.yield
    //CHECK:        }
    //CHECK:        return [[OUTPUT]] : memref<1x16x64x64xf16, #NHWC, @DDR>
}
