//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU3720 || platform-NPU4000

func.func @verifyNNDMA(%arg0: memref<97x257x673xui8>, %arg1: memref<97x257x673xui8>) -> memref<97x257x673xui8> {
// expected-error@+1 {{The size of the DMA transaction 16777217 Byte for a [97, 257, 673] tensor is greater than the limit 16777215 Byte}}
    %0 = VPUIP.NNDMA inputs(%arg0 : memref<97x257x673xui8>) outputs(%arg1 : memref<97x257x673xui8>) -> memref<97x257x673xui8>
    return %arg0 : memref<97x257x673xui8>
}

// -----

func.func @main() {
// expected-error@+1 {{BufferSection 'NetworkInput' is not compatible with memory space '@CMX_NN'}}
    %buf0 = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<10xf16, @CMX_NN>
}

// -----

func.func @main() {
// expected-error@+1 {{Output type must have DDR memory space}}
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<10xf16>
}

// -----

func.func @main() {
// expected-error@+1 {{Output type with DDR memory space cannot have section index}}
    %buf0 = VPURT.DeclareBuffer <DDR> [0] <0> -> memref<10xf16, [@DDR, 0]>
}

// -----

func.func @main() {
// expected-error@+1 {{Output type with DDR memory space cannot have section index}}
    %buf0 = VPURT.DeclareBuffer <DDR> [0] <0> -> memref<10xf16, @DDR>
}

// -----

func.func @main() {
// expected-error@+1 {{Section index is missing}}
    %buf0 = VPURT.DeclareBuffer <CMX_NN> <0> -> memref<10xf16, @CMX_NN>
}

// -----

func.func @main() {
// expected-error@+1 {{Array of section indexes is supported only for distributed buffer type}}
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> memref<10xf16, @CMX_NN>
}

// -----

func.func @main() {
// expected-error@+1 {{Section index '0' and memory space index '1' mismatch}}
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<10xf16, [@CMX_NN, 1]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @main() {
// expected-error@+1 {{Empty section index is not supported}}
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [] <0> -> !InputDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

func.func @main() {
// expected-error@+1 {{Number of clusters '2' and section indexes '1' mismatch}}
    %buf0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !InputDistributed
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func nested @cache_flush(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.task_type = @CACHE_FLUSH}
func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @InvalidCacheHandlingSwKernel(%1: memref<1x1x1x1000xf16, @DDR>, %2: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr  = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

   VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        // expected-error@+1 {{SW Kernel Cache Op should have no operands}}
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
                    @VPU.SW::@cache_flush
                    inputs(%in_ddr as %arg0: memref<1x1x1x1000xf16, @DDR>)
                    outputs(%out_ddr as %arg1: memref<1x1x1x1000xf16, @DDR>)
                    on tile 0 -> memref<1x1x1x1000xf16, @DDR>{
            VPUIP.SW.Kernel.run (%arg0, %arg1)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
        }
    }

    return %2 : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func nested @cache_flush(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.task_type = @CACHE_FLUSH}
func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

func.func @InvalidCacheHandlingFunc(%1: memref<1x1x1x1000xf16, @DDR>, %2: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %in_ddr  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
    %out_ddr  = VPURT.DeclareBuffer <DDR> <2000> -> memref<1x1x1x1000xf16, @DDR>

    %b0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %b1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

   VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%1 : memref<1x1x1x1000xf16, @DDR>) outputs(%in_ddr : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    }

    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        // expected-error@+1 {{SW Kernel Cache Op func should have no inputs}}
        VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 0, 0, 0>}
                    @VPU.SW::@cache_flush
                    inputs()
                    outputs()
                    on tile 0 {
            VPUIP.SW.Kernel.run
        }
    }

    return %2 : memref<1x1x1x1000xf16, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @SubviewWithStrideAttrInDifferentAxis(%arg0: memref<1x3x8x8xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x8xf16, {order = #NHWC}, @DDR> {
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x528x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x4x8xf16, {order = #NHWC}, @DDR>
    %3 = VPUIP.SubView %1 [0, 512, 0, 0] [1, 16, 8, 8] : memref<1x528x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> to memref<1x16x8x8xf16, {order = #NHWC, strides = [33792, 1, 4224, 528]}, [@CMX_NN, 0]>
    // expected-error@+1 {{The output stride([16384 Bit, 16 Bit, 2048 Bit, 256 Bit]) != infered stride([540672 Bit, 16 Bit, 67584 Bit, 8448 Bit])}}
    %4 = VPUIP.SubView %3 [0, 0, 4, 0] [1, 16, 4, 8] : memref<1x16x8x8xf16, {order = #NHWC, strides = [33792, 1, 4224, 528]}, [@CMX_NN, 0]> to memref<1x16x4x8xf16, {order = #NHWC, strides = [1024, 1, 128, 16]}, [@CMX_NN, 0]>
    %5 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%4 : memref<1x16x4x8xf16, {order = #NHWC, strides = [1024, 1, 128, 16]}, [@CMX_NN, 0]>) outputs(%2 : memref<1x16x4x8xf16, {order = #NHWC}, @DDR>) -> memref<1x16x4x8xf16, {order = #NHWC}, @DDR>
    return %5 : memref<1x16x4x8xf16, {order = #NHWC}, @DDR>
}
