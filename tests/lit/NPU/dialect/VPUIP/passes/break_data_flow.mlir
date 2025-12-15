//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --break-data-flow %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @LinearGraph
func.func @LinearGraph(%arg0: memref<10xf16>, %arg1: memref<10xf16>) -> memref<10xf16> {
    %0 = memref.alloc() : memref<10xf16>
    %1 = memref.alloc() : memref<10xf16>
    %token, %results = async.execute -> !async.value<memref<10xf16>> attributes {"async-deps-index" = 0 : i64} {
      %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
          inputs(%arg0 as %input: memref<10xf16>)
          outputs(%0 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
      }
      async.yield %3 : memref<10xf16>
    }
    %token_0, %results_1 = async.execute [%token] (%results as %arg2: !async.value<memref<10xf16>>) -> !async.value<memref<10xf16>> attributes {"async-deps-index" = 1 : i64} {
      %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
          inputs(%arg2 as %input: memref<10xf16>)
          outputs(%1 as %output: memref<10xf16>) on tile 0 -> memref<10xf16> {
          VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<10xf16>, memref<10xf16>
      }
      async.yield %3 : memref<10xf16>
    }
    %2 = async.await %results_1 : !async.value<memref<10xf16>>
    return %2 : memref<10xf16>

    // CHECK:       [[BUF0:%.+]] = memref.alloc() : memref<10xf16>
    // CHECK:       [[BUF1:%.+]] = memref.alloc() : memref<10xf16>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[BUF0]]
    // CHECK:       async.yield [[BUF0]] : memref<10xf16>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          ([[F1]] as [[VAL1:%arg[0-9]]]: !async.value<memref<10xf16>>)
    // CHECK:       VPUIP.SW.Kernel
    // CHECK-SAME:      inputs([[VAL1]]
    // CHECK-SAME:      outputs([[BUF1]]
    // CHECK:       async.yield [[BUF1]] : memref<10xf16>

    // CHECK:       [[VAL3:%.+]] = async.await [[F2]] : !async.value<memref<10xf16>>
    // CHECK:       return [[VAL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputSliceDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!OutputDistributedBuffer = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64
}>

!InputBufferDdr = memref<1x64x8x16xf16, #NHWC, @DDR>
!InputSliceBuffer = memref<1x64x8x16xf16, {order = #NHWC, strides = [16384, 1, 1024, 64]}, @CMX_NN>
!OutputBuffer = memref<1x64x16x16xf16, #NHWC, @CMX_NN>
!OutputBufferDdr = memref<1x64x16x16xf16, #NHWC, @DDR>

// CHECK-LABEL: @VPUIPConcatView
// CHECK-SAME: 		[[ARG0:%.+]]: memref<1x64x8x16xf16, #NHWC, @DDR>
// CHECK-SAME: 		[[ARG1:%.+]]: memref<1x64x16x16xf16, #NHWC, @DDR>
func.func @VPUIPConcatView(%arg0: !InputBufferDdr, %arg1: !OutputBufferDdr) -> !OutputBufferDdr {
    %0 = VPURT.AllocDistributed -> !InputDistributedBuffer

	%token_0, %results_0 = async.execute -> !async.value<!InputSliceDistributedBuffer> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    	%1 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    	%2 = VPUIP.NNDMA inputs(%arg0 : !InputBufferDdr) outputs(%1 : !InputSliceDistributedBuffer) -> !InputSliceDistributedBuffer
		async.yield %2 : !InputSliceDistributedBuffer
	}

	%token_1, %results_1 = async.execute -> !async.value<!InputSliceDistributedBuffer> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    	%1 = VPUIP.SubView %0 [0, 0, 8, 0] [1, 64, 8, 16] : !InputDistributedBuffer to !InputSliceDistributedBuffer
    	%2 = VPUIP.NNDMA inputs(%arg0 : !InputBufferDdr) outputs(%1 : !InputSliceDistributedBuffer) -> !InputSliceDistributedBuffer
		async.yield %2 : !InputSliceDistributedBuffer
	}

	%token_concat, %results__concat = async.execute [%token_0, %token_1] (%results_0 as %arg2: !async.value<!InputSliceDistributedBuffer>,
																		  %results_1 as %arg3: !async.value<!InputSliceDistributedBuffer>)
			-> !async.value<!OutputBufferDdr> attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 1 : i64} {
    	%1 = VPUIP.ConcatView inputs(%arg2, %arg3 : !InputSliceDistributedBuffer, !InputSliceDistributedBuffer) outputs(%0 : !OutputDistributedBuffer) -> !OutputDistributedBuffer
    	%2 = VPUIP.NNDMA inputs(%1 : !OutputDistributedBuffer) outputs(%arg1 : !OutputBufferDdr) -> !OutputBufferDdr
		async.yield %2 : !OutputBufferDdr
	}

	%1 = async.await %results__concat : !async.value<!OutputBufferDdr>
    return %1 : !OutputBufferDdr

	// CHECK: [[ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>

	// CHECK: 		[[T0:%.+]], [[F0:%.+]] = async.execute
    // CHECK:   		[[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]]
    // CHECK:   		{{[^:]+}} = VPUIP.NNDMA
	// CHECK-SAME: 						inputs([[ARG0]]
	// CHECK-SAME: 						outputs([[SUBVIEW0]]
    // CHECK:   		async.yield [[SUBVIEW0]]


    // CHECK: 		[[T1:%.+]], [[F1:%.+]] = async.execute
    // CHECK: 		  	[[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]]
    // CHECK: 		  	{{[^:]+}} = VPUIP.NNDMA
	// CHECK-SAME: 						inputs([[ARG0]]
	// CHECK-SAME: 						outputs([[SUBVIEW0]]
    // CHECK: 		  	async.yield [[SUBVIEW1]]


    // CHECK: 		[[T2:%.+]], [[F2:%.+]] = async.execute [[[T0]], [[T1]]]
    // CHECK: 		    {{[^:]+}} = VPUIP.NNDMA
    // CHECK-SAME:                      inputs([[ALLOC]]
    // CHECK-SAME:                      outputs([[ARG1]]
    // CHECK: 		  async.yield [[ARG1]] : memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK: [[VAL:%.+]] = async.await [[F2]] : !async.value<memref<1x64x16x16xf16, #NHWC, @DDR>>
    // CHECK: return [[VAL]] : memref<1x64x16x16xf16, #NHWC, @DDR>
}
