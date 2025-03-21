//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// -----
// Case 1: DUPLICATED clustering mode, copy DDR to CMX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_ui4  = memref<1x1x1x111xui4, @DDR>
!DDR_f16 = memref<1x1x1x111xf16, @DDR>

!CMX_ui4 = memref<1x1x1x111xui4, @CMX_NN>

!type_Distributed_ui4 = !VPUIP.DistributedBuffer<
    1x1x1x111xui4, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_Distributed_f16 = !VPUIP.DistributedBuffer<
    1x1x1x111xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_Copy(memref<*xui4, @DDR>, memref<*xui4, @CMX_NN>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @DuplicatedClusteringSWCopyCaseDDRtoCMX
// CHECK-SAME: ([[ORIG_INPUT:%.+]]: memref<1x1x1x111xui4, @DDR>, [[ORIG_OUTPUT:%.+]]: memref<1x1x1x111xf16, @DDR>)

func.func @DuplicatedClusteringSWCopyCaseDDRtoCMX(%orig_input: !DDR_ui4, %orig_output: !DDR_f16) -> !DDR_f16 {
    %input_ddr_ui4 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_ui4
    %input_cmx_ui4 = VPURT.DeclareBuffer <CMX_NN> <0> -> !type_Distributed_ui4
    %output_cmx_f16 = VPURT.DeclareBuffer <CMX_NN> <64> -> !type_Distributed_f16

    %barrier_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%barrier_0 : !VPURT.Barrier) {
        %nndma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%orig_input : !DDR_ui4) outputs(%input_ddr_ui4 : !DDR_ui4) -> !DDR_ui4
    }
    VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_2 : !VPURT.Barrier) {
      %sw_copy = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs(%input_ddr_ui4 as %input_kernel_run: !DDR_ui4) outputs(%input_cmx_ui4 as %output_kernel_run: !CMX_ui4) on tile 0 -> !type_Distributed_ui4{
        VPUIP.SW.Kernel.run(%input_kernel_run, %output_kernel_run) : !DDR_ui4, !CMX_ui4
      }
    }
    VPURT.Task waits(%barrier_3 : !VPURT.Barrier) {
      %nndma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%output_cmx_f16 : !type_Distributed_f16) outputs(%orig_output : !DDR_f16) -> !DDR_f16
    }
    return %orig_output : !DDR_f16
}

// CHECK:      [[INPUT_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x111xui4, @DDR>
// CHECK:      [[INPUT_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x111xui4, @DDR>
// CHECK:      [[INPUT_DDR_2:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x111xui4, @DDR>
// CHECK:      [[OUTPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x111xui4, [@CMX_NN, 0]>
// CHECK:      [[OUTPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x111xui4, [@CMX_NN, 1]>
// CHECK:      [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x1x1x111xf16, [@CMX_NN, 0]>

// CHECK:      [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

// CHECK:      VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
// CHECK:        [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[ORIG_INPUT]] : memref<1x1x1x111xui4, @DDR>) outputs([[INPUT_DDR_0]] : memref<1x1x1x111xui4, @DDR>) -> memref<1x1x1x111xui4, @DDR>
// CHECK:      }

// CHECK:      VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:        [[SW_COPY_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_DDR_1]] as [[KERNEL_RUN_INPUT:%.+]]: memref<1x1x1x111xui4, @DDR>) outputs([[OUTPUT_CMX_0]] as [[KERNEL_RUN_OUTPUT:%.+]]: memref<1x1x1x111xui4, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x111xui4, [@CMX_NN, 0]>{
// CHECK:          VPUIP.SW.Kernel.run([[KERNEL_RUN_INPUT]], [[KERNEL_RUN_OUTPUT]]) : memref<1x1x1x111xui4, @DDR>, memref<1x1x1x111xui4, [@CMX_NN, 0]>
// CHECK:        }
// CHECK:      }
// CHECK:      VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:        [[SW_COPY_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_DDR_2]] as [[KERNEL_RUN_INPUT:%.+]]: memref<1x1x1x111xui4, @DDR>) outputs([[OUTPUT_CMX_1]] as [[KERNEL_RUN_OUTPUT:%.+]]: memref<1x1x1x111xui4, [@CMX_NN, 1]>) on tile 1 -> memref<1x1x1x111xui4, [@CMX_NN, 1]>{
// CHECK:          VPUIP.SW.Kernel.run([[KERNEL_RUN_INPUT]], [[KERNEL_RUN_OUTPUT]]) : memref<1x1x1x111xui4, @DDR>, memref<1x1x1x111xui4, [@CMX_NN, 1]>
// CHECK:        }
// CHECK:      }

// CHECK:      VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
// CHECK:        [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[IN_CMX]] : memref<1x1x1x111xf16, [@CMX_NN, 0]>) outputs([[ORIG_OUTPUT]] : memref<1x1x1x111xf16, @DDR>) -> memref<1x1x1x111xf16, @DDR>
// CHECK:      }
// CHECK:    }



// -----
// Case 2: DUPLICATED clustering mode, copy CMX to DDR

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_i8  = memref<1x2x3x4xsi8, @DDR>
!DDR_i4 = memref<1x2x3x4xsi4, @DDR>

!CMX_i4 = memref<1x2x3x4xsi4, @CMX_NN>

!type_Distributed_i8 = !VPUIP.DistributedBuffer<
    1x2x3x4xsi8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!type_Distributed_i4 = !VPUIP.DistributedBuffer<
    1x2x3x4xsi4, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_Copy(memref<*xi4, @CMX_NN>, memref<*xi4, @DDR>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @DuplicatedClusteringSWCopyCaseCMXtoDDR
// CHECK-SAME: ([[ORIG_INPUT:%.+]]: memref<1x2x3x4xsi8, @DDR>, [[ORIG_OUTPUT:%.+]]: memref<1x2x3x4xsi4, @DDR>)

func.func @DuplicatedClusteringSWCopyCaseCMXtoDDR(%orig_input: !DDR_i8, %orig_output: !DDR_i4) -> !DDR_i4 {
    %output_ddr_i4 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_i4
    %output_cmx_i8 = VPURT.DeclareBuffer <CMX_NN> <0> -> !type_Distributed_i8
    %input_cmx_i4 = VPURT.DeclareBuffer <CMX_NN> <64> ->!type_Distributed_i4

    %barrier_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%barrier_0 : !VPURT.Barrier) {
        %nndma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%orig_input : !DDR_i8) outputs(%output_cmx_i8 : !type_Distributed_i8) -> !type_Distributed_i8
    }
    VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_2 : !VPURT.Barrier) {
        %sw_copy = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs(%input_cmx_i4 as %kernel_run_input: !CMX_i4) outputs(%output_ddr_i4 as %kernel_run_output: !DDR_i4) on tile 0 -> !DDR_i4{
            VPUIP.SW.Kernel.run(%kernel_run_input, %kernel_run_output) : !CMX_i4, !DDR_i4
        }
    }
    VPURT.Task waits(%barrier_3 : !VPURT.Barrier) {
        %nndma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%output_ddr_i4 : !DDR_i4) outputs(%orig_output: !DDR_i4) -> !DDR_i4
    }
    return %orig_output : !DDR_i4
}


// CHECK:      [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xsi4, @DDR>
// CHECK:      [[OUTPUT_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xsi4, @DDR>
// CHECK:      [[OUTPUT_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x3x4xsi4, @DDR>

// CHECK:      [[OUTPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x2x3x4xsi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:      [[INPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <64> -> memref<1x2x3x4xsi4, [@CMX_NN, 0]>
// CHECK:      [[INPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <64> -> memref<1x2x3x4xsi4, [@CMX_NN, 1]>

// CHECK:      [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:      [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

// CHECK:      VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
// CHECK:        [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[ORIG_INPUT]] : memref<1x2x3x4xsi8, @DDR>) outputs([[OUTPUT_CMX_0]] : !VPUIP.DistributedBuffer<1x2x3x4xsi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x2x3x4xsi8, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:      }

// CHECK:      VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:        [[SW_COPY_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_CMX_0]] as [[KERNEL_RUN_INPUT:%.+]]: memref<1x2x3x4xsi4, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR_0]] as [[KERNEL_RUN_OUTPUT:%.+]]: memref<1x2x3x4xsi4, @DDR>) on tile 0 -> memref<1x2x3x4xsi4, @DDR>{
// CHECK:          VPUIP.SW.Kernel.run([[KERNEL_RUN_INPUT]], [[KERNEL_RUN_OUTPUT]]) : memref<1x2x3x4xsi4, [@CMX_NN, 0]>, memref<1x2x3x4xsi4, @DDR>
// CHECK:        }
// CHECK:      }
// CHECK:      VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:        [[SW_COPY_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_CMX_1]] as [[KERNEL_RUN_INPUT:%.+]]: memref<1x2x3x4xsi4, [@CMX_NN, 1]>) outputs([[OUTPUT_DDR_1]] as [[KERNEL_RUN_OUTPUT:%.+]]: memref<1x2x3x4xsi4, @DDR>) on tile 1 -> memref<1x2x3x4xsi4, @DDR>{
// CHECK:          VPUIP.SW.Kernel.run([[KERNEL_RUN_INPUT]], [[KERNEL_RUN_OUTPUT]]) : memref<1x2x3x4xsi4, [@CMX_NN, 1]>, memref<1x2x3x4xsi4, @DDR>
// CHECK:        }
// CHECK:      }

// CHECK:      VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
// CHECK:        [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_DDR]] : memref<1x2x3x4xsi4, @DDR>) outputs([[ORIG_OUTPUT]] : memref<1x2x3x4xsi4, @DDR>) -> memref<1x2x3x4xsi4, @DDR>
// CHECK:      }



// -----
// Case 3: SEGMENTED clustering mode, copy DDR to CMX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_ui4  = memref<1x1x20x20xui4, @DDR>
!DDR_i8  = memref<1x1x20x20xsi8, @DDR>

!CMX_ui4 = memref<1x1x20x20xui4, @CMX_NN>

!type_Distributed_ui4 =  !VPUIP.DistributedBuffer<
    1x1x20x20xui4, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!type_Distributed_i8 = !VPUIP.DistributedBuffer<
    1x1x20x20xsi8, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_Copy(memref<*xui4, @DDR>, memref<*xui4, @CMX_NN>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @SegmentedClusteringSWCopyCaseDDRtoCMX
// CHECK-SAME: ([[ORIG_INPUT:%.+]]: memref<1x1x20x20xui4, @DDR>)

func.func @SegmentedClusteringSWCopyCaseDDRtoCMX(%orig_input: !DDR_ui4) -> !DDR_i8 {
    %orig_output = VPURT.DeclareBuffer <DDR> <0> -> !DDR_i8
    %output_ddr_u4 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_ui4
    %output_cmx_ui4 = VPURT.DeclareBuffer <CMX_NN> <0> -> !type_Distributed_ui4
    %input_cmx_i8 = VPURT.DeclareBuffer <CMX_NN> <128> -> !type_Distributed_i8

    %barrier_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%barrier_0 : !VPURT.Barrier) {
        %nndma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%orig_input : !DDR_ui4) outputs(%output_ddr_u4 : !DDR_ui4) -> !DDR_ui4
    }
    VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_2 : !VPURT.Barrier) {
        %sw_copy = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs(%output_ddr_u4 as %kernel_run_input: !DDR_ui4) outputs(%output_cmx_ui4 as %kernel_run_output: !CMX_ui4) on tile 0 -> !type_Distributed_ui4{
            VPUIP.SW.Kernel.run(%kernel_run_input, %kernel_run_output) : !DDR_ui4, !CMX_ui4
        }
    }
    VPURT.Task waits(%barrier_3 : !VPURT.Barrier) {
        %nndma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%input_cmx_i8 : !type_Distributed_i8) outputs(%orig_output : !DDR_i8) -> !DDR_i8
    }
    return %orig_output : !DDR_i8
}


// CHECK:        [[OUTPUT_TILE_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:        [[OUTPUT_TILE_1:%.+]] = VPURT.DeclareBuffer <DDR> <200> -> memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>

// CHECK:        [[OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x20x20xui4, @DDR>
// CHECK:        [[INPUT_DDR_0:%.+]]= VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:        [[INPUT_DDR_1:%.+]]= VPURT.DeclareBuffer <DDR> <100> -> memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>

// CHECK:        [[OUTPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x10x20xui4, [@CMX_NN, 0]>
// CHECK:        [[OUTPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x10x20xui4, [@CMX_NN, 1]>
// CHECK:        [[INPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x10x20xsi8, [@CMX_NN, 0]>
// CHECK:        [[INPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <128> -> memref<1x1x10x20xsi8, [@CMX_NN, 1]>

// CHECK:        [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:        [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:        [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:        [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

// CHECK:        VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
// CHECK:          [[NNDMA:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[ORIG_INPUT]] : memref<1x1x20x20xui4, @DDR>) outputs([[OUTPUT_DDR]] : memref<1x1x20x20xui4, @DDR>) -> memref<1x1x20x20xui4, @DDR>
// CHECK:        }

// CHECK:        VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:          [[SW_COPY_0:%.+]]  = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_DDR_0]]as [[KERNEL_RUN_IN:%.+]]: memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) outputs([[OUTPUT_CMX_0]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x1x10x20xui4, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x10x20xui4, [@CMX_NN, 0]>{
// CHECK:            VPUIP.SW.Kernel.run([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>, memref<1x1x10x20xui4, [@CMX_NN, 0]>
// CHECK:          }
// CHECK:        }
// CHECK:        VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:          [[SW_COPY_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_DDR_1]]as [[KERNEL_RUN_IN:%.+]]: memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) outputs([[OUTPUT_CMX_1]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x1x10x20xui4, [@CMX_NN, 1]>) on tile 1 -> memref<1x1x10x20xui4, [@CMX_NN, 1]>{
// CHECK:            VPUIP.SW.Kernel.run([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x1x10x20xui4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>, memref<1x1x10x20xui4, [@CMX_NN, 1]>
// CHECK:          }
// CHECK:        }

// CHECK:        VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
// CHECK:          [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_CMX_0]] : memref<1x1x10x20xsi8, [@CMX_NN, 0]>) outputs([[OUTPUT_TILE_0]] : memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) -> memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:        }
// CHECK:        VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
// CHECK:          [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[INPUT_CMX_1]] : memref<1x1x10x20xsi8, [@CMX_NN, 1]>) outputs([[OUTPUT_TILE_1]] : memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) -> memref<1x1x10x20xsi8, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:        }



// -----
// Case 4: SEGMENTED clustering mode, copy CMX to DDR

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!DDR_f16  = memref<1x1x20x20xf16, @DDR>
!DDR_i4  = memref<1x1x20x20xsi4, @DDR>

!CMX_i4 = memref<1x1x20x20xsi4, @CMX_NN>

!type_Distributed_f16 =  !VPUIP.DistributedBuffer<
    1x1x20x20xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!type_Distributed_i4 = !VPUIP.DistributedBuffer<
    1x1x20x20xsi4, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_Copy(memref<*xsi4, @CMX_NN>, memref<*xsi4, @DDR>) attributes {VPU.kernel_code = "copy.cpp", VPU.kernel_entry = "copy", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @SegmentedClusteringSWCopyCaseCMXtoDDR
// CHECK-SAME: ([[ORIG_OUTPUT:%.+]]: memref<1x1x20x20xsi4, @DDR>)

func.func @SegmentedClusteringSWCopyCaseCMXtoDDR(%orig_output: !DDR_i4) -> !DDR_i4 {
    %orig_input = VPURT.DeclareBuffer <DDR> <0> -> !DDR_f16
    %output_ddr_i4 = VPURT.DeclareBuffer <DDR> <0> -> !DDR_i4
    %output_cmx_f16 = VPURT.DeclareBuffer <CMX_NN> <0> -> !type_Distributed_f16
    %input_cmx_i4 = VPURT.DeclareBuffer <CMX_NN> <448> -> !type_Distributed_i4

    %barrier_0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %barrier_3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%barrier_0 : !VPURT.Barrier) {
        %nndma_0 = VPUIP.NNDMA {port = 0 : i64} inputs(%orig_input : !DDR_f16) outputs(%output_cmx_f16 : !type_Distributed_f16 ) -> !type_Distributed_f16
    }
    VPURT.Task waits(%barrier_1 : !VPURT.Barrier) updates(%barrier_2 : !VPURT.Barrier) {
        %sw_copy = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs(%input_cmx_i4 as %kernel_run_input: !CMX_i4) outputs(%output_ddr_i4 as %kernel_run_output: !DDR_i4) on tile 0 -> !DDR_i4{
            VPUIP.SW.Kernel.run(%kernel_run_input, %kernel_run_output) : !CMX_i4, !DDR_i4
        }
    }
    VPURT.Task waits(%barrier_3 : !VPURT.Barrier) {
        %nndma_1 = VPUIP.NNDMA {port = 0 : i64} inputs(%output_ddr_i4 : !DDR_i4) outputs(%orig_output : !DDR_i4) -> !DDR_i4
    }
    return %orig_output : !DDR_i4
}


// CHECK:          [[INPUT_TILE_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x10x20xf16, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:          [[INPUT_TILE_1:%.+]] = VPURT.DeclareBuffer <DDR> <400> -> memref<1x1x10x20xf16, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>

// CHECK:          [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x20x20xsi4, @DDR>
// CHECK:          [[OUTPUT_DDR_0:%.+]]  = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:          [[OUTPUT_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>

// CHECK:          [[OUTPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x10x20xf16, [@CMX_NN, 0]>
// CHECK:          [[OUTPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x10x20xf16, [@CMX_NN, 1]>
// CHECK:          [[INPUT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <448> -> memref<1x1x10x20xsi4, [@CMX_NN, 0]>
// CHECK:          [[INPUT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <448> -> memref<1x1x10x20xsi4, [@CMX_NN, 1]>

// CHECK:          [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:          [[BARRIER_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:          [[BARRIER_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
// CHECK:          [[BARRIER_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

// CHECK:          VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
// CHECK:            [[NNDMA_0:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_TILE_0]] : memref<1x1x10x20xf16, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) outputs([[OUTPUT_CMX_0]] : memref<1x1x10x20xf16, [@CMX_NN, 0]>) -> memref<1x1x10x20xf16, [@CMX_NN, 0]>
// CHECK:          }
// CHECK:          VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier) {
// CHECK:            [[NNDMA_1:%.+]] = VPUIP.NNDMA {port = 1 : i64} inputs([[INPUT_TILE_1]] : memref<1x1x10x20xf16, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) outputs([[OUTPUT_CMX_1]] : memref<1x1x10x20xf16, [@CMX_NN, 1]>) -> memref<1x1x10x20xf16, [@CMX_NN, 1]>
// CHECK:          }

// CHECK:          VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:            [[SW_COPY_0:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_CMX_0]] as [[KERNEL_RUN_IN:%.+]]: memref<1x1x10x20xsi4, [@CMX_NN, 0]>) outputs([[OUTPUT_DDR_0]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) on tile 0 -> memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>{
// CHECK:              VPUIP.SW.Kernel.run([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x1x10x20xsi4, [@CMX_NN, 0]>, memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:            }
// CHECK:          }
// CHECK:          VPURT.Task waits([[BARRIER_1]] : !VPURT.Barrier) updates([[BARRIER_2]] : !VPURT.Barrier) {
// CHECK:            [[SW_COPY_1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Copy inputs([[INPUT_CMX_1]] as [[KERNEL_RUN_IN:%.+]]: memref<1x1x10x20xsi4, [@CMX_NN, 1]>) outputs([[OUTPUT_DDR_1]] as [[KERNEL_RUN_OUT:%.+]]: memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>) on tile 1 -> memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>{
// CHECK:              VPUIP.SW.Kernel.run([[KERNEL_RUN_IN]], [[KERNEL_RUN_OUT]]) : memref<1x1x10x20xsi4, [@CMX_NN, 1]>, memref<1x1x10x20xsi4, {order = #NCHW, strides = [400, 400, 20, 1]}, @DDR>
// CHECK:            }
// CHECK:          }

// CHECK:          VPURT.Task waits([[BARRIER_3]] : !VPURT.Barrier) {
// CHECK:            [[NNDMA:%.+]] = VPUIP.NNDMA {port = 0 : i64} inputs([[INPUT_DDR]] : memref<1x1x20x20xsi4, @DDR>) outputs([[ORIG_OUTPUT]] : memref<1x1x20x20xsi4, @DDR>) -> memref<1x1x20x20xsi4, @DDR>
// CHECK:          }
