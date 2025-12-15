//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --optimize-copies="workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

!qElemType = !quant.uniform<u8:f16, 0.012266390931372549:116>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 32], [1, 32, 267, 32], [1, 32, 266, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 32], [1, 32, 269, 32], [1, 32, 267, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!SubviewDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!InputDDRType = memref<1x32x800x32x!qElemType, #NHWC , @DDR>
!OutputDDRType = memref<1x32x800x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @DDR>

// CHECK-LABEL: func.func @OptimizeCopyOpSequence
func.func @OptimizeCopyOpSequence(%arg0: !InputDistributedType, %arg1: !InputDistributedType) {
    %0 = VPURT.AllocDistributed -> !InputDistributedType
    %1 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : !InputDistributedType)
        weights(%arg1 : !InputDistributedType)
        parent_input(%arg0 : !InputDistributedType)
        parent_output(%0 : !InputDistributedType)
        outputs(%0 : !InputDistributedType)
        -> !InputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [31, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [31, 267, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 267, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [31, 266, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }
    %2 = memref.alloc() : !OutputDDRType
    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 32, 800, 31] : !InputDistributedType to !SubviewDistributedType
    %4 = VPUIP.Copy inputs(%3 : !SubviewDistributedType) outputs(%2 : !OutputDDRType) -> !OutputDDRType
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.Copy inputs(%4 : !OutputDDRType) outputs(%5 : !OutputDistributedType) -> !OutputDistributedType

    return

    // CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask

    // CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[NCE_TASK:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK:      [[COPY1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]]
    // CHECK-NOT:  [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY1]]
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

!qElemType = !quant.uniform<u8:f16, 0.012266390931372549:116>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 32], [1, 32, 267, 32], [1, 32, 266, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 32], [1, 32, 269, 32], [1, 32, 267, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!InputDDRType = memref<1x32x800x32x!qElemType, #NHWC , @DDR>
!OutputDDRType = memref<1x32x800x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @DDR>

// CHECK-LABEL: func.func @OptimizeCopyOpSequenceWithSubview
func.func @OptimizeCopyOpSequenceWithSubview(%arg0: !InputDistributedType, %arg1: !InputDistributedType) {
    %0 = VPURT.AllocDistributed -> !InputDistributedType
    %1 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : !InputDistributedType)
        weights(%arg1 : !InputDistributedType)
        parent_input(%arg0 : !InputDistributedType)
        parent_output(%0 : !InputDistributedType)
        outputs(%0 : !InputDistributedType)
        -> !InputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [31, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [31, 267, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 267, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [31, 266, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }
    %2 = memref.alloc() : !InputDDRType
    %3 = VPUIP.Copy inputs(%1 : !InputDistributedType) outputs(%2 : !InputDDRType) -> !InputDDRType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 800, 31] : !InputDDRType to !OutputDDRType
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.Copy inputs(%4 : !OutputDDRType) outputs(%5 : !OutputDistributedType) -> !OutputDistributedType

    return

    // CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK:      [[ALLOC:%.+]] = VPURT.AllocDistributed

    // CHECK-NOT:  [[COPY:%.+]] = VPUIP.Copy inputs([[NCE_TASK]]
    // CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[NCE_TASK:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]]
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 32], [1, 32, 267, 32], [1, 32, 266, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 32], [1, 32, 269, 32], [1, 32, 267, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!InputDDRType = memref<1x32x800x32xf16, #NHWC , @DDR>
!OutputDDRType = memref<1x32x800x31xf16, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @DDR>

// CHECK-LABEL: func.func @NotOptimizeCopyOpSequenceWithSubviewDueToLongSpilling
func.func @NotOptimizeCopyOpSequenceWithSubviewDueToLongSpilling(%arg0: !InputDistributedType, %arg1: !InputDistributedType) {
    // parent op
    %0 = VPURT.AllocDistributed -> !InputDistributedType
    %1 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : !InputDistributedType)
        weights(%arg1 : !InputDistributedType)
        parent_input(%arg0 : !InputDistributedType)
        parent_output(%0 : !InputDistributedType)
        outputs(%0 : !InputDistributedType)
        -> !InputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [31, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [31, 267, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 267, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [31, 266, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }
    %2 = memref.alloc() : !InputDDRType
    %3 = VPUIP.Copy inputs(%1 : !InputDistributedType) outputs(%2 : !InputDDRType) -> !InputDDRType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 800, 31] : !InputDDRType to !OutputDDRType
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.Copy inputs(%4 : !OutputDDRType) outputs(%5 : !OutputDistributedType) -> !OutputDistributedType

    // middle op
    %7 = VPURT.AllocDistributed -> !OutputDistributedType
    %8 = VPURT.AllocDistributed -> !OutputDistributedType
    %9 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%7 : !OutputDistributedType)
        weights(%8 : !OutputDistributedType)
        parent_input(%7 : !OutputDistributedType)
        parent_output(%8 : !OutputDistributedType)
        outputs(%8 : !OutputDistributedType)
        -> !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [30, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 267, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [30, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 268, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [30, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }

    // user op
    %10 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%6 : !OutputDistributedType)
        weights(%9 : !OutputDistributedType)
        parent_input(%6 : !OutputDistributedType)
        parent_output(%9 : !OutputDistributedType)
        outputs(%9 : !OutputDistributedType)
        -> !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [30, 267, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 267, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [30, 268, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 268, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [30, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [30, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }

    return

    // CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-NOT:  [[SUBVIEW:%.+]] = VPUIP.SubView [[NCE_TASK:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[COPY_TO_DDR:%.+]] = VPUIP.Copy inputs([[NCE_TASK]]
    // CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[COPY_TO_DDR:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[COPY_TO_CMX:%.+]] = VPUIP.Copy inputs([[SUBVIEW]]
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

!qElemType = !quant.uniform<u8:f16, 0.012266390931372549:116>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!InputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 32], [1, 32, 267, 32], [1, 32, 266, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 32], [1, 32, 269, 32], [1, 32, 267, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x31x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 267, 31], [1, 32, 267, 31], [1, 32, 266, 31]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 267, 0], [0, 0, 534, 0]],
    memory_shapes = [[1, 32, 268, 31], [1, 32, 269, 31], [1, 32, 267, 31]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 266, 0], [0, 0, 533, 0]]
}>

!IncompatibleDistributedType = !VPUIP.DistributedBuffer<
    1x32x800x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 800, 32], [1, 32, 800, 32], [1, 32, 800, 32]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 32, 800, 32], [1, 32, 800, 32], [1, 32, 800, 32]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDDRType = memref<1x32x800x32x!qElemType, #NHWC , @DDR>
!OutputDDRType = memref<1x32x800x31x!qElemType, {order = #NHWC, strides = [819200, 1, 1024, 32]}, @DDR>

// CHECK-LABEL: func.func @NotOptimizeCopyOpSequenceWithSubviewDueToIncompatibleUsers
func.func @NotOptimizeCopyOpSequenceWithSubviewDueToIncompatibleUsers(%arg0: !InputDistributedType, %arg1: !InputDistributedType) {
    %0 = VPURT.AllocDistributed -> !InputDistributedType
    %1 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : !InputDistributedType)
        weights(%arg1 : !InputDistributedType)
        parent_input(%arg0 : !InputDistributedType)
        parent_output(%0 : !InputDistributedType)
        outputs(%0 : !InputDistributedType)
        -> !InputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [31, 266, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [31, 267, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 267, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [31, 266, 31], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [31, 266, 31], outStart = [0, 1, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.130000e+02 : f64, clamp_high = 1.420000e+02 : f64, scale = 2.5052577257156372E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.130000e+02 : f64, in1_mult = [2.572400e+04], in2_mult = [1.985700e+04]>}
    }
    %2 = memref.alloc() : !InputDDRType
    %3 = VPUIP.Copy inputs(%1 : !InputDistributedType) outputs(%2 : !InputDDRType) -> !InputDDRType
    %4 = VPUIP.SubView %3 [0, 0, 0, 0] [1, 32, 800, 31] : !InputDDRType to !OutputDDRType
    %5 = VPURT.AllocDistributed -> !OutputDistributedType
    %6 = VPUIP.Copy inputs(%4 : !OutputDDRType) outputs(%5 : !OutputDistributedType) -> !OutputDistributedType
    %7 = VPURT.AllocDistributed -> !IncompatibleDistributedType
    %8 = VPUIP.Copy inputs(%3 : !InputDDRType) outputs(%7 : !IncompatibleDistributedType) -> !IncompatibleDistributedType

    return

    // CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-NOT:  [[SUBVIEW:%.+]] = VPUIP.SubView [[NCE_TASK:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[COPY_TO_DDR:%.+]] = VPUIP.Copy inputs([[NCE_TASK]]
    // CHECK:      [[SUBVIEW:%.+]] = VPUIP.SubView [[COPY_TO_DDR:%.+]] [0, 0, 0, 0] [1, 32, 800, 31]
    // CHECK:      [[COPY_TO_CMX1:%.+]] = VPUIP.Copy inputs([[SUBVIEW]]
    // CHECK:      [[COPY_TO_CMX2:%.+]] = VPUIP.Copy inputs([[COPY_TO_DDR]]
}
