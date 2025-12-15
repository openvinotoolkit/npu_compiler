//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-sprlut-to-const %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x16x16xf16,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]],
        memory_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]]}>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x16x16xf16,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]],
        memory_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]]}>

!WeightsDistributedType = !VPUIP.DistributedBuffer<
    16x16x1x1xf16,
    #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @ConvertSprLUTToConstWithDistributedOp
func.func @ConvertSprLUTToConstWithDistributedOp(%data: !InputDistributedType,
                                                 %weights: !WeightsDistributedType)
                                                 -> !OutputDistributedType {
    %conv_cmx_outbuf = VPURT.AllocDistributed -> !OutputDistributedType
    %output = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 366 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
        input(%data : !InputDistributedType)
        weights(%weights : !WeightsDistributedType)
        parent_input(%data : !InputDistributedType)
        parent_output(%conv_cmx_outbuf : !OutputDistributedType)
        outputs(%conv_cmx_outbuf : !OutputDistributedType)
    ->  !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 5, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <TANH>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64, sprlut = dense<0> : tensor<224xui16>>}
    }

    return %output : !OutputDistributedType
}

// CHECK:       [[SPRLUT_CONST:%.+]] = const.Declare memref<224xui16> = dense<0> : tensor<224xui16>
// CHECK:       [[ALLOC_SPRLUT:%.+]] = VPURT.AllocDistributed {alignment = 32 : i64} ->
// CHECK-SAME:      !VPUIP.DistributedBuffer<
// CHECK-SAME:              224xui16,
// CHECK-SAME:              #C, [@CMX_NN, 0], {
// CHECK-SAME:                  mode = "DUPLICATED",
// CHECK-SAME:                  num_clusters = 3 : i64}>
// CHECK:       [[COPY_SPRLUT:%.+]] = VPUIP.Copy inputs([[SPRLUT_CONST]] : memref<224xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_SPRLUT]] : [[COPY_OUT_TYPE:!.+]]) -> [[COPY_OUT_TYPE]]
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:          spr_lookup_table([[COPY_SPRLUT]] : [[COPY_OUT_TYPE]]

// CHECK-NOT:   sprlut

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSprLUTToConstMemRef
func.func @ConvertSprLUTToConstMemRef(%data: memref<1x16x16x16xf16, #NHWC, @CMX_NN>,
                                      %weights: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
                                      -> memref<1x16x16x16xf16, #NHWC, @CMX_NN> {
    %conv_cmx_outbuf = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    %nce_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            minimumHardwareExecutionCost = 366 : i64,
                            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%data: memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
                        weights(%weights: memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
                        parent_input(%data: memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
                        parent_output(%conv_cmx_outbuf : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
                        outputs(%conv_cmx_outbuf : memref<1x16x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x16x16xf16, #NHWC, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 5, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <TANH>,
                                    clamp_low = -3.4028234663852886E+38 : f64,
                                    clamp_high = 3.4028234663852886E+38 : f64,
                                    prelu_alpha = [1.000000e+00],
                                    adder = 0.000000e+00 : f64,
                                    sprlut = dense<0> : tensor<224xui16>>}
    }

    return %nce_output : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// CHECK:       [[SPRLUT_CONST:%.+]] = const.Declare memref<224xui16> = dense<0> : tensor<224xui16>
// CHECK:       [[ALLOC_SPRLUT:%.+]] = memref.alloc() : memref<224xui16, [@CMX_NN, 0]>
// CHECK:       [[COPY_SPRLUT:%.+]] = VPUIP.Copy inputs([[SPRLUT_CONST]] : memref<224xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_SPRLUT]] : memref<224xui16, [@CMX_NN, 0]>) -> memref<224xui16, [@CMX_NN, 0]>
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      spr_lookup_table([[COPY_SPRLUT]] : memref<224xui16, [@CMX_NN, 0]>)

// CHECK-NOT:   sprlut

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributedType = !VPUIP.DistributedBuffer<
    1x16x16x16xf16,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]],
        memory_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]]}>

!InputDistributedType = !VPUIP.DistributedBuffer<
    1x16x16x16xf16,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 3, 1],
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]],
        memory_shapes = [[1, 16, 6, 16], [1, 16, 5, 16], [1, 16, 5, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 11, 0]]}>

!WeightsDistributedType = !VPUIP.DistributedBuffer<
    16x16x1x1xf16,
    #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @ConvertSprLUTToConstDistrBuf
func.func @ConvertSprLUTToConstDistrBuf(%data: !InputDistributedType,
                                        %weights: !WeightsDistributedType)
                                        -> !OutputDistributedType {
    %conv_cmx_outbuf = VPURT.AllocDistributed -> !OutputDistributedType

    %nce_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            minimumHardwareExecutionCost = 366 : i64,
                            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%data: !InputDistributedType)
                        weights(%weights: !WeightsDistributedType)
                        parent_input(%data: !InputDistributedType)
                        parent_output(%conv_cmx_outbuf : !OutputDistributedType)
                        outputs(%conv_cmx_outbuf : !OutputDistributedType) -> !OutputDistributedType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 5, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <TANH>,
                                    clamp_low = -3.4028234663852886E+38 : f64,
                                    clamp_high = 3.4028234663852886E+38 : f64,
                                    prelu_alpha = [1.000000e+00],
                                    adder = 0.000000e+00 : f64,
                                    sprlut = dense<0> : tensor<224xui16>>}
    }

    return %nce_output : !OutputDistributedType
}

// CHECK:       [[SPRLUT_CONST:%.+]] = const.Declare memref<224xui16> = dense<0> : tensor<224xui16>
// CHECK:       [[ALLOC_SPRLUT:%.+]] = VPURT.AllocDistributed {alignment = 32 : i64} ->
// CHECK-SAME:      !VPUIP.DistributedBuffer<
// CHECK-SAME:              224xui16,
// CHECK-SAME:              #C, [@CMX_NN, 0], {
// CHECK-SAME:                  mode = "DUPLICATED",
// CHECK-SAME:                  num_clusters = 3 : i64}>
// CHECK:       [[COPY_SPRLUT:%.+]] = VPUIP.Copy inputs([[SPRLUT_CONST]] : memref<224xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_SPRLUT]] : [[COPY_OUT_TYPE:!.+]]) -> [[COPY_OUT_TYPE]]
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      spr_lookup_table([[COPY_SPRLUT]] : [[COPY_OUT_TYPE]])

// CHECK-NOT:   sprlut
