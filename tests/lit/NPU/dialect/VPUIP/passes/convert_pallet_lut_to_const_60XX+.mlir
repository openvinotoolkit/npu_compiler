//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-pallet-lut-to-const %s | FileCheck %s
// REQUIRES: platform-NPU6010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

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
    16x16x1x1x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 3 : i64,
        uniform_distributed_segments,
        compute_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[16, 16, 1, 1], [16, 16, 1, 1], [16, 16, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-LABEL: @ConvertPalletLUTToConstWithDistributedOp
func.func @ConvertPalletLUTToConstWithDistributedOp(%data: !InputDistributedType,
                                                 %weights: !WeightsDistributedType)
                                                 -> !OutputDistributedType {
    %conv_cmx_outbuf = VPURT.AllocDistributed -> !OutputDistributedType
    %output = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 366 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}>
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
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>}
    }

    return %output : !OutputDistributedType
}

// CHECK:       [[PALLET_LUT_CONST:%.+]] = const.Declare memref<64xui16> = dense<[48128, 47506, 47155, 46674, 46221, 45546, 44500, 0, 11544, 12582, 13280, 13672, 14093, 14464, 14793, 15360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xui16>
// CHECK:       [[ALLOC_PALLET_LUT:%.+]] = VPURT.AllocDistributed {alignment = 32 : i64} ->
// CHECK-SAME:      !VPUIP.DistributedBuffer<
// CHECK-SAME:              64xui16,
// CHECK-SAME:              #C, [@CMX_NN, 0], {
// CHECK-SAME:                  mode = "DUPLICATED",
// CHECK-SAME:                  num_clusters = 3 : i64}>
// CHECK:       [[COPY_LUT_CONST:%.+]] = VPUIP.Copy inputs([[PALLET_LUT_CONST]] : memref<64xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_PALLET_LUT]] : [[COPY_OUT_TYPE:!.+]]) -> [[COPY_OUT_TYPE]]
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:          pallet_lookup_table([[COPY_LUT_CONST]] : [[COPY_OUT_TYPE]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertPalletLUTToConstMemRef
func.func @ConvertPalletLUTToConstMemRef(%data: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>,
                                      %weights: memref<16x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
                                      -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN> {
    %conv_cmx_outbuf = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

    %nce_output = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 366 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                            task_type = #VPUIP.nce_task_type<CONV>
                        }>
                        input(%data: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        weights(%weights: memref<16x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
                        parent_input(%data: memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        parent_output(%conv_cmx_outbuf : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        outputs(%conv_cmx_outbuf : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 5, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [15, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>,
                                    clamp_low = -3.4028234663852886E+38 : f64,
                                    clamp_high = 3.4028234663852886E+38 : f64,
                                    prelu_alpha = [1.000000e+00],
                                    adder = 0.000000e+00 : f64>}
    }

    return %nce_output : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
}

// CHECK:       [[PALLET_LUT_CONST:%.+]] = const.Declare memref<64xui16> = dense<[48128, 47506, 47155, 46674, 46221, 45546, 44500, 0, 11544, 12582, 13280, 13672, 14093, 14464, 14793, 15360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xui16>
// CHECK:       [[ALLOC_PALLET_LUT:%.+]] = memref.alloc() : memref<64xui16, [@CMX_NN, 0]>
// CHECK:       [[COPY_LUT_CONST:%.+]] = VPUIP.Copy inputs([[PALLET_LUT_CONST]] : memref<64xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_PALLET_LUT]] : memref<64xui16, [@CMX_NN, 0]>) -> memref<64xui16, [@CMX_NN, 0]>
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      pallet_lookup_table([[COPY_LUT_CONST]] : memref<64xui16, [@CMX_NN, 0]>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:f8E4M3FN, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertPalletLUTHf8ToConstMemRef
func.func @ConvertPalletLUTHf8ToConstMemRef(%data: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>,
                                      %weights: memref<16x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN> {
    %conv_cmx_outbuf = memref.alloc() : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>

    %nce_output = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 366 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                            task_type = #VPUIP.nce_task_type<CONV>
                        }>
                        input(%data: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        weights(%weights: memref<16x32x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
                        parent_input(%data: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        parent_output(%conv_cmx_outbuf : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
                        outputs(%conv_cmx_outbuf : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [31, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 5, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [31, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 2 : i64, inEnd = [31, 4, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 4, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>,
                                    clamp_low = -3.4028234663852886E+38 : f64,
                                    clamp_high = 3.4028234663852886E+38 : f64,
                                    prelu_alpha = [1.000000e+00],
                                    adder = 0.000000e+00 : f64>}
    }

    return %nce_output : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>
}

// CHECK:       [[PALLET_LUT_CONST:%.+]] = const.Declare memref<64xui16> = dense<[47288, 46003, 45232, 44461, 43433, 42148, 40092, 0, 6682, 8738, 10280, 11051, 11822, 12593, 13364, 14392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xui16>
// CHECK:       [[ALLOC_PALLET_LUT:%.+]] = memref.alloc() : memref<64xui16, [@CMX_NN, 0]>
// CHECK:       [[COPY_LUT_CONST:%.+]] = VPUIP.Copy inputs([[PALLET_LUT_CONST]] : memref<64xui16>)
// CHECK-SAME:                                   outputs([[ALLOC_PALLET_LUT]] : memref<64xui16, [@CMX_NN, 0]>) -> memref<64xui16, [@CMX_NN, 0]>
// CHECK:       VPUIP.NCEClusterTask
// CHECK-SAME:      pallet_lookup_table([[COPY_LUT_CONST]] : memref<64xui16, [@CMX_NN, 0]>)
