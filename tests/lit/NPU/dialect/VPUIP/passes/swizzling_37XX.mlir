//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --swizzling %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SetSwizzlingForDpuToDpuBuffer(%in : memref<1x16x56x56xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
                        -> memref<1x16x56x56xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x56x56xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>) -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
            outputs(%buf3 : memref<1x16x56x56xf16, #NHWC, @DDR>)
             -> memref<1x16x56x56xf16, #NHWC, @DDR>

    return %3 : memref<1x16x56x56xf16, #NHWC, @DDR>

    // Verify that alignment is set only for DPU to DPU buffer

    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DoNotSetSwizzlingDueToCmxUsageIncrease(%in : memref<1x16x176x175xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
                        -> memref<1x16x176x175xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x176x175xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>) -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>) -> memref<1x16x176x175xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %3 = VPUIP.Copy
            inputs(%2 : memref<1x16x176x175xf16, #NHWC, @CMX_NN>)
            outputs(%buf3 : memref<1x16x176x175xf16, #NHWC, @DDR>)
             -> memref<1x16x176x175xf16, #NHWC, @DDR>

    return %3 : memref<1x16x176x175xf16, #NHWC, @DDR>

    // Verify that no swizzling is enabled

    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK-NOT:  VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK:      [[BUF1:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x176x175xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

func.func @SetSwizzlingForDpuToDpuBufferInMultiCluster(%input : !Input_DDR,
                        %weights_table : !WeightsTable_DDR,
                        %weights : !Weights_DDR)
                        -> !Output_DDR {

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %5 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%4 : !OutputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%4 : !OutputDistributed)
        parent_output(%output_buff_2_cmx : !OutputDistributed)
        outputs(%output_buff_2_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %6 = VPUIP.Copy { out_mem_space = @DDR } inputs(%5: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %6 : !Output_DDR

    // Verify that alignment is set only for DPU to DPU buffer

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_W:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX_SWIZZLED:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.Copy
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF_OUT_1_CMX_SWIZZLED]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SetSwizzlingForConstantsMultiClusterDuplicated
func.func @SetSwizzlingForConstantsMultiClusterDuplicated(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
            {
                outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]
            }
        }
        PPE :
        {
        }

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK-DAG:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHT:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN,  {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK:      [[NCE_CT_COPY_WT:%.+]] = VPUIP.Copy
    // CHECK:      [[NCE_CT_COPY_WEIGHTS:%.+]] = VPUIP.Copy
    // CHECK:      [[NCE_CT_NCE_TASK:%.+]] = VPUIP.NCEClusterTask
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!Input_DDR = memref<1x32x3x3xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x32x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x64x3x3xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<64x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @SetSwizzlingForConstantsSOK
func.func @SetSwizzlingForConstantsSOK(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 177 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_cmx : !OutputDistributed)
        outputs(%output_buff_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask {
                cluster_id = 0 : i64, outEnd = [2, 2, 63],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]}
        }
        PPE :
        {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK-DAG:       [[CST_WT:%.+]] = const.Declare memref<64x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK-DAG:       [[CST_W:%.+]] = const.Declare memref<64x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>

    // CHECK:   [[BUF_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}


// -----

!qElemType = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:0, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType5 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType6 = !quant.uniform<u8<0:254>:f16:1, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x32x32x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x16x1x1x!qElemType3, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32x!qElemType, #NHWC, @DDR>
!Output_DDR = memref<1x64x32x32x!qElemType1, #NHWC, @DDR>
!Weights_DDR = memref<64x16x1x1x!qElemType3, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = memref<1x16x32x32x!qElemType, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x64x32x32x!qElemType1, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<64x16x1x1x!qElemType3, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, #NCHW, @CMX_NN>

func.func @SetSwizzlingForQuantConstantsSOK(%input : !Input_DDR) -> !Output_DDR
{
    %weights = const.Declare memref<64x16x1x1x!qElemType3, #NHWC> =
        dense<1.0> : tensor<64x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights_table: memref<64x1x1x4xsi32>) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: memref<64x16x1x1x!qElemType3, #NHWC>) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 177 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_cmx : !OutputDistributed)
        outputs(%output_buff_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask {
                cluster_id = 0 : i64, outEnd = [2, 2, 63],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]}
        }
        PPE :
        {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = VPUIP.Copy { out_mem_space = @DDR } inputs(%4: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %5 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 512 size requirement

    // CHECK:   [[BUF_INPUT:%.+]] = VPURT.AllocDistributed
    // CHECK:   [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:   VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:   [[BUF_OUT_CMX:%.+]] = VPURT.AllocDistributed
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 2100000 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

func.func @DoNotSwizzleDueToAlignmentMemIncrease(%in : memref<1x16x180x180xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>,
                        %weights_0 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
                        -> memref<1x16x180x180xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x180x180xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x180x180xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>) -> memref<1x16x180x180xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        weights(%weights_0 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>) -> memref<1x16x180x180xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }


    %3 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%2 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%2 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>) -> memref<1x16x180x180xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x16x180x180xf16, #NHWC, @DDR>)
             -> memref<1x16x180x180xf16, #NHWC, @DDR>

    return %4 : memref<1x16x180x180xf16, #NHWC, @DDR>

    // Verify that swizzling is only assigned to the output buffer of the first conv
    // The second conv doesn't have output swizzling because of the memory exceeds CMX size when the input is swizzled
    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x16x180x180xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF4:%.+]] = memref.alloc() : memref<1x16x180x180xf16, #NHWC, @DDR>

    // CHECK:      [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x180x180xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK:      [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV0]] : memref<1x16x180x180xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
    // CHECK:      [[CONV2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV1]] : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF3]] : memref<1x16x180x180xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x174x174xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x174x174xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x174x174xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x174x174xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x174x174xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x174x174xf16, #NHWC, [@CMX_NN, 0]>

func.func @SetSwizzlingForConstantButNotActivationDueToCmxSizeLimit(%input : !Input_DDR)
                        -> !Output_DDR {

    %weights_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%1 : !InputDistributed)
        parent_output(%output_buff_1_cmx : !OutputDistributed)
        outputs(%output_buff_1_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %5 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%4 : !OutputDistributed)
        weights(%3 : !WeightsDistributed)
        weight_table(%2 : !WeightsTableDistributed)
        parent_input(%4 : !OutputDistributed)
        parent_output(%output_buff_2_cmx : !OutputDistributed)
        outputs(%output_buff_2_cmx : !OutputDistributed) -> !OutputDistributed
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
        }
        PPE :
        {
        }

    %6 = VPUIP.Copy { out_mem_space = @DDR } inputs(%5: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %6 : !Output_DDR

    // Check that swizzling was enabled only for constants but not on activation due to CMX limitation

    // CHECK:      [[CSR_WT:%.+]] = const.Declare memref<16x1x1x4xsi32
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    // CHECK:      [[CSR_W:%.+]] = const.Declare memref<16x16x1x1xf16
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_W:%.+]] = VPURT.AllocDistributed {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x174x174xf16, #NHWC, @DDR>

    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_WT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_W:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CT_NCE_TASK_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_NCE_TASK_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x174x174xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_COPY_OUTPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> memref<1x16x174x174xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferWithInplace
func.func @SetSwizzlingForDpuToDpuBufferWithInplace(%in0 : memref<1x240x8x98xf16, #NHWC, @DDR>,
                        %weight_table0 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>,
                        %weights0 : memref<80x240x3x3xf16, #NHWC, [@CMX_NN, 0]>,
                        %in1 : memref<1x80x8x98xf16, #NHWC, @DDR>,
                        %weight_table1 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>,
                        %weights1 : memref<80x80x3x3xf16, #NHWC, [@CMX_NN, 0]>)
                        -> memref<1x80x6x96xf16, #NHWC, @DDR> {
    %buf0 = memref.alloc() : memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>
    %buf1 = memref.alloc() : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf2 = memref.alloc() : memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>
    %buf3 = memref.alloc() : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf4 = memref.alloc() : memref<80x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %buf5 = memref.alloc() : memref<80x1x1x4xsi32, [@CMX_NN, 0]>
    %buf6 = memref.alloc() : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
    %buf7 = memref.alloc() : memref<1x80x6x96xf16, #NHWC, @DDR>

    // Conv output as weights for Add
    %0 = VPUIP.Copy
               inputs(%in0 : memref<1x240x8x98xf16, #NHWC, @DDR>)
               outputs(%buf0 : memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPUIP.NCEClusterTask
           {
               constantsFused = true,
               kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
               kernel_size = [3, 3],
               kernel_strides = [1, 1],
               task_type = #VPUIP.nce_task_type<CONV>
           }
           input(%0 : memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>)
           weights(%weights0 : memref<80x240x3x3xf16, #NHWC, [@CMX_NN, 0]>)
           weight_table(%weight_table0 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
           parent_input(%0 : memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>)
           parent_output(%buf1 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           outputs(%buf1 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
           variants :
           {
               DPUTask
                   {
                       inEnd = [97, 7, 239], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                       outEnd = [95, 5, 79], outStart = [0, 0, 0],
                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                   }
           }
           PPE :
           {
               PPETask
                   {
                       ppe = #VPU.PPEStub<>
                   }
           }

    // Conv output as activation for Add
    %2 = VPUIP.Copy
               inputs(%in1 : memref<1x80x8x98xf16, #NHWC, @DDR>)
               outputs(%buf2 : memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPUIP.NCEClusterTask
           {
               constantsFused = true,
               kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
               kernel_size = [3, 3],
               kernel_strides = [1, 1],
               task_type = #VPUIP.nce_task_type<CONV>
           }
           input(%2 : memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>)
           weights(%weights1 : memref<80x80x3x3xf16, #NHWC, [@CMX_NN, 0]>)
           weight_table(%weight_table1 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
           parent_input(%2 : memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>)
           parent_output(%buf3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           outputs(%buf3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
           variants :
           {
               DPUTask
                  {
                      inEnd = [97, 7, 79], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                      outEnd = [95, 5, 79], outStart = [0, 0, 0],
                      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                  }
           }
           PPE :
           {
               PPETask
                   {
                       ppe = #VPU.PPEStub<>
                   }
           }

    // Add with is_inplace = true
    %4 = VPUIP.NCEClusterTask
           {
               is_inplace = true,
               task_type = #VPUIP.nce_task_type<ELTWISE>
           }
           input(%3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           weights(%1 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           parent_input(%3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           parent_output(%buf3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           outputs(%buf3 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
           variants :
           {
               DPUTask
                   {
                       inEnd = [95, 5, 79], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                       outEnd = [95, 5, 79], outStart = [0, 0, 0],
                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                   }
           }
           PPE :
           {
               PPETask
                   {
                       ppe = #VPU.PPEStub<>
                   }
           }

    // DWConv
    %weights2 = const.Declare memref<80x16x1x1xf16, #NHWC> = dense<1.250000e-01> : tensor<80x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weight_table2 = const.Declare memref<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>
    %5 = VPUIP.Copy
               inputs(%weights2 : memref<80x16x1x1xf16, #NHWC>)
               outputs(%buf4 : memref<80x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
                -> memref<80x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.Copy
               inputs(%weight_table2 : memref<80x1x1x4xsi32>)
               outputs(%buf5 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
                -> memref<80x1x1x4xsi32, [@CMX_NN, 0]>
    %7 = VPUIP.NCEClusterTask
           {
               kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
               kernel_size = [1, 1], kernel_strides = [1, 1],
               task_type = #VPUIP.nce_task_type<DWCONV>
           }
           input(%4 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           weights(%5 : memref<80x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
           weight_table(%6 : memref<80x1x1x4xsi32, [@CMX_NN, 0]>)
           parent_input(%4 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           parent_output(%buf6 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
           outputs(%buf6 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
           variants :
           {
               DPUTask
                   {
                       inEnd = [95, 5, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                       outEnd = [95, 5, 63], outStart = [0, 0, 0],
                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                   }
               DPUTask
                   {
                       inEnd = [95, 5, 79], inStart = [0, 0, 64], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                       outEnd = [95, 5, 79], outStart = [0, 0, 64],
                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                   }
           }
           PPE :
           {
               PPETask
                   {
                       ppe = #VPU.PPEStub<>
                   }
           }

    %8 = VPUIP.Copy
            inputs(%7 : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%buf7 : memref<1x80x6x96xf16, #NHWC, @DDR>)
             -> memref<1x80x6x96xf16, #NHWC, @DDR>

    return %8 : memref<1x80x6x96xf16, #NHWC, @DDR>

    // Verify that alignment is set only for DPU to DPU buffer

    // CHECK:      [[BUF0_CONV0_ACT:%.+]] = memref.alloc() : memref<1x240x8x98xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:      [[BUF1_CONV0_OUT:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x80x6x96xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    // CHECK:      [[BUF2_CONV1_ACT:%.+]] = memref.alloc() : memref<1x80x8x98xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:      [[BUF3_CONV1_OUT:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<1x80x6x96xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    // CHECK:      [[BUF4_DWCONV_WEIGHTS:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<80x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    // CHECK:      [[BUF5_DWCONV_WEIGHTTABLE:%.+]] = VPURT.Alloc {alignment = 16384 : i64, swizzlingKey = 5 : i64} -> memref<80x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
    // CHECK:      [[BUF6_DWCONV_OUT:%.+]] = memref.alloc() : memref<1x80x6x96xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:      [[BUF7_COPY_OUT:%.+]] = memref.alloc() : memref<1x80x6x96xf16, #NHWC, @DDR>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1470000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

func.func @CannotSwizzledDueToMultiUserWhichCannotSwizzled(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>,
                        %arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>,
                        %weight_table_0 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weight_table_1 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
                        -> (memref<1x256x2x40xf16, #NHWC, @DDR>, memref<1x256x3x40xf16, #NHWC, @DDR>) {

    %buf0 = memref.alloc() : memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @DDR>
    %buf5 = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
             -> memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy
            inputs(%arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>)
            outputs(%buf1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
             -> memref<1x256x7x80xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NCEClusterTask
        {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 99787 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_0 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x256x5x80xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) -> memref<1x256x2x40xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        }
        PPE :
        {
            PPETask
            {
                ppe = #VPU.PPEStub<>
            }
        }

    %3 = VPUIP.NCEClusterTask
        {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3], kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 99787 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<256x256x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_1 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x256x7x80xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) -> memref<1x256x3x40xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [79, 4, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 1, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        }
        PPE :
        {
            PPETask
            {
                ppe = #VPU.PPEStub<>
            }
        }

    %4 = VPUIP.Copy
            inputs(%2 : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x256x2x40xf16, #NHWC, @DDR>)
             -> memref<1x256x2x40xf16, #NHWC, @DDR>

    %5 = VPUIP.Copy
            inputs(%3 : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
            outputs(%buf5 : memref<1x256x3x40xf16, #NHWC, @DDR>)
             -> memref<1x256x3x40xf16, #NHWC, @DDR>

    return %4, %5 : memref<1x256x2x40xf16, #NHWC, @DDR>, memref<1x256x3x40xf16, #NHWC, @DDR>

    // CHECK:      [[INPUT_1:%.+]] = memref.alloc() : memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[INPUT_2:%.+]] = memref.alloc() : memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_OUT_1:%.+]] = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_OUT_2:%.+]] = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @CMX_NN>
    // CHECK:      [[OUTPUT_1:%.+]] = memref.alloc() : memref<1x256x2x40xf16, #NHWC, @DDR>
    // CHECK:      [[OUTPUT_2:%.+]] = memref.alloc() : memref<1x256x3x40xf16, #NHWC, @DDR>

    // CHECK:      [[COPY_IN_1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x256x5x80xf16, #NHWC, @DDR>) outputs([[INPUT_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) -> memref<1x256x5x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[COPY_IN_2:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x256x7x80xf16, #NHWC, @DDR>) outputs([[INPUT_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) -> memref<1x256x7x80xf16, #NHWC, @CMX_NN>
    // CHECK:      [[CONV_1:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:   input([[COPY_IN_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) weights(%arg4 : memref<256x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg2 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:   parent_input([[COPY_IN_1]] : memref<1x256x5x80xf16, #NHWC, @CMX_NN>) parent_output([[CONV_OUT_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) outputs([[CONV_OUT_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:   -> memref<1x256x2x40xf16, #NHWC, @CMX_NN> variants : {
    // CHECK-NEXT:        DPUTask {cluster_id = 0 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:        DPUTask {cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:      } PPE : {
    // CHECK-NEXT:        PPETask {ppe = #VPU.PPEStub<>}
    // CHECK-NEXT:      }
    // CHECK:      [[CONV_2:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:   input([[COPY_IN_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) weights(%arg4 : memref<256x256x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<256x1x1x4xsi32, #NHWC, @CMX_NN>)
    // CHECK-SAME:   parent_input([[COPY_IN_2]] : memref<1x256x7x80xf16, #NHWC, @CMX_NN>) parent_output([[CONV_OUT_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) outputs([[CONV_OUT_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:   -> memref<1x256x3x40xf16, #NHWC, @CMX_NN> variants : {
    // CHECK-NEXT:        DPUTask {cluster_id = 0 : i64, inEnd = [79, 4, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:        DPUTask {cluster_id = 1 : i64, inEnd = [79, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [39, 0, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK-NEXT:      } PPE : {
    // CHECK-NEXT:        PPETask {ppe = #VPU.PPEStub<>}
    // CHECK-NEXT:      }
    // CHECK:      [[OUT_1:%.+]] = VPUIP.Copy inputs([[CONV_1]] : memref<1x256x2x40xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_1]] : memref<1x256x2x40xf16, #NHWC, @DDR>) -> memref<1x256x2x40xf16, #NHWC, @DDR>
    // CHECK:      [[OUT_2:%.+]] = VPUIP.Copy inputs([[CONV_2]] : memref<1x256x3x40xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_2]] : memref<1x256x3x40xf16, #NHWC, @DDR>) -> memref<1x256x3x40xf16, #NHWC, @DDR>
}
