//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swizzling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsSM_DDR = memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>

!InputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @ConvSparseWeights
func.func @ConvSparseWeights(%in : !Input_DDR) -> !OutputStub_CMX {

    %buf0 = memref.alloc() : !InputStub_CMX
    %buf1 = memref.alloc() : !OutputStub_CMX
    %buf2 = memref.alloc() : !WeightsTableStub_CMX
    %buf3 = memref.alloc() : !WeightsStub_CMX
    %buf4 = memref.alloc() : !WeightsSMStub_CMX

    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %0 = VPUIP.Copy inputs(%weight_table : !WeightsTable_DDR) outputs(%buf2 : !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
    %1 = VPUIP.Copy inputs(%in : !Input_DDR) outputs(%buf0 : !InputStub_CMX) -> !InputStub_CMX
    %2 = VPUIP.Copy inputs(%weights : !Weights_DDR) outputs(%buf3 : !WeightsStub_CMX) -> !WeightsStub_CMX
    %3 = VPUIP.Copy inputs(%weights_sm : !WeightsSM_DDR) outputs(%buf4 : !WeightsSMStub_CMX) -> !WeightsSMStub_CMX

    %4 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputStub_CMX)
        weights(%2 : !WeightsStub_CMX)
        weights_sparsity_map(%3 : !WeightsSMStub_CMX)
        weight_table(%0 : !WeightsTableStub_CMX)
        parent_input(%1 : !InputStub_CMX)
        parent_output(%buf1 : !OutputStub_CMX)
        outputs(%buf1 : !OutputStub_CMX) -> !OutputStub_CMX
        variants : {
            DPUTask { outEnd = [55, 10, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    return %4 : !OutputStub_CMX


    // CHECK:       [[OUTPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[INPUT_MEMREF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[WEIGHT_TABLE_BUFFER:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[WEIGHTS_BUFFER:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[WEIGHTS_SM_BUFFER:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

    // CHECK-DAG:       [[WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:       [[WT:%.*]] = VPUIP.Copy inputs([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:                          outputs([[WEIGHT_TABLE_BUFFER]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:                          -> memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[INPUT:%.*]] = VPUIP.Copy inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:                             outputs([[OUTPUT_MEMREF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:                             -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    // CHECK:       [[W:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:                         outputs([[WEIGHTS_BUFFER]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:                         -> memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[W_SM:%.*]] = VPUIP.Copy inputs([[WEIGHTS_SM]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:                            outputs([[WEIGHTS_SM_BUFFER]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:                            -> memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
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

!WeightsSMDistributed = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW, @CMX_NN, {
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
!WeightsSM_DDR = memref<32x1x1x128xi1, @DDR>
!WeightsTable_DDR = memref<32x1x1x4xsi32, @DDR>
!Output_DDR = memref<1x16x56x56xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsSMStub_CMX = memref<32x1x1x128xi1, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvSparseWeightsMulticlustering
func.func @ConvSparseWeightsMulticlustering(%input : !Input_DDR) -> !Output_DDR
{
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<32x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare !WeightsSM_DDR = dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %weights_sm_cmx = VPURT.AllocDistributed -> !WeightsSMDistributed
    %output_buff_1_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output_buff_2_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %output = memref.alloc() : !Output_DDR

    %1 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

    %2 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weight_table: !WeightsTable_DDR) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

    %3 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

    %4 = VPUIP.Copy { out_mem_space = @CMX_NN } inputs(%weights_sm: !WeightsSM_DDR) outputs(%weights_sm_cmx: !WeightsSMDistributed) -> !WeightsSMDistributed

    %5 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : !InputDistributed)
        weights(%3 : !WeightsDistributed)
        weights_sparsity_map(%4 : !WeightsSMDistributed)
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

    %6 = VPUIP.Copy { out_mem_space = @DDR } inputs(%5: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

    return %6 : !Output_DDR

    // Verify that alignment is set for constants which satisfy the 1024 size requirement

    // CHECK-DAG:       [[CST_WEIGHT_TABLE:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:       [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_WEIGHTS_SM:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0
    // CHECK-SAME:      outputs([[BUF_INPUT]]
    // CHECK:       [[COPY_WT:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CST_WEIGHT_TABLE]]
    // CHECK-SAME:      outputs([[BUF_WT]]
    // CHECK:       [[COPY_WEIGHTS:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CST_WEIGHTS]]
    // CHECK-SAME:      outputs([[BUF_WEIGHTS]]
    // CHECK:       [[NCE_CT_NCE_TASK:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CST_WEIGHTS_SM]]
    // CHECK-SAME:      outputs([[BUF_WEIGHTS_SM]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!WeightsStub = memref<16x16x1x1xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInOutSparse
func.func @SetSwizzlingForDpuToDpuBufferInOutSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0,
                        %weights : !WeightsStub)
                        -> (!IODDR0, !SMDDR0) {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0
    %buf1_sm = memref.alloc() : !SMCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf2_sm = memref.alloc() : !SMCMX0
    %buf3_data = memref.alloc() : !IODDR0
    %buf3_sm = memref.alloc() : !SMDDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1:2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%in0_data : !IOCMX0)
        parent_input_sparsity_map(%in0_sm : !SMCMX0)
        parent_output(%buf1_data : !IOCMX0)
        parent_output_sparsity_map(%buf1_sm : !SMCMX0)
        outputs(%buf1_data : !IOCMX0)
        output_sparsity_map(%buf1_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
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

    %2:2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1#0 : !IOCMX0)
        input_sparsity_map(%1#1 : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%1#0 : !IOCMX0)
        parent_input_sparsity_map(%1#1 : !SMCMX0)
        parent_output(%buf2_data : !IOCMX0)
        parent_output_sparsity_map(%1#1 : !SMCMX0)
        outputs(%buf2_data : !IOCMX0)
        output_sparsity_map(%buf2_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
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
            inputs(%2#0 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0
    %4 = VPUIP.Copy
            inputs(%2#1 : !SMCMX0)
            outputs(%buf3_sm : !SMDDR0)
             -> !SMDDR0

    return %3, %4 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_2_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_3_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_3_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @DDR>

    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_0_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_0_SM]] : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @CMX_NN>

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[COPY_0]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          weights(%arg3 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input([[COPY_0]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_1_DATA]] : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1_DATA]] : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)

    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[NCE_0]]#0 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:          weights(%arg3 : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_input([[NCE_0]]#0 : memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_2_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_2_DATA]] : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy inputs([[NCE_1]]#0 : memref<1x16x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:         outputs([[BUFF_3_DATA]] : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy inputs([[NCE_1]]#1 : memref<1x16x56x56xi1, #NHWC, @CMX_NN>)
    // CHECK-SAME:         outputs([[BUFF_3_SM]] : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @DDR>

    // CHECK:       return [[COPY_2]], [[COPY_3]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!WeightsStub = memref<16x16x1x1xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInSparse
func.func @SetSwizzlingForDpuToDpuBufferInSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0,
                        %weights : !WeightsStub)
                        -> !IODDR0 {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf3_data = memref.alloc() : !IODDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%in0_data : !IOCMX0)
        parent_output(%buf1_data : !IOCMX0)
        outputs(%buf1_data : !IOCMX0)
        -> !IOCMX0
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
        input(%1 : !IOCMX0)
        weights(%weights : !WeightsStub)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%1#0 : !IOCMX0)
        parent_output(%buf2_data : !IOCMX0)
        outputs(%buf2_data : !IOCMX0)
        -> !IOCMX0
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
            inputs(%2 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0

    return %3 : !IODDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @CMX_NN>
    // CHECK:       [[ALLOC_0:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// 1x16x170x171xf16 can fit to CMX, but additional in/out sparsity map can not
!WeightsTable0 = memref<16x1x1x4xsi32, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x170x171xf16, #NHWC, @DDR>
!IOCMX0 = memref<1x16x170x171xf16, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x170x171xi1, #NHWC, @DDR>
!SMCMX0 = memref<1x16x170x171xi1, #NHWC, @CMX_NN>

// CHECK-LABEL: @DoNotSetSwizzlingDueToCmxUsageIncreaseSparse
func.func @DoNotSetSwizzlingDueToCmxUsageIncreaseSparse(%in_data : !IODDR0,
                        %in_sm : !SMDDR0,
                        %weight_table : !WeightsTable0)
                        -> (!IODDR0, !SMDDR0) {

    %buf0_data = memref.alloc() : !IOCMX0
    %buf0_sm = memref.alloc() : !SMCMX0
    %buf1_data = memref.alloc() : !IOCMX0
    %buf1_sm = memref.alloc() : !SMCMX0

    %buf2_data = memref.alloc() : !IOCMX0
    %buf2_sm = memref.alloc() : !SMCMX0
    %buf3_data = memref.alloc() : !IODDR0
    %buf3_sm = memref.alloc() : !SMDDR0

    %in0_data = VPUIP.Copy
            inputs(%in_data : !IODDR0)
            outputs(%buf0_data : !IOCMX0)
             -> !IOCMX0
    %in0_sm = VPUIP.Copy
            inputs(%in_sm : !SMDDR0)
            outputs(%buf0_sm : !SMCMX0)
             -> !SMCMX0

    %1:2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%in0_data : !IOCMX0)
        input_sparsity_map(%in0_sm : !SMCMX0)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%in0_data : !IOCMX0)
        parent_input_sparsity_map(%in0_sm : !SMCMX0)
        parent_output(%buf1_data : !IOCMX0)
        parent_output_sparsity_map(%buf1_sm : !SMCMX0)
        outputs(%buf1_data : !IOCMX0)
        output_sparsity_map(%buf1_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
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

    %2:2 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }
        input(%1#0 : !IOCMX0)
        input_sparsity_map(%1#1 : !SMCMX0)
        weight_table(%weight_table : !WeightsTable0)
        parent_input(%1#0 : !IOCMX0)
        parent_input_sparsity_map(%1#1 : !SMCMX0)
        parent_output(%buf2_data : !IOCMX0)
        parent_output_sparsity_map(%1#1 : !SMCMX0)
        outputs(%buf2_data : !IOCMX0)
        output_sparsity_map(%buf2_sm : !SMCMX0)
        -> !IOCMX0, !SMCMX0
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
            inputs(%2#0 : !IOCMX0)
            outputs(%buf3_data : !IODDR0)
             -> !IODDR0
    %4 = VPUIP.Copy
            inputs(%2#1 : !SMCMX0)
            outputs(%buf3_sm : !SMDDR0)
             -> !SMDDR0

    return %3, %4 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK-NOT:  VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64}
    // CHECK-NOT:  VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64}
    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_3_DATA:%.+]] = memref.alloc() : memref<1x16x170x171xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_3_SM:%.+]] = memref.alloc() : memref<1x16x170x171xi1, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsTableDDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!WeightsTableCMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!DistrWeightsTableCMX = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsDDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!WeightsCMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!DistrWeightsCMX = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!SMCMX0 = memref<1x16x56x56xi1, #NHWC, @CMX_NN>
!SMDDR0 = memref<1x16x56x56xi1, #NHWC, @DDR>
!IODistrCMX0 = !VPUIP.DistributedBuffer<
    1x16x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>


!SMCMX1 = memref<1x16x56x56xi1, #NHWC, [@CMX_NN, 0]>
!SMCMX2 = !VPUIP.DistributedBuffer<
    1x16x56x56xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!IOCMX0 = memref<1x16x56x56xf16, #NHWC, [@CMX_NN, 0]>
!IOCMX1 = memref<1x16x56x56xf16, #NHWC, @CMX_NN>
!IODDR0 = memref<1x16x56x56xf16, #NHWC, @DDR>

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferInMultiClusterSparse
func.func @SetSwizzlingForDpuToDpuBufferInMultiClusterSparse(%arg0: !IODDR0, %arg1: !SMDDR0, %arg2: !WeightsTableDDR, %arg3: !WeightsDDR) -> (!IODDR0, !SMDDR0) {
    %0 = VPURT.AllocDistributed -> !IODistrCMX0
    %1 = VPURT.AllocDistributed -> !SMCMX2
    %2 = VPURT.AllocDistributed -> !DistrWeightsTableCMX
    %3 = VPURT.AllocDistributed -> !DistrWeightsCMX
    %4 = VPURT.AllocDistributed -> !IODistrCMX0
    %5 = VPURT.AllocDistributed -> !SMCMX2
    %6 = VPURT.AllocDistributed -> !IODistrCMX0
    %7 = VPURT.AllocDistributed -> !SMCMX2
    %8 = memref.alloc() : !IODDR0
    %9 = memref.alloc() : !SMDDR0
    %10 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg0 : !IODDR0) outputs(%0 : !IODistrCMX0) -> !IODistrCMX0
    %11 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg1 : !SMDDR0) outputs(%1 : !SMCMX2) -> !SMCMX2
    %12 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg2 : !WeightsTableDDR) outputs(%2 : !DistrWeightsTableCMX) -> !DistrWeightsTableCMX
    %13 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg3 : !WeightsDDR) outputs(%3 : !DistrWeightsCMX) -> !DistrWeightsCMX
    %14:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                kernel_strides = [1, 1],
                                task_type = #VPUIP.nce_task_type<CONV>}
        input(%10 : !IODistrCMX0)
        input_sparsity_map(%11 : !SMCMX2)
        weights(%13 : !DistrWeightsCMX)
        weight_table(%12 : !DistrWeightsTableCMX)
        parent_input(%10 : !IODistrCMX0)
        parent_input_sparsity_map(%11 : !SMCMX2)
        parent_output(%4 : !IODistrCMX0)
        parent_output_sparsity_map(%5 : !SMCMX2)
        outputs(%4 : !IODistrCMX0)
        output_sparsity_map(%5 : !SMCMX2)
        -> !IODistrCMX0, !SMCMX2 variants : {
    DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                kernel_strides = [1, 1],
                                task_type = #VPUIP.nce_task_type<CONV>}
        input(%14#0 : !IODistrCMX0)
        input_sparsity_map(%14#1 : !SMCMX2)
        weights(%13 : !DistrWeightsCMX)
        weight_table(%12 : !DistrWeightsTableCMX)
        parent_input(%14#0 : !IODistrCMX0)
        parent_input_sparsity_map(%14#1 : !SMCMX2)
        parent_output(%6 : !IODistrCMX0)
        parent_output_sparsity_map(%7 : !SMCMX2)
        outputs(%6 : !IODistrCMX0)
        output_sparsity_map(%7 : !SMCMX2)
        -> !IODistrCMX0, !SMCMX2 variants : {
    DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %16 = VPUIP.Copy {out_mem_space = @DDR} inputs(%15#0 : !IODistrCMX0) outputs(%8 : !IODDR0) -> !IODDR0
    %17 = VPUIP.Copy {out_mem_space = @DDR} inputs(%15#1 : !SMCMX2) outputs(%9 : !SMDDR0) -> !SMDDR0
    return %16, %17 : !IODDR0, !SMDDR0

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_W:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_4_DATA:%.+]] = memref.alloc() : memref<1x16x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_4_SM:%.+]] = memref.alloc() : memref<1x16x56x56xi1, #NHWC, @DDR>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg0 : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_0_DATA]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg1 : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_0_SM]] : !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_2:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg2 : memref<16x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_1]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[COPY_3:%.+]] = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg3 : memref<16x16x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_W]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[COPY_0]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weights([[COPY_3]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_input([[COPY_0]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_output([[BUFF_2_DATA]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[BUFF_2_DATA]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          output_sparsity_map([[BUFF_2_SM]] : !VPUIP.DistributedBuffer<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[NCE_0]]#0 : !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          input_sparsity_map([[NCE_0]]#1 : !VPUIP.DistributedBuffer<1x16x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weights([[COPY_3]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_input([[NCE_0]]#0 : !VPUIP.DistributedBuffer<1x16x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          parent_output([[BUFF_3_DATA]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[BUFF_3_DATA]] : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_4:%.+]] = VPUIP.Copy {out_mem_space = @DDR} inputs([[NCE_1]]#0 : !VPUIP.DistributedBuffer<1x16x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:         outputs([[BUFF_4_DATA]] : memref<1x16x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_5:%.+]] = VPUIP.Copy {out_mem_space = @DDR} inputs([[NCE_1]]#1 : !VPUIP.DistributedBuffer<1x16x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:         outputs([[BUFF_4_SM]] : memref<1x16x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:          -> memref<1x16x56x56xi1, #NHWC, @DDR>

    // CHECK:       return [[COPY_4]], [[COPY_5]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!WeightsSm0 = memref<32x1x1x128xi1, @DDR>
!IOCMX0 = memref<1x32x56x56xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTable0 = memref<32x1x1x4xsi32, #NHWC, @DDR>
!SMCMX0 = !VPUIP.DistributedBuffer<
    1x32x56x56xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!IODistrCMX0 = !VPUIP.DistributedBuffer<
    1x32x56x56xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!Weights1 = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTable1 = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

!Weights2 = memref<32x1x1x128xi1, @CMX_NN>
!SMCMX1 = memref<1x32x56x56xi1, #NHWC, @CMX_NN>
!Weights3 = !VPUIP.DistributedBuffer<
    32x1x1x128xi1, #NCHW,
    @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}>

!Weights4 = memref<32x32x1x1xf16, #NHWC, @DDR>
!IOCMX1 = memref<1x32x56x56xf16, #NHWC, @CMX_NN>
!WeightsTable2 = memref<32x1x1x4xsi32, @DDR>
!WeightsTable3 = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightsTable4 = memref<32x1x1x4xsi32, [@CMX_NN, 0]>
!IODDR0 = memref<1x32x56x56xf16, #NHWC, @DDR>
!SMDDR0 = memref<1x32x56x56xi1, #NHWC, @DDR>
!Weights5 = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!SMCMX2 = memref<1x32x56x56xi1, #NHWC, [@CMX_NN, 0]>

// [Track number: E#65141]

// CHECK-LABEL: @SetSwizzlingForDpuToDpuBufferWithWeightsInMultiClusterSparse
func.func @SetSwizzlingForDpuToDpuBufferWithWeightsInMultiClusterSparse(%arg0: !IODDR0, %arg1: !SMDDR0) -> (!IODDR0, !SMDDR0) {
    %cst_wt0 = const.Declare !WeightsTable0 = dense<1> : tensor<32x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %cst_wt2 = const.Declare !WeightsTable2 = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare !Weights4 = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_1 = const.Declare !WeightsSm0 = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    %buf0_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf0_sm = VPURT.AllocDistributed -> !SMCMX0

    %buf1_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf1_sm = VPURT.AllocDistributed -> !SMCMX0

    %buf2_data = VPURT.AllocDistributed -> !IODistrCMX0
    %buf2_sm = VPURT.AllocDistributed -> !SMCMX0

    %convW = VPURT.AllocDistributed -> !Weights1
    %convWSM = VPURT.AllocDistributed -> !Weights3
    %convWT = VPURT.AllocDistributed -> !WeightsTable3

    %maxpoolWT = VPURT.AllocDistributed -> !WeightsTable3

    %outData = memref.alloc() : !IODDR0
    %outSM  = memref.alloc() : !SMDDR0

    %13 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg0 : !IODDR0) outputs(%buf0_data : !IODistrCMX0) -> !IODistrCMX0
    %14 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%arg1 : !SMDDR0) outputs(%buf0_sm : !SMCMX0) -> !SMCMX0
    %15 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%cst_wt0 : !WeightsTable0) outputs(%maxpoolWT : !WeightsTable3) -> !WeightsTable3
    %17 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%cst_0 : !Weights4) outputs(%convW : !Weights1) -> !Weights1
    %18 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%cst_1 : !WeightsSm0) outputs(%convWSM : !Weights3) -> !Weights3
    %19 = VPUIP.Copy {out_mem_space = @CMX_NN} inputs(%cst_wt2 : !WeightsTable2) outputs(%convWT : !WeightsTable3) -> !WeightsTable3
    %20:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                kernel_strides = [1, 1],
                                task_type = #VPUIP.nce_task_type<CONV>}
        input(%13 : !IODistrCMX0)
        input_sparsity_map(%14 : !SMCMX0)
        weights(%17 : !Weights1)
        weights_sparsity_map(%18 : !Weights3)
        weight_table(%19 : !WeightsTable3)
        parent_input(%13 : !IODistrCMX0)
        parent_input_sparsity_map(%14 : !SMCMX0)
        parent_output(%buf1_data : !IODistrCMX0)
        parent_output_sparsity_map(%buf1_sm : !SMCMX0)
        outputs(%buf1_data : !IODistrCMX0)
        output_sparsity_map(%buf1_sm : !SMCMX0)
        -> !IODistrCMX0, !SMCMX0 variants : {
    DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %21:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                                kernel_strides = [1, 1],
                                task_type = #VPUIP.nce_task_type<MAXPOOL>}
        input(%20#0 : !IODistrCMX0)
        input_sparsity_map(%20#1 : !SMCMX0)
        weight_table(%15 : !WeightsTable3)
        parent_input(%20#0 : !IODistrCMX0)
        parent_input_sparsity_map(%20#1 : !SMCMX0)
        parent_output(%buf2_data : !IODistrCMX0)
        parent_output_sparsity_map(%buf2_sm : !SMCMX0)
        outputs(%buf2_data : !IODistrCMX0)
        output_sparsity_map(%buf2_sm : !SMCMX0)
        -> !IODistrCMX0, !SMCMX0 variants : {
    DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [55, 55, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %22 = VPUIP.Copy {out_mem_space = @DDR} inputs(%21#0 : !IODistrCMX0) outputs(%outData : !IODDR0) -> !IODDR0
    %23 = VPUIP.Copy {out_mem_space = @DDR} inputs(%21#1 : !SMCMX0) outputs(%outSM : !SMDDR0) -> !SMDDR0
    return %22, %23 : !IODDR0, !SMDDR0

    // CHECK-DAG:       [[CST_WT1:%.+]] = const.Declare memref<32x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<32x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_WT0:%.+]] = const.Declare memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[CST_W:%.+]] = const.Declare memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_W_SM:%.+]] = const.Declare memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR> = dense<1.000000e+00> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:       [[BUFF_0_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_0_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_4:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_5:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[BUFF_8_DATA:%.+]] = memref.alloc() : memref<1x32x56x56xf16, #NHWC, @DDR>
    // CHECK:       [[BUFF_8_SM:%.+]] = memref.alloc() : memref<1x32x56x56xi1, #NHWC, @DDR>

    // CHECK:       [[COPY_0_ACT:%.+]] = VPUIP.Copy
    // CHECK-SAME:         inputs(%arg0 : memref<1x32x56x56xf16, #NHWC, @DDR>)
    // CHECK-SAME:         outputs([[BUFF_0_DATA]] : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_1_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs(%arg1 : memref<1x32x56x56xi1, #NHWC, @DDR>)
    // CHECK-SAME:           outputs([[BUFF_0_SM]] : !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_WT1:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[CST_WT1]] : memref<32x1x1x4xsi32, #NHWC, @DDR>)
    // CHECK-SAME:           outputs([[BUFF_5]] : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_W:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[CST_W]] : memref<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:           outputs([[BUFF_3_DATA]] : !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_W_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[CST_W_SM]] : memref<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:           outputs([[BUFF_3_SM]] : !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[COPY_WT0:%.+]] = VPUIP.Copy
    // CHECK-SAME:           inputs([[CST_WT0]] : memref<32x1x1x4xsi32, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>)
    // CHECK-SAME:           outputs([[BUFF_4]] : !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[COPY_0_ACT]] : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          input_sparsity_map([[COPY_1_SM]] : !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weights([[COPY_W]] : !VPUIP.DistributedBuffer<32x32x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weights_sparsity_map([[COPY_W_SM]] : !VPUIP.DistributedBuffer<32x1x1x128xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weight_table([[COPY_WT0]] : !VPUIP.DistributedBuffer<32x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[BUFF_1_DATA]] : !VPUIP.DistributedBuffer<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          output_sparsity_map([[BUFF_1_SM]] : !VPUIP.DistributedBuffer<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    // CHECK:       [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[NCE_0]]#0 : !VPUIP.DistributedBuffer<1x32x56x56xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          input_sparsity_map([[NCE_0]]#1 : !VPUIP.DistributedBuffer<1x32x56x56xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          weight_table([[COPY_WT1]] : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          outputs([[BUFF_2_DATA]] : !VPUIP.DistributedBuffer<1x32x56x56xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    // CHECK-SAME:          output_sparsity_map([[BUFF_2_SM]] : !VPUIP.DistributedBuffer<1x32x56x56xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
}
