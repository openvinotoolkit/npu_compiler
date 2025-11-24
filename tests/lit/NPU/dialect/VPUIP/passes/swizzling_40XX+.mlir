//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --swizzling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1470000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

func.func @DoNotSwizzleDueToAlignmentMemIncrease(%in : memref<1x16x149x150xf16, #NHWC, @DDR>,
                        %weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>,
                        %weights_0 : memref<16x16x3x3xf16, #NHWC, @CMX_NN>)
                        -> memref<1x16x149x150xf16, #NHWC, @DDR> {

    %buf0 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%in : memref<1x16x149x150xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
             -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>

    %1 = VPUIP.NCEClusterTask
        {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
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
            kernel_size = [3, 3],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights_0 : memref<16x16x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    outEnd = [55, 55, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
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
        input(%2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table : memref<16x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%2 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>) -> memref<1x16x149x150xf16, #NHWC, @CMX_NN>
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
            inputs(%3 : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x16x149x150xf16, #NHWC, @DDR>)
             -> memref<1x16x149x150xf16, #NHWC, @DDR>

    return %4 : memref<1x16x149x150xf16, #NHWC, @DDR>

    // Verify that swizzling is only assigned to the output buffer of the first conv
    // The second conv doesn't have output swizzling because of the memory exceeds CMX size when the input is swizzled
    // CHECK:      [[BUF0:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF1:%.+]] = VPURT.Alloc {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>
    // CHECK:      [[BUF2:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF3:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @CMX_NN>
    // CHECK:      [[BUF4:%.+]] = memref.alloc() : memref<1x16x149x150xf16, #NHWC, @DDR>

    // CHECK:      [[CONV0:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      outputs([[BUF1]] : memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK:      [[CONV1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV0]] : memref<1x16x149x150xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF2]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
    // CHECK:      [[CONV2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[CONV1]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUF3]] : memref<1x16x149x150xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x148x148xf16, #NHWC, @CMX_NN, {
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
    1x16x148x148xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x148x148xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x148x148xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x16x148x148xf16, #NHWC, [@CMX_NN, 0]>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
!OutputStub_CMX = memref<1x16x148x148xf16, #NHWC, [@CMX_NN, 0]>

config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1470000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

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
    input(%1 : !InputDistributed)
    weights(%3 : !WeightsDistributed)
    weight_table(%2 : !WeightsTableDistributed)
    parent_input(%1 : !InputDistributed)
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
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}
    // CHECK:      [[CSR_W:%.+]] = const.Declare memref<16x16x1x1xf16
    // CHECK-SAME:    swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}

    // CHECK:      [[BUF_INPUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_WT:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_W:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64} -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_1_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_2_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:      [[BUF_OUT_DDR:%.+]] = memref.alloc() : memref<1x16x148x148xf16, #NHWC, @DDR>

    // CHECK:      [[NCE_CT_COPY_INPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_WT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CST_COPY_W:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<16x16x1x1xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:      [[NCE_CT_NCE_TASK_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_NCE_TASK_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:   -> !VPUIP.DistributedBuffer<1x16x148x148xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> variants

    // CHECK:      [[NCE_CT_COPY_OUTPUT:%.+]] = VPUIP.Copy
    // CHECK-SAME:   -> memref<1x16x148x148xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1470000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}
//CHECK : func.func @CannotSwizzledDueToMultiUserWhichCannotSwizzled
func.func @CannotSwizzledDueToMultiUserWhichCannotSwizzled(%arg0 : memref<1x128x5x40xf16, #NHWC, @DDR>,
                        %arg1 : memref<1x128x7x40xf16, #NHWC, @DDR>,
                        %weight_table_0 : memref<128x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weight_table_1 : memref<128x1x1x4xsi32, #NHWC, @CMX_NN>,
                        %weights : memref<128x128x3x3xf16, #NHWC, @CMX_NN>)
                        -> (memref<1x128x2x20xf16, #NHWC, @DDR>, memref<1x128x3x20xf16, #NHWC, @DDR>) {

    %buf0 = memref.alloc() : memref<1x128x5x40xf16, #NHWC, @CMX_NN>
    %buf1 = memref.alloc() : memref<1x128x7x40xf16, #NHWC, @CMX_NN>
    %buf2 = memref.alloc() : memref<1x128x2x20xf16, #NHWC, @CMX_NN>
    %buf3 = memref.alloc() : memref<1x128x3x20xf16, #NHWC, @CMX_NN>
    %buf4 = memref.alloc() : memref<1x128x2x20xf16, #NHWC, @DDR>
    %buf5 = memref.alloc() : memref<1x128x3x20xf16, #NHWC, @DDR>

    %0 = VPUIP.Copy
            inputs(%arg0 : memref<1x128x5x40xf16, #NHWC, @DDR>)
            outputs(%buf0 : memref<1x128x5x40xf16, #NHWC, @CMX_NN>)
             -> memref<1x128x5x40xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.Copy
            inputs(%arg1 : memref<1x128x7x40xf16, #NHWC, @DDR>)
            outputs(%buf1 : memref<1x128x7x40xf16, #NHWC, @CMX_NN>)
             -> memref<1x128x7x40xf16, #NHWC, @CMX_NN>

    %2 = VPUIP.NCEClusterTask
        {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [3, 3],
            kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 99787 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%0 : memref<1x128x5x40xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<128x128x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_0 : memref<128x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%0 : memref<1x128x5x40xf16, #NHWC, @CMX_NN>)
        parent_output(%buf2 : memref<1x128x2x20xf16, #NHWC, @CMX_NN>)
        outputs(%buf2 : memref<1x128x2x20xf16, #NHWC, @CMX_NN>) -> memref<1x128x2x20xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [39, 2, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 0, 127], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [39, 2, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 0, 127], outStart = [0, 0, 0],
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
        input(%1 : memref<1x128x7x40xf16, #NHWC, @CMX_NN>)
        weights(%weights : memref<128x128x3x3xf16, #NHWC, @CMX_NN>)
        weight_table(%weight_table_1 : memref<128x1x1x4xsi32, #NHWC, @CMX_NN>)
        parent_input(%1 : memref<1x128x7x40xf16, #NHWC, @CMX_NN>)
        parent_output(%buf3 : memref<1x128x3x20xf16, #NHWC, @CMX_NN>)
        outputs(%buf3 : memref<1x128x3x20xf16, #NHWC, @CMX_NN>) -> memref<1x128x3x20xf16, #NHWC, @CMX_NN>
        variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, inEnd = [39, 4, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 1, 127], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, inEnd = [39, 2, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [19, 0, 127], outStart = [0, 0, 0],
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
            inputs(%2 : memref<1x128x2x20xf16, #NHWC, @CMX_NN>)
            outputs(%buf4 : memref<1x128x2x20xf16, #NHWC, @DDR>)
             -> memref<1x128x2x20xf16, #NHWC, @DDR>

    %5 = VPUIP.Copy
            inputs(%3 : memref<1x128x3x20xf16, #NHWC, @CMX_NN>)
            outputs(%buf5 : memref<1x128x3x20xf16, #NHWC, @DDR>)
             -> memref<1x128x3x20xf16, #NHWC, @DDR>

    return %4, %5 : memref<1x128x2x20xf16, #NHWC, @DDR>, memref<1x128x3x20xf16, #NHWC, @DDR>

    // CHECK:	 [[INPUT_1:%.+]] = memref.alloc() : memref<1x128x5x40xf16, #NHWC, @CMX_NN>
    // CHECK:    [[INPUT_2:%.+]] = memref.alloc() : memref<1x128x7x40xf16, #NHWC, @CMX_NN>
    // CHECK:    [[CONV_OUT_1:%.+]] = memref.alloc() : memref<1x128x2x20xf16, #NHWC, @CMX_NN>
    // CHECK:    [[CONV_OUT_2:%.+]] = memref.alloc() : memref<1x128x3x20xf16, #NHWC, @CMX_NN>
    // CHECK:    [[OUTPUT_1:%.+]] = memref.alloc() : memref<1x128x2x20xf16, #NHWC, @DDR>
    // CHECK:    [[OUTPUT_2:%.+]] = memref.alloc() : memref<1x128x3x20xf16, #NHWC, @DDR>
    // CHECK:    [[COPY_IN_1:%.+]] = VPUIP.Copy inputs(%arg0 : memref<1x128x5x40xf16, #NHWC, @DDR>) outputs([[INPUT_1]] : memref<1x128x5x40xf16, #NHWC, @CMX_NN>) -> memref<1x128x5x40xf16, #NHWC, @CMX_NN>
    // CHECK:    [[COPY_IN_2:%.+]] = VPUIP.Copy inputs(%arg1 : memref<1x128x7x40xf16, #NHWC, @DDR>) outputs([[INPUT_2]] : memref<1x128x7x40xf16, #NHWC, @CMX_NN>) -> memref<1x128x7x40xf16, #NHWC, @CMX_NN>
    // CHECK:    [[CONV_1:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK:    [[CONV_2:%.+]] = VPUIP.NCEClusterTask {is_superdense, kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 99787 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK:    [[OUT_1:%.+]]= VPUIP.Copy inputs([[CONV_1]] : memref<1x128x2x20xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_1]] : memref<1x128x2x20xf16, #NHWC, @DDR>) -> memref<1x128x2x20xf16, #NHWC, @DDR>
    // CHECK:    [[OUT_2:%.+]] = VPUIP.Copy inputs([[CONV_2]] : memref<1x128x3x20xf16, #NHWC, @CMX_NN>) outputs([[OUTPUT_2]] : memref<1x128x3x20xf16, #NHWC, @DDR>) -> memref<1x128x3x20xf16, #NHWC, @DDR>
}

// -----

// CHECK-LABEL: @DoNotSwizzle5dTensors

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!InputBuffer = !VPUIP.DistributedBuffer<25x1x64x2x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
    memory_shapes = [[7, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4], [6, 1, 64, 2, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
}>

!OutputBuffer = !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
    memory_shapes = [[7, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4], [6, 1, 16, 2, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
}>

!WeightsBuffer = !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
    memory_shapes = [[7, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
}>

!WeightTableBuffer = !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]],
    memory_shapes = [[7, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [13, 0, 0, 0, 0], [19, 0, 0, 0, 0]]
}>

config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1470000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 1 of @DPU
}

func.func @DoNotSwizzle5dTensors(
    %IN: !InputBuffer,
    %OUT: !OutputBuffer
) -> !OutputBuffer {
// CHECK:           [[IN:%arg.+]]: !VPUIP.DistributedBuffer<25x1x64x2x4xf16, #GNHWC, @CMX_NN
// CHECK-SAME:      [[OUT:%arg.+]]: !VPUIP.DistributedBuffer<25x1x16x2x4xf16, #GNHWC, @CMX_NN
    %WEIGHTS_CST = const.Declare memref<25x16x64x1x1xf16, #GNHWC> = dense<1.0> : tensor<1x25x16x64xf16>,
        [#const.Reshape<[25, 16, 64, 1, 1]>, #const.Reorder<#GNHWC>]
    // CHECK-NOT: swizzlingScheme = #VPUIP.SwizzlingSchemeAttr
    // CHECK:   [[WEIGHTS_CST:%.+]] = const.Declare memref<25x16x64x1x1xf16, #GNHWC>

    %WEIGHT_TABLE_CST = const.Declare memref<25x16x1x1x4xsi32> = dense<1> : tensor<25x16x1x1x4xsi32>
    // CHECK:   [[WEIGHT_TABLE_CST:%.+]] = const.Declare memref<25x16x1x1x4xsi32>

    %ALLOC_WEIGHTS = VPURT.AllocDistributed -> !WeightsBuffer
    // CHECK:   [[ALLOC_WEIGHTS:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<25x16x64x1x1xf16, #GNHWC, @CMX_NN,

    %WEIGHTS = VPUIP.Copy
        inputs(%WEIGHTS_CST : memref<25x16x64x1x1xf16, #GNHWC>)
        outputs(%ALLOC_WEIGHTS : !WeightsBuffer) -> !WeightsBuffer
    // CHECK:   [[WEIGHTS:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[WEIGHTS_CST]]
    // CHECK-SAME:  outputs([[ALLOC_WEIGHTS]]

    %ALLOC_WEIGHT_TABLE = VPURT.AllocDistributed -> !WeightTableBuffer
    // CHECK:   [[ALLOC_WEIGHT_TABLE:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<25x16x1x1x4xsi32, #NCDHW, @CMX_NN

    %WEIGHT_TABLE = VPUIP.Copy
        inputs(%WEIGHT_TABLE_CST : memref<25x16x1x1x4xsi32>)
        outputs(%ALLOC_WEIGHT_TABLE : !WeightTableBuffer) -> !WeightTableBuffer
    // CHECK:   [[WEIGHT_TABLE:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[WEIGHT_TABLE_CST]]
    // CHECK-SAME:  outputs([[ALLOC_WEIGHT_TABLE]]

    %MATMUL = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%IN : !InputBuffer)
    weights(%WEIGHTS : !WeightsBuffer)
    weight_table(%WEIGHT_TABLE : !WeightTableBuffer)
    parent_input(%IN : !InputBuffer)
    parent_output(%OUT : !OutputBuffer)
    outputs(%OUT : !OutputBuffer) -> !OutputBuffer
    variants : {
        DPUTask {
            cluster_id = 0 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 1 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 2 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
            cluster_id = 3 : i64,
            inEnd = [3, 1, 63],
            inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [3, 1, 15],
            outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
    } PPE : {
        PPETask {
            ppe = #VPU.PPEStub<>
        }
    }
    // CHECK:   [[MATMUL:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[IN]]
    // CHECK-SAME:  weights([[WEIGHTS]]
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE]]
    // CHECK-SAME:  parent_input([[IN]]
    // CHECK-SAME:  parent_output([[OUT]]
    // CHECK-SAME:  outputs([[OUT]]

    return %OUT : !OutputBuffer
    // CHECK:   return [[OUT]]
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x512x9x9xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    uniform_distributed_segments,
    compute_shapes = [[1, 512, 9, 9], [1, 512, 9, 9], [1, 512, 9, 9], [1, 512, 9, 9]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 512, 9, 9], [1, 512, 9, 9], [1, 512, 9, 9], [1, 512, 9, 9]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputSparsityMapDistributed = !VPUIP.DistributedBuffer<
    1x512x18x18xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputStorageElementTableDistributed = !VPUIP.DistributedBuffer<
    1x1x18x18xi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4,
    alignment = [1, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 18, 18], [1, 1, 18, 18], [1, 1, 18, 18], [1, 1, 18, 18]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 18, 18], [1, 1, 18, 18], [1, 1, 18, 18], [1, 1, 18, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    512x512x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[128, 512, 1, 1], [128, 512, 1, 1], [128, 512, 1, 1], [128, 512, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    memory_shapes = [[128, 512, 1, 1], [128, 512, 1, 1], [128, 512, 1, 1], [128, 512, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]
}, #VPUIP.SparsityCompressionAttr<axis = 0, numElems = dense<1> : tensor<512xi64>, alignment = 16>
>

!WeightsSparsityMapDistributed = !VPUIP.DistributedBuffer<
    512x1x1x512xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[128, 1, 1, 512], [128, 1, 1, 512], [128, 1, 1, 512], [128, 1, 1, 512]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 512], [128, 1, 1, 512], [128, 1, 1, 512], [128, 1, 1, 512]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    512x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0], [256, 0, 0, 0], [384, 0, 0, 0]]
}>

!NCEOutputDistributed = !VPUIP.DistributedBuffer<
    1x512x18x18xf16, {order = #NHWC, strides = [165888, 1, 9216, 512]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 18, 18], [1, 128, 18, 18], [1, 128, 18, 18], [1, 128, 18, 18]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0], [0, 256, 0, 0], [0, 384, 0, 0]],
    memory_shapes = [[1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18], [1, 512, 18, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    6x512x18x18xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[6, 512, 18, 18], [6, 512, 18, 18], [6, 512, 18, 18], [6, 512, 18, 18]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[6, 512, 18, 18], [6, 512, 18, 18], [6, 512, 18, 18], [6, 512, 18, 18]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

func.func @DoNotSwizzleWeightsDueToAlignmentMemCannotFitIntoCMX() -> !OutputDistributed {
    %WEIGHTS_CST = const.Declare memref<512x512x1x1xf16, #NHWC> = dense<1.0> : memref<512x512x1x1xf16, #NHWC>
    %WEIGHTS_SM_CST = const.Declare memref<512x1x1x512xi1> = dense<0> : memref<512x1x1x512xi1>
    %WEIGHTS_TABLE_CST = const.Declare memref<512x1x1x4xsi32> = dense<5> : memref<512x1x1x4xsi32>

    %INPUT_CMX = VPURT.AllocDistributed -> !InputDistributed
    %INPUT_SM_CMX = VPURT.AllocDistributed -> !InputSparsityMapDistributed
    %INPUT_SE_TABLE_CMX = VPURT.AllocDistributed -> !InputStorageElementTableDistributed
    %WEIGHTS_CMX = VPURT.AllocDistributed -> !WeightsDistributed
    %WEIGHTS_SM_CMX = VPURT.AllocDistributed -> !WeightsSparsityMapDistributed
    %WEIGHTS_TABLE_CMX = VPURT.AllocDistributed -> !WeightsTableDistributed

    %OUT_CMX_CONCAT = VPURT.AllocDistributed -> !OutputDistributed
    %OUT_NCE_CMX = VPUIP.SubView %OUT_CMX_CONCAT [0, 0, 0, 0] [1, 512, 18, 18] : !OutputDistributed to !NCEOutputDistributed

    %WEIGHTS_COPY = VPUIP.Copy inputs(%WEIGHTS_CST : memref<512x512x1x1xf16, #NHWC>) outputs(%WEIGHTS_CMX : !WeightsDistributed) -> !WeightsDistributed
    %WEIGHTS_SM_COPY = VPUIP.Copy inputs(%WEIGHTS_SM_CST : memref<512x1x1x512xi1>) outputs(%WEIGHTS_SM_CMX : !WeightsSparsityMapDistributed) -> !WeightsSparsityMapDistributed
    %WEIGHTS_TABLE_COPY = VPUIP.Copy inputs(%WEIGHTS_TABLE_CST : memref<512x1x1x4xsi32>) outputs(%WEIGHTS_TABLE_CMX : !WeightsTableDistributed) -> !WeightsTableDistributed

    %NCE = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        minimumHardwareExecutionCost = 13896,
        task_type = #VPUIP.nce_task_type<CONV>
    }
        input(%INPUT_CMX : !InputDistributed)
        input_sparsity_map(%INPUT_SM_CMX : !InputSparsityMapDistributed)
        input_storage_element_table(%INPUT_SE_TABLE_CMX : !InputStorageElementTableDistributed)
        weights(%WEIGHTS_COPY : !WeightsDistributed)
        weights_sparsity_map(%WEIGHTS_SM_COPY : !WeightsSparsityMapDistributed)
        weight_table(%WEIGHTS_TABLE_COPY : !WeightsTableDistributed)
        parent_input(%INPUT_CMX : !InputDistributed)
        parent_input_sparsity_map(%INPUT_SM_CMX : !InputSparsityMapDistributed)
        parent_input_storage_element_table(%INPUT_SE_TABLE_CMX : !InputStorageElementTableDistributed)
        parent_output(%OUT_NCE_CMX : !NCEOutputDistributed)
        outputs(%OUT_NCE_CMX : !NCEOutputDistributed) -> !NCEOutputDistributed
    variants : {
        DPUTask {cluster_id = 0, inEnd = [17, 17, 511], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [17, 17, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 1, inEnd = [17, 17, 511], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [17, 17, 255], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 2, inEnd = [17, 17, 511], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [17, 17, 383], outStart = [0, 0, 256], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
        DPUTask {cluster_id = 3, inEnd = [17, 17, 511], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [17, 17, 511], outStart = [0, 0, 384], pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>}
    }
    PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648, clamp_high = 2147483647, lrelu_mult = 1, lrelu_shift = 0, fp_prelu_alpha = 1.0>}
    }

    return %OUT_CMX_CONCAT : !OutputDistributed

    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x512x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK:       [[WEIGHTS_SM_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x512xi1, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
    // CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<512x1x1x4xsi32, #NHWC, @CMX_NN, {
    // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!FusedConstDistributed = !VPUIP.DistributedBuffer<
    1x1x1x448384xui8, {order = #NCHW, strides = [448384, 448384, 448384, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384], [1, 1, 1, 448384]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!FusedConstWeightsTableDistributed= !VPUIP.DistributedBuffer<
    1x1x1x4096xui8, {order = #NCHW, strides = [448384, 448384, 448384, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096], [1, 1, 1, 4096]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!FusedConstWeightsDistributed = !VPUIP.DistributedBuffer<
    1x1x1x370560xui8, {order = #NCHW, strides = [448384, 448384, 448384, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560], [1, 1, 1, 370560]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!FusedConstWeightsSMDistributed = !VPUIP.DistributedBuffer<
    1x1x1x73728xui8, {order = #NCHW, strides = [448384, 448384, 448384, 1]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728], [1, 1, 1, 73728]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x256x14x14x!quant.uniform<u8:f16, 0.0056737656686820237>, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
}>

!InputSparsityMapDistributed = !VPUIP.DistributedBuffer<
    1x256x14x14xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x256x3x3x!quant.uniform<u8:f16, 0.0056737656686820237>, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3], [256, 256, 3, 3]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}, #VPUIP.SparsityCompressionAttr<axis = 0, numElems = dense_resource<__elided__> : tensor<256xi64>, alignment = 16>
>

!WeightsSparsityMapDistributed = !VPUIP.DistributedBuffer<
    256x1x1x2304xi1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304], [256, 1, 1, 2304]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4], [256, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x14x14x!quant.uniform<u8:f16, 0.0056737656686820237>, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
}>

!OutputSparsityMapDistributed = !VPUIP.DistributedBuffer<
    1x256x14x14xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 6, 1],
    num_clusters = 6,
    uniform_distributed_segments,
    compute_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 256, 3, 14], [1, 256, 3, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14], [1, 256, 2, 14]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 8, 0], [0, 0, 10, 0], [0, 0, 12, 0]]
}>

func.func @SwizzleOutputWithFusedWeightsConstant() -> !OutputDistributed {
    %FUSED_BUFF = VPURT.AllocDistributed -> !FusedConstDistributed

    %WEIGHTS_TABLE_BUFF = VPUIP.SubView %FUSED_BUFF [0, 0, 0, 0] [1, 1, 1, 4096] : !FusedConstDistributed to !FusedConstWeightsTableDistributed
    %WEIGHTS_TABLE = VPUIP.ViewOp %WEIGHTS_TABLE_BUFF : !FusedConstWeightsTableDistributed to !WeightsTableDistributed

    %WEIGHTS_BUFF = VPUIP.SubView %FUSED_BUFF [0, 0, 0, 4096] [1, 1, 1, 370560] : !FusedConstDistributed to !FusedConstWeightsDistributed
    %WEIGHTS = VPUIP.ViewOp %WEIGHTS_BUFF : !FusedConstWeightsDistributed to !WeightsDistributed

    %WEIGHTS_SM_BUFF = VPUIP.SubView %FUSED_BUFF [0, 0, 0, 374656] [1, 1, 1, 73728] : !FusedConstDistributed to !FusedConstWeightsSMDistributed
    %WEIGHTS_SM = VPUIP.ViewOp %WEIGHTS_SM_BUFF : !FusedConstWeightsSMDistributed to !WeightsSparsityMapDistributed

    %INPUT_CMX = VPURT.AllocDistributed -> !InputDistributed
    %INPUT_SM_CMX = VPURT.AllocDistributed -> !InputSparsityMapDistributed
    %OUT_0_CMX = VPURT.AllocDistributed -> !OutputDistributed
    %OUT_0_SM_CMX = VPURT.AllocDistributed -> !OutputSparsityMapDistributed

    %CONV_0:2 = VPUIP.NCEClusterTask {
        constantsFused = true,
        kernel_padding = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        minimumHardwareExecutionCost = 14570,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }
        input(%INPUT_CMX : !InputDistributed)
        input_sparsity_map(%INPUT_SM_CMX : !InputSparsityMapDistributed)
        weights(%WEIGHTS : !WeightsDistributed)
        weights_sparsity_map(%WEIGHTS_SM : !WeightsSparsityMapDistributed)
        weight_table(%WEIGHTS_TABLE : !WeightsTableDistributed)
        parent_input(%INPUT_CMX : !InputDistributed)
        parent_input_sparsity_map(%INPUT_SM_CMX : !InputSparsityMapDistributed)
        parent_output(%OUT_0_CMX : !OutputDistributed)
        parent_output_sparsity_map(%OUT_0_SM_CMX : !OutputSparsityMapDistributed)
        outputs(%OUT_0_CMX : !OutputDistributed)
        output_sparsity_map(%OUT_0_SM_CMX : !OutputSparsityMapDistributed) -> !OutputDistributed, !OutputSparsityMapDistributed
    variants : {
        DPUTask {cluster_id = 0, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 2, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 0>}
        DPUTask {cluster_id = 1, inEnd = [13, 4, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 2, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 2, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 3, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 4, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 5, inEnd = [13, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 1>}
    }
    PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0, clamp_high = 255, lrelu_mult = 1, lrelu_shift = 0, fp_prelu_alpha = 1.0>}
    }

    %OUT_1_CMX = VPURT.AllocDistributed -> !OutputDistributed
    %OUT_1_SM_CMX = VPURT.AllocDistributed -> !OutputSparsityMapDistributed

    %CONV_1:2 = VPUIP.NCEClusterTask {
        constantsFused = true,
        kernel_padding = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        minimumHardwareExecutionCost = 14570,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        task_type = #VPUIP.nce_task_type<CONV>
    }
        input(%CONV_0#0 : !InputDistributed)
        input_sparsity_map(%CONV_0#1 : !InputSparsityMapDistributed)
        weights(%WEIGHTS : !WeightsDistributed)
        weights_sparsity_map(%WEIGHTS_SM : !WeightsSparsityMapDistributed)
        weight_table(%WEIGHTS_TABLE : !WeightsTableDistributed)
        parent_input(%CONV_0#0 : !InputDistributed)
        parent_input_sparsity_map(%CONV_0#1 : !InputSparsityMapDistributed)
        parent_output(%OUT_1_CMX : !OutputDistributed)
        parent_output_sparsity_map(%OUT_1_SM_CMX : !OutputSparsityMapDistributed)
        outputs(%OUT_1_CMX : !OutputDistributed)
        output_sparsity_map(%OUT_1_SM_CMX : !OutputSparsityMapDistributed) -> !OutputDistributed, !OutputSparsityMapDistributed
    variants : {
        DPUTask {cluster_id = 0, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 2, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 1, bottom = 0>}
        DPUTask {cluster_id = 1, inEnd = [13, 4, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 2, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 2, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 3, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 4, inEnd = [13, 3, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 0>}
        DPUTask {cluster_id = 5, inEnd = [13, 2, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [13, 1, 255], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1, right = 1, top = 0, bottom = 1>}
    }
    PPE : {
        PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0, clamp_high = 255, lrelu_mult = 1, lrelu_shift = 0, fp_prelu_alpha = 1.0>}
    }

    return %CONV_1 : !OutputDistributed

    // CHECK:       [[OUT_0_CMX:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x256x14x14x!qElemType, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN
    // CHECK:       [[OUT_SM_0_CMX:%.+]] = VPURT.AllocDistributed {alignment = 32768 : i64, swizzlingKey = 5 : i64}
    // CHECK-SAME:          -> !VPUIP.DistributedBuffer<1x256x14x14xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN
}
