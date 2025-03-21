//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// NCEClusterTiling operation with memref output
// Original operation before IE2IERT lowering:
// func.func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
//     %tensor_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//     return %tensor_cmx : !type_CMX_tensor
// }

// CHECK-LABEL: @NCEClusterTilingCopyOpTensorResult
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
func.func @NCEClusterTilingCopyOpTensorResult(%input0: !type_DDR_tensor) -> !type_CMX_tensor{
    %tensor_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !type_CMX_tensor {
        %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : !type_DDR_tensor -> !type_CMX_tensor
        VPU.Yield %0
    }

    return %tensor_cmx : !type_CMX_tensor

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[NCE_CT_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[ARG0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[COPY_RES:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEClusterTilingCopyOpDistributedResult
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
func.func @NCEClusterTilingCopyOpDistributedResult(%input0: !type_DDR_tensor) -> !typeCmxDistributed{
    %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
        %0 = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
        VPU.Yield %0
    }

    return %tensor_distributed_cmx : !typeCmxDistributed

    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[NCE_CT_RES:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[ARG0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer

    // CHECK:       [[COPY_RES:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!typeCmxDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!type_DDR_tensor = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!type_DDR_memref = memref<1x32x16x16xf16, #NHWC, @DDR>
!type_CMX_tensor = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!type_CMX_memref = memref<1x32x16x16xf16, #NHWC, @CMX_NN>

// 2 NCEClusterTiling operations with distributed type passed in between
// Original operation before IE2IERT lowering
// func.func @NCEClusterTilingDistributedCopy2CopyOp(%input0: !type_DDR_tensor) -> !type_DDR_tensor {
//     %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
//         %0 = IE.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
//         VPU.Yield %0
//     }
//     %tensor_ddr = VPU.NCE.ClusterTiling(%tensor_distributed_cmx as %arg0: !type_CMX_tensor) -> !type_DDR_tensor {
//         %0 = IE.Copy(%arg0) { out_mem_space = @DDR } : !type_CMX_tensor -> !type_DDR_tensor
//         VPU.Yield %0
//     }
//     return %tensor_ddr : !type_DDR_tensor
// }

// CHECK-LABEL: @NCEClusterTilingDistributedCopy2CopyOp
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
func.func @NCEClusterTilingDistributedCopy2CopyOp(%input0: !type_DDR_tensor) -> !type_DDR_tensor {
    %tensor_distributed_cmx = VPU.NCE.ClusterTiling(%input0 as %arg0: !type_DDR_tensor) -> !typeCmxDistributed {
        %0 = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !type_DDR_tensor -> !type_CMX_tensor
        VPU.Yield %0
    }
    %tensor_ddr = VPU.NCE.ClusterTiling(%tensor_distributed_cmx as %arg0: !type_CMX_tensor) -> !type_DDR_tensor {
        %0 = VPU.Copy(%arg0) { out_mem_space = @DDR } : !type_CMX_tensor -> !type_DDR_tensor
        VPU.Yield %0
    }
    return %tensor_ddr : !type_DDR_tensor

    // CHECK:       [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer

    // CHECK:       [[NCE_CT_RES0:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[ARG0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       outputs([[ALLOC0]] as [[ARG2:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       -> !VPUIP.DistributedBuffer

    // CHECK:       [[COPY_RES0:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[ARG2]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)

    // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[NCE_CT_RES1:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:       inputs([[NCE_CT_RES0]] as [[ARG3:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[ALLOC1]] as [[ARG4:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:       -> memref<1x32x16x16xf16, #NHWC, @DDR>

    // CHECK:       [[COPY_RES1:%.+]] = VPUIP.Copy
    // CHECK-SAME:       inputs([[ARG3]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:       outputs([[ARG4]] : memref<1x32x16x16xf16, #NHWC, @DDR>)

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorDistributed = !VPU.DistributedTensor<
    32x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!SMTensorDistributed = !VPU.DistributedTensor<
    32x1x1x256xi1, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!Tensor_DDR = tensor<32x16x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!SMTensor_DDR = tensor<32x1x1x256xi1, {mem_space = @DDR}>
!Tensor_CMX = tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!SMTensor_CMX = tensor<32x1x1x256xi1, {mem_space = @CMX_NN}>

// CHECK-LABEL: @SparseDistributedCopyCMXToDDR
// CHECK-SAME:  [[ARG0:%.+]]: memref<32x16x3x3xf16, #NHWC, @DDR>
// CHECK-SAME:  [[ARG1:%.+]]: memref<32x1x1x256xi1, @DDR>
func.func @SparseDistributedCopyCMXToDDR(%arg0: !Tensor_DDR, %arg1: !SMTensor_DDR)
        -> !VPU.SparseTensor<data=!TensorDistributed, sparsity_map=!SMTensorDistributed, is_weights> {
    %st = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>

    %tensor_distributed_cmx = VPU.NCE.ClusterTiling (%st as %arg2: !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>)
            -> !VPU.SparseTensor<data=!TensorDistributed, sparsity_map=!SMTensorDistributed, is_weights> {
        %0 = VPU.Copy(%arg2) { out_mem_space = @CMX_NN } : !VPU.SparseTensor<data=!Tensor_DDR, sparsity_map=!SMTensor_DDR, is_weights>
                    -> !VPU.SparseTensor<data=!Tensor_CMX, sparsity_map=!SMTensor_CMX, is_weights>
        VPU.Yield %0
    }
    return %tensor_distributed_cmx : !VPU.SparseTensor<data=!TensorDistributed, sparsity_map=!SMTensorDistributed, is_weights>

    // CHECK:       [[INPUT_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[ARG0]], [[ARG1]]) {is_weights}
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @DDR>, sparsity_map=memref<32x1x1x256xi1, @DDR>, is_weights>

    // CHECK:       [[DATA_DIST_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SM_DIST_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUTPUT_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[DATA_DIST_CMX]], [[SM_DIST_CMX]]) {is_weights}
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 is_weights>

    // CHECK:       [[OUTPUT_SPARSE_COPY_CMX:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:      inputs([[INPUT_SPARSE_DDR]] as [[ARG2:%.+]]: !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @DDR>, sparsity_map=memref<32x1x1x256xi1, @DDR>, is_weights>)
    // CHECK-SAME:      outputs([[OUTPUT_SPARSE_CMX]] as [[ARG3:%.+]]: !VPUIP.SparseBuffer<data=memref<32x16x3x3xf16, #NHWC, @CMX_NN>, sparsity_map=memref<32x1x1x256xi1, @CMX_NN>, is_weights>)
    // CHECK-SAME:          -> !VPUIP.SparseBuffer<data=!VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 sparsity_map=!VPUIP.DistributedBuffer<32x1x1x256xi1, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>,
    // CHECK-SAME:                                 is_weights> {

    // CHECK:           [[COPY_OUT:%.+]] = VPUIP.Copy inputs([[ARG2]]
    // CHECK-SAME:                                    outputs([[ARG3]]

    // CHECK:       }

    // CHECK:       return [[OUTPUT_SPARSE_COPY_CMX]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEClusterTilingWithNCEConv
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>
func.func @NCEClusterTilingWithNCEConv(%arg0: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %wt as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
            }
        VPU.Yield %1
    }

    return %0 : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK-DAG:  [[CST:%.+]] = const.Declare  memref<64x32x3x3xf16, #NHWC, @CMX_NN> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    //CHECK-DAG:  [[CST0:%.+]] = const.Declare memref<64x1x1x4xsi32, @CMX_NN> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    //CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    //CHECK:      [[OUT:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:    inputs([[ARG0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>, [[CST]] as [[ARG2:%.+]]: memref<64x32x3x3xf16, #NHWC, @CMX_NN>, [[CST0]] as [[ARG3:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>)
    //CHECK-SAME:    outputs([[ALLOC]] as [[ARG4:%.+]]: memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:    -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {

    //CHECK:        [[NCE_COV:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME:   {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}

    //CHECK:         DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    //CHECK:     return [[OUT]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEClusterTilingWithNceCompressConv
// CHECK-SAME: [[ARG0:%.+]]: memref<1x4x224x224xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<64x4x7x7xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: -> memref<1x64x112x112xf16, #NHWC, @CMX_NN>
func.func @NCEClusterTilingWithNceCompressConv(%input: tensor<1x4x224x224xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                           %weights: tensor<64x4x7x7xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                           %wt: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x64x112x112xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.ClusterTiling (
            %input as %arg0: tensor<1x4x224x224xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            %weights as %arg1: tensor<64x4x7x7xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            %wt as %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>)
                -> tensor<1x64x112x112xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        %1 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
                cm_sp_pattern = 15 : i64,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
            } -> tensor<1x64x112x112xf16, {order = #NHWC, mem_space = @CMX_NN}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 112, 112] <left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
        }
        VPU.Yield %1
    }

    return %0 : tensor<1x64x112x112xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:      [[SHAPE_CAST_ARG1:%.+]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]}
    // CHECK-SAME:   inputs([[ARG1]] : memref<64x4x7x7xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:   -> memref<64x16x7x7xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x112x112xf16, #NHWC, @CMX_NN>

    // CHECK:       [[SHAPE_CAST_ARG0:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 224, 224]}
    // CHECK-SAME:    inputs([[ARG0]] : memref<1x4x224x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    -> memref<1x16x224x224xf16, #NHWC, @CMX_NN>

    // CHECK:       [[OUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:    inputs([[SHAPE_CAST_ARG0]] as [[ARG3:%.+]]: memref<1x16x224x224xf16, #NHWC, @CMX_NN>, [[SHAPE_CAST_ARG1]] as [[ARG4:%.+]]: memref<64x16x7x7xf16, #NHWC, @CMX_NN>, [[ARG2]] as [[ARG5:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:    outputs([[OUT_BUF]] as [[ARG6:%.+]]: memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    -> memref<1x64x112x112xf16, #NHWC, @CMX_NN> {

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:    input([[ARG3]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    weights([[ARG4]] : memref<64x16x7x7xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    weight_table([[ARG5]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:    parent_input([[ARG3]] : memref<1x16x224x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    parent_output([[ARG6]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    outputs([[ARG6]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:    -> memref<1x64x112x112xf16, #NHWC, @CMX_NN> variants : {

    //CHECK: return [[OUT]] : memref<1x64x112x112xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEClusterTilingWithSparseInPlaceNCEEltwise
// CHECK-SAME:  [[ARG0:%.+]]: !VPUIP.SparseBuffer<data=memref<1x64x368x29xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x368x29xi1, #NHWC, @CMX_NN>>
func.func @NCEClusterTilingWithSparseInPlaceNCEEltwise(%arg0: !VPU.SparseTensor<data=tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                                                                sparsity_map=tensor<1x64x368x29xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
        -> tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.ClusterTiling (%arg0 as %arg1: !VPU.SparseTensor<data=tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                                                  sparsity_map=tensor<1x64x368x29xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
        -> tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        %1 = VPU.NCE.Eltwise(%arg1, %arg1) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 123, 29] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            }
        VPU.Yield %1
    }

    return %0 : tensor<1x64x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<1x64x368x29xf16, #NHWC, @CMX_NN>
    //CHECK:      [[DATA2:%.+]], [[SM2:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 1, 0>} -> memref<1x64x368x29xf16, #NHWC, @CMX_NN>, memref<1x64x368x29xi1, #NHWC, @CMX_NN>
    //CHECK:      [[DATA1:%.+]], [[SM1:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 1, 0>} -> memref<1x64x368x29xf16, #NHWC, @CMX_NN>, memref<1x64x368x29xi1, #NHWC, @CMX_NN>

    //CHECK:      [[OUT:%.+]] = VPUIP.NCEClusterTiling
    //CHECK-SAME:    inputs([[DATA1]] as [[INNER_ARG1:[^:]+]]: memref<1x64x368x29xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:           [[SM1]] as [[INNER_ARG2:[^:]+]]: memref<1x64x368x29xi1, #NHWC, @CMX_NN>,
    //CHECK-SAME:           [[DATA2]] as [[INNER_ARG3:[^:]+]]: memref<1x64x368x29xf16, #NHWC, @CMX_NN>,
    //CHECK-SAME:           [[SM2]] as [[INNER_ARG4:[^:]+]]: memref<1x64x368x29xi1, #NHWC, @CMX_NN>)
    //CHECK-SAME:    outputs([[ALLOC]] as [[INNER_ARG5:[^:]+]]: memref<1x64x368x29xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:    -> memref<1x64x368x29xf16, #NHWC, @CMX_NN> {

    //CHECK:        [[NCE:%.+]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:       eltwise_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>
    //CHECK-SAME:   }
    //CHECK-SAME:       input([[INNER_ARG1]]
    //CHECK-SAME:       input_sparsity_map([[INNER_ARG2]]
    //CHECK-SAME:       weights([[INNER_ARG1]]
    //CHECK-SAME:       weights_sparsity_map([[INNER_ARG2]]
    //CHECK-SAME:       outputs([[INNER_ARG5]]

    //CHECK:     return [[OUT]] : memref<1x64x368x29xf16, #NHWC, @CMX_NN>
}
