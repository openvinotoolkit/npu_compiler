//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz
}
!CopyOutFinal = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!CopyInTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @EraseSOKCopySequence
// CHECK-SAME:    ([[ARG0:%.+]]: !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
func.func @EraseSOKCopySequence(%arg0: !CopyInTensorDistributed) -> !CopyOutFinal {

    %0 = VPU.Copy(%arg0) { out_mem_space = @CMX_NN } : !CopyInTensorDistributed -> !TensorStub_CMX

    %1 = VPU.Copy(%0) { out_mem_space = @CMX_NN } : !TensorStub_CMX -> !CopyOutFinal

    return %1: !CopyOutFinal

// CHECK:    [[DIST_CAST:%.+]] = VPU.DistributedCast([[ARG0]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN,
// CHECK:                                                       {mode = "DUPLICATED", num_clusters = 2 : i64}>)
// CHECK:                           -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN,
// CHECK:                               {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
// CHECK:    return [[DIST_CAST]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN,
// CHECK:                           {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

}
// }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL: @ConvChain
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x28x28xf16, {order = #NHWC}>)
func.func @ConvChain(%arg0: tensor<1x128x28x28xf16, {order = #NHWC}>) -> tensor<1x96x14x14xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    %1 = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
             -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]], compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]], memory_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]], memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]]}>

    %3 = VPU.NCE.Convolution(%0, %1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]}
             : !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments, compute_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]], compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]], memory_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]], memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]]}> -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 48, 28, 28], [1, 48, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]], memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    %4 = VPU.Copy(%3) : !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 48, 28, 28], [1, 48, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]], memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
             -> tensor<1x96x28x28xf16, {order = #NHWC}>

    %5 = VPU.Copy(%4) {out_mem_space = @CMX_NN} : tensor<1x96x28x28xf16, {order = #NHWC}>
             -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    %6 = VPU.Copy(%cst_1) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
             -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    %8 = VPU.NCE.Convolution(%5, %6) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 96, 5, 5], strides = [2, 2]}
             : !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>, !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}> -> !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 7, 14], [1, 96, 7, 14]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]], memory_shapes = [[1, 96, 7, 14], [1, 96, 7, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>

    %9 = VPU.Copy(%8) : !VPU.DistributedTensor<1x96x14x14xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 7, 14], [1, 96, 7, 14]], compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]], memory_shapes = [[1, 96, 7, 14], [1, 96, 7, 14]], memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0]]}>
             -> tensor<1x96x14x14xf16, {order = #NHWC}>

    return %9 : tensor<1x96x14x14xf16, {order = #NHWC}>

// CHECK:    [[CST1:%.+]] = const.Declare tensor<96x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x128x3x3xf16>, [#const.Reor
// CHECK:    [[CST2:%.+]] = const.Declare tensor<96x96x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x5x5xf16>, [#const.Reorder<#NHWC>]

// CHECK:    [[CP_INPUT:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = @CMX_NN} : tensor<1x128x28x28xf16, {order = #NHWC}>
// CHECK:     -> !VPU.DistributedTensor<1x128x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 128, 28, 28], [1, 128, 28, 28]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
// CHECK:    [[CP_WEIGHTS:%.+]] = VPU.Copy([[CST1]]) {out_mem_space = @CMX_NN} : tensor<96x128x3x3xf16, {order = #NHWC}>
// CHECK:     -> !VPU.DistributedTensor<96x128x3x3xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[48, 128, 3, 3], [48, 128, 3, 3]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]]}>

// CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution([[CP_INPUT]], [[CP_WEIGHTS]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 128, 3, 3], strides = [1, 1]}
// CHECK: -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 28, 28], [1, 48, 28, 28]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

//CHECK-NEXT:    [[DIST_CAST:%.+]] = VPU.DistributedCast([[CONV]] : !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 48, 28, 28], [1, 48, 28, 28]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
// CHECK:           -> !VPU.DistributedTensor<1x96x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 96, 28, 28], [1, 96, 28, 28]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK-NEXT:    [[COPY:%.+]] = VPU.Copy([[CST2]]) {out_mem_space = @CMX_NN} : tensor<96x96x5x5xf16, {order = #NHWC}>
// CHECK:           -> !VPU.DistributedTensor<96x96x5x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
// CHECK-SAME{LITERAL}:     compute_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[96, 96, 5, 5], [96, 96, 5, 5]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!CopyOutTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!CopyInTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @EraseSOKCopySequence
// CHECK-SAME:    ([[ARG0:%.+]]: !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
func.func @EraseSOKCopySequence(%arg0: !CopyOutTensorDistributed) -> !CopyInTensorDistributed {
    %spilled_ddr = VPU.Copy(%arg0) { out_mem_space = @DDR } : !CopyOutTensorDistributed -> !Tensor_DDR

    %output = VPU.Copy(%spilled_ddr) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !CopyInTensorDistributed

    return %output: !CopyInTensorDistributed

// CHECK:    [[CAST:%.*]] = VPU.DistributedCast([[ARG0]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) ->
// CHECK-SAME:  !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:    return [[CAST]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseTypeCMX = !VPU.SparseTensor<
    data=tensor<1x64x52x52xf16, {
    mem_space = @CMX_NN,
    order = #NHWC}>,
    sparsity_map=tensor<1x64x52x52xi1, {
    mem_space = @CMX_NN,
    order = #NHWC
}>>
!SparseTypeDDR = !VPU.SparseTensor<data=tensor<1x64x52x52xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x52x52xi1, {order = #NHWC}>>

!InputDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                                               sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

!OutputDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                                                sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

// CHECK-LABEL: @OptimizeCMXDDRCMXCopiesSparseType
// CHECK-SAME:    ([[ARG0:%.+]]: !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN,
// CHECK-SAME:                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
// CHECK-SAME:                  sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN,
// CHECK-SAME:                  {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
func.func @OptimizeCMXDDRCMXCopiesSparseType(%input: !InputDistributedType) -> !OutputDistributedType {
    %0 = VPU.Copy (%input) :!InputDistributedType -> !SparseTypeDDR

    %1 = VPU.Copy (%0) {out_mem_space = @CMX_NN} : !SparseTypeDDR -> !OutputDistributedType

    return %1: !OutputDistributedType


    // CHECK:       [[DISTRIBUTED_CAST:%.*]] = VPU.DistributedCast
    // CHECK-SAME:    ([[ARG0]] : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                         sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:       return [[DISTRIBUTED_CAST]] :
    // CHECK-SAME:      !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:      sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!CopyOutTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!CopyInTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @EraseExtraCopys
// CHECK-SAME:    ([[ARG0:%.+]]: !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>)
func.func @EraseExtraCopys(%arg0: !CopyOutTensorDistributed) -> !CopyInTensorDistributed {

    %dist_to_simple = VPU.Copy(%arg0) { out_mem_space = @DDR } : !CopyOutTensorDistributed -> !Tensor_DDR
    %extra_copy0 = VPU.Copy(%dist_to_simple) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !CopyOutTensorDistributed
    %extra_copy1 = VPU.Copy(%extra_copy0) { out_mem_space = @DDR } : !CopyOutTensorDistributed -> !Tensor_DDR

    %output = VPU.Copy(%extra_copy1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !CopyInTensorDistributed

    return %output: !CopyInTensorDistributed

// CHECK:    [[CAST:%.*]] = VPU.DistributedCast([[ARG0]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) ->
// CHECK-SAME:  !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
// CHECK:    return [[CAST]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

}
