//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitDepthConvWithBigC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x5120x64x4xf16, {order = #NHWC}>)
func.func @SplitDepthConvWithBigC(%arg0: tensor<1x5120x64x4xf16, {order = #NHWC}>) -> tensor<1x5120x64x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [5120, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
             strides = [1, 1]
        } -> tensor<1x5120x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x64x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}>
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution([[ARG_0]], [[CST]]) rawFilterShape [5120, 1, 1, 1] {
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1],
    // CHECK-SAME:               tilingStrategy = [1, 4, 1, 1]} -> tensor<1x5120x64x4xf16, {order = #NHWC}>
    // CHECK:  return [[DWConv]] : tensor<1x5120x64x4xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitSparseDepthConvWithBigC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4080x40x40xf16, {order = #NHWC}>)
func.func @SplitSparseDepthConvWithBigC(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0) rawFilterShape [4080, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution([[ARG_0]], [[INPUT]]) rawFilterShape [4080, 1, 4, 4] {
    // CHECK-SAME:      {{.*}}tilingStrategy = [1, 19, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4080x16x16xf16, {order = #NHWC}>)
func.func @SplitSparseNCEMaxPoolWithBigC(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]]) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitSparseDepthConvWithBigCWithSOK
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4080x40x40xf16, {order = #NHWC}>)
    func.func @SplitSparseDepthConvWithBigCWithSOK(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
        %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.DepthConvolution(%arg0, %cst0) rawFilterShape [4080, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,

                strides = [1, 1]
            } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

        return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

        // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
        // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution([[ARG_0]], [[INPUT]]) rawFilterShape [4080, 1, 4, 4] {
        // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        // CHECK-SAME:       {{.*}}tilingStrategy = [1, 12, 1, 1]
        // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
        // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigCWithSOK
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4080x16x16xf16, {order = #NHWC}>)
func.func @SplitSparseNCEMaxPoolWithBigCWithSOK(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]]) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType = !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
!SparseType1 = !VPU.SparseTensor<data=tensor<1x4064x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4064x16x16xi1, {order = #NHWC}>>

module @Test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitOutputSparseForConvSOKFollowedByConcat
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2032x16x16xf16, {order = #NHWC}>)
    func.func @SplitOutputSparseForConvSOKFollowedByConcat(%arg0: tensor<1x2032x16x16xf16, {order = #NHWC}>) -> tensor<1x4064x16x16xf16, {order = #NHWC}> {
        %s0 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
        %wt0 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
        %maxpool0 = VPU.NCE.MaxPool(%s0, %wt0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            kernel_size = [3, 3],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> !SparseType

        %s1 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
        %wt1 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
        %maxpool1 = VPU.NCE.MaxPool(%s1, %wt1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            kernel_size = [3, 3],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> !SparseType


        %concat = VPU.Concat(%maxpool0, %maxpool1) {static_offsets = [[0, 0, 0, 0], [0, 2032, 0, 0]]} : !SparseType, !SparseType -> !SparseType1

        %wt2 = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
        %maxpool2 = VPU.NCE.MaxPool(%concat, %wt2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            kernel_size = [3, 3],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> !SparseType1

        %result = VPU.Desparsify(%maxpool2) : !SparseType1 -> tensor<1x4064x16x16xf16, {order = #NHWC}>
        return %result : tensor<1x4064x16x16xf16, {order = #NHWC}>

        // CHECK: [[ToSparsity_0:%.+]] = VPU.Sparsify([[ARG_0]]) : tensor<1x2032x16x16xf16, {order = #NHWC}>
        // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
        // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
        // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[ToSparsity_0]], [[WT_0]] )
        // CHECK:              tilingStrategy = [1, 3, 1, 1]

        // CHECK: [[ToSparsity_1:%.+]] = VPU.Sparsify([[ARG_0]]) : tensor<1x2032x16x16xf16, {order = #NHWC}>
        // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
        // CHECK-DAG: [[WT_1:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
        // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[ToSparsity_1]], [[WT_1]] )
        // CHECK-SAME:              tilingStrategy = [1, 3, 1, 1]

        // CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
        // CHECK-DAG: [[WT_2:%.+]] = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
        // CHECK: [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[CONCAT]], [[WT_2]] )
        // CHECK-SAME:              tilingStrategy = [1, 5, 1, 1]
        // CHECK: [[RESULT:%.+]] = VPU.Desparsify([[MAXPOOL_2]])

        // CHECK: return [[RESULT]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitConvWithPadsOnCH
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<1x320x32x32xf16, {order = #NHWC}>
// CHECK-SAME:  [[INPUT1:%.+]]: !VPU.SparseTensor
func.func @SplitConvWithPadsOnCH(%arg0: tensor<1x320x32x32xf16, {order = #NHWC}>, %arg1 : !VPU.SparseTensor<data=tensor<256x320x13x13xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x54144xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880]>: tensor<256xi64>, alignment = 16 : i64>>  ) -> tensor<1x256x32x32xf16, {order = #NHWC}>  {

    %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [256, 320, 13, 13] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 6 : i64, right = 6 : i64, top = 6 : i64, bottom = 6 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x320x32x32xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<256x320x13x13xf16, {order = #NHWC}>, sparsity_map=tensor<256x1x1x54144xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880,2880, 2880, 2880, 2880, 2880, 2880, 2880, 2879, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880]>: tensor<256xi64>, alignment = 16 : i64>> -> tensor<1x256x32x32xf16, {order = #NHWC}>
    return %0 : tensor<1x256x32x32xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[INPUT1]])
    // CHECK-SAME:            tilingStrategy = [1, 4, 2, 1]}
    // CHECK:            -> tensor<1x256x32x32xf16, {order = #NHWC}>
    // return [[CONV]] : tensor<1x256x32x32xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
func.func @InterpSplitOverH(
    %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
            -> tensor<1x64x192x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0, 0, 0>,
        sizes_attr = [192, 320],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
    return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 6]
    // CHECK-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @EnsureOneLinePerCluster
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x512x512x512xf16, {order = #NHWC}
func.func @EnsureOneLinePerCluster(%arg0: tensor<1x512x512x512xf16, {order = #NHWC}>) -> tensor<1x512x512x512xf16, {order = #NHWC}> {
    %0 = VPU.Swish(%arg0) {
        beta_value = 1.000000e+00 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
        tensor<1x512x512x512xf16, {order = #NHWC}> -> tensor<1x512x512x512xf16, {order = #NHWC}>
    return %0 : tensor<1x512x512x512xf16, {order = #NHWC}>

    // CHECK:       [[SWISH:%.+]] = VPU.Swish([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 170, 2]
    // CHECK-SAME:      : tensor<1x512x512x512xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x512x512xf16, {order = #NHWC}>

    // CHECK:  return [[SWISH]] : tensor<1x512x512x512xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.010003063725490195>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @TileableConvSOKCTiling
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x2560x8x8xf16, {order = #NHWC}>
  func.func @TileableConvSOKCTiling(%arg0: tensor<1x2560x8x8xf16, {order = #NHWC}>) -> tensor<1x1296x8x8xf32> {
    %cst_0 = const.Declare tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> : tensor<1296x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %0 = VPU.Dequantize(%cst_0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1296x2560x3x3xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%arg0, %0) rawFilterShape [1296, 2560, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x2560x8x8xf16, {order = #NHWC}>, tensor<1296x2560x3x3xf16, {order = #NHWC}> -> tensor<1x1296x8x8xf32>
    return %1 : tensor<1x1296x8x8xf32>


    // CHECK:   [[WEIGHTS:%.+]] =  const.Declare tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> :
    // CHECK-SAME:  tensor<1296x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]

    // CHECK:   [[DEQUANT:%.+]] =  VPU.Dequantize([[WEIGHTS]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: , tilingStrategy = [27, 1, 1, 1]} : tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1296x2560x3x3xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[ARG]], [[DEQUANT]]){{.*}} {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:  tilingStrategy = [1, 27, 1, 1]
    // CHECK-SAME:  -> tensor<1x1296x8x8xf32>
    // CHECK:       return [[CONV]] : tensor<1x1296x8x8xf32>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.010003063725490195>

module @Test {

config.Resources 3 of @NCE {
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SOHConvCTiling
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x2560x8x8xf16, {order = #NHWC}>
  func.func @SOHConvCTiling(%arg0: tensor<1x2560x8x8xf16, {order = #NHWC}>) -> tensor<1x1280x8x8xf32> {
    %cst_0 = const.Declare tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> : tensor<1280x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %0 = VPU.Dequantize(%cst_0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1280x2560x3x3xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%arg0, %0) rawFilterShape [1280, 2560, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x2560x8x8xf16, {order = #NHWC}>, tensor<1280x2560x3x3xf16, {order = #NHWC}> -> tensor<1x1280x8x8xf32>
    return %1 : tensor<1x1280x8x8xf32>


    // CHECK:   [[WEIGHTS:%.+]] =  const.Declare tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> :
    // CHECK-SAME:  tensor<1280x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]

    // CHECK:   [[DEQUANT:%.+]] =  VPU.Dequantize([[WEIGHTS]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: , tilingStrategy = [20, 2, 1, 1]} : tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1280x2560x3x3xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[ARG]], [[DEQUANT]]){{.*}} {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  tilingStrategy = [1, 80, 1, 1]
    // CHECK-SAME:  -> tensor<1x1280x8x8xf32>
    // CHECK:       return [[CONV]] : tensor<1x1280x8x8xf32>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEConvWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x512x512xf16, {order = #NHWC}>, %weights: tensor<3x16x1x1xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [3, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<3x16x1x1xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>

        // CHECK:       VPU.NCE.Convolution
        // CHECK-SAME:    tilingStrategy = [1, 1, 3, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEDepthConvWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x512x512xf16, {order = #NHWC}>, %weights: tensor<3x1x4x8xf16, {order = #NHWC}>) -> tensor<1x3x509x505xf16, {order = #NHWC}> {
        %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [3, 1, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x3x509x505xf16, {order = #NHWC}>
        return %0 : tensor<1x3x509x505xf16, {order = #NHWC}>

        // CHECK:       VPU.NCE.DepthConvolution
        // CHECK-SAME:    tilingStrategy = [1, 1, 3, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEMaxPoolWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>

        // CHECK:       VPU.NCE.MaxPool
        // CHECK-SAME:    tilingStrategy = [1, 1, 3, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEAvgPoolWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>

        // CHECK:       VPU.NCE.AveragePool
        // CHECK-SAME:    tilingStrategy = [1, 1, 3, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEEltwiseWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>, %arg1: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>

        // CHECK:       VPU.NCE.Eltwise
        // CHECK-SAME:    tilingStrategy = [1, 1, 5, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumAssignedSOH
func.func @NCEReduceSumAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 43, 1]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumAssignedSOW
func.func @NCEReduceSumAssignedSOW(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 32, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumSingleCluster
func.func @NCEReduceSumSingleCluster(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 52, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanAssignedSOH
func.func @NCEReduceMeanAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 43, 1]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanAssignedSOW
func.func @NCEReduceMeanAssignedSOW(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>, op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 32, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanSingleCluster
func.func @NCEReduceMeanSingleCluster(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 52, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEConvolutionWithoutSprLUTFitIntoCMX {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 2107664 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @main(%input: tensor<1x16x256x128xf16, {order = #NHWC}>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<
                mode = <NOOP>,
                clamp_low = -3.4028235e+38 : f32,
                clamp_high = 3.4028235e+38 : f32,
                prelu_alpha = [1.0],
                adder = 0.0 : f32
            >,

            strides = [1, 1]
        } : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
        return %0 : tensor<1x16x256x128xf16, {order = #NHWC}>
    }

// CHECK:       VPU.NCE.Convolution
// CHECK-NOT:       tilingStrategy
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NCEConvolutionWithSprLUTDoesntFitIntoCMX {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 2107664 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @main(%input: tensor<1x16x256x128xf16, {order = #NHWC}>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x256x128xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<
                mode = <NOOP>,
                clamp_low = -3.4028235e+38 : f32,
                clamp_high = 3.4028235e+38 : f32,
                prelu_alpha = [1.0],
                adder = 0.0 : f32,
                sprlut = dense<1> : tensor<1024xi64>
            >,

            strides = [1, 1]
        } : tensor<1x16x256x128xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
        return %0 : tensor<1x16x256x128xf16, {order = #NHWC}>
    }

// CHECK:       VPU.NCE.Convolution
// CHECK:       tilingStrategy = [1, 1, 2, 1]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @ScatterElementsUpdateTiling {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 2107664 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    func.func @main(%arg0 : tensor<1x1x1024x2048xf16>, %arg1 : tensor<1x1x32x2048xsi32>, %arg2 : tensor<1x1x32x2048xf16>) -> tensor<1x1x1024x2048xf16> {
    %0 = VPU.ScatterElementsUpdate(%arg0, %arg1, %arg2) {axis = 2 : i64, reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true}
            : tensor<1x1x1024x2048xf16>, tensor<1x1x32x2048xsi32>, tensor<1x1x32x2048xf16> -> tensor<1x1x1024x2048xf16>
    return %0 : tensor<1x1x1024x2048xf16>
}

// CHECK:       VPU.ScatterElementsUpdate
// CHECK:       tilingStrategy = [1, 1, 1, 5]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
module @executors {
    config.Resources 3 of @NCE {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
// CHECK-LABEL: @NCEMatMulSOGAndPipelined
  func.func @NCEMatMulSOGAndPipelined(%arg0: tensor<32x1x128x1x1xf16, {order = #GNHWC}>) ->  tensor<32x1x1408x1x1xf16, {order = #GNHWC}>{
    %weight = const.Declare tensor<32x1408x128x1x1xf16, {order = #GNHWC}> = dense<10.0> : tensor<32x1408x128x1x1xf16, {order = #GNHWC}>
    %grouped_matmul = VPU.NCE.MatMul(%arg0, %weight) rawFilterShape [32, 1408, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<32x1x1408x1x1xf16, {order = #GNHWC}>

    return %grouped_matmul : tensor<32x1x1408x1x1xf16, {order = #GNHWC}>
    // CHECK:         VPU.NCE.MatMul
    // CHECK-SAME:    tilingStrategy = [3, 1, 1, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 2 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @ConvWithLargeOCTileOverChannelAndHeight
    func.func @ConvWithLargeOCTileOverChannelAndHeight(%arg0: tensor<1x1024x256x4xf16, {order = #NHWC}>) -> tensor<1x18944x256x4xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<18944x1024x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<18944x1024x1x1xf16, {order = #NHWC}>
        %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [18944, 1024, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

            strides = [1, 1]
        } : tensor<1x1024x256x4xf16, {order = #NHWC}>, tensor<18944x1024x1x1xf16, {order = #NHWC}> -> tensor<1x18944x256x4xf16, {order = #NHWC}>
        return %0 : tensor<1x18944x256x4xf16, {order = #NHWC}>

        // CHECK:      VPU.NCE.Convolution
        // CHECK-SAME: tilingStrategy = [1, 15, 32, 1]
    }
}

// -----

// Verify that an op carrying both pinnedStrategy and a pre-set tilingStrategy is not re-tiled
// by TilingStrategyAssignment (the existing tilingStrategy must be preserved verbatim).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @PinnedStrategyWithTilingStrategyIsNotRetiled {

config.Resources 3 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @PinnedStrategyWithTilingStrategyIsNotRetiled
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5120x64x4xf16, {order = #NHWC}>)
func.func @PinnedStrategyWithTilingStrategyIsNotRetiled(
        %input: tensor<1x5120x64x4xf16, {order = #NHWC}>)
            -> tensor<1x5120x64x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // Op has pinnedStrategy (UnitAttr) + an explicit tilingStrategy pre-set by the PTC/manual-strategy
    // reader.  TSA must leave tilingStrategy unchanged (keep [1, 2, 1, 1] rather than computing
    // [1, 4, 1, 1] that ISOLATED tiling would derive independently).
    %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [5120, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            pinnedStrategy,
            tilingStrategy = [1, 2, 1, 1]
        } -> tensor<1x5120x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x64x4xf16, {order = #NHWC}>

    // The tilingStrategy value [1, 2, 1, 1] must be preserved – NOT overwritten by TSA.
    // CHECK:       [[RESULT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], {{%.+}})
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
}

}

// -----

// Verify that an op carrying pinnedStrategy but NO tilingStrategy still receives a tiling
// assignment from TilingStrategyAssignment (pinnedStrategy alone must not suppress TSA).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @PinnedStrategyWithoutTilingStrategyStillGetsTiled {

config.Resources 3 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @PinnedStrategyWithoutTilingStrategyStillGetsTiled
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5120x64x4xf16, {order = #NHWC}>)
func.func @PinnedStrategyWithoutTilingStrategyStillGetsTiled(
        %input: tensor<1x5120x64x4xf16, {order = #NHWC}>)
            -> tensor<1x5120x64x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // Op has pinnedStrategy but no tilingStrategy.  This models a PTC hit that supplied only
    // a spatial (multiCluster) strategy.  TSA must still compute and assign tilingStrategy.
    %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [5120, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            pinnedStrategy
        } -> tensor<1x5120x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x64x4xf16, {order = #NHWC}>

    // TSA must produce a tilingStrategy attribute on this op.
    // CHECK:       [[RESULT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], {{%.+}})
    // CHECK-SAME:      tilingStrategy = [1, 4, 1, 1]
}

}
