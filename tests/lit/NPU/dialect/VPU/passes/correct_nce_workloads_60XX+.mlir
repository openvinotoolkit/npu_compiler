//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --correct-NCE-workloads %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConv4x4
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x96x40x40xf16, {order = #NHWC}>)
func.func @DepthConv4x4(%arg0: tensor<1x96x40x40xf16, {order = #NHWC}>) -> tensor<1x96x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<96x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x40x40xf16, {order = #NHWC}>
        -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<96x1x4x4xf16, {order = #NHWC}>
        -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [96, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x96x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x96x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]]) rawFilterShape [96, 1, 4, 4] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x96x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x96x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvLarge
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x512x40x40xf16, {order = #NHWC}>)
func.func @DepthConvLarge(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [512, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 496, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 496, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]]) rawFilterShape [512, 1, 4, 4] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 256, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 224, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 480, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 496, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvMultipleSplits
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x512x40x40xf16, {order = #NHWC}>)
func.func @DepthConvMultipleSplits(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [512, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]])  rawFilterShape [512, 1, 4, 4] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 16, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:          VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvLargeNoChange
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
func.func @ConvLargeNoChange(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> tensor<1x384x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.Convolution(%0, %1) rawFilterShape [384, 64, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        ->  tensor<1x384x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x384x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // No change to the variants

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]]) rawFilterShape [384, 64, 4, 4] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x384x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x384x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvSparseOutput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x96x40x40xf16, {order = #NHWC}>)
func.func @DepthConvSparseOutput(%arg0: tensor<1x96x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x40x40xf16, {order = #NHWC}> -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<96x1x4x4xf16, {order = #NHWC}> -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [96, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        ->  !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>

    return %5 : !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]]) rawFilterShape [96, 1, 4, 4] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>

    // CHECK:       return [[VAL5]] : !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv   Conv   Conv
//   \   /    \  /
//  Concat  Concat
//    |       |
//   Conv    Conv

// CHECK-LABEL: @MultipleSparseProducers
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
func.func @MultipleSparseProducers(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> (tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %cst_weights_0 = const.Declare tensor<384x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x1x1xf16>, [#const.Reorder<#NHWC>]

    %input = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_0 = VPU.Copy(%cst_weights_0) {out_mem_space = @CMX_NN} : tensor<384x64x1x1xf16, {order = #NHWC}> -> tensor<384x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %conv0 = VPU.NCE.Convolution(%input, %weights_0) rawFilterShape [384, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv1 = VPU.NCE.Convolution(%input, %weights_0) rawFilterShape [384, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv2 = VPU.NCE.Convolution(%input, %weights_0) rawFilterShape [384, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %concat0 = VPU.Concat(%conv0, %conv1) {static_offsets = [[0, 0, 0, 0], [0, 384, 0, 0]]}
        : !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
          !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x768x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>

    %concat1 = VPU.Concat(%conv1, %conv2) {static_offsets = [[0, 0, 0, 0], [0, 384, 0, 0]]}
        : !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
          !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x768x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>

    %cst_weights_1 = const.Declare tensor<1024x768x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x768x1x1xf16>, [#const.Reorder<#NHWC>]

    %weights_1 = VPU.Copy(%cst_weights_1) {out_mem_space = @CMX_NN} : tensor<1024x768x1x1xf16, {order = #NHWC}> -> tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %conv3 = VPU.NCE.Convolution(%concat0, %weights_1) rawFilterShape [1024, 768, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x768x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>, tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv4 = VPU.NCE.Convolution(%concat1, %weights_1) rawFilterShape [1024, 768, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x768x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>>, tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    return %conv3, %conv4 : tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // All producer ops have the same number of channels per variants

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>


    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>

    // CHECK:       [[VAL4:%.+]] = VPU.Concat([[VAL1]], [[VAL2]])
    // CHECK:       [[VAL5:%.+]] = VPU.Concat([[VAL2]], [[VAL3]])

    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Convolution([[VAL4]]
    // CHECK:       [[VAL7:%.+]] = VPU.NCE.Convolution([[VAL5]]

    // CHECK:       return [[VAL6]], [[VAL7]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv  DepthConv
//   \     /
//    Concat
//      |
//     Conv

// CHECK-LABEL: @MixedSparseProducers
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
func.func @MixedSparseProducers(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %cst_weights_0 = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_1 = const.Declare tensor<384x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x1x4x4xf16>, [#const.Reorder<#NHWC>]

    %input = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_0 = VPU.Copy(%cst_weights_0) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_1 = VPU.Copy(%cst_weights_1) {out_mem_space = @CMX_NN} : tensor<384x1x4x4xf16, {order = #NHWC}> -> tensor<384x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %conv0 = VPU.NCE.Convolution(%input, %weights_0) rawFilterShape [384, 64, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %depth_conv0 = VPU.NCE.DepthConvolution(%input, %weights_1) rawFilterShape [384, 1, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %concat = VPU.Concat(%conv0, %depth_conv0) {static_offsets = [[0, 0, 0, 0], [0, 384, 0, 0]]}
        : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
          !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x768x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>

    %cst_weights_2 = const.Declare tensor<1024x768x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x768x1x1xf16>, [#const.Reorder<#NHWC>]

    %weights_2 = VPU.Copy(%cst_weights_2) {out_mem_space = @CMX_NN} : tensor<1024x768x1x1xf16, {order = #NHWC}> -> tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %conv1 = VPU.NCE.Convolution(%concat, %weights_2) rawFilterShape [1024, 768, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x768x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>, tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    return %conv1 : tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // All producer ops have the same number of channels per variants

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 288, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 288, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] pad [0, 0, 0, 0] <CUBOID_4x16>

    // CHECK:       [[VAL4:%.+]] = VPU.Concat([[VAL1]], [[VAL2]])

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]]

    // CHECK:       return [[VAL5]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConv3x3
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x384x30x30xf16, {order = #NHWC}>)
func.func @DepthConv3x3(%arg0: tensor<1x384x30x30xf16, {order = #NHWC}>) -> tensor<1x384x28x28xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<384x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<384x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x384x30x30xf16, {order = #NHWC}>
        -> tensor<1x384x30x30xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<384x16x1x1xf16, {order = #NHWC}>
        -> tensor<384x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [384, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x384x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 7, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x384x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x384x28x28xf16, {order = #NHWC}>

    return %5 : tensor<1x384x28x28xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<384x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x384x30x30xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]]) rawFilterShape [384, 1, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x384x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 256, 7, 28] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:               VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 128, 7, 28] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x384x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x384x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeWorkloadForDepthwiseConv(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x54x54xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = [@CMX_NN, 0]} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [128, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} -> tensor<1x128x54x54xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] pad [0, 1, 0, 1] <CUBOID_16x16>
    }

    %5 = VPU.Copy(%4) : tensor<1x128x54x54xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        -> tensor<1x128x54x54xf16, {order = #NHWC}>

    return %5 : tensor<1x128x54x54xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] pad [0, 1, 0, 1] <CUBOID_16x16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0017310915969488189:127>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @OptimizeWorkloadForDepthwiseConvWithU8(%arg0: tensor<1x128x56x56x!qElemType, {order = #NHWC}>) -> tensor<1x128x54x54x!qElemType, {order = #NHWC}> {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x128x56x56x!qElemType, {order = #NHWC}> -> tensor<1x128x56x56x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = [@CMX_NN, 0]} : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [128, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} -> tensor<1x128x54x54x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] pad [0, 1, 0, 1] <CUBOID_16x16>
    }

    %5 = VPU.Copy(%4) : tensor<1x128x54x54x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        -> tensor<1x128x54x54x!qElemType, {order = #NHWC}>

    return %5 : tensor<1x128x54x54x!qElemType, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 28, 28] pad [0, 1, 0, 1] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    256x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x256x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithL1aOpt256Channels
func.func @DepthConvWithL1aOpt256Channels(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<256x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<256x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [256, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // split workload into any of [16, 32, 64, 96, 128, 160, 192, 224, 256] to enable small kernel optimization
    // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:  VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x160x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    160x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x160x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithL1aOpt16
func.func @DepthConvWithL1aOpt16(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<160x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<160x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<160x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [160, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 80, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // split workload into size 16&64 to enable small kernel optimization
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 144, 0, 0] outSizes [1, 16, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x128x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    128x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[64, 16, 1, 1], [64, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]],
    memory_shapes = [[64, 16, 1, 1], [64, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]]
}>

!Output_CMX = !VPU.DistributedTensor<
    1x128x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 24, 42], [1, 64, 24, 42]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 128, 24, 42], [1, 128, 24, 42]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @DepthConvWithL1aOpt128Channels
func.func @DepthConvWithL1aOpt128Channels(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<128x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [128, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:       VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:  VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x2048x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    2048x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x2048x24x42xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @DepthConvWithL1aOpt2048Channels
func.func @DepthConvWithL1aOpt2048Channels(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<2048x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<2048x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<2048x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [2048, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 1024, 0, 0] outSizes [1, 1024, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 512, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 768, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 1024, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 1280, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 1536, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 1792, 0, 0] outSizes [1, 256, 24, 42] pad [1, 1, 1, 1] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvWithSmallSpatialCompute
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x64x16x16xf16, {order = #NHWC}>
// Small dense spatial compute and 1x1 kernel triggers errata condition
func.func @DepthConvWithSmallSpatialCompute(%arg0: tensor<1x64x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x16x16xf16, {order = #NHWC}>
        -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<64x16x1x1xf16, {order = #NHWC}>
        -> tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1) rawFilterShape [64, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]
        } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x16x16xf16, {order = #NHWC}>

    return %5 : tensor<1x64x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]]) rawFilterShape [64, 1, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_4x16>
    // CHECK:               VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_4x16>

    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x64x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x64x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x80x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    80x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x80x24x24xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @SplitOverHeightDepthConv
// Spatial Width of 24 doesn't trigger the Errata
func.func @SplitOverHeightDepthConv(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<80x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<80x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [80, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 12, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 12, 0] outSizes [1, 80, 24, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 12, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 12, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 12, 0] outSizes [1, 64, 24, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 64, 12, 0] outSizes [1, 16, 24, 24] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.DistributedTensor<
    1x80x24x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

!Weights_CMX = !VPU.DistributedTensor<
    80x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x80x24x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments
}>

// CHECK-LABEL: @SplitOverHeightDepthConvWithSmallSpatialCompute
// Spatial Width of 16 and per cluster height of 12 triggers the Errata
func.func @SplitOverHeightDepthConvWithSmallSpatialCompute(%arg0: !Input_CMX) -> !Output_CMX {
    %cst0 = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<80x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<80x16x1x1xf16, {order = #NHWC}> -> !Weights_CMX

    %2 = VPU.NCE.DepthConvolution(%arg0, %0) rawFilterShape [80, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,

            strides = [1, 1]} -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 12, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 0, 12, 0] outSizes [1, 80, 24, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %2 : !Output_CMX


    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 12, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 12, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 12, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 12, 0] outSizes [1, 32, 24, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 12, 0] outSizes [1, 32, 24, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 12, 0] outSizes [1, 16, 24, 16] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 1 : i64}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputType = !VPU.SparseTensor<
    data=tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    sparsity_map=tensor<1x64x4x4xi1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
>

// CHECK-LABEL: @DepthConvSparseInputAndSmallSpatialCompute
// Because of sparse input the Errata condition is not triggered
func.func @DepthConvSparseInputAndSmallSpatialCompute(
    %ACT: !SparseInputType,
    %FILT: tensor<64x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
) -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
    // CHECK:   [[ACT:%.+]]: !VPU.SparseTensor{{.+}}, [[FILT:%.+]]: tensor<64x16x1x1xf16{{.+}}>
    %DWCONV = VPU.NCE.DepthConvolution(%ACT, %FILT) rawFilterShape [64, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        minimumHardwareExecutionCost = 790 : i64,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<
            mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64
        >,

        strides = [1, 1]
    } -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        VPU.DPU.Workload
            outOffsets [0, 0, 0, 0]
            outSizes [1, 64, 16, 16]
            pad [1, 1, 1, 1]
            <CUBOID_16x16>
    }
    // CHECK:   [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[ACT]], [[FILT]])
    // CHECK-NEXT:  VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0]
    // CHECK-SAME:      outSizes [1, 64, 16, 16]

    return %DWCONV : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   return [[DWCONV]] : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolNoSmallKernelOpt
func.func @AvgPoolNoSmallKernelOpt(%arg0: tensor<1x128x56x56xf16, {order = #NHWC}>) -> tensor<1x128x56x56xf16, {order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x128x56x56xf16, {order = #NHWC}> -> tensor<1x128x56x56xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    %1 = VPU.NCE.AveragePool(%0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x128x56x56xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 56, 56] pad [1, 1, 1, 1] <CUBOID_16x16>
    }
    %2 = VPU.Copy(%1) : tensor<1x128x56x56xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        -> tensor<1x128x56x56xf16, {order = #NHWC}>

    return %2 : tensor<1x128x56x56xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.AveragePool
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 56, 56] pad [1, 1, 1, 1] <CUBOID_16x16>
    // CHECK-NOT:     VPU.DPU.Workload outOffsets [0, 32, 0, 0]
}
