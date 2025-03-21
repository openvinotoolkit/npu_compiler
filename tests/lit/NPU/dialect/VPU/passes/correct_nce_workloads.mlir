//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --correct-NCE-workloads %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConv
func.func @DepthConv(%arg0: tensor<1x96x40x40xf16, {order = #NHWC}>) -> tensor<1x96x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<96x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<96x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x40x40xf16, {order = #NHWC}>
        -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<96x1x4x4xf16, {order = #NHWC}>
        -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32, {order = #NHWC}>
        -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [96, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x96x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x96x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:               VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x96x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x96x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvLarge
func.func @DepthConvLarge(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<512x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<512x1x1x4xsi32, {order = #NHWC}>
        -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 496, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 496, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 384, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 448, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 480, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 496, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x37x37xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvMultipleSplits
func.func @DepthConvMultipleSplits(%arg0: tensor<1x512x40x40xf16, {order = #NHWC}>) -> tensor<1x512x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<512x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<512x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x512x40x40xf16, {order = #NHWC}>
        -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<512x1x4x4xf16, {order = #NHWC}>
        -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<512x1x1x4xsi32, {order = #NHWC}>
        -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 1, 4, 4],
            strides = [1, 1]
        } -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 80, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 96, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x512x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<512x1x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x512x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<512x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 80, 0, 0] outSizes [1, 16, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    //CHECK:                VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL5:%.+]] = VPU.Copy([[VAL4]])
    // CHECK-SAME:      -> tensor<1x512x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL5]] : tensor<1x512x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvLargeNoChange
func.func @ConvLargeNoChange(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> tensor<1x384x37x37xf16, {order = #NHWC}> {
    %cst0 = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<384x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<384x1x1x4xsi32, {order = #NHWC}> -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.Convolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 4, 4],
            strides = [1, 1]
        } -> tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>
        ->  tensor<1x384x37x37xf16, {order = #NHWC}>

    return %5 : tensor<1x384x37x37xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // No change to the variants

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[VAL4:%.+]] = VPU.Copy([[VAL3]])
    // CHECK-SAME:      -> tensor<1x384x37x37xf16, {order = #NHWC}>

    // CHECK:       return [[VAL4]] : tensor<1x384x37x37xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvSparseOutput
func.func @DepthConvSparseOutput(%arg0: tensor<1x96x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<96x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x40x40xf16, {order = #NHWC}> -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%cst0) {out_mem_space = @CMX_NN} : tensor<96x1x4x4xf16, {order = #NHWC}> -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %2 = VPU.Copy(%wt) {out_mem_space = @CMX_NN} : tensor<96x1x1x4xsi32, {order = #NHWC}> -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.NCE.DepthConvolution(%0, %1, %2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [96, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %5 = VPU.Copy(%4) : !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        ->  !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>

    return %5 : !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<96x1x4x4xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<96x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL1:%.+]] = VPU.Copy([[CST]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<96x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL4:%.+]] = VPU.NCE.DepthConvolution([[VAL0]], [[VAL1]], [[VAL2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x96x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x96x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:               VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
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
func.func @MultipleSparseProducers(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> (tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %cst_weights_0 = const.Declare tensor<384x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_table_0 = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<384x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %input = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_0 = VPU.Copy(%cst_weights_0) {out_mem_space = @CMX_NN} : tensor<384x64x1x1xf16, {order = #NHWC}> -> tensor<384x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_table_0 = VPU.Copy(%cst_weights_table_0) {out_mem_space = @CMX_NN} : tensor<384x1x1x4xsi32, {order = #NHWC}> -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %conv0 = VPU.NCE.Convolution(%input, %weights_0, %weights_table_0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv1 = VPU.NCE.Convolution(%input, %weights_0, %weights_table_0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv2 = VPU.NCE.Convolution(%input, %weights_0, %weights_table_0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 1, 1],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x40x40xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
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
    %cst_weights_table_1 = const.Declare tensor<1024x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<1024x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %weights_1 = VPU.Copy(%cst_weights_1) {out_mem_space = @CMX_NN} : tensor<1024x768x1x1xf16, {order = #NHWC}> -> tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_table_1 = VPU.Copy(%cst_weights_table_1) {out_mem_space = @CMX_NN} : tensor<1024x1x1x4xsi32, {order = #NHWC}> -> tensor<1024x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %conv3 = VPU.NCE.Convolution(%concat0, %weights_1, %weights_table_1) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1024, 768, 1, 1],
            strides = [1, 1]
        } -> tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %conv4 = VPU.NCE.Convolution(%concat1, %weights_1, %weights_table_1) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1024, 768, 1, 1],
            strides = [1, 1]
        } -> tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    return %conv3, %conv4 : tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x1024x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // All producer ops have the same number of channels per variants

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>


    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>

    // CHECK:       [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>

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
func.func @MixedSparseProducers(%arg0: tensor<1x64x40x40xf16, {order = #NHWC}>) -> tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %cst_weights_0 = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_1 = const.Declare tensor<384x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_table_0 = const.Declare tensor<384x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<384x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %input = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_0 = VPU.Copy(%cst_weights_0) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_1 = VPU.Copy(%cst_weights_1) {out_mem_space = @CMX_NN} : tensor<384x1x4x4xf16, {order = #NHWC}> -> tensor<384x1x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_table_0 = VPU.Copy(%cst_weights_table_0) {out_mem_space = @CMX_NN} : tensor<384x1x1x4xsi32, {order = #NHWC}> -> tensor<384x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %conv0 = VPU.NCE.Convolution(%input, %weights_0, %weights_table_0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %depth_conv0 = VPU.NCE.DepthConvolution(%input, %weights_1, %weights_table_0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 128, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 64, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
                VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %concat = VPU.Concat(%conv0, %depth_conv0) {static_offsets = [[0, 0, 0, 0], [0, 384, 0, 0]]}
        : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>,
          !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        -> !VPU.SparseTensor<data=tensor<1x768x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x768x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>

    %cst_weights_2 = const.Declare tensor<1024x768x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x768x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_weights_table_2 = const.Declare tensor<1024x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<1024x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %weights_2 = VPU.Copy(%cst_weights_2) {out_mem_space = @CMX_NN} : tensor<1024x768x1x1xf16, {order = #NHWC}> -> tensor<1024x768x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights_table_2 = VPU.Copy(%cst_weights_table_2) {out_mem_space = @CMX_NN} : tensor<1024x1x1x4xsi32, {order = #NHWC}> -> tensor<1024x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>

    %conv1 = VPU.NCE.Convolution(%concat, %weights_2, %weights_table_2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1024, 768, 1, 1],
            strides = [1, 1]
        } -> tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 1024, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    return %conv1 : tensor<1x1024x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // All producer ops have the same number of channels per variants

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 288, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL0]]
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 128, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 160, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 224, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 256, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 288, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 320, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK-NEXT:      VPU.DPU.Workload outOffsets [0, 352, 0, 0] outSizes [1, 32, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>

    // CHECK:       [[VAL4:%.+]] = VPU.Concat([[VAL1]], [[VAL2]])

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]]

    // CHECK:       return [[VAL5]]
}
