//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --convert-nce-interpolate-to-dw --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpAssignedSOHAsDwConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x96x10x10xf16, {order = #NHWC}>)
func.func @InterpAssignedSOHAsDwConv(%arg0: tensor<1x96x10x10xf16, {order = #NHWC}>) -> tensor<1x96x20x20xf16, {order = #NHWC}> {
    // In full model interp kernel will be [[0.00625, 0.125, 0.00625], [0.125, 0.25, 0.125], [0.00625, 0.125, 0.00625]]
    // but given the large size of the weights, I opted to put dummy 1.0 weights. It should not matter for the pass, as it doesn't
    // look at original interpolate weights content
    %weights = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x96x22x22xi1> = dense<1> : tensor<1x96x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [96], dataShape = [1, 96, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 96, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x96x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        rawFilterShape = [96, 96, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x96x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x96x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG-SAME{LITERAL}:          [[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x96x22x22xi1> = dense<true> : tensor<1x96x22x22xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 96, 10, 10]
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:           scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [96]
    // CHECK-SAME:      } -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      rawFilterShape = [96, 1, 3, 3],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inElemType = !quant.uniform<u8:f16, 0.57452542174096199:128>
// CHECK-DAG: [[IN_TYPE:!.+]] = !quant.uniform<u8:f16, 0.57452542174096199:128>
!outElemType = !quant.uniform<u8:f16, 0.54435472675398289:128>
// CHECK-DAG: [[OUT_TYPE:!.+]] = !quant.uniform<u8:f16, 0.54435472675398289:128>
!weightsElemType = !quant.uniform<u8:f16, 6.250000e-02>
// CHECK-DAG: [[W_TYPE:!.+]] = !quant.uniform<u8:f16, 6.250000e-02>

module {

config.Resources 2 of @NCE at 1.300000e+03 MHz

// CHECK: @InterpAssignedSOKAsDwConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x96x10x10x[[IN_TYPE]], {order = #NHWC}>)
func.func @InterpAssignedSOKAsDwConv(%arg0: tensor<1x96x10x10x!inElemType, {order = #NHWC}>) -> tensor<1x96x20x20x!outElemType, {order = #NHWC}> {
    // In full model interp kernel will be [[1, 2, 1], [2, 4, 2], [1, 2, 1]] but, given the large size of the weights,
    // I opted to put dummy 1 weights. It should not matter for the pass, as it doesn't look at original interpolate weights content
    %weights = const.Declare tensor<96x96x3x3x!weightsElemType, {order = #NHWC}> = dense<1.0>
        : tensor<96x96x3x3xf32>, [#const.CastElemType<!weightsElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x96x22x22xi1> = dense<1> : tensor<1x96x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = !inElemType, seDepth = 1, seSize = [96], dataShape = [1, 96, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 96, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x96x10x10x!inElemType, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        rawFilterShape = [96, 96, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x96x20x20x!outElemType, {order = #NHWC}>

    return %interpolate : tensor<1x96x20x20x!outElemType, {order = #NHWC}>

    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x96x22x22xi1> = dense<true> : tensor<1x96x22x22xi1>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x16x1x1x[[W_TYPE]], {order = #NHWC}>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00]]],
    // CHECK-DAG-SAME{LITERAL}:          [[[1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00]]],
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable {dataElemType = [[IN_TYPE]], dataShape = [1, 96, 10, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:           scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    // CHECK-SAME:       seDepth = 6 : i64, seSize = [16, 16, 16, 16, 16, 16]
    // CHECK-SAME:      } -> tensor<1x6x22x22xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      rawFilterShape = [96, 1, 3, 3],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x20x20x[[OUT_TYPE]], {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x20x20x[[OUT_TYPE]], {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @InterpAssignedClusteringAsDwConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x96x10x10xf16, {order = #NHWC}>)
func.func @InterpAssignedClusteringAsDwConv(%arg0: tensor<1x96x10x10xf16, {order = #NHWC}>) -> tensor<1x96x20x20xf16, {order = #NHWC}> {
    // In full model interp kernel will be [[0.00625, 0.125, 0.00625], [0.125, 0.25, 0.125], [0.00625, 0.125, 0.00625]]
    // but given the large size of the weights, I opted to put dummy 1.0 weights. It should not matter for the pass, as it doesn't
    // look at original interpolate weights content
    %weights = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x96x22x22xi1> = dense<1> : tensor<1x96x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [96], dataShape = [1, 96, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 96, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x96x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        rawFilterShape = [96, 96, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x96x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x96x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x96x22x22xi1> = dense<true> : tensor<1x96x22x22xi1>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG-SAME{LITERAL}:          [[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = f16, dataShape = [1, 96, 10, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:           scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 96, 22, 22]>
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [96]
    // CHECK-SAME:      } -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      rawFilterShape = [96, 1, 3, 3],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpAssignedHKSwitchAsDwConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32x10x10xf16, {order = #NHWC}>)
func.func @InterpAssignedHKSwitchAsDwConv(%arg0: tensor<1x32x10x10xf16, {order = #NHWC}>) -> tensor<1x32x20x20xf16, {order = #NHWC}> {
    // In full model interp kernel will be [[0.00625, 0.125, 0.00625], [0.125, 0.25, 0.125], [0.00625, 0.125, 0.00625]]
    // but given the large size of the weights, I opted to put dummy 1.0 weights. It should not matter for the pass, as it doesn't
    // look at original interpolate weights content
    %weights = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x32x22x22xi1> = dense<1> : tensor<1x32x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [32], dataShape = [1, 32, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 32, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x32x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x32x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
        rawFilterShape = [32, 32, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x32x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x32x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x32x22x22xi1> = dense<true> : tensor<1x32x22x22xi1>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG-SAME{LITERAL}:          [[[6.250000e-02, 1.250000e-01, 6.250000e-02], [1.250000e-01, 2.500000e-01, 1.250000e-01], [6.250000e-02, 1.250000e-01, 6.250000e-02]]],
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = f16, dataShape = [1, 32, 10, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:           scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 22, 22]>
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [32]
    // CHECK-SAME:      } -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>,
    // CHECK-SAME:      rawFilterShape = [32, 1, 3, 3],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x32x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x32x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inElemType = !quant.uniform<u8:f16, 0.57452542174096199:128>
// CHECK-DAG: [[IN_TYPE:!.+]] = !quant.uniform<u8:f16, 0.57452542174096199:128>
!outElemType = !quant.uniform<u8:f16, 0.54435472675398289:128>
// CHECK-DAG: [[OUT_TYPE:!.+]] = !quant.uniform<u8:f16, 0.54435472675398289:128>
!weightsElemType = !quant.uniform<u8:f16, 6.250000e-02>
// CHECK-DAG: [[W_TYPE:!.+]] = !quant.uniform<u8:f16, 6.250000e-02>

// CHECK: @SingleClusterInterpAsDwConv
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x10x10x[[IN_TYPE]], {order = #NHWC}>)
func.func @SingleClusterInterpAsDwConv(%arg0: tensor<1x64x10x10x!inElemType, {order = #NHWC}>) -> tensor<1x64x20x20x!outElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x3x3x!weightsElemType, {order = #NHWC}>
        = dense<1.0> : tensor<64x64x3x3xf32>, [#const.CastElemType<!weightsElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x22x22xi1> = dense<1> : tensor<1x64x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = !inElemType, seDepth = 1, seSize = [64], dataShape = [1, 64, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x64x10x10x!inElemType, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x20x20x!outElemType, {order = #NHWC}>

    return %interpolate : tensor<1x64x20x20x!outElemType, {order = #NHWC}>

    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x64x22x22xi1> = dense<true> : tensor<1x64x22x22xi1>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1x[[W_TYPE]], {order = #NHWC}>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00]]],
    // CHECK-DAG-SAME{LITERAL}:          [[[1.000000e+00, 2.000000e+00, 1.000000e+00], [2.000000e+00, 4.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00, 1.000000e+00]]],
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = [[IN_TYPE]], dataShape = [1, 64, 10, 10],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>
    // CHECK-SAME:       seDepth = 1 : i64, seSize = [64]
    // CHECK-SAME:      } -> tensor<1x1x22x22xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      rawFilterShape = [64, 1, 3, 3],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x20x20x[[OUT_TYPE]], {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x64x20x20x[[OUT_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SingleClusterInterpNotSupportedChannel
func.func @SingleClusterInterpNotSupportedChannel(%arg0: tensor<1x128x10x10xf16, {order = #NHWC}>) -> tensor<1x128x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x128x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x128x22x22xi1> = dense<1> : tensor<1x128x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [128], dataShape = [1, 128, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x128x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x128x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [128, 128, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x128x20x20xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.NCE.DepthConvolution
    // CHECK: VPU.NCE.Interpolate
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SingleClusterInterpNotSupportedWorkloadWithOptimization
func.func @SingleClusterInterpNotSupportedWorkloadWithOptimization(%arg0: tensor<1x64x10x10xf16, {order = #NHWC}>) -> tensor<1x64x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x22x22xi1> = dense<1> : tensor<1x64x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [64], dataShape = [1, 64, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x64x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x20x20xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.NCE.DepthConvolution
    // CHECK: VPU.NCE.Interpolate
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {

config.Resources 2 of @NCE at 1.300000e+03 MHz
// CHECK-LABEL: @SOKInterpTooManyTilesNeeded
func.func @SOKInterpTooManyTilesNeeded(%arg0: tensor<1x512x10x10xf16, {order = #NHWC}>) -> tensor<1x512x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x512x22x22xi1> = dense<1> : tensor<1x512x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [512], dataShape = [1, 512, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 512, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 512, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x512x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x512x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 512, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        rawFilterShape = [512, 512, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x512x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x512x20x20xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.NCE.DepthConvolution
    // CHECK: VPU.NCE.Interpolate
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOHInterpTooManyTilesNeeded
func.func @SOHInterpTooManyTilesNeeded(%arg0: tensor<1x128x10x10xf16, {order = #NHWC}>) -> tensor<1x128x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x128x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x128x22x22xi1> = dense<1> : tensor<1x128x22x22xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [128], dataShape = [1, 128, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 22, 22]>
    } -> tensor<1x1x22x22xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 22, 22]>
    } -> !VPU.SparseTensor<data=tensor<1x128x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x128x22x22xi1>,
                           storage_element_table=tensor<1x1x22x22xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>,
                                              scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 22, 22]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        rawFilterShape = [128, 128, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x128x20x20xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.NCE.DepthConvolution
    // CHECK: VPU.NCE.Interpolate
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {

config.Resources 2 of @NCE at 1.300000e+03 MHz

// CHECK-LABEL: @SOKInterpNotSupportedWorkload
func.func @SOKInterpNotSupportedWorkload(%arg0: tensor<1x64x10x10xf16, {order = #NHWC}>) -> tensor<1x64x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x64x30x30xi1> = dense<1> : tensor<1x64x30x30xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [64], dataShape = [1, 64, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 3.0, 3.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 30, 30]>
    } -> tensor<1x1x30x30xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 3.0, 3.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 30, 30]>
    } -> !VPU.SparseTensor<data=tensor<1x64x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x30x30xi1>,
                           storage_element_table=tensor<1x1x30x30xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 3.0, 3.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 30, 30]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [3, 3],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x30x30xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x30x30xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.NCE.DepthConvolution
    // CHECK: VPU.NCE.Interpolate
}
}
