//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ParsePrintGroupSparseTensorPartial
func.func @ParsePrintGroupSparseTensorPartial(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = dense<1> : tensor<1x32x16x16xi1>
    %1 = VPU.GroupSparseTensor(%arg0, %0)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    %2:2 = builtin.unrealized_conversion_cast %1 : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    return %2#0 : tensor<1x32x16x16xf16>

    // CHECK-DAG:       [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = dense<true> : tensor<1x32x16x16xi1>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[SM]])
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK:       [[VAL1:%.*]]:2 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseTensor
func.func @ParsePrintGroupSparseTensor(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x16x16xi1> = dense<1> : tensor<1x32x16x16xi1>
    %1 = const.Declare tensor<1x32x1x1xi32> = dense<1> : tensor<1x32x1x1xi32>
    %2 = VPU.GroupSparseTensor(%arg0, %0, %1)
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    %3:3 = builtin.unrealized_conversion_cast %2
            : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
            to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    return %3#0 : tensor<1x32x16x16xf16>

    // CHECK-DAG:   [[SM:%.*]] = const.Declare tensor<1x32x16x16xi1> = dense<true> : tensor<1x32x16x16xi1>
    // CHECK-DAG:   [[SE:%.*]] = const.Declare tensor<1x32x1x1xi32> = dense<1> : tensor<1x32x1x1xi32>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[SM]], [[SE]])
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK:       [[VAL1:%.*]]:3 = builtin.unrealized_conversion_cast
    // CHECK-SAME:                      [[VAL0]] : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x16x16xi1>, storage_element_table=tensor<1x32x1x1xi32>>
    // CHECK-SAME:                      to tensor<1x32x16x16xf16>, tensor<1x32x16x16xi1>, tensor<1x32x1x1xi32>
    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseTensorWeights
func.func @ParsePrintGroupSparseTensorWeights(%arg0: tensor<16x32x1x1xf16>)
        -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                             #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>> {
    %cst_sm = const.Declare tensor<16x1x1x128xi1> = dense<1> : tensor<16x1x1x128xi1>
    %2 = VPU.GroupSparseTensor(%arg0, %cst_sm) {
            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>,
            is_weights
        } -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                               #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>
    return %2 : !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
                                  #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>

    // CHECK-DAG:   [[CST_SM:%.*]] = const.Declare tensor<16x1x1x128xi1> = dense<true> : tensor<16x1x1x128xi1>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[CST_SM]]) {
    // CHECK:            is_weights
    // CHECK:            sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>
    // CHECK:        } -> !VPU.SparseTensor<data=tensor<16x32x1x1xf16>, sparsity_map=tensor<16x1x1x128xi1>, is_weights
    // CHECK-SAME:                          #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK:       return [[VAL0]]
}

// -----

// CHECK-LABEL: @ParsePrintGroupSparseSeAttrTensor
func.func @ParsePrintGroupSparseSeAttrTensor(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x32x16x16xf16> {
    %0 = const.Declare tensor<1x32x32x32xi1> = dense<1> : tensor<1x32x32x32xi1>
    %1 = const.Declare tensor<1x1x32x32xi32> = dense<1> : tensor<1x1x32x32xi32>

    %2 = VPU.GroupSparseTensor(%arg0, %0, %1) {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>}
            -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x32x32xi1>, storage_element_table=tensor<1x1x32x32xi32>, #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>>
    %3:3 = builtin.unrealized_conversion_cast %2
            : !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x32x32xi1>, storage_element_table=tensor<1x1x32x32xi32>, #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>>
            to tensor<1x32x16x16xf16>, tensor<1x32x32x32xi1>, tensor<1x1x32x32xi32>
    return %3#0 : tensor<1x32x16x16xf16>

    // CHECK-DAG:   [[SM:%.*]] = const.Declare tensor<1x32x32x32xi1> = dense<true> : tensor<1x32x32x32xi1>
    // CHECK-DAG:   [[SE:%.*]] = const.Declare tensor<1x1x32x32xi32> = dense<1> : tensor<1x1x32x32xi32>
    // CHECK:       [[VAL0:%.*]] = VPU.GroupSparseTensor(%arg0, [[SM]], [[SE]])
    // CHECK-SAME:                 {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>}
    // CHECK-SAME:                 -> !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x32x32xi1>, storage_element_table=tensor<1x1x32x32xi32>, #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>>

    // CHECK:       [[VAL1:%.*]]:3 = builtin.unrealized_conversion_cast [[VAL0]] :
    // CHECK-SAME:                   !VPU.SparseTensor<data=tensor<1x32x16x16xf16>, sparsity_map=tensor<1x32x32x32xi1>, storage_element_table=tensor<1x1x32x32xi32>, #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 32, 32, 32]>>
    // CHECK-SAME:                   to tensor<1x32x16x16xf16>, tensor<1x32x32x32xi1>, tensor<1x1x32x32xi32>

    // CHECK:       return [[VAL1]]#0 : tensor<1x32x16x16xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeSlice
func.func @CanonicalizeSlice(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights> {
    %cst = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_sparse = VPU.GroupSparseTensor(%cst, %cst_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>

    %cst_sparse_slice = VPU.Slice %cst_sparse [0, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    return %cst_sparse_slice : !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK-DAG:  [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:  [[WEIGHTS_SM:%.*]] = const.Declare tensor<64x1x1x384xi1> =
    // CHECK-SAME:     dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:      [[WEIGHTS_SPARSE:%.*]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>
    // CHECK:      return [[WEIGHTS_SPARSE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeMultipleSlice
func.func @CanonicalizeMultipleSlice(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>)
        -> (!VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>,
            !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>) {
    %cst = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %cst_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %cst_sparse = VPU.GroupSparseTensor(%cst, %cst_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>

    %cst_sparse_slice_0 = VPU.Slice %cst_sparse [0, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    %cst_sparse_slice_1 = VPU.Slice %cst_sparse [64, 0, 0, 0] [64, 32, 3, 3] :
        !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights> to
        !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    return %cst_sparse_slice_0, %cst_sparse_slice_1
        : !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>,
          !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK-DAG:  [[WEIGHTS_SLICE_1:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:  [[WEIGHTS_SM_SLICE_1:%.*]] = const.Declare tensor<64x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[64, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK-DAG:  [[WEIGHTS_SLICE_0:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> =  dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:  [[WEIGHTS_SM_SLICE_0:%.*]] = const.Declare tensor<64x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 32, 3, 3]>, #const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:      [[WEIGHTS_SPARSE_SLICE_1:%.*]] = VPU.GroupSparseTensor([[WEIGHTS_SLICE_1]], [[WEIGHTS_SM_SLICE_1]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK:      [[WEIGHTS_SPARSE_SLICE_0:%.*]] = VPU.GroupSparseTensor([[WEIGHTS_SLICE_0]], [[WEIGHTS_SM_SLICE_0]]) {is_weights}
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<64x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x384xi1>, is_weights>

    // CHECK:      return [[WEIGHTS_SPARSE_SLICE_0]], [[WEIGHTS_SPARSE_SLICE_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeSEAttrSlice
func.func @CanonicalizeSEAttrSlice(%arg0: tensor<1x64x4x4xf16, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x64x5x9xi1>,
                             storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>> {
    %data_cmx = VPU.Copy(%arg0)  {out_mem_space = @CMX_NN} : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %cst_sm = const.Declare tensor<1x64x9x9xi1> = dense<true> : tensor<1x64x9x9xi1>
    %se_table = VPU.StorageElementTable { dataElemType = f16, dataShape=[1, 64, 4, 4], seDepth = 1 : i64, seSize = 64 : i64, seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]> } -> tensor<1x1x9x9xi32, {order = #NHWC}>

    %sparse = VPU.GroupSparseTensor(%data_cmx, %cst_sm, %se_table)  {seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>}
        -> !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x64x9x9xi1>,
                             storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>>

    %sparse_slice = VPU.Slice %sparse [0, 0, 0, 0] [1, 64, 5, 9] :
        !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x9x9xi1>,
                          storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x5x9xi1>,
                          storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    return %sparse_slice : !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                             sparsity_map=tensor<1x64x5x9xi1>,
                                             storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       [[SM:%.*]] = const.Declare tensor<1x64x5x9xi1> = dense<true> : tensor<1x64x9x9xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 5, 9]>]
    // CHECK:       [[DATA:%.*]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[SE_TABLE:%.*]] = VPU.StorageElementTable {
    // CHECK-SAME:                       dataElemType = f16,
    // CHECK-SAME:                       dataShape = [1, 64, 3, 4],
    // CHECK-SAME:                       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>,
    // CHECK-SAME:                       seDepth = 1 : i64,
    // CHECK-SAME:                       seSize = 64 : i64}
    // CHECK-SAME:                     -> tensor<1x1x5x9xi32, {order = #NHWC}>

    // CHECK:       [[DATA_SLICE:%.*]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 3, 4] : tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:                          to tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[GROUP_OP:%.*]] = VPU.GroupSparseTensor([[DATA_SLICE]], [[SM]], [[SE_TABLE]])
    // CHECK-SAME:                      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>}
    // CHECK-SAME:                     -> !VPU.SparseTensor<
    // CHECK-SAME:                          data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                          sparsity_map=tensor<1x64x5x9xi1>,
    // CHECK-SAME:                          storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       return [[GROUP_OP]] : !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x64x5x9xi1>, storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CanonicalizeSEAttrMultipleSlice
func.func @CanonicalizeSEAttrMultipleSlice(%arg0: tensor<1x64x4x4xf16, {order = #NHWC}>)
        -> (!VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x64x5x9xi1>,
                             storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
            !VPU.SparseTensor<data=tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                             sparsity_map=tensor<1x64x4x9xi1>,
                                             storage_element_table=tensor<1x1x4x9xi32, {order = #NHWC}>,
                                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>) {
    %data_cmx = VPU.Copy(%arg0)  {out_mem_space = @CMX_NN} : tensor<1x64x4x4xf16, {order = #NHWC}> -> tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %cst_sm = const.Declare tensor<1x64x9x9xi1> = dense<true> : tensor<1x64x9x9xi1>
    %se_table = VPU.StorageElementTable { dataElemType = f16, dataShape=[1, 64, 4, 4], seDepth = 1 : i64, seSize = 64 : i64, seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>} -> tensor<1x1x9x9xi32, {order = #NHWC}>

    %sparse = VPU.GroupSparseTensor(%data_cmx, %cst_sm, %se_table)  {seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>}
        -> !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                             sparsity_map=tensor<1x64x9x9xi1>,
                             storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>>

    // Tile over H
    %sparse_slice_0 = VPU.Slice %sparse [0, 0, 0, 0] [1, 64, 5, 9] :
        !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x9x9xi1>,
                          storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x5x9xi1>,
                          storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    %sparse_slice_1 = VPU.Slice %sparse [0, 0, 5, 0] [1, 64, 4, 9] :
        !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x9x9xi1>,
                          storage_element_table=tensor<1x1x9x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPU.SparseTensor<data=tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                          sparsity_map=tensor<1x64x4x9xi1>,
                          storage_element_table=tensor<1x1x4x9xi32, {order = #NHWC}>,
                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    return %sparse_slice_0, %sparse_slice_1: !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                             sparsity_map=tensor<1x64x5x9xi1>,
                                             storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
                                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
                                             !VPU.SparseTensor<data=tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                             sparsity_map=tensor<1x64x4x9xi1>,
                                             storage_element_table=tensor<1x1x4x9xi32, {order = #NHWC}>,
                                             #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    // Sparsity map is constant, SliceOp was fused into it
    // CHECK-DAG:       [[SM_SLICE_0:%.*]] = const.Declare tensor<1x64x4x9xi1> = dense<true> : tensor<1x64x9x9xi1>, [#const.SubView<[0, 0, 5, 0], [1, 64, 4, 9]>]
    // CHECK-DAG:       [[SM_SLICE_1:%.*]] = const.Declare tensor<1x64x5x9xi1> = dense<true> : tensor<1x64x9x9xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 5, 9]>]

    // CHECK:       [[DATA:%.*]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x64x4x4xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // VPU.StorageElementTable is constant operation, SliceOp was fused into it
    // CHECK:       [[SET_SLICE_0:%.*]] = VPU.StorageElementTable {
    // CHECK-SAME:                          dataElemType = f16,
    // CHECK-SAME:                          dataShape = [1, 64, 3, 4],
    // CHECK-SAME:                          seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>,
    // CHECK-SAME:                          seDepth = 1 : i64,
    // CHECK-SAME:                          seSize = 64 : i64}
    // CHECK-SAME:                        -> tensor<1x1x5x9xi32, {order = #NHWC}>

    // CHECK:       [[SET_SLICE_1:%.*]] = VPU.StorageElementTable {
    // CHECK-SAME:                          dataElemType = f16,
    // CHECK-SAME:                          dataShape = [1, 64, 2, 4],
    // CHECK-SAME:                          seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>,
    // CHECK-SAME:                          seDepth = 1 : i64,
    // CHECK-SAME:                          seSize = 64 : i64}
    // CHECK-SAME:                          -> tensor<1x1x4x9xi32, {order = #NHWC}>

    // CHECK:       [[DATA_SLICE_0:%.*]] = VPU.Slice [[DATA]] [0, 0, 2, 0] [1, 64, 2, 4] : tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:                          to tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // Two VPU.GroupSparseTensor operations have been created, one for each slice
    // CHECK:       [[GROUP_SLICE_0:%.*]] = VPU.GroupSparseTensor([[DATA_SLICE_0]], [[SM_SLICE_0]], [[SET_SLICE_1]])
    // CHECK-SAME:                          {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>}
    // CHECK-SAME:                          -> !VPU.SparseTensor<
    // CHECK-SAME:                              data=tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                              sparsity_map=tensor<1x64x4x9xi1>,
    // CHECK-SAME:                              storage_element_table=tensor<1x1x4x9xi32, {order = #NHWC}>,
    // CHECK-SAME:                              #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    // CHECK:       [[DATA_SLICE_1:%.*]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 3, 4] : tensor<1x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:                          to tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[GROUP_SLICE_1:%.*]] = VPU.GroupSparseTensor([[DATA_SLICE_1]], [[SM_SLICE_1]], [[SET_SLICE_0]])
    // CHECK-SAME:                              {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>}
    // CHECK-SAME:                          -> !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x64x5x9xi1>, storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       return [[GROUP_SLICE_1]], [[GROUP_SLICE_0]] :
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x64x3x4xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x64x5x9xi1>, storage_element_table=tensor<1x1x5x9xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x64x2x4xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x64x4x9xi1>, storage_element_table=tensor<1x1x4x9xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>
}
