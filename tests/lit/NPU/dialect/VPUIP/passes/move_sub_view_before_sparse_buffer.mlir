//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --move-subview-before-sparse-buffer --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveSubviewUp
func.func @MoveSubviewUp(%arg0: memref<1x64x4x4xf16, #NHWC>,
                                   %arg1: !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                           sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                           storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
    -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                           sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                           storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>> {
    // Constant sparsity map
    %cst_sm = const.Declare memref<1x64x9x9xi1, {order = #NHWC}> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>

    // Stprage element table op
    %se_table = VPUIP.StorageElementTable { dataElemType = f16, dataShape=[1, 64, 4, 4], seDepth = 1 : i64, seSize = 64 : i64, seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]> } -> memref<1x1x9x9xi32, #NHWC>

    // Group all buffers
    %sparse = VPUIP.GroupSparseBuffer(%arg0, %cst_sm, %se_table)  {seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>}
        -> !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>>

    // Apply subview to grouped buffer
    %sparse_subview = VPUIP.SubView %sparse [0, 0, 0, 0] [1, 64, 5, 9] :
        !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                            sparsity_map=memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                            storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // Copy output, is is necessary, otherwise function result type will be changed after the pass
    %result = VPUIP.Copy inputs(%sparse_subview : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                                    sparsity_map=memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
                            outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
                            -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    return %result : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       [[SM:%.*]] = const.Declare memref<1x64x5x9xi1, #NHWC> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [1, 64, 5, 9]>]

    // CHECK:       [[SE_TABLE:%.*]] = VPUIP.StorageElementTable {
    // CHECK-SAME:                       dataElemType = f16,
    // CHECK-SAME:                       dataShape = [1, 64, 3, 4],
    // CHECK-SAME:                       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>,
    // CHECK-SAME:                       seDepth = 1 : i64,
    // CHECK-SAME:                       seSize = 64 : i64}
    // CHECK-SAME:                     -> memref<1x1x5x9xi32, #NHWC>

    // CHECK:       [[DATA_SLICE:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 3, 4] : memref<1x64x4x4xf16, #NHWC>
    // CHECK-SAME:                          to memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>

    // CHECK:       [[GROUP_OP:%.*]] = VPUIP.GroupSparseBuffer([[DATA_SLICE]], [[SM]], [[SE_TABLE]])
    // CHECK-SAME:                      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>}
    // CHECK-SAME:                     -> !VPUIP.SparseBuffer<
    // CHECK-SAME:                          data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>
    // CHECK-SAME:                          sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                          storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       [[COPY_OP:%.*]] = VPUIP.Copy inputs([[GROUP_OP]] :
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
    // CHECK-SAME:                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
    // CHECK-SAME:                               outputs(%arg1 :
    // CHECK-SAME:                                    !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
    // CHECK-SAME:                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
    // CHECK-SAME:                      -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
    // CHECK-SAME:                          sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                          storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:       return [[COPY_OP]] : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
    // CHECK-SAME:          sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:          storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:          #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveSubviewUpCheckReinfer
// CHECK-SAME:     ([[ARG0:%.+]]: memref<1x64x4x4xf16, #NHWC>, [[ARG1:%.+]]: memref<1x64x3x4xf16, #NHWC>)
func.func @MoveSubviewUpCheckReinfer(%arg0: memref<1x64x4x4xf16, #NHWC>,
                                     %arg1: memref<1x64x3x4xf16, #NHWC>)
    -> memref<1x64x3x4xf16, #NHWC> {
    // Constant sparsity map
    %cst_sm = const.Declare memref<1x64x9x9xi1, {order = #NHWC}> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>

    // Stprage element table op
    %se_table = VPUIP.StorageElementTable { dataElemType = f16, dataShape=[1, 64, 4, 4], seDepth = 1 : i64, seSize = 64 : i64, seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]> } -> memref<1x1x9x9xi32, #NHWC>

    // Group all buffers
    %sparse = VPUIP.GroupSparseBuffer(%arg0, %cst_sm, %se_table)  {seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>}
        -> !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>>

    // Apply subview to grouped buffer
    %sparse_subview = VPUIP.SubView %sparse [0, 0, 0, 0] [1, 64, 5, 9] :
        !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                            sparsity_map=memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                            storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // UngroupSparseBuffer has its own infer return type function, check that type is updated after all optimizations and operations are folded
    %1, %2, %3 = VPUIP.UngroupSparseBuffer(%sparse_subview) {resultSegmentSizes = array<i32: 1, 1, 1>} -> memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>, memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>, memref<1x1x5x9xi32, #NHWC>

    %result = VPUIP.Copy inputs(%1 : memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>)
                            outputs(%arg1 : memref<1x64x3x4xf16, #NHWC>)
                            -> memref<1x64x3x4xf16, #NHWC>

    return %result : memref<1x64x3x4xf16, #NHWC>

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 64, 3, 4] : memref<1x64x4x4xf16, #NHWC> to memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>
    // CHECK:       [[GROUP:%.+]] = VPUIP.GroupSparseBuffer([[SUBVIEW]]
    // CHECK:       [[DATA:%.+]], [[SM:%.+]], [[SE:%.+]] = VPUIP.UngroupSparseBuffer([[GROUP]])
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy inputs([[DATA]] : memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>) outputs([[ARG1]] : memref<1x64x3x4xf16, #NHWC>) -> memref<1x64x3x4xf16, #NHWC>
    // CHECK:       return [[COPY]] : memref<1x64x3x4xf16, #NHWC>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MoveMultipleSubviewUp
func.func @MoveMultipleSubviewUp(
            %arg0: memref<1x64x4x4xf16, #NHWC>,
            %arg1: !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
            %arg2: !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
                                sparsity_map=memref<1x64x4x9xi1, #NHWC>,
                                storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>)
        -> (!VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
            !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
                                sparsity_map=memref<1x64x4x9xi1, #NHWC>,
                                storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>) {
    %cst_sm = const.Declare memref<1x64x9x9xi1, {order = #NHWC}> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>
    %se_table = VPUIP.StorageElementTable { dataElemType = f16, dataShape=[1, 64, 4, 4], seDepth = 1 : i64, seSize = 64 : i64, seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>} -> memref<1x1x9x9xi32, #NHWC>

    %sparse = VPUIP.GroupSparseBuffer(%arg0, %cst_sm, %se_table)  {seAttr =  #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>}
        -> !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                               sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                               storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>>

    // Tile over H
    %sparse_slice_0 = VPUIP.SubView %sparse [0, 0, 0, 0] [1, 64, 5, 9] :
        !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                            sparsity_map=memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                            storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    %result_0 = VPUIP.Copy inputs(%sparse_slice_0 : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                                    sparsity_map=memref<1x64x5x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
                            outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
                            -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    %sparse_slice_1 = VPUIP.SubView %sparse [0, 0, 5, 0] [1, 64, 4, 9] :
        !VPUIP.SparseBuffer<data=memref<1x64x4x4xf16, #NHWC>,
                            sparsity_map=memref<1x64x9x9xi1, {order = #NHWC}>,
                            storage_element_table=memref<1x1x9x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 9, 9]>> to
        !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                            sparsity_map=memref<1x64x4x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                            storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                            #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    %result_1 = VPUIP.Copy inputs(%sparse_slice_1 : !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
                                    sparsity_map=memref<1x64x4x9xi1, {order = #NHWC, strides = [5184, 1, 576, 64]}>,
                                    storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>)
                            outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x4x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>)
                            -> !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x4x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    return %result_0, %result_1: !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x5x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x5x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
                                 !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
                                    sparsity_map=memref<1x64x4x9xi1, #NHWC>,
                                    storage_element_table=memref<1x1x4x9xi32, #NHWC>,
                                    #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1., 1., 2., 2.], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    // CHECK:           [[CONST_SM_0:%.*]] = const.Declare memref<1x64x4x9xi1, #NHWC> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>, [#const.SubView<[0, 0, 5, 0], [1, 64, 4, 9]>]
    // CHECK:           [[CONST_SM_1:%.*]] = const.Declare memref<1x64x5x9xi1, #NHWC> = dense<true> : tensor<1x64x9x9xi1, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [1, 64, 5, 9]>]
    // CHECK:           [[SET_OP_0:%.*]] = VPUIP.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 2, 4],
    // CHECK-SAME:              seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>, seDepth = 1 : i64, seSize = 64 : i64} -> memref<1x1x4x9xi32, #NHWC>
    // CHECK:           [[SET_OP_1:%.*]] = VPUIP.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 3, 4],
    // CHECK-SAME:              seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>, seDepth = 1 : i64, seSize = 64 : i64} -> memref<1x1x5x9xi32, #NHWC>

    // CHECK:           [[SUBVIEW_DATA_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 64, 3, 4] : memref<1x64x4x4xf16, #NHWC>
    // CHECK-SAME:              to memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>

    // CHECK:           [[GROUP_1:%.*]] = VPUIP.GroupSparseBuffer([[SUBVIEW_DATA_1:%.*]], [[CONST_SM_1]], [[SET_OP_1:%.*]])
    // CHECK-SAME:              {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                        offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>}
    // CHECK-SAME:              -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
    // CHECK-SAME:                      sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                      storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:           [[COPY_1:%.*]] = VPUIP.Copy inputs([[GROUP_1:%.*]] : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
    // CHECK-SAME:              sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:              storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:              #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
    // CHECK-SAME:              outputs(%arg1 : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
    // CHECK-SAME:                              sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                              storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                              #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                      offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>)
    // CHECK-SAME:              -> !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>,
    // CHECK-SAME:                      sparsity_map=memref<1x64x5x9xi1, #NHWC>,
    // CHECK-SAME:                      storage_element_table=memref<1x1x5x9xi32, #NHWC>,
    // CHECK-SAME:                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                              offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>

    // CHECK:           [[SUBVIEW_DATA_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 2, 0] [1, 64, 2, 4] : memref<1x64x4x4xf16, #NHWC>
    // CHECK-SAME:              to memref<1x64x2x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>

    // CHECK:           [[GROUP_0:%.*]] = VPUIP.GroupSparseBuffer([[SUBVIEW_DATA_0]], [[CONST_SM_0]], [[SET_OP_0:%.*]])
    // CHECK-SAME:              {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                        offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>}
    // CHECK-SAME:              -> !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
    // CHECK-SAME:                      sparsity_map=memref<1x64x4x9xi1, #NHWC>,
    // CHECK-SAME:                      storage_element_table=memref<1x1x4x9xi32, #NHWC>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    // CHECK:           [[COPY_0:%.*]] = VPUIP.Copy inputs([[GROUP_0:%.*]] : !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, {order = #NHWC, strides = [1024, 1, 256, 64]}>,
    // CHECK-SAME:              sparsity_map=memref<1x64x4x9xi1, #NHWC>, storage_element_table=memref<1x1x4x9xi32, #NHWC>,
    // CHECK-SAME:              #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>)
    // CHECK-SAME:              outputs(%arg2 : !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
    // CHECK-SAME:                      sparsity_map=memref<1x64x4x9xi1, #NHWC>,
    // CHECK-SAME:                      storage_element_table=memref<1x1x4x9xi32, #NHWC>,
    // CHECK-SAME:                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                              offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>)
    // CHECK-SAME:              -> !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>,
    // CHECK-SAME:                      sparsity_map=memref<1x64x4x9xi1, #NHWC>,
    // CHECK-SAME:                      storage_element_table=memref<1x1x4x9xi32, #NHWC>,
    // CHECK-SAME:                      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>

    // CHECK:           return [[COPY_1]], [[COPY_0]] : !VPUIP.SparseBuffer<data=memref<1x64x3x4xf16, #NHWC>, sparsity_map=memref<1x64x5x9xi1, #NHWC>, storage_element_table=memref<1x1x5x9xi32, #NHWC>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 0, 0], sizes = [1, 64, 5, 9]>>,
    // CHECK-SAME:                                      !VPUIP.SparseBuffer<data=memref<1x64x2x4xf16, #NHWC>, sparsity_map=memref<1x64x4x9xi1, #NHWC>, storage_element_table=memref<1x1x4x9xi32, #NHWC>, #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], offsets = [0, 0, 1, 0], sizes = [1, 64, 4, 9]>>
}
