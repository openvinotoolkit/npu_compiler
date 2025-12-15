//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-vpuops-to-upstream %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: ConvertSliceToExtractSlice
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>)
func.func @ConvertSliceToExtractSlice(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>)
    -> tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}> {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 12, -9223372036854775808, -9223372036854775808] 
        : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}> 
        to tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

    return %0 : tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK-NOT: VPU.Slice

    // CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: [[SLICE:%.+]] = tensor.extract_slice
    // CHECK-SAME:      [0, 0, 0, 0] [1, 12, [[DIM_H]], [[DIM_W]]] [1, 1, 1, 1]
    // return [[SLICE]] : tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

!OutputSparseTensor = !VPU.SparseTensor<
    data=tensor<1x4x8x16xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x1x1x512xi1, {order = #NHWC, mem_space = @CMX_NN}>,
    is_weights
>

// CHECK-LABEL: NotConvertSparseSliceToExtractSlice
func.func @NotConvertSparseSliceToExtractSlice(%arg0: tensor<1x32x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>, %arg1: tensor<1x32x16x16xi1, {order = #NHWC, mem_space = @CMX_NN}>) -> !OutputSparseTensor {
    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1) {is_weights} -> !InputSparseTensor
    %output = VPU.Slice %input_sparse [0, 0, 0, 0] [1, 4, 8, 16]: !InputSparseTensor to !OutputSparseTensor
    return %output: !OutputSparseTensor

    // CHECK-NOT: tensor.extract_slice
}
