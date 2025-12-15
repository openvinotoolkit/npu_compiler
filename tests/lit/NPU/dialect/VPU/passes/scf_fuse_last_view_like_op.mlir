//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --scf-fuse-last-viewlike-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map0 = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map5 = affine_map<(d0) -> (d0 + 1)>

!convInDynamic = tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
!convOutDynamic = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
!convInSlicedDynamic = tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
!convInSlicedPaddedDynamic = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 514, 66]> : tensor<4xsi64>, order = #NHWC}>
!convOutSlicedDynamic = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
!permuteOutDynamic = tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>

// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 512)>

// CHECK-LABEL:   @FusePermuteCastToScf
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 64]> : tensor<4xsi64>, order = #NHWC}>
func.func @FusePermuteCastToScf(%arg0: !convInDynamic) -> !permuteOutDynamic {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %pad = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %dim = tensor.dim %arg0, %c2 : !convInDynamic
    %0 = tensor.empty(%dim) : !convOutDynamic

    %1 = scf.for %arg1 = %c0 to %dim step %c512 iter_args(%arg2 = %0) -> (!convOutDynamic) {
      %2 = affine.min #map0(%arg1)[%dim]
      %3 = affine.max #map1(%arg1)
      %4 = affine.max #map2(%arg1)
      %5 = affine.min #map3()[%4]
      %6 = affine.max #map4(%2, %3)
      %7 = affine.min #map3()[%6]
      %8 = affine.apply #map5(%2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, 0] [1, 32, %8, 64] [1, 1, 1, 1]
        : !convInDynamic to !convInSlicedDynamic
      %padded = tensor.pad %extracted_slice low[0, 0, %5, 1] high[0, 0, %7, 1] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %pad : f16
      } : !convInSlicedDynamic to !convInSlicedPaddedDynamic
      %9 = VPU.NCE.Convolution(%padded, %cst) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]
      } : !convInSlicedPaddedDynamic, tensor<256x32x3x3xf16, {order = #NHWC}>
        -> !convOutSlicedDynamic
      %inserted_slice = tensor.insert_slice %9 into %arg2[0, 0, %arg1, 0] [1, 256, %2, 64] [1, 1, 1, 1]
        : !convOutSlicedDynamic into !convOutDynamic
      scf.yield %inserted_slice : !convOutDynamic
    }
    %2 = VPU.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NCHW} : !convOutDynamic -> !permuteOutDynamic

    return %2 : !permuteOutDynamic

    // CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[TWO_CST:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[STEP:%.+]] = arith.constant 512 : index

    // CHECK: [[HEIGHT_DIM:%.+]] = tensor.dim [[INPUT]], [[TWO_CST]]
    // CHECK: [[OUT_BUFF:%.+]] = tensor.empty([[HEIGHT_DIM]])
    // CHECK-SAME: : tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[ZERO_CST]] to [[HEIGHT_DIM]] step [[STEP]] iter_args([[OUT:%.+]] = [[OUT_BUFF]])
    // CHECK-SAME:  -> (tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>)

    // CHECK: [[OUT_SIZE:%.+]] = affine.min #[[$MAP]]([[IDX]])[[[HEIGHT_DIM]]]

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK: [[PERMUTE:%.+]] = VPU.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME: : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 512, 64]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME: -> tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 64, 256]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[OUT]][0, [[IDX]], 0, 0] [1, [[OUT_SIZE]], 64, 256] [1, 1, 1, 1]
    // CHECK-SAME: : tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 512, 64, 256]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME: into tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: scf.yield [[INSERT]] : tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: return [[SCF]] : tensor<1x?x64x256xf16, {bounds = #const.OpaqueI64Elements<[1, 1024, 64, 256]> : tensor<4xsi64>, order = #NCHW}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#CNWH = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

!convertInDynamic = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
!convertOutDynamic = tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
!convertInSlicedDynamic = tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NHWC}>
!convertOutSlicedDynamic = tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NHWC}>
!permuteOutDynamic = tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>

#map = affine_map<(d0)[s0] -> (128, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (512, -d0 + s0)>

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (128, -d0 + s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (512, -d0 + s0)>
// CHECK-DAG: #[[$CNWH:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

// CHECK-LABEL: @FusePermuteCastToNestedScf
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NHWC}>
func.func @FusePermuteCastToNestedScf(%arg0: !convertInDynamic) -> !permuteOutDynamic {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index

    %dim = tensor.dim %arg0, %c2 : !convertInDynamic
    %dim_0 = tensor.dim %arg0, %c3 : !convertInDynamic
    %0 = tensor.empty(%dim, %dim_0) : !convertOutDynamic

    %1 = scf.for %arg1 = %c0 to %dim step %c128 iter_args(%arg2 = %0) -> (!convertOutDynamic) {
      %2 = scf.for %arg3 = %c0 to %dim_0 step %c512 iter_args(%arg4 = %arg2) -> (!convertOutDynamic) {
        %3 = affine.min #map(%arg1)[%dim]
        %4 = affine.min #map1(%arg3)[%dim_0]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, %arg3] [1, 1, %3, %4] [1, 1, 1, 1]
            : !convertInDynamic to !convertInSlicedDynamic
        %5 = VPU.Convert(%extracted_slice) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
            : !convertInSlicedDynamic -> !convertOutSlicedDynamic
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, %arg3] [1, 1, %3, %4] [1, 1, 1, 1]
            : !convertOutSlicedDynamic into !convertOutDynamic
        scf.yield %inserted_slice : !convertOutDynamic
      }
      scf.yield %2 : !convertOutDynamic
    }
    %2 = VPU.PermuteCast(%1) {dst_order = #NWCH, mem_perm = #CNWH} : !convertOutDynamic -> !permuteOutDynamic
    return %2 : !permuteOutDynamic

    // CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[TWO_CST:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[THREE_CST:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[STEP_H:%.+]] = arith.constant 128 : index
    // CHECK-DAG: [[STEP_W:%.+]] = arith.constant 512 : index

    // CHECK: [[HEIGHT_DIM:%.+]] = tensor.dim [[INPUT]], [[TWO_CST]]
    // CHECK: [[WIDTH_DIM:%.+]] = tensor.dim [[INPUT]], [[THREE_CST]]
    // CHECK: [[OUT_BUFF:%.+]] = tensor.empty([[HEIGHT_DIM]], [[WIDTH_DIM]])
    // CHECK-SAME: : tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>

    // CHECK: [[SCF_H:%.+]] = scf.for [[IDX_H:%.+]] = [[ZERO_CST]] to [[HEIGHT_DIM]] step [[STEP_H]] iter_args([[OUT:%.+]] = [[OUT_BUFF]])
    // CHECK-SAME:  -> (tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>)

    // CHECK: [[SCF_W:%.+]] = scf.for [[IDX_W:%.+]] = [[ZERO_CST]] to [[WIDTH_DIM]] step [[STEP_W]] iter_args([[OUT0:%.+]] = [[OUT]])
    // CHECK-SAME:  -> (tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>)

    // CHECK: [[OUT_SIZE_H:%.+]] = affine.min #[[$MAP]]([[IDX_H]])[[[HEIGHT_DIM]]]
    // CHECK: [[OUT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[IDX_W]])[[[WIDTH_DIM]]]

    // CHECK: [[CONVERT:%.+]] = VPU.Convert
    // CHECK: [[PERMUTE:%.+]] = VPU.PermuteCast([[CONVERT]]) {dst_order = #NWCH, mem_perm = #[[$CNWH]]}
    // CHECK-SAME: : tensor<1x1x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME: -> tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[128, 1, 512, 1]> : tensor<4xsi64>, order = #NWCH}>

    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[OUT0]][[[IDX_H]], 0, [[IDX_W]], 0] [[[OUT_SIZE_H]], 1, [[OUT_SIZE_W]], 1] [1, 1, 1, 1]
    // CHECK-SAME: : tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[128, 1, 512, 1]> : tensor<4xsi64>, order = #NWCH}>
    // CHECK-SAME: into tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>

    // CHECK: scf.yield [[INSERT]] : tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>

    // CHECK: return [[SCF_H]] : tensor<?x1x?x1xf32, {bounds = #const.OpaqueI64Elements<[1024, 1, 2048, 1]> : tensor<4xsi64>, order = #NWCH}>

}
