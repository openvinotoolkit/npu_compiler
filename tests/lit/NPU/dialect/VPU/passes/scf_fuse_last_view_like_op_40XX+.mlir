//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --scf-fuse-last-viewlike-op %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


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

// CHECK: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 512)>

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
      %9 = VPU.NCE.Convolution(%padded, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
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

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (128, -d0 + s0)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (512, -d0 + s0)>
// CHECK-DAG: #[[$CNWH:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

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

// -----
// CHECK-LABEL: @FusingLastSliceOperationIntoScfForLoop
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @FusingLastSliceOperationIntoScfForLoop(%arg0: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> {
  %c30 = arith.constant 30 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  %0 = tensor.empty(%dim) : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  %1 = scf.for %arg2 = %c0 to %dim step %c30 iter_args(%arg3 = %0) -> (tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 30)>(%arg2)[%dim]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 16, %3, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 16, %3, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %6 = VPU.NCE.Eltwise(%extracted_slice, %extracted_slice_0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %inserted_slice = tensor.insert_slice %6 into %arg3[0, 0, %arg2, 0] [1, 16, %3, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 30, 1280]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    scf.yield %inserted_slice : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK: [[OUT:%.+]] = tensor.empty({{.+}}) : tensor<1x3x?x1280xf16,
  // CHECK: [[LOOP:%.+]] = scf.for [[IDX:%.+]] = {{.+}} to [[DIM:%.+]] step {{.+}} iter_args([[OUT0:%.+]] = [[OUT]]) -> (tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>)
  // CHECK:   [[SLICE_IN_LOOP:%.+]] = VPU.Slice
  // CHECK:   [[INSERT:%.+]] = tensor.insert_slice [[SLICE_IN_LOOP]] into [[OUT0]][0, 0, [[IDX]], 0] [1, 3, {{.+}}, 1280] [1, 1, 1, 1] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 30, 1280]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK:   scf.yield [[INSERT]] : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK: }
  // CHECK-NOT: VPU.Slice
  // CHECK: return [[LOOP]]

  %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 3, -9223372036854775808, 1280] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
  return %2 : tensor<1x3x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0)[s0] -> (-d0 + s0, 90)>

!qElemType = !quant.uniform<u8:f16, 0.0067687005388970467:49>

!qInDynamic = tensor<1x3x?x1920x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!qInSliced = tensor<1x3x?x1920x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 1920]> : tensor<4xsi64>, order = #NHWC}>
!castOutDynamic = tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @FuseQuantizeCastToScf
func.func @FuseQuantizeCastToScf(%arg0: !qInDynamic) -> !castOutDynamic {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c90 = arith.constant 90 : index
    %dim = tensor.dim %arg0, %c2 : !qInDynamic
    %0 = tensor.empty(%dim) : !qInDynamic

    %1 = scf.for %arg1 = %c0 to %dim step %c90 iter_args(%arg2 = %0) -> (!qInDynamic) {
      %3 = affine.min #map(%arg1)[%dim]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0] [1, 3, %3, 1920] [1, 1, 1, 1]
        : !qInDynamic to !qInSliced
      %inserted_slice = tensor.insert_slice %extracted_slice into %arg2[0, 0, %arg1, 0] [1, 3, %3, 1920] [1, 1, 1, 1]
        : !qInSliced into !qInDynamic
      scf.yield %inserted_slice : !qInDynamic
    }
    %2 = VPU.QuantizeCast(%1) {dstElemType = ui8} : !qInDynamic -> !castOutDynamic
    return %2 : !castOutDynamic

    // CHECK-DAG: [[ZERO_CST:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[TWO_CST:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[STEP:%.+]] = arith.constant 90 : index

    // CHECK: [[HEIGHT_DIM:%.+]] = tensor.dim %arg0, [[TWO_CST]]
    // CHECK: [[OUT_BUFF:%.+]] = tensor.empty([[HEIGHT_DIM]])
    // CHECK-SAME: : tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[SCF:%.+]] = scf.for [[IDX:%.+]] = [[ZERO_CST]] to [[HEIGHT_DIM]] step [[STEP]] iter_args([[OUT:%.+]] = [[OUT_BUFF]])
    // CHECK-SAME:  -> (tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)

    // CHECK: [[EXTRACT:%.+]] = tensor.extract_slice
    // CHECK: [[QC:%.+]] = VPU.QuantizeCast([[EXTRACT]]) {dstElemType = ui8}
    // CHECK-SAME: -> tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: [[INSERT:%.+]] = tensor.insert_slice [[QC]] into [[OUT]][0, 0, [[IDX]], 0] [1, 3, {{.+}}, 1920] [1, 1, 1, 1]
    // CHECK-SAME: : tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME: into tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: scf.yield [[INSERT]] : tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK: return [[SCF]] : tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#mapH = affine_map<(d0)[s0] -> (-d0 + s0, 90)>
#mapW = affine_map<(d0)[s0] -> (-d0 + s0, 480)>

!qElemType = !quant.uniform<u8:f16, 0.0067687005388970467:49>

!quantInputFull = tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!quantInputTile = tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 480]> : tensor<4xsi64>, order = #NHWC}>
!permutedFull = tensor<1x?x?x3x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
!permutedTile = tensor<1x?x?x3x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>
!ui8OutputFull = tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
!ui8OutputTile = tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>

// Verifies that a view-like op chain (PermuteCast -> QuantizeCast) is fused

// CHECK: #[[$MAPH:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 90)>
// CHECK: #[[$MAPW:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 480)>

// CHECK-LABEL:   @FusePermuteCastAndQuantizeCastChainToScf
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
func.func @FusePermuteCastAndQuantizeCastChainToScf(%arg0: !quantInputFull) -> !ui8OutputFull {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c90 = arith.constant 90 : index
    %c480 = arith.constant 480 : index
    %dimH = tensor.dim %arg0, %c2 : !quantInputFull
    %dimW = tensor.dim %arg0, %c3 : !quantInputFull
    %empty = tensor.empty(%dimH, %dimW) : !quantInputFull

    %loop_h = scf.for %h = %c0 to %dimH step %c90 iter_args(%accH = %empty) -> (!quantInputFull) {
      %loop_w = scf.for %w = %c0 to %dimW step %c480 iter_args(%accW = %accH) -> (!quantInputFull) {
        %tileH = affine.min #mapH(%h)[%dimH]
        %tileW = affine.min #mapW(%w)[%dimW]
        %extracted = tensor.extract_slice %arg0[0, 0, %h, %w] [1, 3, %tileH, %tileW] [1, 1, 1, 1]
          : !quantInputFull to !quantInputTile
        %inserted = tensor.insert_slice %extracted into %accW[0, 0, %h, %w] [1, 3, %tileH, %tileW] [1, 1, 1, 1]
          : !quantInputTile into !quantInputFull
        scf.yield %inserted : !quantInputFull
      }
      scf.yield %loop_w : !quantInputFull
    }
    %permute_cast = VPU.PermuteCast(%loop_h) {dst_order = #NCHW, mem_perm = #NCHW} : !quantInputFull -> !permutedFull
    %quantize_cast = VPU.QuantizeCast(%permute_cast) {dstElemType = ui8} : !permutedFull -> !ui8OutputFull
    return %quantize_cast : !ui8OutputFull

    // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK-DAG: [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK-DAG: [[CST_90:%.+]] = arith.constant 90 : index
    // CHECK-DAG: [[CST_480:%.+]] = arith.constant 480 : index

    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CST_2]]
    // CHECK-SAME: tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[CST_3]]
    // CHECK-SAME: tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[OUTPUT_BUFF:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]])
    // CHECK-SAME: tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK: [[LOOP_H:%.+]] = scf.for [[IV_H:%.+]] = [[CST_0]] to [[DIM_H]] step [[CST_90]] iter_args([[ACC_H:%.+]] = [[OUTPUT_BUFF]])
    // CHECK-SAME: -> (tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>)
    // CHECK:   [[LOOP_W:%.+]] = scf.for [[IV_W:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_480]] iter_args([[ACC_W:%.+]] = [[ACC_H]])
    // CHECK-SAME: -> (tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>)

    // CHECK:       [[TILE_H:%.+]] = affine.min #[[$MAPH]]([[IV_H]]){{\[}}[[DIM_H]]]
    // CHECK:       [[TILE_W:%.+]] = affine.min #[[$MAPW]]([[IV_W]]){{\[}}[[DIM_W]]]
    // CHECK:       [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[IV_H]], [[IV_W]]] [1, 3, [[TILE_H]], [[TILE_W]]] [1, 1, 1, 1]
    // CHECK-SAME:    tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:    tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[EXTRACTED_SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:    tensor<1x3x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:    tensor<1x?x?x3x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[QUANTIZE_CAST:%.+]] = VPU.QuantizeCast([[PERMUTE_CAST]]) {dstElemType = ui8}
    // CHECK-SAME:    tensor<1x?x?x3x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:    tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[QUANTIZE_CAST]] into [[ACC_W]][0, [[IV_H]], [[IV_W]], 0] [1, [[TILE_H]], [[TILE_W]], 3] [1, 1, 1, 1]
    // CHECK-SAME:    tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK-SAME:    tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       scf.yield [[INSERTED_SLICE]] : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     scf.yield [[LOOP_W]] : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK: return [[LOOP_H]] : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map = affine_map<(d0)[s0] -> (-d0 + s0, 90)>

!qElemType = !quant.uniform<u8:f16, 0.0067687005388970467:49>

!argFull = tensor<1x?x1920x3xui8, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 3]> : tensor<4xsi64>, order = #NCHW}>
!permutedFull = tensor<1x3x?x1920xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!quantInputFull = tensor<1x3x?x1920x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
!quantInputTile = tensor<1x3x?x1920x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 90, 1920]> : tensor<4xsi64>, order = #NHWC}>

// View-like ops whose operand is a block argument (PermuteCast here) must be skipped by the pass
// without crashing on a null defining op.

// CHECK-LABEL: @SkipViewLikeOnBlockArgument
func.func @SkipViewLikeOnBlockArgument(%arg0: !argFull) -> !quantInputFull {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c90 = arith.constant 90 : index

    %permuted = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : !argFull -> !permutedFull
    %quantized = VPU.QuantizeCast(%permuted) {dstElemType = !qElemType} : !permutedFull -> !quantInputFull

    %dim = tensor.dim %quantized, %c2 : !quantInputFull
    %empty = tensor.empty(%dim) : !quantInputFull
    %loop = scf.for %iv = %c0 to %dim step %c90 iter_args(%acc = %empty) -> (!quantInputFull) {
        %tile = affine.min #map(%iv)[%dim]
        %slice = tensor.extract_slice %quantized[0, 0, %iv, 0] [1, 3, %tile, 1920] [1, 1, 1, 1]
            : !quantInputFull to !quantInputTile
        %inserted = tensor.insert_slice %slice into %acc[0, 0, %iv, 0] [1, 3, %tile, 1920] [1, 1, 1, 1]
            : !quantInputTile into !quantInputFull
        scf.yield %inserted : !quantInputFull
    }
    return %loop : !quantInputFull

    // The leading PermuteCast/QuantizeCast on %arg0 are preserved unchanged.
    // CHECK: [[PERMUTED:%.+]] = VPU.PermuteCast(%arg0)
    // CHECK: [[QUANTIZED:%.+]] = VPU.QuantizeCast([[PERMUTED]])
    // CHECK: [[LOOP:%.+]] = scf.for
    // CHECK: return [[LOOP]]
}
