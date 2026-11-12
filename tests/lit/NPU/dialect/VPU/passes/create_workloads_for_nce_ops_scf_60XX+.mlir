//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU6010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// NCE.MatMul 5D inside scf.forall — verifies 5D workload path produces a single
// workload with the full 5D shape and cluster_id = 0.
// CHECK-LABEL: @MatMulInsideForall
func.func @MatMulInsideForall(
        %arg0: tensor<1x1x16x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>,
        %weights: tensor<1x32x16x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>)
        -> tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> {

    %out = tensor.empty() : tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    %result = scf.forall (%gid) = (0) to (6) step (1)
        shared_outs(%o = %out) -> (tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>) {

        %mm = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [1, 32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } : tensor<1x1x16x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>, tensor<1x32x16x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
          -> tensor<1x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %mm into %o[%gid, 0, 0, 0, 0] [1, 1, 32, 4, 1] [1, 1, 1, 1, 1]
                : tensor<1x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
                into tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
        }
    }

    return %result : tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.MatMul
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [1, 1, 16, 4, 1] outOffsets [0, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1] pad [0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// DW small-spatial-compute errata (Errata E#18039459346, NPU60XX+ only): 1x1-kernel MaxPool
// tiled dynamically over H (step=24 over 40 -> tiles {24, 16}). The remainder tile (H=16, W=16)
// is genuinely <=16x16 and must trigger the errata's channel split into {32, 32}; the full tile
// (H=24, W=16) must NOT split.
//
// This requires WorkloadSizeConstraint::doesDWOperationNeedWorkloadSplit to use the pass's own
// per-branch resolved shape (inputShapeOverride) instead of reading operand(0)'s own IR type,
// which is dynamically shaped here (mlir::ShapedType::kDynamic for H) and would otherwise
// trivially satisfy the errata's `<=16` check for every branch, over-splitting the H=24 tile too.

!InBoundedH = tensor<1x64x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 24, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!OutBoundedH = tensor<1x64x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 24, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!AccTypeH = tensor<1x64x40x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @MaxPoolSmallSpatialErrataTileOverH
func.func @MaxPoolSmallSpatialErrataTileOverH(
        %arg0: tensor<1x64x40x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x40x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %acc = tensor.empty() : !AccTypeH
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c40 = arith.constant 40 : index

    %result = scf.for %iv = %c0 to %c40 step %c24 iter_args(%out = %acc) -> (!AccTypeH) {
        %tile_sz = affine.min affine_map<(d0) -> (24, 40 - d0)>(%iv)

        %slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, %tile_sz, 16] [1, 1, 1, 1]
            : tensor<1x64x40x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to !InBoundedH

        %pool = VPU.NCE.MaxPool(%slice) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1],
            kernel_size = [1, 1]
        } : !InBoundedH -> !OutBoundedH

        %ins = tensor.insert_slice %pool into %out[0, 0, %iv, 0] [1, 64, %tile_sz, 16] [1, 1, 1, 1]
            : !OutBoundedH into !AccTypeH

        scf.yield %ins : !AccTypeH
    }

    return %result : !AccTypeH

    // CHECK:       scf.for [[IV:%.+]] = {{.+}} to {{.+}} step {{.+}} iter_args({{.+}}) -> (tensor<1x64x40x16xf16
    // CHECK:         [[TILESZ:%.+]] = affine.min
    // CHECK:         tensor.extract_slice %arg0[0, 0, [[IV]], 0] [1, 64, [[TILESZ]], 16]
    // CHECK:         [[C2:%.+]] = arith.constant 2 : index
    // CHECK:         [[DIM:%.+]] = tensor.dim {{%.+}}, [[C2]]
    // CHECK:         [[C24:%.+]] = arith.constant 24 : index
    // CHECK:         [[CMP:%.+]] = arith.cmpi eq, [[DIM]], [[C24]]
    // CHECK:         VPU.NCE.MaxPool
    // CHECK-SAME:      -> tensor<1x64x?x16xf16
    // CHECK:           scf.if [[CMP]] {
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 64, 24, 16] outOffsets [0, 0, 0, 0] outSizes [1, 64, 24, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           } else {
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:             VPU.DPU.Workload inOffsets [0, 32, 0, 0] inSizes [1, 32, 16, 16] outOffsets [0, 32, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           }
    // CHECK:         tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, [[IV]], 0] [1, 64, [[TILESZ]], 16]
}
