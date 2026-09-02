//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvInsideForall
func.func @ConvInsideForall(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %conv into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.Convolution
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Convolution with SplitOverKernel multiclustering strategy inside scf.forall.
// SOK splits the output channels across clusters: each cluster processes a weight slice
// and produces a portion of the output channels.
// CHECK-LABEL: @ConvInsideForallSOK
func.func @ConvInsideForallSOK(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                               %weights: tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %offset_c = affine.apply affine_map<(d0) -> (d0 * 32)>(%tid)

        %w_slice = tensor.extract_slice %weights[%offset_c, 0, 0, 0] [32, 16, 1, 1] [1, 1, 1, 1]
            : tensor<64x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %conv into %o[0, %offset_c, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         tensor.extract_slice %arg1[{{%.+}}, 0, 0, 0] [32, 16, 1, 1]
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:         tensor.parallel_insert_slice {{%.+}} into {{%.+}}[0, {{%.+}}, 0, 0]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Convolution with SplitOverHeight multiclustering strategy inside scf.forall.
// SOH splits the activation height across clusters: each cluster processes a height slice.
// CHECK-LABEL: @ConvInsideForallSOH
func.func @ConvInsideForallSOH(%arg0: tensor<1x16x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                               %weights: tensor<64x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<1x64x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<1x64x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %offset_h = affine.apply affine_map<(d0) -> (d0 * 16)>(%tid)

        %a_slice = tensor.extract_slice %arg0[0, 0, %offset_h, 0] [1, 16, 16, 16] [1, 1, 1, 1]
            : tensor<1x16x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %conv = VPU.NCE.Convolution(%a_slice, %weights) rawFilterShape [64, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<64x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %conv into %o[0, 0, %offset_h, 0] [1, 64, 16, 16] [1, 1, 1, 1]
                : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<1x64x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<1x64x32x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         tensor.extract_slice %arg0[0, 0, {{%.+}}, 0] [1, 16, 16, 16]
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] pad [1, 1, 1, 1] <CUBOID_16x16>
    // CHECK:         tensor.parallel_insert_slice {{%.+}} into {{%.+}}[0, 0, {{%.+}}, 0]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Convolution with Clustering multiclustering strategy inside scf.forall.
// CHECK-LABEL: @ConvInsideForallClustering
func.func @ConvInsideForallClustering(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                      %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %conv into %o[%tid, 0, 0, 0] [1, 32, 16, 16] [1, 1, 1, 1]
                : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvInsideForallDynamicH
func.func @ConvInsideForallDynamicH(%arg0: tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
                                    %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<2x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1]
        } : tensor<1x16x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
            tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

        %cast = tensor.cast %conv : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 11, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
                                  to tensor<1x32x11x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %cast into %o[%tid, 0, 0, 0] [1, 32, 11, 16] [1, 1, 1, 1]
                : tensor<1x32x11x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<2x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<2x32x22x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // The pass should compute workloads using the bounded shape [1, 32, 11, 16]
    // and the output type should remain dynamic after the pass
    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      -> tensor<1x32x?x16xf16
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 11, 16] outOffsets [0, 0, 0, 0] outSizes [1, 32, 11, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvOutsideForallNotTouched
func.func @ConvOutsideForallNotTouched(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                       %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %conv : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // The SCF pass should NOT generate workloads for ops outside scf.forall
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NOT:     VPU.DPU.Workload
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolInsideForallDynamicC
func.func @MaxPoolInsideForallDynamicC(
        %arg0: tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<2x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %pool = VPU.NCE.MaxPool(%arg0) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
        } : tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

        %cast = tensor.cast %pool : tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
                                  to tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %cast into %o[%tid, 0, 0, 0] [1, 48, 16, 16] [1, 1, 1, 1]
                : tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<2x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<2x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // Dynamic channel split: MaxPool with dynamic channels (bound=48)
    // should produce an scf.for loop inside the workload region that splits
    // channels into valid segments from {64, 32, 16} dynamically.
    //
    // CHECK:       scf.forall
    // CHECK:         [[C1:%.+]] = arith.constant 1 : index
    // CHECK:         [[DIM:%.+]] = tensor.dim %arg0, [[C1]]
    // CHECK:         [[C16:%.+]] = arith.constant 16 : index
    // CHECK:         [[K:%.+]] = arith.divui [[DIM]], [[C16]] : index
    // CHECK:         VPU.NCE.MaxPool
    // CHECK-SAME:      -> tensor<1x?x16x16xf16
    // CHECK:           scf.for [[IV:%.+]] = {{.+}} to {{.+}} step {{.+}} iter_args([[CHOFF:%.+]] = {{.+}}) -> (index)
    // CHECK:             [[CMP1:%.+]] = arith.cmpi ult, [[IV]]
    // CHECK:             [[SEL1:%.+]] = arith.select [[CMP1]]
    // CHECK:             [[CMP2:%.+]] = arith.cmpi ult, [[IV]]
    // CHECK:             [[CHSZ:%.+]] = arith.select [[CMP2]]
    // CHECK:             VPU.DPU.Workload inOffsets [0, [[CHOFF]], 0, 0] inSizes [1, [[CHSZ]], 16, 16] outOffsets [0, [[CHOFF]], 0, 0] outSizes [1, [[CHSZ]], 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:             [[NEXT:%.+]] = arith.addi [[CHOFF]], [[CHSZ]] : index
    // CHECK:             scf.yield [[NEXT]] : index
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// scf.for step=256 over 640 channels → tiles {256, 256, 128}.
// 256 channels → depthwise split {64, 64, 64, 64} = 4 workloads
// 128 channels → depthwise split {64, 64} = 2 workloads

!InBounded256 = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!OutBounded256 = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!AccType256 = tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @MaxPoolPerTileWorkloads256vs128
func.func @MaxPoolPerTileWorkloads256vs128(
        %arg0: tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %acc = tensor.empty() : !AccType256
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c640 = arith.constant 640 : index

    // scf.for tiling over channels: step=256, total=640 → tiles of 256, 256, 128
    %result = scf.for %iv = %c0 to %c640 step %c256 iter_args(%out = %acc) -> (!AccType256) {
        %tile_sz = affine.min affine_map<(d0) -> (256, 640 - d0)>(%iv)

        %slice = tensor.extract_slice %arg0[0, %iv, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
            : tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to !InBounded256

        %pool = VPU.NCE.MaxPool(%slice) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [1, 1],
            kernel_size = [1, 1]
        } : !InBounded256 -> !OutBounded256

        %ins = tensor.insert_slice %pool into %out[0, %iv, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
            : !OutBounded256 into !AccType256

        scf.yield %ins : !AccType256
    }

    return %result : !AccType256

    // CHECK:       scf.for
    // CHECK:         [[C1:%.+]] = arith.constant 1 : index
    // CHECK:         [[DIM:%.+]] = tensor.dim {{%.+}}, [[C1]]
    // CHECK:         [[C256:%.+]] = arith.constant 256 : index
    // CHECK:         [[CMP:%.+]] = arith.cmpi eq, [[DIM]], [[C256]]
    // CHECK:         VPU.NCE.MaxPool
    // CHECK-SAME:      -> tensor<1x?x16x16xf16
    // CHECK:           scf.if [[CMP]] {
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK:           } else {
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, 64, 16, 16]
    // CHECK-NOT:         VPU.DPU.Workload
    // CHECK:           }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv 3x3 with strides=[2,2] and pad=[1,1,1,1] inside scf.forall with SOH.
// SOH with strides: each cluster processes a height slice of the activation
// and produces a portion of the output height.
// CHECK-LABEL: @ConvStridedSOH
func.func @ConvStridedSOH(%arg0: tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                           %weights: tensor<64x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %offset_h_in = affine.apply affine_map<(d0) -> (d0 * 16)>(%tid)
        %offset_h_out = affine.apply affine_map<(d0) -> (d0 * 8)>(%tid)

        %a_slice = tensor.extract_slice %arg0[0, 0, %offset_h_in, 0] [1, 16, 16, 32] [1, 1, 1, 1]
            : tensor<1x16x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to tensor<1x16x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %conv = VPU.NCE.Convolution(%a_slice, %weights) rawFilterShape [64, 16, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
            strides = [2, 2]
        } : tensor<1x16x16x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<64x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x64x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %conv into %o[0, 0, %offset_h_out, 0] [1, 64, 8, 16] [1, 1, 1, 1]
                : tensor<1x64x8x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         tensor.extract_slice %arg0[0, 0, {{%.+}}, 0] [1, 16, 16, 32]
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:      strides = [2, 2]
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 32] outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] pad [1, 0, 1, 0] <CUBOID_16x16>
    // CHECK:         tensor.parallel_insert_slice {{%.+}} into {{%.+}}[0, 0, {{%.+}}, 0]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// NCE.Permute inside scf.forall — verifies the Permute "correction" logic
// (outSizes uses original channels, not expandedChannels).
// CHECK-LABEL: @PermuteInsideForall
func.func @PermuteInsideForall(%arg0: tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>)
        -> tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %perm = VPU.NCE.Permute(%arg0) {
            dstElemType = f16,
            dstOrder = #NHWC,
            expandedChannels = 4 : i64,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>
        } : tensor<1x3x224x224xf16, {mem_space = @CMX_NN}> -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %perm into %o[%tid, 0, 0, 0] [1, 4, 224, 224] [1, 1, 1, 1]
                : tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
                into tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
        }
    }

    return %result : tensor<2x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.Permute
    // CHECK-SAME:      expandedChannels = 4
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 3, 224, 224] outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// NCE.MatMul 5D inside scf.forall — verifies 5D workload path produces a single
// workload with the full 5D shape and cluster_id = 0.
// CHECK-LABEL: @MatMulInsideForall
func.func @MatMulInsideForall(
        %arg0: tensor<1x1x16x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>,
        %weights: tensor<1x32x16x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>)
        -> tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> {

    %wt = const.Declare tensor<1x32x1x1x4xsi32, {mem_space = @CMX_NN, order = #GNCHW}> = dense<1> : tensor<1x32x1x1x4xsi32, {mem_space = @CMX_NN}>

    %out = tensor.empty() : tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    %result = scf.forall (%gid) = (0) to (6) step (1)
        shared_outs(%o = %out) -> (tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>) {

        %mm = VPU.NCE.MatMul(%arg0, %weights, %wt : tensor<1x32x1x1x4xsi32, {mem_space = @CMX_NN, order = #GNCHW}>) rawFilterShape [1, 32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>,
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
