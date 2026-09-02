//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvInsideForallStatic
func.func @ConvInsideForallStatic(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                  %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %out = tensor.empty() : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.forall (%tid) = (0) to (2) step (1)
        shared_outs(%o = %out) -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 16] outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] pad [1, 1, 1, 1] <CUBOID_{{.+}}>
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
    // CHECK:             arith.cmpi ult, [[IV]], {{.+}} : index
    // CHECK:             arith.select {{.+}} : index
    // CHECK:             VPU.DPU.Workload inOffsets [0, [[CHOFF]], 0, 0] inSizes [1, [[CHSZ:%.+]], 16, 16] outOffsets [0, [[CHOFF]], 0, 0] outSizes [1, [[CHSZ]], 16, 16] pad [0, 0, 0, 0] <CUBOID_{{.+}}>
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

    %result = scf.for %iv = %c0 to %c640 step %c256 iter_args(%out = %acc) -> (!AccType256) {
        %tile_sz = affine.min affine_map<(d0) -> (256, 640 - d0)>(%iv)

        %slice = tensor.extract_slice %arg0[0, %iv, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
            : tensor<1x640x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to !InBounded256

        %pool = VPU.NCE.MaxPool(%slice) {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
    // CHECK:           } else {
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
    // CHECK:             VPU.DPU.Workload{{.*}}outSizes [1, {{32|64}}, 16, 16]
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
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
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 16, 32] outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] pad [1, 0, 1, 0] <CUBOID_{{.+}}>
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
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

// Nested scf.for (height tiling, step=24 over 48) + scf.forall (SOH multiclustering
// across 6 clusters with dynamic bounds). Conv 3x3 stride=1 with tensor.pad providing
// spatial padding. Models the post-vertical-fusion IR for SOH convolutions.
// CHECK-LABEL: @ConvSOHTileOverH
func.func @ConvSOHTileOverH(%arg0: tensor<1x32x48x48xf16, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
  %c23 = arith.constant 23 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  %c0 = arith.constant 0 : index
  %cst_w = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x32x48x48xf16, {order = #NHWC}>

  %1 = scf.for %arg1 = %c0 to %c48 step %c24 iter_args(%arg2 = %0) -> (tensor<1x32x48x48xf16, {order = #NHWC}>) {
    %start_h = affine.max affine_map<(d0) -> (0, d0 - 1)>(%arg1)
    %raw_top = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%arg1)
    %top_pad = affine.min affine_map<()[s0] -> (1, s0)>()[%raw_top]
    %raw_bot = affine.max affine_map<(d0) -> (0, d0 - 22)>(%start_h)
    %bot_pad = affine.min affine_map<()[s0] -> (1, s0)>()[%raw_bot]

    %extracted_slice = tensor.extract_slice %arg0[0, 0, %start_h, 0] [1, 32, 25, 48] [1, 1, 1, 1]
      : tensor<1x32x48x48xf16, {order = #NHWC}> to tensor<1x32x25x48xf16, {order = #NHWC}>

    %sum_pad = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%top_pad, %bot_pad)
    %out_h = arith.addi %sum_pad, %c23 : index
    %forall_out = tensor.empty(%out_h) : tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
    %step_inner = affine.apply affine_map<(d0) -> (d0 ceildiv 6)>(%out_h)

    %forall_result = scf.forall (%arg3) = (0) to (%out_h) step (%step_inner)
        shared_outs(%arg4 = %forall_out) -> (tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>) {

      %inner_tile = affine.min affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>(%arg3, %out_h)[%out_h]
      %inner_start = affine.max affine_map<(d0, d1) -> (0, d0 - d1)>(%arg3, %top_pad)
      %inner_top = affine.max affine_map<(d0, d1) -> (0, d0 - d1)>(%top_pad, %inner_start)
      %inner_bot = affine.max affine_map<(d0, d1, d2) -> (0, d0 - d1 + d2 - 23)>(%arg3, %top_pad, %inner_tile)
      %inner_extract = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>(%inner_top, %inner_bot, %inner_tile)

      %inner_slice = tensor.extract_slice %extracted_slice[0, 0, %inner_start, 0] [1, 32, %inner_extract, 48] [1, 1, 1, 1]
        : tensor<1x32x25x48xf16, {order = #NHWC}> to tensor<1x32x?x48xf16, {order = #NHWC}>

      %copy_act = VPU.Copy(%inner_slice) {out_mem_space = @CMX_NN}
        : tensor<1x32x?x48xf16, {order = #NHWC}> -> tensor<1x32x?x48xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %copy_w = VPU.Copy(%cst_w) {out_mem_space = @CMX_NN}
        : tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

      %padded = tensor.pad %copy_act low[0, 0, %inner_top, 1] high[0, 0, %inner_bot, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %cst : f16
      } : tensor<1x32x?x48xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to tensor<1x32x?x50xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 6, 50]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %copy_w) rawFilterShape [32, 32, 3, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
      } : tensor<1x32x?x50xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 6, 50]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
          tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 4, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %conv into %arg4[0, 0, %arg3, 0] [1, 32, %inner_tile, 48] [1, 1, 1, 1]
          : tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 4, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
          into tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      }
    }

    %cast = tensor.cast %forall_result
      : tensor<1x32x?x48xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 48]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      to tensor<1x32x24x48xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %copy_out = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x32x24x48xf16, {mem_space = @CMX_NN, order = #NHWC}>
      -> tensor<1x32x24x48xf16, {order = #NHWC}>

    %ins = tensor.insert_slice %copy_out into %arg2[0, 0, %arg1, 0] [1, 32, 24, 48] [1, 1, 1, 1]
      : tensor<1x32x24x48xf16, {order = #NHWC}> into tensor<1x32x48x48xf16, {order = #NHWC}>

    scf.yield %ins : tensor<1x32x48x48xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x32x48x48xf16, {order = #NHWC}>

    // The pass falls back to bounded shapes for the dynamic forall, producing
    // a single workload from the bounded input [1,32,6,50] and output [1,32,4,48].
    // CHECK:       scf.for
    // CHECK:         tensor.extract_slice
    // CHECK:         scf.forall
    // CHECK:           tensor.extract_slice
    // CHECK:           VPU.Copy
    // CHECK:           VPU.Copy
    // CHECK:           tensor.pad
    // CHECK:           VPU.NCE.Convolution
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 6, 49] outOffsets [0, 0, 0, 0] outSizes [1, 32, 4, 48] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           tensor.parallel_insert_slice
    // CHECK:         tensor.cast
    // CHECK:         VPU.Copy
    // CHECK:         tensor.insert_slice
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Nested scf.for (height tiling, step=24 over 48) + scf.forall (SOH multiclustering
// with dynamic bounds). Conv 3x3 stride=2 with tensor.pad providing spatial padding.
// Stride-2 halves the output spatial dimensions. NCE pad=[0,0,0,0].
// CHECK-LABEL: @ConvStride2SOHTileOverH
func.func @ConvStride2SOHTileOverH(%arg0: tensor<1x16x48x32xf16, {order = #NHWC}>) -> tensor<1x32x24x16xf16, {order = #NHWC}> {
  %cst = arith.constant 0.000000e+00 : f16
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  %c0 = arith.constant 0 : index
  %cst_w = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x32x24x16xf16, {order = #NHWC}>

  %1 = scf.for %arg1 = %c0 to %c48 step %c24 iter_args(%arg2 = %0) -> (tensor<1x32x24x16xf16, {order = #NHWC}>) {
    %start_h = affine.max affine_map<(d0) -> (0, d0 - 1)>(%arg1)
    %raw_top = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%arg1)
    %top_pad = affine.min affine_map<()[s0] -> (1, s0)>()[%raw_top]
    %raw_bot = affine.max affine_map<(d0) -> (0, d0 - 22)>(%start_h)
    %bot_pad = affine.min affine_map<()[s0] -> (1, s0)>()[%raw_bot]

    %extracted_slice = tensor.extract_slice %arg0[0, 0, %start_h, 0] [1, 16, 25, 32] [1, 1, 1, 1]
      : tensor<1x16x48x32xf16, {order = #NHWC}> to tensor<1x16x25x32xf16, {order = #NHWC}>

    %sum_pad = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%top_pad, %bot_pad)
    %out_h = affine.apply affine_map<(d0) -> ((d0 + 22) floordiv 2 + 1)>(%sum_pad)
    %forall_out = tensor.empty(%out_h) : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
    %step_inner = affine.apply affine_map<(d0) -> (d0 ceildiv 6)>(%out_h)

    %forall_result = scf.forall (%arg3) = (0) to (%out_h) step (%step_inner)
        shared_outs(%arg4 = %forall_out) -> (tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>) {

      %inner_tile = affine.min affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>(%arg3, %out_h)[%out_h]
      %inner_start = affine.max affine_map<(d0, d1) -> (0, 2 * d0 - d1)>(%arg3, %top_pad)
      %inner_top = affine.max affine_map<(d0, d1) -> (0, d0 - 2 * d1)>(%top_pad, %arg3)
      %inner_bot = affine.max affine_map<(d0, d1, d2) -> (0, 2 * d0 + 2 * d2 - d1 - 24)>(%arg3, %top_pad, %inner_tile)
      %inner_extract = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + 2 * d2 + 1)>(%inner_top, %inner_bot, %inner_tile)

      %inner_slice = tensor.extract_slice %extracted_slice[0, 0, %inner_start, 0] [1, 16, %inner_extract, 32] [1, 1, 1, 1]
        : tensor<1x16x25x32xf16, {order = #NHWC}> to tensor<1x16x?x32xf16, {order = #NHWC}>

      %copy_act = VPU.Copy(%inner_slice) {out_mem_space = @CMX_NN}
        : tensor<1x16x?x32xf16, {order = #NHWC}> -> tensor<1x16x?x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %copy_w = VPU.Copy(%cst_w) {out_mem_space = @CMX_NN}
        : tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

      %padded = tensor.pad %copy_act low[0, 0, %inner_top, 1] high[0, 0, %inner_bot, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %cst : f16
      } : tensor<1x16x?x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
        to tensor<1x16x?x34xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 34]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%padded, %copy_w) rawFilterShape [32, 16, 3, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [2, 2]
      } : tensor<1x16x?x34xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 5, 34]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
          tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 2, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %conv into %arg4[0, 0, %arg3, 0] [1, 32, %inner_tile, 16] [1, 1, 1, 1]
          : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 2, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
          into tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      }
    }

    %cast = tensor.cast %forall_result
      : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      to tensor<1x32x12x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %copy_out = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x32x12x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
      -> tensor<1x32x12x16xf16, {order = #NHWC}>

    %out_off = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%arg1)
    %ins = tensor.insert_slice %copy_out into %arg2[0, 0, %out_off, 0] [1, 32, 12, 16] [1, 1, 1, 1]
      : tensor<1x32x12x16xf16, {order = #NHWC}> into tensor<1x32x24x16xf16, {order = #NHWC}>

    scf.yield %ins : tensor<1x32x24x16xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x32x24x16xf16, {order = #NHWC}>

    // Stride-2 version: bounded padded input [1,16,5,34], output [1,32,2,16].
    // CHECK:       scf.for
    // CHECK:         tensor.extract_slice
    // CHECK:         scf.forall
    // CHECK:           tensor.extract_slice
    // CHECK:           VPU.Copy
    // CHECK:           VPU.Copy
    // CHECK:           tensor.pad
    // CHECK:           VPU.NCE.Convolution
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:        strides = [2, 2]
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 5, 32] outOffsets [0, 0, 0, 0] outSizes [1, 32, 2, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           tensor.parallel_insert_slice
    // CHECK:         tensor.cast
    // CHECK:         VPU.Copy
    // CHECK:         tensor.insert_slice
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv 3x3 inside scf.for with tensor.pad providing spatial padding externally.
// The NCE op has pad=[0,0,0,0] because padding is applied by tensor.pad.
// This models the SCF tiling pattern where border-aware padding is computed
// dynamically per tile (e.g., top pad for first tile, bottom pad for last tile).
// The pass correctly accounts for tensor.pad when computing the NCE input shape.
// CHECK-LABEL: @ConvWithExternalPadScfFor
func.func @ConvWithExternalPadScfFor(
        %arg0: tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights: tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %acc = tensor.empty() : tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c50 = arith.constant 50 : index
    %c100 = arith.constant 100 : index

    %result = scf.for %iv = %c0 to %c100 step %c50 iter_args(%out = %acc) -> (tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
        %start_h = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%iv)

        %slice = tensor.extract_slice %arg0[0, 0, %start_h, 0] [1, 32, 51, 16] [1, 1, 1, 1]
            : tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to tensor<1x32x51x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %top_pad = affine.max affine_map<(d0) -> (1 - d0, 0)>(%iv)
        %bot_pad = affine.max affine_map<(d0) -> (d0 + 50 - 99, 0)>(%iv)

        %cst_pad = arith.constant 0.0 : f16
        %padded = tensor.pad %slice low[0, 0, %top_pad, 1] high[0, 0, %bot_pad, 1] {
        ^bb0(%a: index, %b: index, %c: index, %d: index):
            tensor.yield %cst_pad : f16
        } : tensor<1x32x51x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
          to tensor<1x32x?x18xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 52, 18]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

        %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [32, 32, 3, 3] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } : tensor<1x32x?x18xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 52, 18]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
            tensor<32x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

        %cast = tensor.cast %conv : tensor<1x32x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
                                  to tensor<1x32x50x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %ins = tensor.insert_slice %cast into %out[0, 0, %iv, 0] [1, 32, 50, 16] [1, 1, 1, 1]
            : tensor<1x32x50x16xf16, {mem_space = @CMX_NN, order = #NHWC}> into tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.yield %ins : tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    }

    return %result : tensor<1x32x100x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // The pass detects tensor.pad on the activation and uses pad=[0,0,0,0] for
    // the NCE workloads. The padded input shape (52x18 bounded) maps to a
    // single workload covering the full output tile.
    // CHECK:       scf.for
    // CHECK:         tensor.extract_slice
    // CHECK:         tensor.pad
    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 52, 17] outOffsets [0, 0, 0, 0] outSizes [1, 32, 50, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv 3x3 inside nested scf.for (width tiling, step=50 over 100) and scf.forall
// (height SOH tiling, step=16 over 32). tensor.pad provides IV-dependent spatial
// padding: first tile in each dim gets top/left=1, last tile gets bottom/right=1.
// NCE op uses pad=[0,0,0,0]. Tests nested loop handling in the pass.
// CHECK-LABEL: @ConvNestedScfForWidthForallHeight
func.func @ConvNestedScfForWidthForallHeight(
        %arg0: tensor<1x16x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %cst = arith.constant 0.000000e+00 : f16
    %acc = tensor.empty() : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c50 = arith.constant 50 : index
    %c100 = arith.constant 100 : index

    %result = scf.for %iv_w = %c0 to %c100 step %c50 iter_args(%out_w = %acc) -> (tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        // Width boundary padding from outer loop IV
        %start_w = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%iv_w)
        %left_pad = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%iv_w)
        %right_pad = affine.max affine_map<(d0) -> (d0 - 49, 0)>(%iv_w)

        %forall_out = tensor.empty() : tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %tile_w = scf.forall (%iv_h) = (0) to (32) step (16)
            shared_outs(%fo = %forall_out) -> (tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

            // Height boundary padding from inner loop IV
            %top_pad = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%iv_h)
            %bot_pad = affine.max affine_map<(d0) -> (d0 - 15, 0)>(%iv_h)
            %start_h = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%iv_h)

            %in_slice = tensor.extract_slice %arg0[0, 0, %start_h, %start_w] [1, 16, 17, 51] [1, 1, 1, 1]
                : tensor<1x16x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
                to tensor<1x16x17x51xf16, {mem_space = @CMX_NN, order = #NHWC}>

            %padded = tensor.pad %in_slice low[0, 0, %top_pad, %left_pad] high[0, 0, %bot_pad, %right_pad] {
            ^bb0(%a: index, %b: index, %c: index, %d: index):
                tensor.yield %cst : f16
            } : tensor<1x16x17x51xf16, {mem_space = @CMX_NN, order = #NHWC}>
              to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 19, 53]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

            %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [32, 16, 3, 3] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                strides = [1, 1]
            } : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 19, 53]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
                tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 17, 51]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

            %cast = tensor.cast %conv
                : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 17, 51]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
                to tensor<1x32x16x50xf16, {mem_space = @CMX_NN, order = #NHWC}>

            scf.forall.in_parallel {
                tensor.parallel_insert_slice %cast into %fo[0, 0, %iv_h, 0] [1, 32, 16, 50] [1, 1, 1, 1]
                    : tensor<1x32x16x50xf16, {mem_space = @CMX_NN, order = #NHWC}>
                    into tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }

        %ins_w = tensor.insert_slice %tile_w into %out_w[0, 0, 0, %iv_w] [1, 32, 32, 50] [1, 1, 1, 1]
            : tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}> into tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.yield %ins_w : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    }

    return %result : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       scf.for
    // CHECK:         scf.forall
    // CHECK:           tensor.extract_slice %arg0[0, 0, {{%.+}}, {{%.+}}] [1, 16, 17, 51]
    // CHECK:           tensor.pad
    // CHECK:           VPU.NCE.Convolution
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 18, 52] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 50] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           tensor.parallel_insert_slice {{%.+}} into {{%.+}}[0, 0, {{%.+}}, 0]
    // CHECK:       tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, 0, {{%.+}}]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Conv 3x3 inside nested scf.for (width tiling, step=50 over 100) and scf.forall
// (height tiling, step=11 over 32 -> 3 clusters with tiles 11, 11, 10).
// tensor.pad provides IV-dependent spatial padding.
// NCE op uses pad=[0,0,0,0]. Tests uneven height splitting across clusters.
// The pass generates scf.if to select between the two workload shapes.
// CHECK-LABEL: @ConvNestedScfForWidthForallHeightUneven
func.func @ConvNestedScfForWidthForallHeightUneven(
        %arg0: tensor<1x16x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        %weights: tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %cst = arith.constant 0.000000e+00 : f16
    %acc = tensor.empty() : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c50 = arith.constant 50 : index
    %c100 = arith.constant 100 : index

    %result = scf.for %iv_w = %c0 to %c100 step %c50 iter_args(%out_w = %acc) -> (tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

        // Width boundary padding from outer loop IV
        %start_w = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%iv_w)
        %left_pad = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%iv_w)
        %right_pad = affine.max affine_map<(d0) -> (d0 - 49, 0)>(%iv_w)

        %forall_out = tensor.empty() : tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %tile_w = scf.forall (%iv_h) = (0) to (32) step (11)
            shared_outs(%fo = %forall_out) -> (tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

            // Tile height: 11 for first two clusters, 10 for last cluster
            %tile_h = affine.min affine_map<(d0) -> (11, 32 - d0)>(%iv_h)
            %start_h = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%iv_h)
            %top_pad = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%iv_h)
            %bot_pad = affine.max affine_map<(d0, d1) -> (d0 + d1 - 31, 0)>(%iv_h, %tile_h)
            %extract_h = affine.apply affine_map<(d0, d1, d2) -> (d0 + 2 - d1 - d2)>(%tile_h, %top_pad, %bot_pad)

            %in_slice = tensor.extract_slice %arg0[0, 0, %start_h, %start_w] [1, 16, %extract_h, 51] [1, 1, 1, 1]
                : tensor<1x16x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
                to tensor<1x16x?x51xf16, {mem_space = @CMX_NN, order = #NHWC}>

            %padded = tensor.pad %in_slice low[0, 0, %top_pad, %left_pad] high[0, 0, %bot_pad, %right_pad] {
            ^bb0(%a: index, %b: index, %c: index, %d: index):
                tensor.yield %cst : f16
            } : tensor<1x16x?x51xf16, {mem_space = @CMX_NN, order = #NHWC}>
              to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 14, 53]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

            %conv = VPU.NCE.Convolution(%padded, %weights) rawFilterShape [32, 16, 3, 3] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                strides = [1, 1]
            } : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 14, 53]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
                tensor<32x16x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
              -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 51]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

            scf.forall.in_parallel {
                tensor.parallel_insert_slice %conv into %fo[0, 0, %iv_h, 0] [1, 32, %tile_h, %c50] [1, 1, 1, 1]
                    : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 12, 51]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
                    into tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}>
            }
        }

        %ins_w = tensor.insert_slice %tile_w into %out_w[0, 0, 0, %iv_w] [1, 32, 32, 50] [1, 1, 1, 1]
            : tensor<1x32x32x50xf16, {mem_space = @CMX_NN, order = #NHWC}> into tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

        scf.yield %ins_w : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    }

    return %result : tensor<1x32x32x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // Uneven height splitting: tiles 11, 11, 10.
    // Two different workload shapes selected at runtime via scf.if.
    // CHECK:       scf.for
    // CHECK:         scf.forall
    // CHECK:           tensor.extract_slice %arg0[0, 0, {{%.+}}, {{%.+}}] [1, 16, {{%.+}}, 51]
    // CHECK:           tensor.pad
    // CHECK:           [[C13:%.+]] = arith.constant 13 : index
    // CHECK:           [[CMP_H:%.+]] = arith.cmpi eq, {{%.+}}, [[C13]] : index
    // CHECK:           [[C52:%.+]] = arith.constant 52 : index
    // CHECK:           [[CMP_W:%.+]] = arith.cmpi eq, {{%.+}}, [[C52]] : index
    // CHECK:           [[COND:%.+]] = arith.andi [[CMP_H]], [[CMP_W]] : i1
    // CHECK:           VPU.NCE.Convolution
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:             scf.if [[COND]] {
    // CHECK:               VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 13, 52] outOffsets [0, 0, 0, 0] outSizes [1, 32, 11, 50] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:             } else {
    // CHECK:               VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 12, 52] outOffsets [0, 0, 0, 0] outSizes [1, 32, 10, 50] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:             }
    // CHECK:           tensor.parallel_insert_slice {{%.+}} into {{%.+}}[0, 0, {{%.+}}, 0]
    // CHECK:       tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, 0, {{%.+}}]
}
