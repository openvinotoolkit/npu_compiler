//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SplitNCEPermute
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
func.func @SplitNCEPermute(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       VPU.NCE.Permute([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    // CHECK-SAME:      } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {

    // CHECK:       DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224]
    // CHECK-SAME:      pad [0, 0, 0, 0]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedInputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }
    func.func @main(%input: tensor<1x3x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<16x1x1x512xf16, {order = #NHWC}>)
            -> tensor<1x16x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 3, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

            strides = [1, 1]
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<16x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 37, 73] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x16x4x8xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [3, 16, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

            strides = [1, 1]
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<3x16x4x8xf16, {order = #NHWC}> -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedInputOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x3x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x1x1x512xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [3, 3, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

            strides = [1, 1]
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<3x1x1x512xf16, {order = #NHWC}> -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEDepthConvWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x1x4x8xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.DepthConvolution(%input, %weights) rawFilterShape [3, 1, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

            strides = [1, 1]
        } -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEMaxPoolWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.MaxPool
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEAvgPoolWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.AveragePool
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEEltwiseWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            input_padding = [0, 13, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.Eltwise
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] pad [0, 0, 0, 0] <CUBOID_8x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// NCE ops inside an scf.for must be skipped. In the product pipeline WorkloadsForNCEOpsSCF
// (create-workloads-for-nce-ops-scf) runs immediately before this pass and has already given them
// workloads; this test encodes that precondition directly in the input rather than running that pass,
// so the convolution in the loop comes in with one workload and must come out with the same one.
// The identical convolution outside the loop is a control: this pass does split that one, which shows
// the pass is working and skips the first only because it is in the loop.

// CHECK-LABEL: @SkipNCEOpInsideSCFFor
func.func @SkipNCEOpInsideSCFFor(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %cst = const.Declare tensor<144x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<144x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_ctrl = const.Declare tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %c0 = arith.constant 0 : index
    %c144 = arith.constant 144 : index
    %c48 = arith.constant 48 : index
    %out = tensor.empty() : tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %result = scf.for %k = %c0 to %c144 step %c48 iter_args(%acc = %out) -> (tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
        %tile_sz = affine.min affine_map<(d0) -> (-d0 + 144, 48)>(%k)

        %w_slice = tensor.extract_slice %cst[%k, 0, 0, 0] [%tile_sz, 16, 1, 1] [1, 1, 1, 1]
            : tensor<144x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
            to tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[144, 16, 1, 1]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

        %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [%tile_sz, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[144, 16, 1, 1]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
            index
          -> tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}> {
            // Input workload, matching what WorkloadsForNCEOpsSCF produces in the product pipeline.
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
        }

        %insert = tensor.insert_slice %conv into %acc[0, %k, 0, 0] [1, %tile_sz, 16, 16] [1, 1, 1, 1]
            : tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 144, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
            into tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        scf.yield %insert : tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    }

    %ctrl = VPU.NCE.Convolution(%arg0, %cst_ctrl) rawFilterShape [48, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
      -> tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %result, %ctrl : tensor<1x144x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x48x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // Keeps the one workload it came in with, no cost estimate
    // CHECK:       scf.for
    // CHECK:         VPU.NCE.Convolution
    // CHECK-NOT:       minimumHardwareExecutionCost
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 16, 16] pad [0, 0, 0, 0] <CUBOID_16x16>
    // no second workload added
    // CHECK-NOT:       VPU.DPU.Workload
    // CHECK:         scf.yield

    // The control convolution outside the loop is split as usual
    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 48, 16, 16] pad [0, 0, 0, 0]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipNCEOpInsideSCFForall
func.func @SkipNCEOpInsideSCFForall(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                    %weights: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> (tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
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

    %ctrl = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
      -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %result, %ctrl : tensor<2x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // This one comes in with no workloads, so none may be added either
    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.Convolution
    // CHECK-NOT:       minimumHardwareExecutionCost
    // CHECK-NOT:       VPU.DPU.Workload
    // CHECK:         scf.forall.in_parallel

    // The control convolution outside the loop is split as usual
    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0]
}
