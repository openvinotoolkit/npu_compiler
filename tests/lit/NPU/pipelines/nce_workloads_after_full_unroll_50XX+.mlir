//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --cost-model-analysis-construct --nce-workloads="scf-tiling=true enable-print-statistics=false" %s | FileCheck %s
// REQUIRES: platform-NPU5010

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#output_size = affine_map<(d0) -> (-d0 + 1280, 427)>
#input_offset = affine_map<(d0) -> (0, d0 * 2 - 1)>
#input_pad = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#clamp_pad = affine_map<()[s0] -> (1, s0)>
#input_height = affine_map<(d0) -> (-d0 + 33)>
#input_width = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK-LABEL: @WorkloadShapeAfterFullUnroll
// CHECK-SAME: [[INPUT:%[^:]+]]: tensor<1x4x1600x2560xf32, {order = #NHWC}>
func.func @WorkloadShapeAfterFullUnroll(%arg0: tensor<1x4x1600x2560xf32, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c800 = arith.constant 800 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1280 = arith.constant 1280 : index
    %c427 = arith.constant 427 : index
    %weights = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %padding_value = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    %result = scf.for %height = %c0 to %c800 step %c16 iter_args(%height_output = %empty) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
        %width_result = scf.for %width = %c0 to %c1280 step %c427 iter_args(%output = %height_output) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
            %output_width = affine.min #output_size(%width)
            %height_offset = affine.max #input_offset(%height)
            %height_pad_value = affine.max #input_pad(%height)
            %height_pad = affine.min #clamp_pad()[%height_pad_value]
            %slice_height = affine.apply #input_height(%height_pad)
            %width_offset = affine.max #input_offset(%width)
            %width_pad_value = affine.max #input_pad(%width)
            %width_pad = affine.min #clamp_pad()[%width_pad_value]
            %slice_width = affine.apply #input_width(%output_width, %width_pad)
            %slice = tensor.extract_slice %arg0[0, 0, %height_offset, %width_offset] [1, 4, %slice_height, %slice_width] [1, 1, 1, 1] : tensor<1x4x1600x2560xf32, {order = #NHWC}> to tensor<1x4x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
            %converted = VPU.Convert(%slice) {dstElemType = f16} : tensor<1x4x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
            %copied = VPU.Copy(%converted) {out_mem_space = [@CMX_NN, 0]} : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
            %padded = tensor.pad %copied low[0, 0, %height_pad, %width_pad] high[0, 0, 0, 0] {
            ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
                tensor.yield %padding_value : f16
            } : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
            %weights_cmx = VPU.Copy(%weights) {out_mem_space = [@CMX_NN, 0]} : tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
            %conv = VPU.NCE.Convolution(%padded, %weights_cmx) rawFilterShape [32, 4, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [2, 2]} : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
            %cast = tensor.cast %conv : tensor<1x32x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x16x?x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
            %copied_output = VPU.Copy(%cast) : tensor<1x32x16x?x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x16x?x!qElemType, {order = #NHWC}>
            %inserted = tensor.insert_slice %copied_output into %output[0, 0, %height, %width] [1, 32, 16, %output_width] [1, 1, 1, 1] : tensor<1x32x16x?x!qElemType, {order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
            scf.yield %inserted : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
        }
        scf.yield %width_result : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %result : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
}

// The first unrolled tile has a 16x427 NCE result. Its workload must cover the
// same output shape; without tracing through VPU.Convert it incorrectly uses 16x426.
// CHECK: [[SLICE:%[^ ]+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 854]
// CHECK: [[CONVERTED:%[^ ]+]] = VPU.Convert([[SLICE]])
// CHECK: [[CONV:%[^ ]+]] = VPU.NCE.Convolution
// CHECK-SAME: -> tensor<1x32x16x427x!qElemType
// CHECK: VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 4, 32, 854] outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 427] pad [1, 0, 1, 0] <CUBOID_16x16>
