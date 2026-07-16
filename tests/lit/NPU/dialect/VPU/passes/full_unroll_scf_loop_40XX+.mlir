//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (-d0 + 2560, 38)>

// CHECK: func.func @UnrollConvert([[INPUT:%.+]]: tensor<1x4x1600x2560xf32, {order = #NHWC}>)
func.func @UnrollConvert(%arg0: tensor<1x4x1600x2560xf32, {order = #NHWC}>) -> tensor<1x4x1600x2560xf16, {order = #NHWC}> {
    %c38 = arith.constant 38 : index
    %c2560 = arith.constant 2560 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x4x1600x2560xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c2560 step %c38 iter_args(%arg2 = %0) -> (tensor<1x4x1600x2560xf16, {order = #NHWC}>) {
      %map = affine.min #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg1] [1, 4, 1600, %map] [1, 1, 1, 1] : tensor<1x4x1600x2560xf32, {order = #NHWC}> to tensor<1x4x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %40 = VPU.Convert(%extracted_slice) {dstElemType = f16} : tensor<1x4x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %40 into %arg2[0, 0, 0, %arg1] [1, 4, 1600, %map] [1, 1, 1, 1] : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x4x1600x2560xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x4x1600x2560xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x4x1600x2560xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 1600, 38]
    //CHECK: [[CONVERT0:%.+]] = VPU.Convert([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 38] [1, 4, 1600, 38]
    //CHECK: [[CONVERT1:%.+]] = VPU.Convert([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONVERT0]], [[CONVERT1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 38],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 2508], [0, 0, 0, 2546]]}
}

// -----

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

#map1 = affine_map<(d0) -> (-d0 + 1280, 15)>
#map2 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map3 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK: func.func @UnrollConvolution([[INPUT:%.+]]: tensor<1x4x1600x2560xf16, {order = #NHWC}>)
func.func @UnrollConvolution(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c15 = arith.constant 15 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf16>, [#const.Reorder<#NHWC>]

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c15 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %size1 = affine.min #map1(%arg1)
      %offset0 = affine.max #map2(%arg1)
      %value = affine.max #map3(%arg1)
      %pad = affine.min #map4()[%value]
      %size0 = affine.apply #map5(%size1, %pad)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %offset0] [1, 4, 1600, %size0] [1, 1, 1, 1] : tensor<1x4x1600x2560xf16, {order = #NHWC}> to tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
      %padded = tensor.pad %extracted_slice low[0, 0, 1, %pad] high[0, 0, 0, 0] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>
      %conv = VPU.NCE.Convolution(%padded, %cst_0) rawFilterShape [32, 4, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>,  strides = [2, 2]} : tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %conv into %arg2[0, 0, 0, %arg1] [1, 32, 800, %size1] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 1600, 30]
    //CHECK-NOT: tensor.pad
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]]
    //CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 29] [1, 4, 1600, 31]
    //CHECK-NOT: tensor.pad
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]]
    //CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 15],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 1260], [0, 0, 0, 1275]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 1280, 28)>

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

// CHECK: func.func @UnrollEltwise([[INPUT0:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>,
// CHECK-SAME:                     [[INPUT1:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>)
func.func @UnrollEltwise(%arg0: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>, %arg1: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c28 = arith.constant 28 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c1280 step %c28 iter_args(%arg3 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %size = affine.min #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %eltwise = VPU.NCE.Eltwise(%extracted_slice, %extracted_slice_1) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, scale = 3.5136938095092773E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.380000e+02 : f64, in1_mult = [2.934300e+04], in2_mult = [2.934300e+04]>} -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %eltwise into %arg3[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }


    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]]
    //CHECK-NOT: tensor.insert_slice
    //CHECK: [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 28] [1, 32, 800, 28]
    //CHECK: [[SLICE3:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 28] [1, 32, 800, 28]
    //CHECK: [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[SLICE2]], [[SLICE3]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE0]], [[ELTWISE1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 28],
    //CHECK-SAME{LITERAL}:  [0, 0, 0, 1232], [0, 0, 0, 1260]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 2)>

// CHECK: func.func @UnrollDepthToSpace([[INPUT:%.+]]: tensor<1x16x800x1280xf16, {order = #NHWC}>)
func.func @UnrollDepthToSpace(%arg0: tensor<1x16x800x1280xf16, {order = #NHWC}>) -> tensor<1x4x1600x2560xf32, {order = #NHWC}> {
    %c20 = arith.constant 20 : index
    %c1600 = arith.constant 1600 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1600 step %c20 iter_args(%arg2 = %0) -> (tensor<1x4x1600x2560xf32, {order = #NHWC}>) {
      %offset = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %offset, 0] [1, 16, 10, 1280] [1, 1, 1, 1] : tensor<1x16x800x1280xf16, {order = #NHWC}> to tensor<1x16x10x1280xf16, {order = #NHWC}>
      %40 = VPU.DepthToSpace(%extracted_slice) {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x10x1280xf16, {order = #NHWC}> -> tensor<1x4x20x2560xf32, {order = #NHWC}>
      %inserted_slice = tensor.insert_slice %40 into %arg2[0, 0, %arg1, 0] [1, 4, 20, 2560] [1, 1, 1, 1] : tensor<1x4x20x2560xf32, {order = #NHWC}> into tensor<1x4x1600x2560xf32, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    }

    return %1 : tensor<1x4x1600x2560xf32, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 10, 1280]
    //CHECK: [[D2S0:%.+]] = VPU.DepthToSpace([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 10, 0] [1, 16, 10, 1280]
    //CHECK: [[D2S1:%.+]] = VPU.DepthToSpace([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[D2S0]], [[D2S1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 20, 0],
    //CHECK-SAME{LITERAL}:  [0, 0, 1560, 0], [0, 0, 1580, 0]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 254)>
#map4 = affine_map<(d0) -> (0, d0 - 358)>
#map5 = affine_map<(d0, d1) -> (-d0 - d1 + 122)>

// CHECK: func.func @Unroll2DMaxpool([[INPUT:%.+]]: tensor<1x16x512x480xf16, {order = #NHWC}>)
 func.func @Unroll2DMaxpool(%arg0: tensor<1x16x512x480xf16, {order = #NHWC}>) -> tensor<1x16x512x480xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c120 = arith.constant 120 : index
    %c256 = arith.constant 256 : index
    %c480 = arith.constant 480 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x16x512x480xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c512 step %c256 iter_args(%arg2 = %0) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %c480 step %c120 iter_args(%arg4 = %arg2) -> (tensor<1x16x512x480xf16, {order = #NHWC}>) {
        %3 = affine.max #map(%arg1)
        %4 = affine.max #map1(%arg1)
        %5 = affine.min #map2()[%4]
        %6 = affine.max #map3(%3)
        %7 = affine.min #map2()[%6]
        %8 = affine.max #map(%arg3)
        %9 = affine.max #map1(%arg3)
        %10 = affine.min #map2()[%9]
        %11 = affine.max #map4(%8)
        %12 = affine.min #map2()[%11]
        %13 = affine.apply #map5(%10, %12)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, %8] [1, 16, 257, %13] [1, 1, 1, 1] : tensor<1x16x512x480xf16, {order = #NHWC}> to tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %padded = tensor.pad %extracted_slice low[0, 0, %5, %10] high[0, 0, %7, %12] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %cst : f16
        } : tensor<1x16x257x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 259, 482]> : tensor<4xsi64>, order = #NHWC}>
        %14 = VPU.NCE.MaxPool(%padded) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [3, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}>
        %cast = tensor.cast %14 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 257, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x120xf16, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg4[0, 0, %arg1, %arg3] [1, 16, 256, 120] [1, 1, 1, 1] : tensor<1x16x256x120xf16, {order = #NHWC}> into tensor<1x16x512x480xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x512x480xf16, {order = #NHWC}>
      }
      scf.yield %2 : tensor<1x16x512x480xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x16x512x480xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 257, 121]
    //CHECK-NOT: tensor.pad
    //CHECK: [[MAXPOOL0:%.+]] = VPU.NCE.MaxPool([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 119] [1, 16, 257, 122]
    //CHECK-NOT: tensor.pad
    //CHECK: [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[SLICE1]]
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL0]], [[MAXPOOL1]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 120], [0, 0, 0, 240], [0, 0, 0, 360], [0, 0, 256, 0], [0, 0, 256, 120], [0, 0, 256, 240], [0, 0, 256, 360]]}
}

// -----


config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> ((d0 floordiv 6) * 5 + 130)>

// CHECK: func.func @UnrollUnevenConvolution([[INPUT:%.+]]: tensor<1x96x800x1280xf16, {order = #NHWC}>)
func.func @UnrollUnevenConvolution(%arg0: tensor<1x96x800x1280xf16, {order = #NHWC}>) -> tensor<1x16x800x1280xf16, {order = #NHWC}> {
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c780 = arith.constant 780 : index
    %c800 = arith.constant 800 : index
    %c0 = arith.constant 0 : index

    %cst_0 = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = tensor.empty() : tensor<1x16x800x1280xf16, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c800 step %c6 iter_args(%arg2 = %0) -> (tensor<1x16x800x1280xf16, {order = #NHWC}>) {
      %2 = arith.cmpi ult, %arg1, %c780 : index
      %3 = arith.select %2, %c6, %c5 : index
      %4 = scf.if %2 -> (index) {
        scf.yield %arg1 : index
      } else {
        %5 = affine.apply #map(%arg1)
        scf.yield %5 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %4, 0] [1, 96, %3, 1280] [1, 1, 1, 1] : tensor<1x96x800x1280xf16, {order = #NHWC}> to tensor<1x96x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %42 = VPU.NCE.Convolution(%extracted_slice, %cst_0) rawFilterShape [16, 96, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x96x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x96x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %42 into %arg2[0, 0, %4, 0] [1, 16, %3, 1280] [1, 1, 1, 1] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x800x1280xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x800x1280xf16, {order = #NHWC}>
    }


    return %1 : tensor<1x16x800x1280xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 96, 6, 1280]
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 6, 0] [1, 96, 6, 1280]
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]]

    //CHECK: [[SLICEN:%.+]] = VPU.Slice [[INPUT]] [0, 0, 780, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN:%.+]] = VPU.NCE.Convolution([[SLICEN]]
    //CHECK: [[SLICEN1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 785, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN1:%.+]] = VPU.NCE.Convolution([[SLICEN1]]
    //CHECK: [[SLICEN2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 790, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN2:%.+]] = VPU.NCE.Convolution([[SLICEN2]]
    //CHECK: [[SLICEN3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 795, 0] [1, 96, 5, 1280]
    //CHECK: [[CONVN3:%.+]] = VPU.NCE.Convolution([[SLICEN3]]

    //CHECK-NOT: tensor.insert_slice
    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]],
    //CHECK-SAME:            [[CONVN]], [[CONVN1]], [[CONVN2]], [[CONVN3]])
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0]
    //CHECK-SAME{LITERAL}:  [0, 0, 780, 0], [0, 0, 785, 0], [0, 0, 790, 0], [0, 0, 795, 0]]}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> ((d0 floordiv 28) * 27 + 38)>

!qElemType = !quant.uniform<u8:f16, 0.0050920921213486615:109>


// CHECK: func.func @UnrollloopWithDMAs([[INPUT:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>)
func.func @UnrollloopWithDMAs(%arg0: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c1064 = arith.constant 1064 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c28 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %2 = arith.cmpi ult, %arg1, %c1064 : index
      %3 = arith.select %2, %c28, %c27 : index
      %4 = scf.if %2 -> (index) {
        scf.yield %arg1 : index
      } else {
        %9 = affine.apply #map(%arg1)
        scf.yield %9 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %5 = VPU.Copy(%extracted_slice) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %6 = VPU.Copy(%extracted_slice_1) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %7 = VPU.NCE.Eltwise(%5, %6) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, scale = 2.3410655558109283E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.090000e+02 : f64, in1_mult = [2.863000e+04], in2_mult = [2.863000e+04]>} -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
      %8 = VPU.Copy(%7) : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %8 into %arg2[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 800, 28]
    //CHECK: [[COPY0:%.+]] = VPU.Copy([[SLICE0]])
    //CHECK: [[COPY1:%.+]] = VPU.Copy([[SLICE1]])
    //CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[COPY0]], [[COPY1]])
    //CHECK: [[COPY2:%.+]] = VPU.Copy([[ELTWISE0]])
    //CHECK-NOT: tensor.insert_slice

    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[COPY2]],
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 28]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
#map5 = affine_map<(d0) -> (d0 ceildiv 6)>
#map6 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>
#map7 = affine_map<(d0, d1) -> (0, d0 - d1)>
#map8 = affine_map<(d0, d1, d2) -> (0, d0 - d1 + d2 - 31)>
#map9 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>

!paddedConvInTiledType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerConvOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!convOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-LABEL:   @SOHConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOHConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c31 = arith.constant 31 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1]
      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    %7 = affine.apply #map4(%4, %6)
    %8 = arith.addi %7, %c31 : index
    %9 = tensor.empty(%8) : !convOutTiledType
    %10 = affine.apply #map5(%8)
    %11 = scf.forall (%arg3) = (0) to (%8) step (%10) shared_outs(%arg4 = %9) -> (!convOutTiledType) {
      %13 = affine.min #map6(%arg3, %8)[%8]
      %14 = affine.max #map7(%arg3, %4)
      %15 = affine.max #map7(%4, %14)
      %16 = affine.max #map8(%arg3, %4, %13)
      %17 = affine.apply #map9(%15, %16, %13)

      %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, 0, %14, 0] [1, 32, %17, 64] [1, 1, 1, 1]
        : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {order = #NHWC}>

      %copy_act = VPU.Copy(%extracted_slice_1) {out_mem_space = @CMX_NN}
        : tensor<1x32x?x64xf16, {order = #NHWC}> -> tensor<1x32x?x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

      %padded = tensor.pad %copy_act low[0, 0, %15, 1] high[0, 0, %16, 1] {
      ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
        tensor.yield %cst : f16
      } : tensor<1x32x?x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to !paddedConvInTiledType

      %18 = VPU.NCE.Convolution(%padded, %copy_weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
         strides = [1, 1]}
        : !paddedConvInTiledType, tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> !innerConvOutTiledType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %18 into %arg4[0, 0, %arg3, 0] [1, 256, %13, 64] [1, 1, 1, 1]
          : !innerConvOutTiledType into !convOutTiledType
      }
    }

    %cast = tensor.cast %11
      : !convOutTiledType to tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %copy_output = VPU.Copy(%cast) {out_mem_space = @DDR}
      : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    %inserted_slice = tensor.insert_slice %copy_output into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK-SAME:    -> !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:           mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 7, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 4, 64]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 7, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 4, 64]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 11, 0], [0, 0, 17, 0], [0, 0, 23, 0], [0, 0, 29, 0]]}>

// CHECK:       [[COPY_WEIGHTS0:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:    -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:           mode = "DUPLICATED", num_clusters = 6 : i64,
// CHECK-SAME{LITERAL}:  compute_shapes = [[256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY_TILE0]], [[COPY_WEIGHTS0]])
// CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
// CHECK-SAME:    : !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME:      !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64,
// CHECK-SAME:    -> !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 2, 64]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 2, 64]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]]}>

// CHECK:       [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]])
// CHECK-SAME:    : !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME:    -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
// CHECK-SAME:    tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:       [[COPY_TILE1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK-SAME:    -> !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:           mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 3, 64]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 8, 64], [1, 32, 3, 64]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]]}>

// CHECK:       [[COPY_WEIGHTS1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:    : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:    -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:           mode = "DUPLICATED", num_clusters = 6 : i64,
// CHECK-SAME{LITERAL}:  compute_shapes = [[256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3], [256, 32, 3, 3]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY_TILE1]], [[COPY_WEIGHTS1]])
// CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
// CHECK-SAME:    : !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME:      !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64,
// CHECK-SAME:    -> !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME{LITERAL}:  compute_shapes = [[1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 2, 64]],
// CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]],
// CHECK-SAME{LITERAL}:  memory_shapes = [[1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 6, 64], [1, 256, 2, 64]],
// CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0], [0, 0, 18, 0], [0, 0, 24, 0], [0, 0, 30, 0]]}>

// CHECK:       [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV1]])
// CHECK-SAME:    : !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
// CHECK-SAME:    -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 32, 0{{\]\]}}}
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
#map5 = affine_map<(d0) -> (-d0 + 256, 96)>

!convInTiledType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!convOutTiledDDRType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

!innerConvOutTiledType = tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!castedConvOutTiledType = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL:   @SOKConvTileOverH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SOKConvTileOverH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c31 = arith.constant 31 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 32, 33, 64] [1, 1, 1, 1]
      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    %7 = affine.apply #map4(%4, %6)
    %8 = arith.addi %7, %c31 : index
    %9 = tensor.empty(%8) : !convOutTiledDDRType

    %10 = scf.forall (%arg3) = (0) to (256) step (96) shared_outs(%arg4 = %9) -> (!convOutTiledDDRType) {
      %11 = affine.min #map5(%arg3)

      %copy_act = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : tensor<1x32x33x64xf16, {order = #NHWC}> -> tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %padded = tensor.pad %copy_act low[0, 0, %4, 1] high[0, 0, %6, 1] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %cst : f16
      } : tensor<1x32x33x64xf16, {mem_space = @CMX_NN, order = #NHWC}> to !convInTiledType

      %extracted_slice_2 = tensor.extract_slice %cst_0[%arg3, 0, 0, 0] [%11, 32, 3, 3] [1, 1, 1, 1]
        : tensor<256x32x3x3xf16, {order = #NHWC}>
        to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
      %copy_weights = VPU.Copy(%extracted_slice_2) {out_mem_space = @CMX_NN}
        : tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

      %15 = VPU.NCE.Convolution(%padded, %copy_weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
         strides = [1, 1]
      } : !convInTiledType,
          tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      -> !innerConvOutTiledType

      %copy_output = VPU.Copy(%15) {out_mem_space = @DDR}
      : !innerConvOutTiledType -> tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      %cast_3 = tensor.cast %copy_output
        : tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
        to !castedConvOutTiledType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast_3 into %arg4[0, %arg3, 0, 0] [1, %11, 64, 64] [1, 1, 1, 1]
          : !castedConvOutTiledType into !convOutTiledDDRType
      }
    }

    %cast = tensor.cast %10 : !convOutTiledDDRType to tensor<1x256x32x64xf16, {order = #NHWC}>
    %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:         [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK:         [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
// CHECK-SAME:      tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:         [[COPY_TILE0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "DUPLICATED", num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 33, 64], [1, 32, 33, 64], [1, 32, 33, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 33, 64], [1, 32, 33, 64], [1, 32, 33, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:         [[COPY_W_TILE0:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1],
// CHECK-SAME{LITERAL}:    compute_shapes = [[96, 32, 3, 3], [96, 32, 3, 3], [64, 32, 3, 3]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[96, 32, 3, 3], [96, 32, 3, 3], [64, 32, 3, 3]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}>

// CHECK:         [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY_TILE0]], [[COPY_W_TILE0]])
// CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1],
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 96, 32, 64], [1, 96, 32, 64], [1, 64, 32, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 96, 32, 64], [1, 96, 32, 64], [1, 64, 32, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]]}>

// CHECK:         [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1]
// CHECK-SAME:      -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:         [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
// CHECK-SAME:      tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:         [[COPY_TILE1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x32x33x64xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x32x33x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "DUPLICATED", num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 33, 64], [1, 32, 33, 64], [1, 32, 33, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 33, 64], [1, 32, 33, 64], [1, 32, 33, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

// CHECK:         [[COPY_W_TILE1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1],
// CHECK-SAME{LITERAL}:    compute_shapes = [[96, 32, 3, 3], [96, 32, 3, 3], [64, 32, 3, 3]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[96, 32, 3, 3], [96, 32, 3, 3], [64, 32, 3, 3]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}>

// CHECK:         [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY_TILE1]], [[COPY_W_TILE1]])
// CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1],
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 96, 32, 64], [1, 96, 32, 64], [1, 64, 32, 64]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 96, 32, 64], [1, 96, 32, 64], [1, 64, 32, 64]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]]}>

// CHECK:         [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV1]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x256x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1]
// CHECK-SAME:      -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 32, 0{{\]\]}}}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
#map1 = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 126)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
#map5 = affine_map<(d0) -> (-d0 + 160, 54)>
#map6 = affine_map<(d0) -> (0, d0 - 1)>
#map7 = affine_map<(d0) -> (-d0 + 1, 0)>
#map8 = affine_map<(d0, d1) -> (0, d0 + d1 - 159)>
#map9 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
#map10 = affine_map<(d0) -> (-d0, 0)>
#map11 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2)>
#map12 = affine_map<(d0) -> (-d0 + 320, 107)>
#map13 = affine_map<(d0) -> (d0 floordiv 2)>

!inputConvType = tensor<1x32x160x256xf16, {order = #NHWC}>
!outputD2SType = tensor<1x4x320x512xf16, {order = #NHWC}>
!inD2SType = tensor<1x16x160x128xf16, {order = #NHWC}>

!inputConvTiledDDRType = tensor<1x32x160x129xf16, {order = #NHWC}>

!outConvTiledType = tensor<1x16x160x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 256]> : tensor<4xsi64>, order = #NHWC}>
!outD2STiledDDRType = tensor<1x4x320x256xf16, {order = #NHWC}>

!innerConvInputType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 56, 258]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerConvOutputType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerConvOutputDDRType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, order = #NHWC}>
!innerConvOutCastedType = tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 54, 256]> : tensor<4xsi64>, order = #NHWC}>

!innerD2SInType = tensor<1x16x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 160, 128]> : tensor<4xsi64>, order = #NHWC}>
!innerD2SOutType = tensor<1x4x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 320, 256]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @ChainConvD2SWithMC
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x160x256xf16, {order = #NHWC}>
func.func @ChainConvD2SWithMC(%arg0: !inputConvType) -> !outputD2SType {
  %c129 = arith.constant 129 : index
  %c127 = arith.constant 127 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

  %0 = tensor.empty() : !outputD2SType
  %1 = scf.for %arg1 = %c0 to %c512 step %c256 iter_args(%arg2 = %0) -> (!outputD2SType) {
    %2 = affine.max #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.min #map2()[%3]
    %5 = affine.max #map3(%2)
    %6 = affine.min #map2()[%5]
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %2] [1, 32, 160, 129] [1, 1, 1, 1]
      : !inputConvType to !inputConvTiledDDRType

    %7 = affine.apply #map4(%4, %6)
    %8 = arith.addi %7, %c127 : index
    %9 = tensor.empty(%8) : !outConvTiledType
    %10 = scf.forall (%arg3) = (0) to (160) step (54) shared_outs(%arg4 = %9) -> (!outConvTiledType) {
      %13 = affine.min #map5(%arg3)
      %14 = arith.addi %7, %c129 : index
      %15 = affine.max #map6(%arg3)
      %16 = affine.max #map7(%15)
      %17 = affine.max #map8(%arg3, %13)
      %18 = affine.apply #map9(%16, %17, %13)
      %19 = affine.max #map10(%4)
      %20 = affine.apply #map11(%14, %4, %6)

      %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, 0, %15, %19] [1, 32, %18, %20] [1, 1, 1, 1]
        : tensor<1x32x160x129xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {order = #NHWC}>

      %copy_act = VPU.Copy(%extracted_slice_1) {out_mem_space = @CMX_NN}
        : tensor<1x32x?x?xf16, {order = #NHWC}> -> tensor<1x32x?x?xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %padded = tensor.pad %copy_act low[0, 0, %16, %4] high[0, 0, %17, %6] {
      ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
        tensor.yield %cst : f16
      } : tensor<1x32x?x?xf16, {mem_space = @CMX_NN, order = #NHWC}> to !innerConvInputType

      %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<16x32x3x3xf16, {order = #NHWC}> -> tensor<16x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
      %conv = VPU.NCE.Convolution(%padded, %copy_weights) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : !innerConvInputType, tensor<16x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !innerConvOutputType
      %copy_output = VPU.Copy(%conv) {out_mem_space = @DDR}
        : !innerConvOutputType -> !innerConvOutputDDRType
      %cast_2 = tensor.cast %copy_output : !innerConvOutputDDRType to !innerConvOutCastedType

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast_2 into %arg4[0, 0, %arg3, 0] [1, 16, %13, 256] [1, 1, 1, 1]
          : !innerConvOutCastedType into !outConvTiledType
      }
    }
    %cast = tensor.cast %10 : !outConvTiledType to !inD2SType
    %11 = tensor.empty() : !outD2STiledDDRType
    %12 = scf.forall (%arg3) = (0) to (320) step (107) shared_outs(%arg4 = %11)
        -> (!outD2STiledDDRType) {
      %13 = affine.min #map12(%arg3)
      %14 = affine.apply #map13(%arg3)
      %15 = affine.apply #map13(%13)
      %extracted_slice_1 = tensor.extract_slice %cast[0, 0, %14, 0] [1, 16, %15, 128] [1, 1, 1, 1]
        : !inD2SType to !innerD2SInType
      %16 = VPU.DepthToSpace(%extracted_slice_1)
        {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : !innerD2SInType -> !innerD2SOutType

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg4[0, 0, %arg3, 0] [1, 4, %13, 256] [1, 1, 1, 1]
          : !innerD2SOutType into !outD2STiledDDRType
      }
    }

    %inserted_slice = tensor.insert_slice %12 into %arg2[0, 0, 0, %arg1] [1, 4, 320, 256] [1, 1, 1, 1]
      : !outD2STiledDDRType into !outputD2SType
    scf.yield %inserted_slice : !outputD2SType
  }
  return %1 : !outputD2SType

// CHECK:         [[TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 160, 129]
// CHECK-SAME:      : tensor<1x32x160x256xf16, {order = #NHWC}> to tensor<1x32x160x129xf16, {order = #NHWC}>

// CHECK:         [[COPY_INPUT0:%.+]] = VPU.Copy([[TILE0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x32x160x129xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x32x160x129xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 55, 129], [1, 32, 56, 129], [1, 32, 53, 129]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 55, 129], [1, 32, 56, 129], [1, 32, 53, 129]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]]}>

// CHECK:         [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY_INPUT0]], {{[^:]+}})
// CHECK-SAME:           pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 1 : i64>
// CHECK-SAME:      : !VPU.DistributedTensor<1x32x160x129xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:        !VPU.DistributedTensor<16x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 16, 54, 128], [1, 16, 54, 128], [1, 16, 52, 128]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 108, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 16, 54, 128], [1, 16, 54, 128], [1, 16, 52, 128]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 108, 0]]

// CHECK:         [[COPY_OUT0:%.+]] = VPU.Copy([[CONV0:%.+]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:      -> tensor<1x16x160x128xf16, {order = #NHWC}>

// CHECK:         [[COPY_IN_D2S0:%.+]] = VPU.Copy([[COPY_OUT0:%.+]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x16x160x128xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 53, 128], [1, 16, 53, 128], [1, 16, 53, 128]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 53, 128], [1, 16, 53, 128], [1, 16, 53, 128]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]]

// CHECK:          [[D2S0:%.+]] = VPU.DepthToSpace([[COPY_IN_D2S0]])
// CHECK-SAME:       : !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:       -> !VPU.DistributedTensor<1x4x320x256xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 4, 107, 256], [1, 4, 107, 256], [1, 4, 106, 256]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 107, 0], [0, 0, 214, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 4, 107, 256], [1, 4, 107, 256], [1, 4, 106, 256]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 107, 0], [0, 0, 214, 0]]

// CHECK:          [[COPY_OUT_D2S0:%.+]] = VPU.Copy([[D2S0]])
// CHECK-SAME:        : !VPU.DistributedTensor<1x4x320x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:        -> tensor<1x4x320x256xf16, {order = #NHWC}>


// CHECK:         [[TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 127] [1, 32, 160, 129]
// CHECK-SAME:      : tensor<1x32x160x256xf16, {order = #NHWC}> to tensor<1x32x160x129xf16, {order = #NHWC}>

// CHECK:         [[COPY_CONV_INPUT1:%.+]] = VPU.Copy([[TILE1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x32x160x129xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x32x160x129xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64

// CHECK:         [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY_CONV_INPUT1]], {{%.+}})
// CHECK-SAME:      : !VPU.DistributedTensor<1x32x160x129xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64

// CHECK:         [[COPY_OUT1:%.+]] = VPU.Copy([[CONV1]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:      -> tensor<1x16x160x128xf16, {order = #NHWC}>

// CHECK:         [[COPY_IN_D2S1:%.+]] = VPU.Copy([[COPY_OUT1:%.+]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x16x160x128xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 53, 128], [1, 16, 53, 128], [1, 16, 53, 128]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 53, 128], [1, 16, 53, 128], [1, 16, 53, 128]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 53, 0], [0, 0, 107, 0]]

// CHECK:          [[D2S1:%.+]] = VPU.DepthToSpace([[COPY_IN_D2S1]])
// CHECK-SAME:       : !VPU.DistributedTensor<1x16x160x128xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:       -> !VPU.DistributedTensor<1x4x320x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64

// CHECK:          [[COPY_OUT_D2S1:%.+]] = VPU.Copy([[D2S1]])
// CHECK-SAME:        : !VPU.DistributedTensor<1x4x320x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:        -> tensor<1x4x320x256xf16, {order = #NHWC}>


// CHECK:       VPU.Concat([[COPY_OUT_D2S0]], [[COPY_OUT_D2S1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 256{{\]\]}}}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<()[s0] -> (1, s0)>
#map3 = affine_map<(d0) -> (0, d0 - 30)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
#map5 = affine_map<(d0) -> (d0 ceildiv 3)>
#map6 = affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 3)>
#map7 = affine_map<(d0, d1) -> (0, d0 - d1)>
#map8 = affine_map<(d0, d1, d2) -> (0, d0 - d1 + d2 - 31)>
#map9 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 2)>
#map10 = affine_map<(d0) -> (-d0, 0)>
#map11 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2)>

!paddedInputTiledType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outputTiledType = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerInputTiledType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 24, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerOutputTiledType = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!innerOutputTiledDDRType = tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, order = #NHWC}>

!castedOutputTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 22, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @TwoAxisTilingNCEConvSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @TwoAxisTilingNCEConvSOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %c33 = arith.constant 33 : index
  %c31 = arith.constant 31 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst_0 = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.for %arg1 = %c0 to %c64 step %c32 iter_args(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = scf.for %arg3 = %c0 to %c64 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %3 = affine.max #map(%arg1)
      %4 = affine.max #map1(%arg1)
      %5 = affine.min #map2()[%4]
      %6 = affine.max #map3(%3)
      %7 = affine.min #map2()[%6]
      %8 = affine.max #map(%arg3)
      %9 = affine.max #map1(%arg3)
      %10 = affine.min #map2()[%9]
      %11 = affine.max #map3(%8)
      %12 = affine.min #map2()[%11]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, %8] [1, 32, 33, 33] [1, 1, 1, 1]
        : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x33xf16, {order = #NHWC}>

      %13 = affine.apply #map4(%5, %7)
      %14 = arith.addi %13, %c31 : index
      %15 = affine.apply #map4(%10, %12)
      %16 = arith.addi %15, %c31 : index

      %17 = tensor.empty(%14, %16) : !outputTiledType
      %18 = affine.apply #map5(%14)
      %19 = scf.forall (%arg5) = (0) to (%14) step (%18) shared_outs(%arg6 = %17) -> (!outputTiledType) {
        %20 = affine.min #map6(%arg5, %14)[%14]
        %21 = arith.addi %15, %c33 : index
        %22 = affine.max #map7(%arg5, %5)
        %23 = affine.max #map7(%5, %22)
        %24 = affine.max #map8(%arg5, %5, %20)
        %25 = affine.apply #map9(%23, %24, %20)
        %26 = affine.max #map10(%10)
        %27 = affine.apply #map11(%21, %10, %12)

        %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0, %22, %26] [1, 32, %25, %27] [1, 1, 1, 1]
          : tensor<1x32x33x33xf16, {order = #NHWC}> to tensor<1x32x?x?xf16, {order = #NHWC}>
        %copy_act = VPU.Copy(%extracted_slice_2) {out_mem_space = @CMX_NN}
        : tensor<1x32x?x?xf16, {order = #NHWC}> -> tensor<1x32x?x?xf16, {mem_space = @CMX_NN, order = #NHWC}>

        %padded = tensor.pad %copy_act low[0, 0, %23, %10] high[0, 0, %24, %12] {
          ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
            tensor.yield %cst : f16
        } : tensor<1x32x?x?xf16, {mem_space = @CMX_NN, order = #NHWC}> to !innerInputTiledType

        %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        %28 = VPU.NCE.Convolution(%padded, %copy_weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>, 
          strides = [1, 1], tiling_index = 0 : i64
        } : !innerInputTiledType, tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> !innerOutputTiledType

        %copy_output = VPU.Copy(%28) {out_mem_space = @DDR}
          : !innerOutputTiledType -> !innerOutputTiledDDRType
        %cast_3 = tensor.cast %copy_output : !innerOutputTiledDDRType to !castedOutputTiledType
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %cast_3 into %arg6[0, 0, %arg5, 0] [1, 256, %20, 64] [1, 1, 1, 1]
            : !castedOutputTiledType into !outputTiledType
        }
      }
      %cast = tensor.cast %19 : !outputTiledType to tensor<1x256x32x32xf16, {order = #NHWC}>

      %inserted_slice = tensor.insert_slice %cast into %arg4[0, 0, %arg1, %arg3] [1, 256, 32, 32] [1, 1, 1, 1]
        : tensor<1x256x32x32xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
    }
    scf.yield %2 : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:         [[TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 33]
// CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x33xf16, {order = #NHWC}>

// CHECK:         [[COPY_INNER_CONV_INPUT0:%.+]] = VPU.Copy([[TILE_0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<1x32x33x33xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x32x33x33xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:             mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 12, 33], [1, 32, 13, 33], [1, 32, 12, 33]],
// CHECK-SAME{LITERAL}:    compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0]],
// CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 12, 33], [1, 32, 13, 33], [1, 32, 12, 33]],
// CHECK-SAME{LITERAL}:    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0]]}>

// CHECK:         [[COPY_WEIGHTS:%.+]] = VPU.Copy({{[^:]+}}) {out_mem_space = @CMX_NN}
// CHECK-SAME:      : tensor<256x32x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "DUPLICATED", num_clusters = 3 : i64

// CHECK:         [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY_INNER_CONV_INPUT0]], [[COPY_WEIGHTS]])
// CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
// CHECK-SAME:      : !VPU.DistributedTensor<1x32x33x33xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:        !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64
// CHECK-SAME:      -> !VPU.DistributedTensor<1x256x32x32xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

// CHECK:         [[COPY_OUT_0:%.+]] = VPU.Copy([[CONV0]])
// CHECK-SAME:      : !VPU.DistributedTensor<1x256x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME:      -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:         [[TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 31] [1, 32, 33, 33]

// CHECK:         [[COPY_INNER_CONV_INPUT1:%.+]] = VPU.Copy([[TILE_1]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x33x33xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 12, 33], [1, 32, 13, 33], [1, 32, 12, 33]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 12, 33], [1, 32, 13, 33], [1, 32, 12, 33]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 21, 0]]}>

// CHECK:         [[CONV_1:%.+]] = VPU.NCE.Convolution([[COPY_INNER_CONV_INPUT1]], {{[^:]+}})
// CHECK-SAME:            pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
// CHECK-SAME:      -> !VPU.DistributedTensor<1x256x32x32xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

// CHECK:         [[COPY_OUT_1:%.+]] = VPU.Copy([[CONV_1]])

// CHECK:         [[TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 33]
// CHECK:         [[COPY_TILE2:%.+]] = VPU.Copy([[TILE_2]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x33x33xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 13, 33], [1, 32, 13, 33], [1, 32, 11, 33]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 13, 33], [1, 32, 13, 33], [1, 32, 11, 33]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

// CHECK:         [[CONV_2:%.+]] = VPU.NCE.Convolution([[COPY_TILE2]], {{[^:]+}})
// CHECK-SAME:            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>
// CHECK-SAME:       -> !VPU.DistributedTensor<1x256x32x32xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]

// CHECK:         [[COPY_OUT_2:%.+]] = VPU.Copy([[CONV_2]])
// CHECK-SAME:      -> tensor<1x256x32x32xf16, {order = #NHWC}>

// CHECK:         [[TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 31] [1, 32, 33, 33]
// CHECK:         [[COPY_TILE3:%.+]] = VPU.Copy([[TILE_3]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:        -> !VPU.DistributedTensor<1x32x33x33xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 32, 13, 33], [1, 32, 13, 33], [1, 32, 11, 33]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 32, 13, 33], [1, 32, 13, 33], [1, 32, 11, 33]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

// CHECK:         [[INNER_CONV_3:%.+]] = VPU.NCE.Convolution([[COPY_TILE3]]
// CHECK-SAME:         pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
// CHECK-SAME:       -> !VPU.DistributedTensor<1x256x32x32xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:            mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64
// CHECK-SAME{LITERAL}:   compute_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
// CHECK-SAME{LITERAL}:   memory_shapes = [[1, 256, 11, 32], [1, 256, 11, 32], [1, 256, 10, 32]],
// CHECK-SAME{LITERAL}:   memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

// CHECK:         [[COPY_OUT_3:%.+]] = VPU.Copy([[INNER_CONV_3]])

// CHECK:       VPU.Concat([[COPY_OUT_0]], [[COPY_OUT_1]], [[COPY_OUT_2]], [[COPY_OUT_3]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32], [0, 0, 32, 0], [0, 0, 32, 32]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> ((d0 floordiv 5) * 4 + 4)>
#map1 = affine_map<(d0) -> (-d0 + 640, 96)>
#map2 = affine_map<(d0) -> (d0 - 1, 0)>
#map3 = affine_map<(d0) -> (-(d0 - 1), 0)>
#map4 = affine_map<()[s0] -> (s0, 1)>
#map5 = affine_map<(d0, d1) -> (d1 + d0 + 2 - 32, 0)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

// CHECK-LABEL: @UnrollCTiledConvolution
// CHECK-SAME:       [[INPUT0:%arg[0-9]]]: tensor<1x640x32x32xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT1:%arg[0-9]]]: tensor<640x640x3x3xf16, {order = #NHWC}>)
func.func @UnrollCTiledConvolution(%arg0: tensor<1x640x32x32xf16, {order = #NHWC}>, %arg1: tensor<640x640x3x3xf16, {order = #NHWC}>) -> tensor<1x640x32x32xf16, {order = #NHWC}> {
    %0 = tensor.empty() : tensor<1x640x32x32xf16, {order = #NHWC}>
    %c0 = arith.constant 0 : index
    %c640 = arith.constant 640 : index
    %c32 = arith.constant 32 : index
    %c96 = arith.constant 96 : index
    %c5 = arith.constant 5 : index
    %1 = scf.for %arg2 = %c0 to %c640 step %c96 iter_args(%arg3 = %0) -> (tensor<1x640x32x32xf16, {order = #NHWC}>) {
      %2 = scf.for %arg4 = %c0 to %c32 step %c5 iter_args(%arg5 = %arg3) -> (tensor<1x640x32x32xf16, {order = #NHWC}>) {
        %c20 = arith.constant 20 : index
        %c4 = arith.constant 4 : index
        %3 = arith.cmpi ult, %arg4, %c20 : index
        %4:2 = scf.if %3 -> (index, index) {
          scf.yield %arg4, %c5 : index, index
        } else {
          %13 = affine.apply #map(%arg4)
          scf.yield %13, %c4 : index, index
        }
        %c640_0 = arith.constant 640 : index
        %5 = affine.min #map1(%arg2)
        %c32_1 = arith.constant 32 : index
        %6 = affine.max #map2(%4#0)
        %7 = affine.max #map3(%4#0)
        %8 = affine.min #map4()[%7]
        %9 = affine.max #map5(%4#1, %6)
        %10 = affine.min #map4()[%9]
        %11 = affine.apply #map6(%4#1, %8, %10)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %6] [1, 640, 32, %11] [1, 1, 1, 1] : tensor<1x640x32x32xf16, {order = #NHWC}> to tensor<1x640x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
        %extracted_slice_2 = tensor.extract_slice %arg1[%arg2, 0, 0, 0] [%5, 640, 3, 3] [1, 1, 1, 1] : tensor<640x640x3x3xf16, {order = #NHWC}> to tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
        %cst_4 = arith.constant 0.000000e+00 : f16
        %padded = tensor.pad %extracted_slice low[0, 0, 1, %8] high[0, 0, 1, %10] {
        ^bb0(%arg6: index, %arg7: index, %arg8: index, %arg9: index):
          tensor.yield %cst_4 : f16
        } : tensor<1x640x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x640x34x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 34, 34]> : tensor<4xsi64>, order = #NHWC}>
        %12 = VPU.NCE.Convolution(%padded, %extracted_slice_2) rawFilterShape [640, 640, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,  strides = [1, 1], tiling_loop_index = 0 : i64} : tensor<1x640x34x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 34, 34]> : tensor<4xsi64>, order = #NHWC}>, tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x?x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %12 into %arg5[0, %arg2, 0, %4#0] [1, %5, 32, %4#1] [1, 1, 1, 1] : tensor<1x?x32x?xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x640x32x32xf16, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x640x32x32xf16, {order = #NHWC}>
      }
      scf.yield %2 : tensor<1x640x32x32xf16, {order = #NHWC}>
    }
    return %1 : tensor<1x640x32x32xf16, {order = #NHWC}>

    //CHECK:    [[SLICE_INPUT0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 640, 32, 6]
    //CHECK:    [[SLICE_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [96, 640, 3, 3]

    //CHECK:    [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE_INPUT0]], [[SLICE_INPUT1]])
    //CHECK: rawFilterShape [640, 640, 3, 3]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 256, 48)>

!weightsType = tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!weightsTypeDDR = tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
!outType = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outTypeDDR = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NCEConvSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NCEConvSOK(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (256) step (48) shared_outs(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)
    %extracted_slice = tensor.extract_slice %cst[%arg1, 0, 0, 0] [%2, 32, 3, 3] [1, 1, 1, 1]
      : tensor<256x32x3x3xf16, {order = #NHWC}> to !weightsTypeDDR
    %3 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
      : tensor<1x32x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %4 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
      : !weightsTypeDDR -> !weightsType

    %5 = VPU.NCE.Convolution(%3, %4) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      ppe = #VPU.PPEStub<>,  strides = [1, 1]
    } : tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>, !weightsType
      -> !outType
    %6 = VPU.Copy(%5) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %6 into %arg2[0, %arg1, 0, 0] [1, %2, 64, 64] [1, 1, 1, 1]
        : !outTypeDDR into tensor<1x256x64x64xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

  // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

  // CHECK:       [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-SAME:      -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:           mode = "DUPLICATED", num_clusters = 6 : i64,
  // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 32, 64, 64], [1, 32, 64, 64], [1, 32, 64, 64], [1, 32, 64, 64], [1, 32, 64, 64], [1, 32, 64, 64]],
  // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:      : tensor<256x32x3x3xf16, {order = #NHWC}>
  // CHECK-SAME:      -> !VPU.DistributedTensor<256x32x3x3xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [6, 1, 1, 1], num_clusters = 6 : i64, alignment = [16, 1, 1, 1],
  // CHECK-SAME{LITERAL}:      compute_shapes = [[48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [16, 32, 3, 3]],
  // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [240, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:      memory_shapes = [[48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [48, 32, 3, 3], [16, 32, 3, 3]],
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0], [144, 0, 0, 0], [192, 0, 0, 0], [240, 0, 0, 0]]}>

  // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[ACT_COPY]], [[WEIGHTS_COPY]])
  // CHECK-SAME:       pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:         -> !VPU.DistributedTensor<1x256x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:               mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1],
  // CHECK-SAME{LITERAL}:      compute_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 16, 64, 64]],
  // CHECK-SAME{LITERAL}:      compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 240, 0, 0]],
  // CHECK-SAME{LITERAL}:      memory_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 48, 64, 64], [1, 16, 64, 64]],
  // CHECK-SAME{LITERAL}:      memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0], [0, 144, 0, 0], [0, 192, 0, 0], [0, 240, 0, 0]]}>

  // CHECK:        [[COPY_OUT:%.+]] = VPU.Copy([[CONV]])
  // CHECK-SAME:       -> tensor<1x256x64x64xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 128, 48)>

!weightsType = tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 16, 1, 1]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!weightsTypeDDR = tensor<?x16x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
!actType = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!actTypeDDR = tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NCEDWConvSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
func.func @NCEDWConvSOK(%arg0: tensor<1x128x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (128) step (48) shared_outs(%arg2 = %0) -> (tensor<1x128x64x64xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)

    %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0, 0] [1, %2, 64, 64] [1, 1, 1, 1]
        : tensor<1x128x64x64xf16, {order = #NHWC}> to !actTypeDDR
    %extracted_slice_0 = tensor.extract_slice %cst[%arg1, 0, 0, 0] [%2, 16, 1, 1] [1, 1, 1, 1]
        : tensor<128x16x1x1xf16, {order = #NHWC}> to !weightsTypeDDR

    %3 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType
    %4 = VPU.Copy(%extracted_slice_0) {out_mem_space = @CMX_NN} : !weightsTypeDDR -> !weightsType

    %5 = VPU.NCE.DepthConvolution(%3, %4) rawFilterShape [128, 1, 3, 3] {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
    } -> !actType
    %6 = VPU.Copy(%5) : !actType -> !actTypeDDR

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %6 into %arg2[0, %arg1, 0, 0] [1, %2, 64, 64] [1, 1, 1, 1] : !actTypeDDR into tensor<1x128x64x64xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x128x64x64xf16, {order = #NHWC}>

  // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>

  // CHECK:       [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:       : tensor<1x128x64x64xf16, {order = #NHWC}>
  // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 32, 64, 64]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 32, 64, 64]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]}>

  // CHECK:       [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:       : tensor<128x16x1x1xf16, {order = #NHWC}>
  // CHECK-SAME:       -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64, alignment = [16, 1, 1, 1],
  // CHECK-SAME{LITERAL}:     compute_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [32, 16, 1, 1]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [32, 16, 1, 1]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0], [96, 0, 0, 0]]}>

  // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[ACT_COPY]], [[WEIGHTS_COPY]])
  // CHECK-SAME:       pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
  // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, alignment = [1, 16, 1, 1],
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 32, 64, 64]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 48, 64, 64], [1, 48, 64, 64], [1, 32, 64, 64]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]}>

  // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[DWCONV]])
  // CHECK-SAME:        -> tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPoolSOK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x12x12xf16, {order = #NHWC}>
func.func @NCEPoolSOK(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x12x12xf16, {order = #NHCW}> {
  %0 = tensor.empty() : tensor<1x64x12x12xf16, {order = #NHCW}>
  %1 = scf.forall (%arg1) = (0) to (64) step (16) shared_outs(%arg2 = %0) -> (tensor<1x64x12x12xf16, {order = #NHCW}>) {
    %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0, 0] [1, 16, 12, 12] [1, 1, 1, 1]
        : tensor<1x64x12x12xf16, {order = #NHWC}> to tensor<1x16x12x12xf16, {order = #NHWC}>
    %2 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : tensor<1x16x12x12xf16, {order = #NHWC}> -> tensor<1x16x12x12xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %3 = VPU.NCE.AveragePool(%2) {
      kernel_size = [2, 2], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, strides = [1, 1]
    } -> tensor<1x16x12x12xf16, {mem_space = @CMX_NN, order = #NHCW}>
    %4 = VPU.Copy(%3)
        : tensor<1x16x12x12xf16, {mem_space = @CMX_NN, order = #NHCW}> -> tensor<1x16x12x12xf16, {order = #NHCW}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[0, %arg1, 0, 0] [1, 16, 12, 12] [1, 1, 1, 1]
        : tensor<1x16x12x12xf16, {order = #NHCW}> into tensor<1x64x12x12xf16, {order = #NHCW}>
    }
  }
  return %1 : tensor<1x64x12x12xf16, {order = #NHCW}>

  // CHECK:       [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:       : tensor<1x64x12x12xf16, {order = #NHWC}>
  // CHECK-SAME:       -> !VPU.DistributedTensor<1x64x12x12xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>

  // CHECK:       [[AVEPOOL:%.+]] = VPU.NCE.AveragePool([[ACT_COPY]])
  // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:      -> !VPU.DistributedTensor<1x64x12x12xf16, #NHCW, @CMX_NN, {
  // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12], [1, 16, 12, 12]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [0, 48, 0, 0]]}>

  // CHECK:       [[COPY_OUT:%.+]] = VPU.Copy([[AVEPOOL]])
  // CHECK-SAME:      -> tensor<1x64x12x12xf16, {order = #NHCW}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 47, 16)>

!actType = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 47, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NCHW}>
!actTypeDDR = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 47, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @NCEPermuteSOKWithChannelAlignment
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x47x16x16xf16>
func.func @NCEPermuteSOKWithChannelAlignment(%arg0: tensor<1x47x16x16xf16>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x48x16x16xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (48) step (16) shared_outs(%arg2 = %0) -> (tensor<1x48x16x16xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)

    %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0, 0] [1, %2, 16, 16] [1, 1, 1, 1]
        : tensor<1x47x16x16xf16> to !actTypeDDR
    %3 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType

    %4 = VPU.NCE.Permute(%3) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

    %5 = VPU.Copy(%4) : tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NHWC}>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg2[0, %arg1, 0, 0] [1, 16, 16, 16] [1, 1, 1, 1]
        : tensor<1x16x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 16]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x48x16x16xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x48x16x16xf16, {order = #NHWC}>

  // CHECK:          [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:        : tensor<1x47x16x16xf16>
  // CHECK-SAME:        -> !VPU.DistributedTensor<1x47x16x16xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 15, 16, 16]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 15, 16, 16]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>

  // CHECK:          [[PERMUTE:%.+]] = VPU.NCE.Permute([[ACT_COPY]])
  // CHECK-SAME:             expandedChannels = 48 : i64
  // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x16x16xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 16, 16, 16], [1, 16, 16, 16], [1, 16, 16, 16]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>

  // CHECK:          [[OUT_COPY:%.+]] = VPU.Copy([[PERMUTE]])
  // CHECK-SAME:          -> tensor<1x48x16x16xf16, {order = #NHWC}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 60, 20)>

!actType = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 60, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NCHW}>
!actTypeDDR = tensor<1x?x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 60, 16, 16]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @NCEPermuteSOKNoChannelAlignment
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x60x16x16xf16>
func.func @NCEPermuteSOKNoChannelAlignment(%arg0: tensor<1x60x16x16xf16>) -> tensor<1x60x16x16xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x60x16x16xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (60) step (20) shared_outs(%arg2 = %0) -> (tensor<1x60x16x16xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)

    %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0, 0] [1, %2, 16, 16] [1, 1, 1, 1]
        : tensor<1x60x16x16xf16> to !actTypeDDR
    %3 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : !actTypeDDR -> !actType

    %4 = VPU.NCE.Permute(%3) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 20 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x20x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

    %5 = VPU.Copy(%4) : tensor<1x20x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x20x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 16, 16]> : tensor<4xsi64>, order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg2[0, %arg1, 0, 0] [1, 20, 16, 16] [1, 1, 1, 1]
          : tensor<1x20x16x16xf16, {bounds = #const.OpaqueI64Elements<[1, 20, 16, 16]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x60x16x16xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x60x16x16xf16, {order = #NHWC}>

// CHECK:           [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:        : tensor<1x60x16x16xf16>
// CHECK-SAME:        -> !VPU.DistributedTensor<1x60x16x16xf16, #NCHW, @CMX_NN, {
// CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 20, 16, 16], [1, 20, 16, 16], [1, 20, 16, 16]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0], [0, 40, 0, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 20, 16, 16], [1, 20, 16, 16], [1, 20, 16, 16]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0], [0, 40, 0, 0]]}>

// CHECK:           [[PERMUTE:%.+]] = VPU.NCE.Permute([[ACT_COPY]])
// CHECK-SAME:             expandedChannels = 60 : i64
// CHECK-SAME:         -> !VPU.DistributedTensor<1x60x16x16xf16, #NHWC, @CMX_NN, {
// CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
// CHECK-SAME{LITERAL}:     compute_shapes = [[1, 20, 16, 16], [1, 20, 16, 16], [1, 20, 16, 16]],
// CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 20, 0, 0], [0, 40, 0, 0]],
// CHECK-SAME{LITERAL}:     memory_shapes = [[1, 20, 16, 16], [1, 20, 16, 16], [1, 20, 16, 16]],
// CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 20, 0, 0], [0, 40, 0, 0]]}>

// CHECK:           [[OUT_COPY:%.+]] = VPU.Copy([[PERMUTE]])
// CHECK-SAME:       -> tensor<1x60x16x16xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 31, 7)>
#map1 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map2 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 * 2 + d1 - 61)>
#map5 = affine_map<(d0, d1, d2) -> (d0 * 2 - d1 - d2 + 3)>

!inTypeDDR = tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
!inType = tensor<1x32x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!inTypePadded = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 66, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!outType = tensor<1x256x?x31xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 31, 31]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outTypeDDR = tensor<1x256x?x31xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 31, 31]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NCEConvSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NCEConvSOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x31x31xf16, {order = #NHWC}> {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = const.Declare tensor<256x32x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x5x5xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x256x31x31xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (31) step (7) shared_outs(%arg2 = %0) -> (tensor<1x256x31x31xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.max #map2(%arg1)
    %5 = affine.min #map3()[%4]
    %6 = affine.max #map4(%2, %3)
    %7 = affine.min #map3()[%6]
    %8 = affine.apply #map5(%2, %5, %7)

    %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, 0] [1, 32, %8, 64] [1, 1, 1, 1]
        : tensor<1x32x64x64xf16, {order = #NHWC}> to !inTypeDDR
    %9 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN} : !inTypeDDR -> !inType

    %padded = tensor.pad %9 low[0, 0, %5, 1] high[0, 0, %7, 1] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f16
    } : !inType to !inTypePadded

    %10 = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN} : tensor<256x32x5x5xf16, {order = #NHWC}>
        -> tensor<256x32x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %11 = VPU.NCE.Convolution(%padded, %10) rawFilterShape [256, 32, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [2, 2]
    } : !inTypePadded, tensor<256x32x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !outType

    %12 = VPU.Copy(%11) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %12 into %arg2[0, 0, %arg1, 0] [1, 256, %2, 31] [1, 1, 1, 1]
          : !outTypeDDR into tensor<1x256x31x31xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x256x31x31xf16, {order = #NHWC}>

  // CHECK:           [[WEIGHTS:%.+]] = const.Declare tensor<256x32x5x5xf16, {order = #NHWC}>
  // CHECK:           [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:         : tensor<1x32x64x64xf16, {order = #NHWC}>
  // CHECK-SAME:         -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:              mode = "OVERLAPPED", num_tiles = [1, 1, 5, 1], num_clusters = 5 : i64,
  // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 16, 64], [1, 32, 17, 64], [1, 32, 17, 64], [1, 32, 17, 64], [1, 32, 9, 64]],
  // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 27, 0], [0, 0, 41, 0], [0, 0, 55, 0]],
  // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 16, 64], [1, 32, 17, 64], [1, 32, 17, 64], [1, 32, 17, 64], [1, 32, 9, 64]],
  // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 13, 0], [0, 0, 27, 0], [0, 0, 41, 0], [0, 0, 55, 0]]}>

  // CHECK:           [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN} : tensor<256x32x5x5xf16, {order = #NHWC}>
  // CHECK-SAME:          -> !VPU.DistributedTensor<256x32x5x5xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:               mode = "DUPLICATED", num_clusters = 5 : i64

  // CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[ACT_COPY]], [[WEIGHTS_COPY]]) rawFilterShape [256, 32, 5, 5] {
  // CHECK-SAME:                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:           : !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 5, 1], num_clusters = 5 : i64
  // CHECK-SAME:             !VPU.DistributedTensor<256x32x5x5xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 5 : i64
  // CHECK-SAME:           -> !VPU.DistributedTensor<1x256x31x31xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                  mode = "OVERLAPPED", num_tiles = [1, 1, 5, 1], num_clusters = 5 : i64,
  // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 3, 31]],
  // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0]],
  // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 7, 31], [1, 256, 3, 31]],
  // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 21, 0], [0, 0, 28, 0]]}>

  // CHECK:           [[OUT_COPY:%.+]] = VPU.Copy([[CONV]])
  // CHECK-SAME:           -> tensor<1x256x31x31xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 32, 11)>
#map1 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map2 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (d0 * 2 - d1)>

!inTypeDDR = tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
!inType = tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!inTypePadded = tensor<1x128x?x65xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 65, 65]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!outType = tensor<1x128x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 32, 32]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outTypeDDR = tensor<1x128x?x32xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 32, 32]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NCEDWConvSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
func.func @NCEDWConvSOH(%arg0: tensor<1x128x64x64xf16, {order = #NHWC}>) -> tensor<1x128x32x32xf16, {order = #NHWC}> {
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = tensor.empty() : tensor<1x128x32x32xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (32) step (11) shared_outs(%arg2 = %0) -> (tensor<1x128x32x32xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)
    %3 = affine.max #map1(%arg1)
    %4 = affine.max #map2(%arg1)
    %5 = affine.min #map3()[%4]
    %6 = affine.apply #map4(%2, %5)

    %extracted_slice = tensor.extract_slice %arg0[0, 0, %3, 0] [1, 128, %6, 64] [1, 1, 1, 1]
        : tensor<1x128x64x64xf16, {order = #NHWC}> to !inTypeDDR
    %7 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : !inTypeDDR -> !inType

    %padded = tensor.pad %7 low[0, 0, %5, 1] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f16
    } : !inType to !inTypePadded

    %8 = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<128x16x1x1xf16, {order = #NHWC}> -> tensor<128x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %9 = VPU.NCE.DepthConvolution(%padded, %8) rawFilterShape [128, 1, 2, 2]{
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [2, 2]
    } -> !outType

    %10 = VPU.Copy(%9) : !outType -> !outTypeDDR

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg2[0, 0, %arg1, 0] [1, 128, %2, 32] [1, 1, 1, 1] : !outTypeDDR into tensor<1x128x32x32xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x128x32x32xf16, {order = #NHWC}>

  // CHECK:             [[WEIGHTS:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
  // CHECK:             [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:            : tensor<1x128x64x64xf16, {order = #NHWC}>
  // CHECK-SAME:            -> !VPU.DistributedTensor<1x128x64x64xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                  mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
  // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 128, 21, 64], [1, 128, 22, 64], [1, 128, 20, 64]],
  // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0]],
  // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 128, 21, 64], [1, 128, 22, 64], [1, 128, 20, 64]],
  // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 43, 0]]}>

  // CHECK:             [[WEIGHTS_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:           : tensor<128x16x1x1xf16, {order = #NHWC}>
  // CHECK-SAME:           -> !VPU.DistributedTensor<128x16x1x1xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                 mode = "DUPLICATED", num_clusters = 3 : i64

  // CHECK:             [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[ACT_COPY]], [[WEIGHTS_COPY]]) rawFilterShape [128, 1, 2, 2] {
  // CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
  // CHECK-SAME:            -> !VPU.DistributedTensor<1x128x32x32xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                  mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
  // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 128, 11, 32], [1, 128, 11, 32], [1, 128, 10, 32]],
  // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]],
  // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 128, 11, 32], [1, 128, 11, 32], [1, 128, 10, 32]],
  // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0]]}>

  // CHECK:             [[OUT_COPY:%.+]] = VPU.Copy([[DWCONV]])
  // CHECK-SAME:            -> tensor<1x128x32x32xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEPoolSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x12x12xf16, {order = #NHWC}>
func.func @NCEPoolSOH(%arg0: tensor<1x64x12x12xf16, {order = #NHWC}>) -> tensor<1x64x12x12xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x64x12x12xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (12) step (3) shared_outs(%arg2 = %0) -> (tensor<1x64x12x12xf16, {order = #NHWC}>) {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0] [1, 64, 3, 12] [1, 1, 1, 1]
        : tensor<1x64x12x12xf16, {order = #NHWC}> to tensor<1x64x3x12xf16, {order = #NHWC}>
    %2 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : tensor<1x64x3x12xf16, {order = #NHWC}> -> tensor<1x64x3x12xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %3 = VPU.NCE.MaxPool(%2) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } -> tensor<1x64x3x12xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %4 = VPU.Copy(%3) : tensor<1x64x3x12xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x3x12xf16, {order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[0, 0, %arg1, 0] [1, 64, 3, 12] [1, 1, 1, 1]
          : tensor<1x64x3x12xf16, {order = #NHWC}> into tensor<1x64x12x12xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x64x12x12xf16, {order = #NHWC}>

  // CHECK:           [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:              : tensor<1x64x12x12xf16, {order = #NHWC}>
  // CHECK-SAME:              -> !VPU.DistributedTensor<1x64x12x12xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                    mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
  // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12]],
  // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
  // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12]],
  // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

  // CHECK:           [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ACT_COPY]]) {
  // CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:           -> !VPU.DistributedTensor<1x64x12x12xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
  // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12]],
  // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
  // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12], [1, 64, 3, 12]],
  // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

  // CHECK:           [[OUT_COPY:%.+]] = VPU.Copy([[MAXPOOL]])
  // CHECK-SAME:          -> tensor<1x64x12x12xf16, {order = #NHWC}>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 16, 6)>

!inTypeDDR = tensor<1x43x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 43, 16, 16]> : tensor<4xsi64>, order = #NCHW}>
!inType = tensor<1x43x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 43, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NCHW}>
!outType = tensor<1x48x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!outTypeDDR = tensor<1x48x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 48, 16, 16]> : tensor<4xsi64>, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x43x16x16xf16>
func.func @NCEPermuteSOH(%arg0: tensor<1x43x16x16xf16>) -> tensor<1x48x16x16xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x48x16x16xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (16) step (6) shared_outs(%arg2 = %0) -> (tensor<1x48x16x16xf16, {order = #NHWC}>) {
    %2 = affine.min #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0] [1, 43, %2, 16] [1, 1, 1, 1]
        : tensor<1x43x16x16xf16> to !inTypeDDR
    %3 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN}
        : !inTypeDDR -> !inType

    %4 = VPU.NCE.Permute(%3) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEStub<>}
        -> !outType

    %5 = VPU.Copy(%4) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg2[0, 0, %arg1, 0] [1, 48, %2, 16] [1, 1, 1, 1]
          : !outTypeDDR into tensor<1x48x16x16xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x48x16x16xf16, {order = #NHWC}>

  // CHECK:             [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:          : tensor<1x43x16x16xf16>
  // CHECK-SAME:          -> !VPU.DistributedTensor<1x43x16x16xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:                mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
  // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 43, 6, 16], [1, 43, 6, 16], [1, 43, 4, 16]],
  // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]],
  // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 43, 6, 16], [1, 43, 6, 16], [1, 43, 4, 16]],
  // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]]}>

  // CHECK:             [[PERMUTE:%.+]] = VPU.NCE.Permute([[ACT_COPY]])
  // CHECK-SAME:              expandedChannels = 48 : i64
  // CHECK-SAME:          -> !VPU.DistributedTensor<1x48x16x16xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
  // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 48, 6, 16], [1, 48, 6, 16], [1, 48, 4, 16]],
  // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]],
  // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 48, 6, 16], [1, 48, 6, 16], [1, 48, 4, 16]],
  // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 12, 0]]}>

  // CHECK:             [[OUT_COPY:%.+]] = VPU.Copy([[PERMUTE]])
  // CHECK-SAME:            -> tensor<1x48x16x16xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 4)>

// CHECK-LABEL: @DepthToSpaceSOH
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x12x270xf16, {order = #NHWC}>
func.func @DepthToSpaceSOH(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x8x48x1080xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (48) step (12) shared_outs(%arg2 = %0) -> (tensor<1x8x48x1080xf16, {order = #NHWC}>) {
    %2 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %2, 0] [1, 128, 3, 270] [1, 1, 1, 1]
        : tensor<1x128x12x270xf16, {order = #NHWC}> to tensor<1x128x3x270xf16, {order = #NHWC}>
    %3 = VPU.DepthToSpace(%extracted_slice) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : tensor<1x128x3x270xf16, {order = #NHWC}> -> tensor<1x8x12x1080xf16, {order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[0, 0, %arg1, 0] [1, 8, 12, 1080] [1, 1, 1, 1]
          : tensor<1x8x12x1080xf16, {order = #NHWC}> into tensor<1x8x48x1080xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x8x48x1080xf16, {order = #NHWC}>

  // CHECK:              [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:            : tensor<1x128x12x270xf16, {order = #NHWC}>
  // CHECK-SAME:            -> !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
  // CHECK-SAME{LITERAL}:       compute_shapes = [[1, 128, 3, 270], [1, 128, 3, 270], [1, 128, 3, 270], [1, 128, 3, 270]],
  // CHECK-SAME{LITERAL}:       compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]],
  // CHECK-SAME{LITERAL}:       memory_shapes = [[1, 128, 3, 270], [1, 128, 3, 270], [1, 128, 3, 270], [1, 128, 3, 270]],
  // CHECK-SAME{LITERAL}:       memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 6, 0], [0, 0, 9, 0]]}>

  // CHECK:              [[D2S:%.+]] = VPU.DepthToSpace([[ACT_COPY]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
  // CHECK-SAME:               : !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                    mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
  // CHECK-SAME:               -> !VPU.DistributedTensor<1x8x48x1080xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                    mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
  // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 12, 1080], [1, 8, 12, 1080], [1, 8, 12, 1080], [1, 8, 12, 1080]],
  // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0], [0, 0, 36, 0]],
  // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 12, 1080], [1, 8, 12, 1080], [1, 8, 12, 1080], [1, 8, 12, 1080]],
  // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 12, 0], [0, 0, 24, 0], [0, 0, 36, 0]]}>

  // CHECK:              [[OUT_COPY:%.+]] = VPU.Copy([[D2S]])
  // CHECK-SAME:              -> tensor<1x8x48x1080xf16, {order = #NHWC}>
}


// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (d0 floordiv 4)>

// CHECK-LABEL: @DepthToSpaceSOW
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x12x270xf16, {order = #NHWC}>
func.func @DepthToSpaceSOW(%arg0: tensor<1x128x12x270xf16, {order = #NHWC}>) -> tensor<1x8x48x1080xf16, {order = #NHWC}> {
  %0 = tensor.empty() : tensor<1x8x48x1080xf16, {order = #NHWC}>
  %1 = scf.forall (%arg1) = (0) to (1080) step (216) shared_outs(%arg2 = %0) -> (tensor<1x8x48x1080xf16, {order = #NHWC}>) {
    %2 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %2] [1, 128, 12, 54] [1, 1, 1, 1]
        : tensor<1x128x12x270xf16, {order = #NHWC}> to tensor<1x128x12x54xf16, {order = #NHWC}>
    %3 = VPU.DepthToSpace(%extracted_slice) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : tensor<1x128x12x54xf16, {order = #NHWC}> -> tensor<1x8x48x216xf16, {order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[0, 0, 0, %arg1] [1, 8, 48, 216] [1, 1, 1, 1]
          : tensor<1x8x48x216xf16, {order = #NHWC}> into tensor<1x8x48x1080xf16, {order = #NHWC}>
    }
  }
  return %1 : tensor<1x8x48x1080xf16, {order = #NHWC}>

  // CHECK:          [[ACT_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:             : tensor<1x128x12x270xf16, {order = #NHWC}>
  // CHECK-SAME:             -> !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                    mode = "SEGMENTED", num_tiles = [1, 1, 1, 5], num_clusters = 5 : i64,
  // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54]],
  // CHECK-SAME{LITERAL}:           compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 54], [0, 0, 0, 108], [0, 0, 0, 162], [0, 0, 0, 216]],
  // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54], [1, 128, 12, 54]],
  // CHECK-SAME{LITERAL}:           memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 54], [0, 0, 0, 108], [0, 0, 0, 162], [0, 0, 0, 216]]}>

  // CHECK:          [[D2S:%.+]] = VPU.DepthToSpace([[ACT_COPY]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
  // CHECK-SAME:            : !VPU.DistributedTensor<1x128x12x270xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                  mode = "SEGMENTED", num_tiles = [1, 1, 1, 5], num_clusters = 5 : i64
  // CHECK-SAME:            -> !VPU.DistributedTensor<1x8x48x1080xf16, #NHWC, @CMX_NN, {
  // CHECK-SAME:                  mode = "SEGMENTED", num_tiles = [1, 1, 1, 5], num_clusters = 5 : i64,
  // CHECK-SAME{LITERAL}:         compute_shapes = [[1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216]],
  // CHECK-SAME{LITERAL}:         compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 216], [0, 0, 0, 432], [0, 0, 0, 648], [0, 0, 0, 864]],
  // CHECK-SAME{LITERAL}:         memory_shapes = [[1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216], [1, 8, 48, 216]],
  // CHECK-SAME{LITERAL}:         memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 216], [0, 0, 0, 432], [0, 0, 0, 648], [0, 0, 0, 864]]}>
  // CHECK:          [[OUT_COPY:%.+]] = VPU.Copy([[D2S]])
  // CHECK-SAME:            -> tensor<1x8x48x1080xf16, {order = #NHWC}>
}

// -----

// Verify multiclustering unroll for LSTMGates (multi-output SW op).
// The scf.forall splits the H dimension (dim 2) across 4 clusters.
// Both outputs (hiddenState, cellState) must be distributed and copied out.

#map2 = affine_map<(d0) -> (-d0 + 128, 32)>

// CHECK-LABEL: @UnrollMCLSTMGates
// CHECK-SAME:       [[GATES:%[^:]+]]: tensor<1x1x128x512xf16>
// CHECK-SAME:       [[CELL:%[^:]+]]: tensor<1x1x128x128xf16>
func.func @UnrollMCLSTMGates(
    %arg0: tensor<1x1x128x512xf16>,
    %arg1: tensor<1x1x128x128xf16>
) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
  %out_h = tensor.empty() : tensor<1x1x128x128xf16>
  %out_c = tensor.empty() : tensor<1x1x128x128xf16>

  %result:2 = scf.forall (%iv) = (0) to (128) step (32)
      shared_outs(%sh_h = %out_h, %sh_c = %out_c)
      -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
    %tile_size = affine.min #map2(%iv)

    %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, %tile_size, 512] [1, 1, 1, 1]
        : tensor<1x1x128x512xf16>
          to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>}>
    %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, %tile_size, 128] [1, 1, 1, 1]
        : tensor<1x1x128x128xf16>
          to tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>

    %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell)
        : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 512]> : tensor<4xsi64>}>,
          tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>
       -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>,
          tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %h into %sh_h[0, 0, %iv, 0] [1, 1, %tile_size, 128] [1, 1, 1, 1]
          : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>
            into tensor<1x1x128x128xf16>
      tensor.parallel_insert_slice %c into %sh_c[0, 0, %iv, 0] [1, 1, %tile_size, 128] [1, 1, 1, 1]
          : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 128, 128]> : tensor<4xsi64>}>
            into tensor<1x1x128x128xf16>
    }
  }
  return %result#0, %result#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
}

// CHECK-NOT: scf.forall

// Input copies: both inputs are distributed (SEGMENTED on H, 4 clusters)
// CHECK:      [[GATES_COPY:%.+]] = VPU.Copy([[GATES]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:     : tensor<1x1x128x512xf16>
// CHECK-SAME:     -> !VPU.DistributedTensor<1x1x128x512xf16, #NCHW, @CMX_NN, {
// CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

// CHECK:      [[CELL_COPY:%.+]] = VPU.Copy([[CELL]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:     : tensor<1x1x128x128xf16>
// CHECK-SAME:     -> !VPU.DistributedTensor<1x1x128x128xf16, #NCHW, @CMX_NN, {
// CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
// CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128]],
// CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

// Compute op with distributed outputs
// CHECK:      [[H_DIST:%.+]], [[C_DIST:%.+]] = VPU.LSTMGates([[GATES_COPY]], [[CELL_COPY]])
// CHECK-SAME:     -> !VPU.DistributedTensor<1x1x128x128xf16, #NCHW, @CMX_NN, {
// CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
// CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128]],
// CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]

// Output copies: one per result
// CHECK:      [[H_OUT:%.+]] = VPU.Copy([[H_DIST]])
// CHECK-SAME:     -> tensor<1x1x128x128xf16>
// CHECK:      [[C_OUT:%.+]] = VPU.Copy([[C_DIST]])
// CHECK-SAME:     -> tensor<1x1x128x128xf16>

// CHECK:      return [[H_OUT]], [[C_OUT]]

// -----

// CHECK: func.func @UnrollLSTMGatesNoMC([[GATES:%.+]]: tensor<1x1x128x512xf16>,
// CHECK-SAME:                           [[CELL:%.+]]: tensor<1x1x128x128xf16>)
func.func @UnrollLSTMGatesNoMC(
    %arg0: tensor<1x1x128x512xf16>,
    %arg1: tensor<1x1x128x128xf16>
) -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {
  %c0   = arith.constant 0   : index
  %c128 = arith.constant 128 : index
  %c32  = arith.constant 32  : index
  %out0 = tensor.empty() : tensor<1x1x128x128xf16>
  %out1 = tensor.empty() : tensor<1x1x128x128xf16>

  // Tiling loop: 4 tiles of size 32 along the H dimension (dim 2). No multiClusterStrategy is set.
  %result:2 = scf.for %iv = %c0 to %c128 step %c32
      iter_args(%acc_h = %out0, %acc_c = %out1)
      -> (tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>) {

    %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, 32, 512] [1, 1, 1, 1]
        : tensor<1x1x128x512xf16> to tensor<1x1x32x512xf16>
    %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
        : tensor<1x1x128x128xf16> to tensor<1x1x32x128xf16>

    %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell)
        : tensor<1x1x32x512xf16>, tensor<1x1x32x128xf16>
       -> tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>

    %ins_h = tensor.insert_slice %h into %acc_h[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
        : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>
    %ins_c = tensor.insert_slice %c into %acc_c[0, 0, %iv, 0] [1, 1, 32, 128] [1, 1, 1, 1]
        : tensor<1x1x32x128xf16> into tensor<1x1x128x128xf16>

    scf.yield %ins_h, %ins_c : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>
  }

  return %result#0, %result#1 : tensor<1x1x128x128xf16>, tensor<1x1x128x128xf16>

  // CHECK-NOT: scf.for
  // CHECK-NOT: tensor.extract_slice

  // CHECK: [[GATES0:%.+]] = VPU.Slice [[GATES]] [0, 0, 0, 0] [1, 1, 32, 512]
  // CHECK: [[CELL0:%.+]]  = VPU.Slice [[CELL]]  [0, 0, 0, 0] [1, 1, 32, 128]
  // CHECK: [[H0:%.+]], [[C0:%.+]] = VPU.LSTMGates([[GATES0]], [[CELL0]])
  // CHECK-NOT: multiClusterStrategy
  // CHECK: [[GATES1:%.+]] = VPU.Slice [[GATES]] [0, 0, 32, 0] [1, 1, 32, 512]
  // CHECK: [[CELL1:%.+]]  = VPU.Slice [[CELL]]  [0, 0, 32, 0] [1, 1, 32, 128]
  // CHECK: [[H1:%.+]], [[C1:%.+]] = VPU.LSTMGates([[GATES1]], [[CELL1]])
  // CHECK: [[GATES2:%.+]] = VPU.Slice [[GATES]] [0, 0, 64, 0] [1, 1, 32, 512]
  // CHECK: [[CELL2:%.+]]  = VPU.Slice [[CELL]]  [0, 0, 64, 0] [1, 1, 32, 128]
  // CHECK: [[H2:%.+]], [[C2:%.+]] = VPU.LSTMGates([[GATES2]], [[CELL2]])
  // CHECK: [[GATES3:%.+]] = VPU.Slice [[GATES]] [0, 0, 96, 0] [1, 1, 32, 512]
  // CHECK: [[CELL3:%.+]]  = VPU.Slice [[CELL]]  [0, 0, 96, 0] [1, 1, 32, 128]
  // CHECK: [[H3:%.+]], [[C3:%.+]] = VPU.LSTMGates([[GATES3]], [[CELL3]])

  // CHECK-NOT: tensor.insert_slice

  // CHECK: [[CONCAT_H:%.+]] = VPU.Concat([[H0]], [[H1]], [[H2]], [[H3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
  // CHECK: [[CONCAT_C:%.+]] = VPU.Concat([[C0]], [[C1]], [[C2]], [[C3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_tile = affine_map<(d0) -> (-d0 + 1024, 205)>
#map_mc = affine_map<(d0) -> (-d0 + 205, 35)>

// CHECK-LABEL: @MCLSTMGatesLargeShape
// CHECK-SAME:         [[ARG0:%.+]]: tensor<1x1x1024x2048xf16>
// CHECK-SAME:         [[ARG1:%.+]]: tensor<1x1x1024x512xf16>
  func.func @MCLSTMGatesLargeShape(%arg0: tensor<1x1x1024x2048xf16>, %arg1: tensor<1x1x1024x512xf16>) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c205 = arith.constant 205 : index
    %0 = tensor.empty() : tensor<1x1x1024x512xf16>
    %1:2 = scf.for %arg2 = %c0 to %c1024 step %c205 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>) {
      %tile_sz = affine.min #map_tile(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, 0] [1, 1, %tile_sz, 2048] [1, 1, 1, 1] : tensor<1x1x1024x2048xf16> to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %arg2, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1] : tensor<1x1x1024x512xf16> to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
      %mc_step = affine.apply affine_map<(d0) -> (d0 ceildiv 6)>(%tile_sz)
      %2 = tensor.empty(%tile_sz) : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
      %3:2 = scf.forall (%arg5) = (0) to (%tile_sz) step (%mc_step) shared_outs(%arg6 = %2, %arg7 = %2) -> (tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>) {
        %4 = affine.min affine_map<(d0, d1)[s0] -> (-d0 + s0, d1 ceildiv 6)>(%arg5, %tile_sz)[%tile_sz]
        %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0, %arg5, 0] [1, 1, %4, 2048] [1, 1, 1, 1] : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, 0, %arg5, 0] [1, 1, %4, 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
        %outputHiddenState, %outputCellState = VPU.LSTMGates(%extracted_slice_2, %extracted_slice_3) : tensor<1x1x?x2048xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 2048]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %outputHiddenState into %arg6[0, 0, %arg5, 0] [1, 1, %4, 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
          tensor.parallel_insert_slice %outputCellState into %arg7[0, 0, %arg5, 0] [1, 1, %4, 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}>
        }
      }
      %ins_h = tensor.insert_slice %3#0 into %arg3[0, 0, %arg2, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x512xf16>
      %ins_c = tensor.insert_slice %3#1 into %arg4[0, 0, %arg2, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1] : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 512]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x512xf16>
      scf.yield %ins_h, %ins_c : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
    }
    return %1#0, %1#1 : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>

  // CHECK-NOT: scf.for
  // CHECK-NOT: scf.forall

  // Shared output accumulator (uninitialised; filled by insert_slice chains).
  // CHECK: [[ACC:%.+]] = tensor.empty() : tensor<1x1x1024x512xf16>

  // CHECK: [[G0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 1, 205, 2048]
  // CHECK: [[S0:%.+]] = VPU.Slice [[ARG1]] [0, 0, 0, 0] [1, 1, 205, 512]
  // CHECK: [[G0_DIST:%.+]] = VPU.Copy([[G0]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x1x205x2048xf16>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x2048xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 30, 2048]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 140, 0], [0, 0, 175, 0]],
  // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 35, 2048], [1, 1, 30, 2048]],
  // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 140, 0], [0, 0, 175, 0]]}>
  // CHECK: [[S0_DIST:%.+]] = VPU.Copy([[S0]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x1x205x512xf16>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x512xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 30, 512]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 140, 0], [0, 0, 175, 0]],
  // CHECK-SAME{LITERAL}: memory_shapes = [[1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 30, 512]],
  // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 35, 0], [0, 0, 70, 0], [0, 0, 105, 0], [0, 0, 140, 0], [0, 0, 175, 0]]}>
  // CHECK: [[H0_DIST:%.+]], [[C0_DIST:%.+]] = VPU.LSTMGates([[G0_DIST]], [[S0_DIST]])
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x512xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 35, 512], [1, 1, 30, 512]]
  // CHECK: [[H0_CPY:%.+]] = VPU.Copy([[H0_DIST]])
  // CHECK-SAME:     -> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast [[H0_CPY]]
  // CHECK: [[C0_CPY:%.+]] = VPU.Copy([[C0_DIST]])
  // CHECK-SAME:     -> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast [[C0_CPY]]
  // CHECK: tensor.insert_slice {{.*}} into [[ACC]][0, 0, {{.*}}, 0] [1, 1, {{.*}}, 512]
  // CHECK: tensor.insert_slice {{.*}} into [[ACC]][0, 0, {{.*}}, 0] [1, 1, {{.*}}, 512]

  // CHECK: VPU.Slice [[ARG0]] [0, 0, 205, 0] [1, 1, 205, 2048]
  // CHECK: VPU.Slice [[ARG1]] [0, 0, 205, 0] [1, 1, 205, 512]
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.LSTMGates
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast

  // CHECK: VPU.Slice [[ARG0]] [0, 0, 410, 0] [1, 1, 205, 2048]
  // CHECK: VPU.Slice [[ARG1]] [0, 0, 410, 0] [1, 1, 205, 512]
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.LSTMGates
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast

  // CHECK: VPU.Slice [[ARG0]] [0, 0, 615, 0] [1, 1, 205, 2048]
  // CHECK: VPU.Slice [[ARG1]] [0, 0, 615, 0] [1, 1, 205, 512]
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x2048xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy{{.*}}{out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x205x512xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.LSTMGates
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast
  // CHECK: VPU.Copy{{.*}}-> tensor<1x1x205x512xf16>
  // CHECK: tensor.cast

  // CHECK: [[G4:%.+]] = VPU.Slice [[ARG0]] [0, 0, 820, 0] [1, 1, 204, 2048]
  // CHECK: [[S4:%.+]] = VPU.Slice [[ARG1]] [0, 0, 820, 0] [1, 1, 204, 512]
  // CHECK: [[G4_DIST:%.+]] = VPU.Copy([[G4]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x1x204x2048xf16>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x204x2048xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 34, 2048], [1, 1, 34, 2048], [1, 1, 34, 2048], [1, 1, 34, 2048], [1, 1, 34, 2048], [1, 1, 34, 2048]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 34, 0], [0, 0, 68, 0], [0, 0, 102, 0], [0, 0, 136, 0], [0, 0, 170, 0]]
  // CHECK: [[S4_DIST:%.+]] = VPU.Copy([[S4]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x1x204x512xf16>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x204x512xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 34, 0], [0, 0, 68, 0], [0, 0, 102, 0], [0, 0, 136, 0], [0, 0, 170, 0]]
  // CHECK: [[H4_DIST:%.+]], [[C4_DIST:%.+]] = VPU.LSTMGates([[G4_DIST]], [[S4_DIST]])
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x204x512xf16, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512], [1, 1, 34, 512]]
  // CHECK: [[H4_CPY:%.+]] = VPU.Copy([[H4_DIST]])
  // CHECK-SAME:     -> tensor<1x1x204x512xf16>
  // CHECK: tensor.cast [[H4_CPY]]
  // CHECK: [[C4_CPY:%.+]] = VPU.Copy([[C4_DIST]])
  // CHECK-SAME:     -> tensor<1x1x204x512xf16>
  // CHECK: tensor.cast [[C4_CPY]]

  // CHECK: [[H_OUT:%.+]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0, {{.*}}, 0] [1, 1, {{.*}}, 512]
  // CHECK: [[C_OUT:%.+]] = tensor.insert_slice {{.*}} into {{.*}}[0, 0, {{.*}}, 0] [1, 1, {{.*}}, 512]
  // CHECK: return [[H_OUT]], [[C_OUT]] : tensor<1x1x1024x512xf16>, tensor<1x1x1024x512xf16>
  }

// -----

// Verify multiclustering unroll for TopK (multi-output SW op).
// The scf.forall splits the H dimension (dim 2) across 4 clusters.
// Both outputs (values, indices) must be distributed and copied out.

#map3 = affine_map<(d0) -> (-d0 + 128, 32)>

// CHECK-LABEL: @UnrollMCTopK
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x64x128x128xf32>
func.func @UnrollMCTopK(
    %arg0: tensor<1x64x128x128xf32>
) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
  %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
  %out_vals = tensor.empty() : tensor<1x8x128x128xf32>
  %out_inds = tensor.empty() : tensor<1x8x128x128xsi32>

  %result:2 = scf.forall (%iv) = (0) to (128) step (32)
      shared_outs(%sh_v = %out_vals, %sh_i = %out_inds)
      -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
    %tile_size = affine.min #map3(%iv)

    %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, %tile_size, 128] [1, 1, 1, 1]
        : tensor<1x64x128x128xf32>
          to tensor<1x64x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 128]> : tensor<4xsi64>}>

    %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
        axis = 1 : i64,
        element_type = si32,
        k_value = 8 : i64,
        mode = #IE.topk_mode<MAX>,
        sort = #IE.topk_sort_type<SORT_INDICES>
    } : tensor<1x64x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 64, 128, 128]> : tensor<4xsi64>}>,
        tensor<1x1x1x1024xui8>
     -> tensor<1x8x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 128]> : tensor<4xsi64>}>,
        tensor<1x8x?x128xsi32, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 128]> : tensor<4xsi64>}>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %vals into %sh_v[0, 0, %iv, 0] [1, 8, %tile_size, 128] [1, 1, 1, 1]
          : tensor<1x8x?x128xf32, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 128]> : tensor<4xsi64>}>
            into tensor<1x8x128x128xf32>
      tensor.parallel_insert_slice %inds into %sh_i[0, 0, %iv, 0] [1, 8, %tile_size, 128] [1, 1, 1, 1]
          : tensor<1x8x?x128xsi32, {bounds = #const.OpaqueI64Elements<[1, 8, 128, 128]> : tensor<4xsi64>}>
            into tensor<1x8x128x128xsi32>
    }
  }

  return %result#0, %result#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

  // CHECK-NOT: scf.forall

  // Input copy: activation is distributed (SEGMENTED on H, 4 clusters)
  // CHECK:      [[IN_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x64x128x128xf32>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x64x128x128xf32, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 64, 32, 128], [1, 64, 32, 128], [1, 64, 32, 128], [1, 64, 32, 128]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
  // CHECK-SAME{LITERAL}: memory_shapes = [[1, 64, 32, 128], [1, 64, 32, 128], [1, 64, 32, 128], [1, 64, 32, 128]],
  // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

  // k_buf: DUPLICATED (non-tiled, constant input)
  // CHECK:      [[K_COPY:%.+]] = VPU.Copy
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1x1024xui8, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "DUPLICATED"

  // Compute op with distributed outputs (values and indices)
  // CHECK:      [[V_DIST:%.+]], [[I_DIST:%.+]] = VPU.TopK([[IN_COPY]], [[K_COPY]])
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x8x128x128xf32, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 8, 32, 128], [1, 8, 32, 128], [1, 8, 32, 128], [1, 8, 32, 128]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]

  // Output copies: one per result
  // CHECK:      [[V_OUT:%.+]] = VPU.Copy([[V_DIST]])
  // CHECK-SAME:     -> tensor<1x8x128x128xf32>
  // CHECK:      [[I_OUT:%.+]] = VPU.Copy([[I_DIST]])
  // CHECK-SAME:     -> tensor<1x8x128x128xsi32>

  // CHECK:      return [[V_OUT]], [[I_OUT]]
}

// -----

// CHECK: func.func @UnrollTopKNoMC([[INPUT:%.+]]: tensor<1x64x128x128xf32>)
func.func @UnrollTopKNoMC(
    %arg0: tensor<1x64x128x128xf32>
) -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {
  %c0   = arith.constant 0   : index
  %c128 = arith.constant 128 : index
  %c32  = arith.constant 32  : index
  %k_buf = VPU.Empty : tensor<1x1x1x1024xui8>
  %out_vals = tensor.empty() : tensor<1x8x128x128xf32>
  %out_inds = tensor.empty() : tensor<1x8x128x128xsi32>

  // Tiling loop: 4 tiles of size 32 along the H dimension (dim 2). No multiClusterStrategy is set.
  %result:2 = scf.for %iv = %c0 to %c128 step %c32
      iter_args(%acc_v = %out_vals, %acc_i = %out_inds)
      -> (tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>) {

    %slice_in = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 64, 32, 128] [1, 1, 1, 1]
        : tensor<1x64x128x128xf32> to tensor<1x64x32x128xf32>

    %vals, %inds = VPU.TopK(%slice_in, %k_buf) {
        axis = 1 : i64,
        element_type = si32,
        k_value = 8 : i64,
        mode = #IE.topk_mode<MAX>,
        sort = #IE.topk_sort_type<SORT_INDICES>
    } : tensor<1x64x32x128xf32>, tensor<1x1x1x1024xui8>
      -> tensor<1x8x32x128xf32>, tensor<1x8x32x128xsi32>

    %ins_v = tensor.insert_slice %vals into %acc_v[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
        : tensor<1x8x32x128xf32> into tensor<1x8x128x128xf32>
    %ins_i = tensor.insert_slice %inds into %acc_i[0, 0, %iv, 0] [1, 8, 32, 128] [1, 1, 1, 1]
        : tensor<1x8x32x128xsi32> into tensor<1x8x128x128xsi32>

    scf.yield %ins_v, %ins_i : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>
  }

  return %result#0, %result#1 : tensor<1x8x128x128xf32>, tensor<1x8x128x128xsi32>

  // CHECK-NOT: scf.for
  // CHECK-NOT: tensor.extract_slice

  // k_buf is defined once; all four TopK tiles share it.
  // CHECK: [[K_BUF:%.+]] = VPU.Empty : tensor<1x1x1x1024xui8>

  // CHECK: [[IN0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 32, 128]
  // CHECK: [[V0:%.+]], [[I0:%.+]] = VPU.TopK([[IN0]], [[K_BUF]])
  // CHECK-SAME: axis = 1 : i64, element_type = si32, k_value = 8 : i64
  // CHECK-NOT: multiClusterStrategy
  // CHECK: [[IN1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 32, 0] [1, 64, 32, 128]
  // CHECK: [[V1:%.+]], [[I1:%.+]] = VPU.TopK([[IN1]], [[K_BUF]])
  // CHECK: [[IN2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 64, 0] [1, 64, 32, 128]
  // CHECK: [[V2:%.+]], [[I2:%.+]] = VPU.TopK([[IN2]], [[K_BUF]])
  // CHECK: [[IN3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 96, 0] [1, 64, 32, 128]
  // CHECK: [[V3:%.+]], [[I3:%.+]] = VPU.TopK([[IN3]], [[K_BUF]])

  // CHECK-NOT: tensor.insert_slice

  // CHECK: [[VALS:%.+]] = VPU.Concat([[V0]], [[V1]], [[V2]], [[V3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
  // CHECK: [[INDS:%.+]] = VPU.Concat([[I0]], [[I1]], [[I2]], [[I3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0) -> (-d0 + 1024, 171)>

// CHECK-LABEL: @MCTopKLargeShape
// CHECK-SAME:         [[INPUT:%.+]]: tensor<1x512x4096x4096xf32>
func.func @MCTopKLargeShape(%arg0: tensor<1x512x4096x4096xf32>) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>) {
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %0 = VPU.Empty : tensor<1x1x1x8192xui8>
    %1 = tensor.empty() : tensor<1x1x4096x4096xf32>
    %2 = tensor.empty() : tensor<1x1x4096x4096xsi32>
    %3:2 = scf.for %arg1 = %c0 to %c4096 step %c1024 iter_args(%arg2 = %1, %arg3 = %2) -> (tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0] [1, 512, 1024, 4096] [1, 1, 1, 1] : tensor<1x512x4096x4096xf32> to tensor<1x512x1024x4096xf32>
      %4 = tensor.empty() : tensor<1x1x1024x4096xf32>
      %5 = tensor.empty() : tensor<1x1x1024x4096xsi32>
      %6:2 = scf.forall (%arg4) = (0) to (1024) step (171) shared_outs(%arg5 = %4, %arg6 = %5) -> (tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xsi32>) {
        %7 = affine.min #map(%arg4)
        %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, 0, %arg4, 0] [1, 512, %7, 4096] [1, 1, 1, 1] : tensor<1x512x1024x4096xf32> to tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>
        %output_values, %output_indices = VPU.TopK(%extracted_slice_1, %0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x512x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 512, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x1x8192xui8> -> tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %output_values into %arg5[0, 0, %arg4, 0] [1, 1, %7, 4096] [1, 1, 1, 1] : tensor<1x1x?x4096xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x4096xf32>
          tensor.parallel_insert_slice %output_indices into %arg6[0, 0, %arg4, 0] [1, 1, %7, 4096] [1, 1, 1, 1] : tensor<1x1x?x4096xsi32, {bounds = #const.OpaqueI64Elements<[1, 1, 1024, 4096]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x1x1024x4096xsi32>
        }
      }
      %inserted_slice = tensor.insert_slice %6#0 into %arg2[0, 0, %arg1, 0] [1, 1, 1024, 4096] [1, 1, 1, 1] : tensor<1x1x1024x4096xf32> into tensor<1x1x4096x4096xf32>
      %inserted_slice_0 = tensor.insert_slice %6#1 into %arg3[0, 0, %arg1, 0] [1, 1, 1024, 4096] [1, 1, 1, 1] : tensor<1x1x1024x4096xsi32> into tensor<1x1x4096x4096xsi32>
      scf.yield %inserted_slice, %inserted_slice_0 : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>
    }
    return %3#0, %3#1 : tensor<1x1x4096x4096xf32>, tensor<1x1x4096x4096xsi32>

  // CHECK-NOT: scf.for
  // CHECK-NOT: scf.forall

  // k_buf is defined once; all four TopK tiles share it.
  // CHECK: [[K_BUF:%.+]] = VPU.Empty : tensor<1x1x1x8192xui8>

  // Tile 0: H offset 0
  // CHECK: [[IN0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 1024, 4096]
  // CHECK: [[IN0_DIST:%.+]] = VPU.Copy([[IN0]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     : tensor<1x512x1024x4096xf32>
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x512x1024x4096xf32, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 169, 4096]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 171, 0], [0, 0, 342, 0], [0, 0, 513, 0], [0, 0, 684, 0], [0, 0, 855, 0]],
  // CHECK-SAME{LITERAL}: memory_shapes = [[1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 171, 4096], [1, 512, 169, 4096]],
  // CHECK-SAME{LITERAL}: memory_offsets = [[0, 0, 0, 0], [0, 0, 171, 0], [0, 0, 342, 0], [0, 0, 513, 0], [0, 0, 684, 0], [0, 0, 855, 0]]}>
  // CHECK: [[K0_DIST:%.+]] = VPU.Copy([[K_BUF]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1x8192xui8, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "DUPLICATED", num_clusters = 6 : i64
  // CHECK: [[V0_DIST:%.+]], [[I0_DIST:%.+]] = VPU.TopK([[IN0_DIST]], [[K0_DIST]])
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1024x4096xf32, #NCHW, @CMX_NN, {
  // CHECK-SAME:          mode = "SEGMENTED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64
  // CHECK-SAME{LITERAL}: compute_shapes = [[1, 1, 171, 4096], [1, 1, 171, 4096], [1, 1, 171, 4096], [1, 1, 171, 4096], [1, 1, 171, 4096], [1, 1, 169, 4096]],
  // CHECK-SAME{LITERAL}: compute_offsets = [[0, 0, 0, 0], [0, 0, 171, 0], [0, 0, 342, 0], [0, 0, 513, 0], [0, 0, 684, 0], [0, 0, 855, 0]]
  // CHECK: [[V0:%.+]] = VPU.Copy([[V0_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xf32>
  // CHECK: [[I0:%.+]] = VPU.Copy([[I0_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xsi32>

  // Tile 1: H offset 1024
  // CHECK: [[IN1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 1024, 0] [1, 512, 1024, 4096]
  // CHECK: [[IN1_DIST:%.+]] = VPU.Copy([[IN1]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x512x1024x4096xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy([[K_BUF]]) {out_mem_space = @CMX_NN}
  // CHECK: [[V1_DIST:%.+]], [[I1_DIST:%.+]] = VPU.TopK([[IN1_DIST]],
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1024x4096xf32
  // CHECK: [[V1:%.+]] = VPU.Copy([[V1_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xf32>
  // CHECK: [[I1:%.+]] = VPU.Copy([[I1_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xsi32>

  // Tile 2: H offset 2048
  // CHECK: [[IN2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 2048, 0] [1, 512, 1024, 4096]
  // CHECK: [[IN2_DIST:%.+]] = VPU.Copy([[IN2]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x512x1024x4096xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy([[K_BUF]]) {out_mem_space = @CMX_NN}
  // CHECK: [[V2_DIST:%.+]], [[I2_DIST:%.+]] = VPU.TopK([[IN2_DIST]],
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1024x4096xf32
  // CHECK: [[V2:%.+]] = VPU.Copy([[V2_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xf32>
  // CHECK: [[I2:%.+]] = VPU.Copy([[I2_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xsi32>

  // Tile 3: H offset 3072
  // CHECK: [[IN3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 3072, 0] [1, 512, 1024, 4096]
  // CHECK: [[IN3_DIST:%.+]] = VPU.Copy([[IN3]]) {out_mem_space = @CMX_NN}
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x512x1024x4096xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED"
  // CHECK: VPU.Copy([[K_BUF]]) {out_mem_space = @CMX_NN}
  // CHECK: [[V3_DIST:%.+]], [[I3_DIST:%.+]] = VPU.TopK([[IN3_DIST]],
  // CHECK-SAME:     -> !VPU.DistributedTensor<1x1x1024x4096xf32
  // CHECK: [[V3:%.+]] = VPU.Copy([[V3_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xf32>
  // CHECK: [[I3:%.+]] = VPU.Copy([[I3_DIST]])
  // CHECK-SAME:     -> tensor<1x1x1024x4096xsi32>

  // Both outputs are concatenated separately along the H dimension.
  // CHECK: [[VALS:%.+]] = VPU.Concat([[V0]], [[V1]], [[V2]], [[V3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]]}
  // CHECK-SAME:     : tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xf32>, tensor<1x1x1024x4096xf32>
  // CHECK-SAME:       -> tensor<1x1x4096x4096xf32>
  // CHECK: [[INDS:%.+]] = VPU.Concat([[I0]], [[I1]], [[I2]], [[I3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 1024, 0], [0, 0, 2048, 0], [0, 0, 3072, 0]]}
  // CHECK-SAME:     : tensor<1x1x1024x4096xsi32>, tensor<1x1x1024x4096xsi32>, tensor<1x1x1024x4096xsi32>, tensor<1x1x1024x4096xsi32>
  // CHECK-SAME:       -> tensor<1x1x4096x4096xsi32>
  // CHECK: return [[VALS]], [[INDS]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map_dyn_conv = affine_map<(d0) -> (-d0 + 640, 96)>

// CHECK-LABEL:   @UnrollDynamicConvolution
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x640x32x32xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>)
func.func @UnrollDynamicConvolution(%arg0: tensor<1x640x32x32xf16, {order = #NHWC}>, %arg1: tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x640x32x32xf16, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c640 = arith.constant 640 : index
    %c96 = arith.constant 96 : index

    %0 = tensor.empty() : tensor<1x640x32x32xf16, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c640 step %c96 iter_args(%arg3 = %0) -> (tensor<1x640x32x32xf16, {order = #NHWC}>) {
      %2 = affine.min #map_dyn_conv(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 640, 32, 32] [1, 1, 1, 1] : tensor<1x640x32x32xf16, {order = #NHWC}> to tensor<1x640x32x32xf16, {order = #NHWC}>
      %extracted_slice_weights = tensor.extract_slice %arg1[%arg2, 0, 0, 0] [%2, 640, 3, 3] [1, 1, 1, 1] : tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}> to tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

      %conv = VPU.NCE.Convolution(%extracted_slice, %extracted_slice_weights) rawFilterShape [%2, 640, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, strides = [1, 1]} : tensor<1x640x32x32xf16, {order = #NHWC}>, tensor<?x640x3x3xf16, {bounds = #const.OpaqueI64Elements<[640, 640, 3, 3]> : tensor<4xsi64>, order = #NHWC}>, index -> tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %conv into %arg3[0, %arg2, 0, 0] [1, %2, 32, 32] [1, 1, 1, 1] : tensor<1x?x32x32xf16, {bounds = #const.OpaqueI64Elements<[1, 640, 32, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x640x32x32xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x640x32x32xf16, {order = #NHWC}>
    }

    return %1 : tensor<1x640x32x32xf16, {order = #NHWC}>

    //CHECK-NOT: tensor.extract_slice
    //CHECK: [[SLICEW0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW0]]) rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW1:%.+]] = VPU.Slice [[WEIGHTS]] [96, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW1]])  rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW2:%.+]] = VPU.Slice [[WEIGHTS]] [192, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW2]])  rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW3:%.+]] = VPU.Slice [[WEIGHTS]] [288, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV3:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW3]])  rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW4:%.+]] = VPU.Slice [[WEIGHTS]] [384, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV4:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW4]])  rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW5:%.+]] = VPU.Slice [[WEIGHTS]] [480, 0, 0, 0] [96, 640, 3, 3]
    //CHECK: [[CONV5:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW5]]) rawFilterShape [%c96, 640, 3, 3] 
    //CHECK: [[SLICEW6:%.+]] = VPU.Slice [[WEIGHTS]] [576, 0, 0, 0] [64, 640, 3, 3]
    //CHECK: [[CONV6:%.+]] = VPU.NCE.Convolution([[INPUT]], [[SLICEW6]])  rawFilterShape [%c64, 640, 3, 3]

    //CHECK-NOT: tensor.insert_slice
    //CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]], [[CONV4]], [[CONV5]], [[CONV6]])
    //CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0], [0, 288, 0, 0], [0, 384, 0, 0], [0, 480, 0, 0], [0, 576, 0, 0]]}
    //CHECK: return [[CONCAT]] : tensor<1x640x32x32xf16, {order = #NHWC}>
}