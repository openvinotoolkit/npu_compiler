//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> ((d0 floordiv 4) * 3 + 1)>

// CHECK-LABEL:    func.func @ParsePrintDynamicWorkload

func.func @ParsePrintDynamicWorkload(%arg0: tensor<1x16x1x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                      %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x1x10xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %0 = tensor.empty() : tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c10 step %c4 iter_args(%arg3 = %0) -> (tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
      %2 = arith.cmpi ult, %arg2, %c4 : index
      %3 = arith.select %2, %c4, %c3 : index
      %4 = scf.if %2 -> (index) {
        scf.yield %arg2 : index
      } else {
        %6 = affine.apply #map(%arg2)
        scf.yield %6 : index
      }
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %4] [1, 16, 1, %3] [1, 1, 1, 1] : tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 10]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
      %5 = VPU.NCE.MaxPool(%extracted_slice, %arg1 )
        {resultSegmentSizes = array<i32: 1, 0, 0, 0>, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>, strides = [1, 1], tiling_index = 0 : i64}
         -> tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 10]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>{
            VPU.DPU.Workload outOffsets [0, 0, 0, %4] outSizes [1, 16, 1, %3] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
         }
      %inserted_slice = tensor.insert_slice %5 into %arg3[0, 0, 0, %4] [1, 16, 1, %3] [1, 1, 1, 1] : tensor<1x16x1x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 10]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}> into tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>
    }
    return %1 : tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>


    //CHECK: [[LOOP_END:%.+]] = arith.constant 10 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_STEP_REMAINDER:%.+]] = arith.constant 3 : index
    //CHECK: [[LOOP_STEP_MAIN:%.+]] = arith.constant 4 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x1x10xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP_MAIN]]

    //CHECK:                [[CMPI:%.+]] = arith.cmpi ult, [[LOOP_ITER]], [[LOOP_STEP_MAIN]] : index
    //CHECK:                [[SELECT_STEP:%.+]] = arith.select [[CMPI]], [[LOOP_STEP_MAIN]], [[LOOP_STEP_REMAINDER]] : index
    //CHECK:                [[SELECT_OFFSET:%.+]] = scf.if [[CMPI]]

    //CHECK:  VPU.DPU.Workload outOffsets [0, 0, 0, [[SELECT_OFFSET]]] outSizes [1, 16, 1, [[SELECT_STEP]]] pad [0, 0, 0, 0] <VECTOR_FP16>
}

// -----

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#map1 = affine_map<(d0) -> (-d0 + 1280, 15)>
#map2 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map3 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK-LABEL:    func.func @ParsePrintDynamicPaddedWorkload

func.func @ParsePrintDynamicPaddedWorkload(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {

    %c15 = arith.constant 15 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf16>, [#const.Reorder<#NHWC>]

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c15 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>) {
      %size1 = affine.min #map1(%arg1)
      %offset0 = affine.max #map2(%arg1)
      %value = affine.max #map3(%arg1)
      %pad = affine.min #map4()[%value]
      %size0 = affine.apply #map5(%size1, %pad)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %offset0] [1, 4, 1600, %size0] [1, 1, 1, 1] : tensor<1x4x1600x2560xf16, {order = #NHWC, mem_space = @CMX_NN}> to tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>
      %padded = tensor.pad %extracted_slice low[0, 0, 1, %pad] high[0, 0, 0, 0] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}> to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>
      %conv = VPU.NCE.Convolution(%padded, %cst_0) rawFilterShape [32, 4, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>,
           strides = [2, 2]
      } : tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>,
          tensor<32x1x1x144xf16, {order = #NHWC}>
          -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}>{
        VPU.DPU.Workload outOffsets [0, 0, 0, %arg1] outSizes [1, 32, 800, %size1] pad [%pad, 0, 1, 0] <CUBOID_4x16>
      }
      %inserted_slice = tensor.insert_slice %conv into %arg2[0, 0, 0, %arg1] [1, 32, 800, %size1] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC, mem_space = @CMX_NN}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>
    }
    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>

    //CHECK-DAG: [[LOOP_END:%.+]] = arith.constant 1280 : index
    //CHECK-DAG: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK-DAG: [[LOOP_STEP:%.+]] = arith.constant 15 : index

    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x800x1280x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]

    //CHECK:                [[SIZE:%.+]] = affine.min #map([[LOOP_ITER]])
    //CHECK                 [[VALUE:%.+]] = affine.max #map2({{%.+}})
    //CHECK                 [[PAD:%.+]] = affine.min #map3()[[[VALUE]]]

    // VPU.DPU.Workload outOffsets [0, 0, 0, [[LOOP_ITER]]] outSizes [1, 16, 1, [[SIZE]]] pad [[[PAD]], 0, 1, 0] <VECTOR_FP16>
}
