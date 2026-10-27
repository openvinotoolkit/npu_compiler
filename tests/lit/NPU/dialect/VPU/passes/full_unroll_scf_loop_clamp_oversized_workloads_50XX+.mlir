//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 1280, 640)>

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

// An unpadded Eltwise carries the bounded (full-tensor) 1280-wide workload produced before unrolling. Eltwise has no
// tensor.pad parent, so restore-padding translation skips it; the safety-net clamp must resize the workload to each
// tile's concrete 640-wide output once the loop is unrolled.
// CHECK-LABEL: @ClampOversizedEltwiseWorkload
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT1:%.+]]: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>)
func.func @ClampOversizedEltwiseWorkload(%arg0: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>, %arg1: tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %c640 = arith.constant 640 : index
    %c1280 = arith.constant 1280 : index
    %c0 = arith.constant 0 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %c1280 step %c640 iter_args(%arg3 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
      %size = affine.min #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType, {order = #NHWC}> to tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
      %eltwise = VPU.NCE.Eltwise(%extracted_slice, %extracted_slice_1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.380000e+02 : f64, clamp_high = 1.170000e+02 : f64, scale = 3.5136938095092773E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.380000e+02 : f64, in1_mult = [2.934300e+04], in2_mult = [2.934300e+04]>} -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> {
        VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 800, 1280] outOffsets [0, 0, 0, 0] outSizes [1, 32, 800, 1280] pad [0, 0, 0, 0] <CUBOID_8x16>
      }
      %inserted_slice = tensor.insert_slice %eltwise into %arg3[0, 0, 0, %arg2] [1, 32, 800, %size] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    }

    return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    // First tile: 640-wide output; the oversized 1280 workload is clamped to 640.
    // CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 800, 640]
    // CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 800, 640]
    // CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE0]], [[SLICE1]])
    // CHECK-SAME: -> tensor<1x32x800x640x!qElemType
    // CHECK: VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 800, 640] outOffsets [0, 0, 0, 0] outSizes [1, 32, 800, 640] pad [0, 0, 0, 0] <CUBOID_8x16>

    // Second tile: same clamp at width offset 640.
    // CHECK: [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 640] [1, 32, 800, 640]
    // CHECK: [[SLICE3:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 640] [1, 32, 800, 640]
    // CHECK: [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[SLICE2]], [[SLICE3]])
    // CHECK-SAME: -> tensor<1x32x800x640x!qElemType
    // CHECK: VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 800, 640] outOffsets [0, 0, 0, 0] outSizes [1, 32, 800, 640] pad [0, 0, 0, 0] <CUBOID_8x16>

    // CHECK: VPU.Concat([[ELTWISE0]], [[ELTWISE1]])
    // CHECK-NOT: outSizes [1, 32, 800, 1280]
}
