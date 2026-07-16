//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --move-tensor-ops-to-cmx %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010


config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0034980668741113998:117>

// CHECK-LABEL: @MoveTensorPaddingToCMX
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x1600x2560xf16, {order = #NHWC}>
func.func @MoveTensorPaddingToCMX(%arg0: tensor<1x4x1600x2560xf16, {order = #NHWC}>) -> tensor<1x32x800x1280x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf32>, [#const.CastElemType<f16>, #const.LayoutCast<#NHWC>]

    %c0 = arith.constant 0 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %c1140 = arith.constant 1140 : index
    %cst = arith.constant 0.0 : f16
    %c1280 = arith.constant 1280 : index

    %0 = tensor.empty() : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %c1280 step %c15 iter_args(%arg2 = %0) -> (tensor<1x32x800x1280x!qElemType, {order = #NHWC}>) {
        %2 = arith.cmpi ult, %arg1, %c1140 : index
        %3 = arith.select %2, %c15, %c14 : index
        %4 = scf.if %2 -> (index) {
            scf.yield %arg1 : index
        } else {
            %13 = affine.apply affine_map<(d0) -> ((d0 floordiv 15) * 14 + 76)>(%arg1)
            scf.yield %13 : index
        }
        %5 = affine.max affine_map<(d0) -> (0, d0 * 2 - 1)>(%4)
        %6 = affine.max affine_map<(d0) -> (d0 * -2 + 1, 0)>(%4)
        %7 = affine.min affine_map<()[s0] -> (1, s0)>()[%6]
        %8 = affine.apply affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>(%3, %7)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %5] [1, 4, 1600, %8] [1, 1, 1, 1] : tensor<1x4x1600x2560xf16, {order = #NHWC}> to tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %padded = tensor.pad %extracted_slice low[0, 0, 1, %7] high[0, 0, 0, 0] {
            ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
            tensor.yield %cst : f16
        } : tensor<1x4x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}>
        %9 = VPU.Copy(%padded) {out_mem_space = [@CMX_NN, 0]} : tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %10 = VPU.Copy(%cst_0) {out_mem_space = [@CMX_NN, 0]} : tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %11 = VPU.NCE.Convolution(%9, %10) rawFilterShape [32, 4, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>,
             strides = [2, 2]} : tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]>
            : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
            -> tensor<1x32x800x?x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %12 = VPU.Copy(%11) : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %12 into %arg2[0, 0, 0, %4] [1, 32, 800, %3] [1, 1, 1, 1] : tensor<1x32x800x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>
  }

  return %1 : tensor<1x32x800x1280x!qElemType, {order = #NHWC}>

    //CHECK:   [[EXTRACT_SLICE:%.+]] = tensor.extract_slice [[INPUT]]
    //CHECK:   [[COPY_INPUT:%.+]] = VPU.Copy([[EXTRACT_SLICE]])
    //CHECK:   [[PAD:%.+]] = tensor.pad [[COPY_INPUT]]
    //CHECK:   to tensor<1x4x1601x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0]
    //CHECK:   VPU.NCE.Convolution([[PAD]]
}

// -----

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

!convInTiledPaddedDDRType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, order = #NHWC}>
!convInTiledPaddedType = tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

!innerConvOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!convOutTiledType = tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-LABEL:   @SOHConvTileOverH
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
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
      %12 = affine.min #map6(%arg3, %8)[%8]
      %13 = affine.max #map7(%arg3, %4)
      %14 = affine.max #map7(%4, %13)
      %15 = affine.max #map8(%arg3, %4, %12)
      %16 = affine.apply #map9(%14, %15, %12)
      %extracted_slice_1 = tensor.extract_slice %extracted_slice[0, 0, %13, 0] [1, 32, %16, 64] [1, 1, 1, 1]
        : tensor<1x32x33x64xf16, {order = #NHWC}> to tensor<1x32x?x64xf16, {order = #NHWC}>

      %padded = tensor.pad %extracted_slice_1 low[0, 0, %14, 1] high[0, 0, %15, 1] {
      ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
        tensor.yield %cst : f16
      } : tensor<1x32x?x64xf16, {order = #NHWC}> to !convInTiledPaddedDDRType

      %copy_act = VPU.Copy(%padded) {out_mem_space = @CMX_NN}
        : !convInTiledPaddedDDRType -> !convInTiledPaddedType
      %copy_weights = VPU.Copy(%cst_0) {out_mem_space = @CMX_NN}
        : tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

      %17 = VPU.NCE.Convolution(%copy_act, %copy_weights) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
         strides = [1, 1]}
        : !convInTiledPaddedType, tensor<256x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> !innerConvOutTiledType
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %17 into %arg4[0, 0, %arg3, 0] [1, 256, %12, 64] [1, 1, 1, 1]
          : !innerConvOutTiledType into !convOutTiledType
      }
    }

    %copy_output = VPU.Copy(%11) {out_mem_space = @DDR}
      : !convOutTiledType
      -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    %cast = tensor.cast %copy_output
      : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x256x32x64xf16, {order = #NHWC}>

    %inserted_slice = tensor.insert_slice %cast into %arg2[0, 0, %arg1, 0] [1, 256, 32, 64] [1, 1, 1, 1]
      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
    scf.yield %inserted_slice : tensor<1x256x64x64xf16, {order = #NHWC}>
  }
  return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

// CHECK:       [[SCF_FOR:%.+]] = scf.for

// CHECK:       [[IN_SLICE:%.+]] = tensor.extract_slice [[INPUT]]
// CHECK-SAME:    to tensor<1x32x33x64xf16, {order = #NHWC}>

// CHECK:       [[CONV:%.+]] = scf.forall
// CHECK:         [[IN_MC0:%.+]] = tensor.extract_slice [[IN_SLICE]]
// CHECK-SAME:      to tensor<1x32x?x64xf16, {order = #NHWC}>

// CHECK:         [[IN_COPY:%.+]] = VPU.Copy([[IN_MC0]]) {out_mem_space = @CMX_NN}
// CHECK-SAME:      -> tensor<1x32x?x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:         [[IN_PAD:%.+]] = tensor.pad [[IN_COPY]]
// CHECK:         } : tensor<1x32x?x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:      to tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:         [[INNER_CONV0:%.+]] = VPU.NCE.Convolution([[IN_PAD]], {{%.+}})
// CHECK-SAME:      : tensor<1x32x?x66xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 13, 66]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:      -> tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 11, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

// CHECK:        [[CAST:%.+]] = tensor.cast [[CONV]]
// CHECK-SAME:     : tensor<1x256x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
// CHECK-SAME:     to tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:        [[OUT_COPY:%.+]] = VPU.Copy([[CAST]]) {out_mem_space = @DDR}
// CHECK-SAME:     : tensor<1x256x32x64xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>

// CHECK:         tensor.insert_slice [[OUT_COPY]]
// CHECK-SAME:      : tensor<1x256x32x64xf16, {order = #NHWC}> into tensor<1x256x64x64xf16, {order = #NHWC}>
}
