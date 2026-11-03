//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --pass-pipeline="builtin.module(builtin.module(assign-tiles-for-sw-io-dma-modules))" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>

// NPU5010 has 3 tiles, so MinTiles is 3 and a single-tile module with SW IO DMA ops only is compiled for 3 tiles

// CHECK-LABEL: module @MinTilesThree
module @MinTilesThree {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU5010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 3 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 1571840 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func0(%arg0: !Input, %arg1: tensor<4xf32>) -> !Input {
      %0 = Core.ReinterpretCast(%arg0) : !Input -> !BoundedInput
      %1 = VPU.Empty : !AuxBuffer
      %2 = VPU.InterpolateDMA(%0, %arg1, %1) {
              attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>,
                                     antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
              axes_attr = [0, 1, 2, 3]} : !BoundedInput, tensor<4xf32>, !AuxBuffer -> !BoundedInput
      %3 = Core.ReinterpretCast(%2) : !BoundedInput -> !Input
      return %3 : !Input

      // CHECK:       VPU.InterpolateDMA
      // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    }
  }
}
