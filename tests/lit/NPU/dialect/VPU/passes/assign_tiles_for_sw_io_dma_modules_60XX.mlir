//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --pass-pipeline="builtin.module(builtin.module(assign-tiles-for-sw-io-dma-modules))" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>

// CHECK-LABEL: module @SwIoDmaOnly
module @SwIoDmaOnly {
  // CHECK: config.Resources 1 of @NCE
  config.Resources 1 of @NCE at 2.100000e+03 MHz {
    config.MemoryResource 2096128 bytes of @CMX_NN
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }

  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 3 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func0(%arg0: !Input, %arg1: tensor<4xf32>) -> (!Input, !Input) {
      %0 = Core.ReinterpretCast(%arg0) : !Input -> !BoundedInput
      %1 = VPU.Empty : !AuxBuffer
      %2 = VPU.InterpolateDMA(%0, %arg1, %1) {
              attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>,
                                     antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
              axes_attr = [0, 1, 2, 3]} : !BoundedInput, tensor<4xf32>, !AuxBuffer -> !BoundedInput
      %3 = VPU.Empty : !AuxBuffer
      %4 = VPU.InterpolateDMA(%2, %arg1, %3) {
              attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>,
                                     antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
              axes_attr = [0, 1, 2, 3]} : !BoundedInput, tensor<4xf32>, !AuxBuffer -> !BoundedInput
      %5 = Core.ReinterpretCast(%2) : !BoundedInput -> !Input
      %6 = Core.ReinterpretCast(%4) : !BoundedInput -> !Input
      return %5, %6 : !Input, !Input

      // CHECK:       VPU.InterpolateDMA
      // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
      // CHECK:       VPU.InterpolateDMA
      // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    }
  }

  // CHECK: module @Module1
  module @Module1 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 1 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func1(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
      %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
      return %0 : tensor<1x8x80x960xf16>

      // CHECK-NOT:   multiClusterStrategy
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>

// Platforms without MinTiles keep the module tile count

// CHECK-LABEL: module @NoMinTilesOption
module @NoMinTilesOption {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    // CHECK: config.Resources 1 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
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
      // CHECK-NOT:       multiClusterStrategy
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>

// CHECK-LABEL: module @SwIoDmaWithOtherCompute
module @SwIoDmaWithOtherCompute {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 1 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func0(%arg0: !Input, %arg1: tensor<4xf32>, %arg2: tensor<1x8x80x960xf16>) -> (!Input, tensor<1x8x80x960xf16>) {
      %0 = Core.ReinterpretCast(%arg0) : !Input -> !BoundedInput
      %1 = VPU.Empty : !AuxBuffer
      %2 = VPU.InterpolateDMA(%0, %arg1, %1) {
              attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>,
                                     antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
              axes_attr = [0, 1, 2, 3]} : !BoundedInput, tensor<4xf32>, !AuxBuffer -> !BoundedInput
      %3 = Core.ReinterpretCast(%2) : !BoundedInput -> !Input
      %4 = VPU.Sigmoid(%arg2) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
      return %3, %4 : !Input, tensor<1x8x80x960xf16>

      // CHECK:       VPU.InterpolateDMA
      // CHECK-NOT:       multiClusterStrategy
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>

// Modules compiled for more than one tile keep their tile count

// CHECK-LABEL: module @MultiTileModule
module @MultiTileModule {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 2 of @NCE
    config.Resources 2 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
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
      // CHECK-NOT:       multiClusterStrategy
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!PadInput = tensor<2x3x4x?xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 20]> : tensor<4xsi64>, order = #NCHW}>
!PadOutput = tensor<5x6x11x?xf16, {bounds = #const.OpaqueI64Elements<[5, 6, 11, 27]> : tensor<4xsi64>, order = #NCHW}>
!PadAuxBuffer = tensor<1x1x1x1049600xui8>

// Ops without Clustering support run on a single tile, so a module with only such ops is not changed

// CHECK-LABEL: module @SingleTileSwIoDmaOnly
module @SingleTileSwIoDmaOnly {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 1 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func0(%arg0: !PadInput) -> !PadOutput {
      %0 = VPU.Empty : !PadAuxBuffer
      %1 = VPU.PadDma(%arg0, %0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f32, pads_begin_attr = [1, 2, 3, 4], pads_end_attr = [2, 1, 4, 3]}
              : !PadInput, !PadAuxBuffer -> !PadOutput
      return %1 : !PadOutput

      // CHECK:       VPU.PadDma
      // CHECK-NOT:       multiClusterStrategy
    }
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!Input = tensor<1x3x?x?xf16>
!BoundedInput = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NCHW}>
!AuxBuffer = tensor<1x1x1x1324212xui8>
!PadInput = tensor<2x3x4x?xf16, {bounds = #const.OpaqueI64Elements<[2, 3, 4, 20]> : tensor<4xsi64>, order = #NCHW}>
!PadOutput = tensor<5x6x11x?xf16, {bounds = #const.OpaqueI64Elements<[5, 6, 11, 27]> : tensor<4xsi64>, order = #NCHW}>
!PadAuxBuffer = tensor<1x1x1x1049600xui8>

// Ops with Clustering support get it; the others stay on a single tile in the multi-tile module

// CHECK-LABEL: module @MixedSwIoDmaOps
module @MixedSwIoDmaOps {
  // CHECK: module @Module0
  module @Module0 attributes {config.compilationMode = #config.compilation_mode<HostCompile_Interpreter>, config.platform = #config.platform<NPU6010>} {
    config.PipelineOptions @Options {
      config.Option @config.MinTiles : 3 : ui64
    }
    // CHECK: config.Resources 3 of @NCE
    config.Resources 1 of @NCE at 2.100000e+03 MHz {
      config.MemoryResource 2096128 bytes of @CMX_NN
      config.ExecutorResource 2 of @SHAVE_ACT
      config.ExecutorResource 1 of @DPU
    }

    func.func @main_func0(%arg0: !Input, %arg1: tensor<4xf32>, %arg2: !PadInput) -> (!Input, !PadOutput) {
      %0 = Core.ReinterpretCast(%arg0) : !Input -> !BoundedInput
      %1 = VPU.Empty : !AuxBuffer
      %2 = VPU.InterpolateDMA(%0, %arg1, %1) {
              attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>,
                                     antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
              axes_attr = [0, 1, 2, 3]} : !BoundedInput, tensor<4xf32>, !AuxBuffer -> !BoundedInput
      %3 = Core.ReinterpretCast(%2) : !BoundedInput -> !Input
      %4 = VPU.Empty : !PadAuxBuffer
      %5 = VPU.PadDma(%arg2, %4) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f32, pads_begin_attr = [1, 2, 3, 4], pads_end_attr = [2, 1, 4, 3]}
              : !PadInput, !PadAuxBuffer -> !PadOutput
      return %3, %5 : !Input, !PadOutput

      // CHECK:       VPU.InterpolateDMA
      // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
      // CHECK:       VPU.PadDma
      // CHECK-NOT:       multiClusterStrategy
    }
  }
}
