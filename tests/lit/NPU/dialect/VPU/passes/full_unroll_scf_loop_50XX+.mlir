//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 112, 32)>
#map1 = affine_map<(d0) -> (((d0 + 15) floordiv 16) * 16)>

!actType = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>
!actTypeDDR = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHWC}>
!outType = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHCW}>
!outTypeDDR = tensor<1x?x12x12xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 12, 12]> : tensor<4xsi64>, order = #NHCW}>

// CHECK-LABEL: @NCEEltwiseSOK
// CHECK-SAME:       [[INPUT0:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
// CHECK-SAME:       [[INPUT1:%[^:]+]]: tensor<1x112x12x12xf16, {order = #NHWC}>
func.func @NCEEltwiseSOK(%arg0: tensor<1x112x12x12xf16, {order = #NHWC}>, %arg1: tensor<1x112x12x12xf16, {order = #NHWC}>) -> tensor<1x112x12x12xf16, {order = #NHCW}> {
  %0 = tensor.empty() : tensor<1x112x12x12xf16, {order = #NHCW}>
  %1 = scf.forall (%arg2) = (0) to (112) step (32) shared_outs(%arg3 = %0) -> (tensor<1x112x12x12xf16, {order = #NHCW}>) {
    %2 = affine.min #map(%arg2)
    %3 = affine.apply #map1(%2)

    %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0] [1, %3, 12, 12] [1, 1, 1, 1]
        : tensor<1x112x12x12xf16, {order = #NHWC}> to !actTypeDDR
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg2, 0, 0] [1, %3, 12, 12] [1, 1, 1, 1]
        : tensor<1x112x12x12xf16, {order = #NHWC}> to !actTypeDDR

    %4 = VPU.Copy(%extracted_slice) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType
    %5 = VPU.Copy(%extracted_slice_0) {out_mem_space = @CMX_NN} : !actTypeDDR -> !actType

    %6 = VPU.NCE.Eltwise(%4, %5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        -> !outType

    %7 = VPU.Copy(%6) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2, 0, 0] [1, %2, 12, 12] [1, 1, 1, 1]
          : !outTypeDDR into tensor<1x112x12x12xf16, {order = #NHCW}>
    }
  }
  return %1 : tensor<1x112x12x12xf16, {order = #NHCW}>

    // CHECK:        [[IN0_COPY:%.+]] = VPU.Copy([[INPUT0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[IN1_COPY:%.+]] = VPU.Copy([[INPUT1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:          : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN0_COPY]], [[IN1_COPY]])
    // CHECK-SAME:           {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x12x12xf16, #NHCW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[OUT_COPY:%.+]] = VPU.Copy([[ELTWISE]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x112x12x12xf16, #NHCW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
    // CHECK-SAME:       -> tensor<1x112x12x12xf16, {order = #NHCW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_lstm_soh = affine_map<(d0) -> (-d0 + 112, 32)>

// Multiclustering: multi-result LSTMGates with SplitOverHeight and uneven tile sizes.
// H=112 distributed over 4 clusters with step=32 gives tiles [32, 32, 32, 16].
// Validates that full-unroll-scf-loop correctly emits one SEGMENTED VPU.Copy per
// input and one SEGMENTED VPU.Copy per output result (hidden state and cell state).

// CHECK-LABEL: @LSTMGatesSOHUnevenClusters
// CHECK-SAME:       [[GATES:%[^:]+]]: tensor<1x1x112x512xf16>
// CHECK-SAME:       [[CELL:%[^:]+]]: tensor<1x1x112x128xf16>
func.func @LSTMGatesSOHUnevenClusters(
    %arg0: tensor<1x1x112x512xf16>,
    %arg1: tensor<1x1x112x128xf16>
) -> (tensor<1x1x112x128xf16>, tensor<1x1x112x128xf16>) {
  %empty_h = tensor.empty() : tensor<1x1x112x128xf16>
  %empty_c = tensor.empty() : tensor<1x1x112x128xf16>
  %result:2 = scf.forall (%iv) = (0) to (112) step (32)
      shared_outs(%out_h = %empty_h, %out_c = %empty_c)
      -> (tensor<1x1x112x128xf16>, tensor<1x1x112x128xf16>) {
    %tile_sz = affine.min #map_lstm_soh(%iv)
    %slice_gates = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 1, %tile_sz, 512] [1, 1, 1, 1]
        : tensor<1x1x112x512xf16>
        to tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 512]> : tensor<4xsi64>, order = #NCHW}>
    %slice_cell = tensor.extract_slice %arg1[0, 0, %iv, 0] [1, 1, %tile_sz, 128] [1, 1, 1, 1]
        : tensor<1x1x112x128xf16>
        to tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>
    %h, %c = VPU.LSTMGates(%slice_gates, %slice_cell)
        : tensor<1x1x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 512]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>,
           tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %h into %out_h[0, 0, %iv, 0] [1, 1, %tile_sz, 128] [1, 1, 1, 1]
          : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>
          into tensor<1x1x112x128xf16>
      tensor.parallel_insert_slice %c into %out_c[0, 0, %iv, 0] [1, 1, %tile_sz, 128] [1, 1, 1, 1]
          : tensor<1x1x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 1, 112, 128]> : tensor<4xsi64>, order = #NCHW}>
          into tensor<1x1x112x128xf16>
    }
  }
  return %result#0, %result#1 : tensor<1x1x112x128xf16>, tensor<1x1x112x128xf16>

    // CHECK-NOT: scf.forall

    // CHECK:        [[GATES_COPY:%.+]] = VPU.Copy([[GATES]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x1x112x512xf16>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x1x112x512xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 16, 512]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 32, 512], [1, 1, 16, 512]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

    // CHECK:        [[CELL_COPY:%.+]] = VPU.Copy([[CELL]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x1x112x128xf16>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x1x112x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

    // CHECK:        [[H:%.+]], [[C:%.+]] = VPU.LSTMGates([[GATES_COPY]], [[CELL_COPY]])
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x1x112x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>,
    // CHECK-SAME:              !VPU.DistributedTensor<1x1x112x128xf16, #NCHW, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64,
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 32, 128], [1, 1, 16, 128]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>

    // CHECK:        [[H_OUT:%.+]] = VPU.Copy([[H]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x1x112x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]
    // CHECK-SAME:       -> tensor<1x1x112x128xf16>

    // CHECK:        [[C_OUT:%.+]] = VPU.Copy([[C]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x1x112x128xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1]
    // CHECK-SAME:       -> tensor<1x1x112x128xf16>

    // CHECK:        return [[H_OUT]], [[C_OUT]] : tensor<1x1x112x128xf16>, tensor<1x1x112x128xf16>
}

// -----

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
    config.Option @config.AutoPaddingIDU : true
}

!qElemType10 = !quant.uniform<u8:f16, 0.0034980668741113998:117>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 1280, 427)>
#map1 = affine_map<(d0) -> (0, d0 * 2 - 1)>
#map2 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0) -> (-d0 + 33)>
#map5 = affine_map<(d0, d1) -> (d0 * 2 - d1 + 1)>

// CHECK-LABEL: @UnrollTwoDimLoopWithCast
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x4x1600x2560xf32, {order = #NHWC}>
func.func @UnrollTwoDimLoopWithCast(%arg0: tensor<1x4x1600x2560xf32, {order = #NHWC}>) -> tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>
 {
    %c800 = arith.constant 800 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1280 = arith.constant 1280 : index
    %c427 = arith.constant 427 : index

    %cst_0 = const.Declare tensor<32x1x1x144xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1x144xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst = arith.constant 0.000000e+00 : f16

    %1 = tensor.empty() : tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>
    %2 = scf.for %arg1 = %c0 to %c800 step %c16 iter_args(%arg2 = %1) -> (tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>) {
      %3 = scf.for %arg3 = %c0 to %c1280 step %c427 iter_args(%arg4 = %arg2) -> (tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>) {
        %4 = affine.min #map(%arg3)
        %5 = affine.max #map1(%arg1)
        %6 = affine.max #map2(%arg1)
        %7 = affine.min #map3()[%6]
        %8 = affine.apply #map4(%7)
        %9 = affine.max #map1(%arg3)
        %10 = affine.max #map2(%arg3)
        %11 = affine.min #map3()[%10]
        %12 = affine.apply #map5(%4, %11)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %5, %9] [1, 4, %8, %12] [1, 1, 1, 1] : tensor<1x4x1600x2560xf32, {order = #NHWC}> to tensor<1x4x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %13 = VPU.Convert(%extracted_slice) {dstElemType = f16} : tensor<1x4x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %14 = VPU.Copy(%13) {out_mem_space = [@CMX_NN, 0]} : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %padded = tensor.pad %14 low[0, 0, %7, %11] high[0, 0, 0, 0] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %cst : f16
        } : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1600, 2560]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %15 = VPU.Copy(%cst_0) {out_mem_space = [@CMX_NN, 0]} : tensor<32x1x1x144xf16, {order = #NHWC}> -> tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output = VPU.NCE.Convolution(%padded, %15) rawFilterShape [32, 4, 3, 3] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.170000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [2, 2]} : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 1601, 2561]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<32x1x1x144xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType10, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %cast = tensor.cast %output : tensor<1x32x?x?x!qElemType10, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x16x?x!qElemType10, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %17 = VPU.Copy(%cast) : tensor<1x32x16x?x!qElemType10, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x16x?x!qElemType10, {order = #NHWC}>
        %inserted_slice = tensor.insert_slice %17 into %arg4[0, 0, %arg1, %arg3] [1, 32, 16, %4] [1, 1, 1, 1] : tensor<1x32x16x?x!qElemType10, {order = #NHWC}> into tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>
      }
      scf.yield %3 : tensor<1x32x800x1280x!qElemType10, {order = #NHWC}>
    }

    return %2: tensor<1x32x800x1280x!quant.uniform<u8:f16, 0.0034980668741113998:117>, {order = #NHWC}>

    //CHECK: VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 854]
    //CHECK: VPU.Convert
    //CHECK: VPU.Copy
    //CHECK: VPU.Copy
    //CHECK: VPU.NCE.Convolution
    //CHECK:  VPU.Copy

    //CHECK: VPU.Concat
}


// -----

!qElemType = !quant.uniform<u8:f16, 5.7474947443195417E-4:128>
!qElemType1 = !quant.uniform<u8:f16, 5.4446090670192944E-4:128>
!qElemType2 = !quant.uniform<u8:f16, 5.4758985837300616E-4:128>
!qElemType15 = !quant.uniform<u8:f16, 0.0048445047116747091:109>
!qElemType17 = !quant.uniform<u8:f16, 0.004490463640175614:142>
!qElemType18 = !quant.uniform<u8:f16, 0.0060971145536385333:117>
!qElemType19 = !quant.uniform<u8:f16, 0.0078253755382463042:157>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map9 = affine_map<(d0) -> (0, d0 - 1)>
#map10 = affine_map<(d0) -> (-d0 + 1, 0)>
#map18 = affine_map<(d0) -> (-d0 + 2560, 366)>
#map19 = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
#map20 = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
#map21 = affine_map<(d0) -> (0, d0 - 782)>
#map22 = affine_map<(d0, d1) -> (-d0 - d1 + 18)>
#map23 = affine_map<(d0, d1) -> (d0 + d1 floordiv 2 - 1278, 0)>
#map24 = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 floordiv 2 + 2)>
#map25 = affine_map<(d0, d1, d2) -> (0, d0 - d1 - d2 - 780)>
#map26 = affine_map<(d0, d1, d2, d3) -> (-d0 - d1 - d2 - d3 + 20)>
#map27 = affine_map<(d0, d1, d2, d3) -> (d0 - d1 - d2 + d3 floordiv 2 - 1276, 0)>
#map28 = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 - d2 - d3 + d4 floordiv 2 + 4)>
#map29 = affine_map<(d0, d1, d2, d3, d4) -> (0, d0 - d1 - d2 - d3 - d4 - 778)>
#map30 = affine_map<(d0, d1, d2, d3, d4, d5) -> (-d0 - d1 - d2 - d3 - d4 - d5 + 22)>
#map31 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 - d1 - d2 - d3 - d4 + d5 floordiv 2 - 1274, 0)>
#map32 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (-d0 - d1 - d2 - d3 - d4 - d5 + d6 floordiv 2 + 6)>

// CHECK-LABEL: @UnrollLoopDepthToSpaceCast
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x800x1280x!qElemType,
// CHECK-SAME:      [[INPUT1:%.+]]: tensor<1x32x800x1280x!qElemType1,
func.func @UnrollLoopDepthToSpaceCast(%input0: tensor<1x32x800x1280x!qElemType15, {order = #NHWC}>, %input1: tensor<1x32x800x1280x!qElemType17, {order = #NHWC}>) -> tensor<1x4x1600x2560xf32, {order = #NHWC}> {
    %c-99_i8 = arith.constant -99 : i8
    %c-114_i8 = arith.constant -114 : i8
    %c117_i8 = arith.constant 117 : i8
    %c366 = arith.constant 366 : index
    %c32 = arith.constant 32 : index
    %c2560 = arith.constant 2560 : index
    %c1600 = arith.constant 1600 : index
    %c0 = arith.constant 0 : index
    %cst_13 = const.Declare tensor<16x96x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x96x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_14 = const.Declare tensor<96x32x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<96x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_15 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_16 = const.Declare tensor<32x32x3x3x!qElemType2, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %13 = tensor.empty() : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    %14 = scf.for %arg1 = %c0 to %c1600 step %c32 iter_args(%arg2 = %13) -> (tensor<1x4x1600x2560xf32, {order = #NHWC}>) {
      %16 = scf.for %arg3 = %c0 to %c2560 step %c366 iter_args(%arg4 = %arg2) -> (tensor<1x4x1600x2560xf32, {order = #NHWC}>) {
        %17 = affine.min #map18(%arg3)
        %18 = affine.max #map19(%arg1)
        %19 = affine.max #map20(%arg1)
        %20 = affine.min #map3()[%19]
        %21 = affine.max #map21(%18)
        %22 = affine.min #map3()[%21]
        %23 = affine.apply #map22(%20, %22)
        %24 = affine.max #map19(%arg3)
        %25 = affine.max #map20(%arg3)
        %26 = affine.min #map3()[%25]
        %27 = affine.max #map23(%24, %17)
        %28 = affine.min #map3()[%27]
        %29 = affine.apply #map24(%26, %28, %17)
        %extracted_slice = tensor.extract_slice %input0[0, 0, %18, %24] [1, 32, %23, %29] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType15, {order = #NHWC}> to tensor<1x32x?x?x!qElemType15, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %30 = affine.max #map9(%18)
        %31 = affine.max #map10(%18)
        %32 = affine.min #map3()[%31]
        %33 = affine.max #map25(%30, %20, %22)
        %34 = affine.min #map3()[%33]
        %35 = affine.apply #map26(%32, %34, %20, %22)
        %36 = affine.max #map9(%24)
        %37 = affine.max #map10(%24)
        %38 = affine.min #map3()[%37]
        %39 = affine.max #map27(%36, %26, %28, %17)
        %40 = affine.min #map3()[%39]
        %41 = affine.apply #map28(%38, %40, %26, %28, %17)
        %extracted_slice_24 = tensor.extract_slice %input1[0, 0, %30, %36] [1, 32, %35, %41] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType17, {order = #NHWC}> to tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %42 = affine.max #map9(%30)
        %43 = affine.max #map10(%30)
        %44 = affine.min #map3()[%43]
        %45 = affine.max #map29(%42, %32, %34, %20, %22)
        %46 = affine.min #map3()[%45]
        %47 = affine.apply #map30(%44, %46, %32, %34, %20, %22)
        %48 = affine.max #map9(%36)
        %49 = affine.max #map10(%36)
        %50 = affine.min #map3()[%49]
        %51 = affine.max #map31(%48, %38, %40, %26, %28, %17)
        %52 = affine.min #map3()[%51]
        %53 = affine.apply #map32(%50, %52, %38, %40, %26, %28, %17)
        %extracted_slice_25 = tensor.extract_slice %input1[0, 0, %42, %48] [1, 32, %47, %53] [1, 1, 1, 1] : tensor<1x32x800x1280x!qElemType17, {order = #NHWC}> to tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %54 = builtin.unrealized_conversion_cast %c-114_i8 : i8 to !qElemType17
        %55 = VPU.Copy(%extracted_slice_25) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %padded = tensor.pad %55 low[0, 0, %44, %50] high[0, 0, %46, %52] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %54 : !qElemType17
        } : tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %56 = VPU.Copy(%cst_16) {out_mem_space = [@CMX_NN, 0]} : tensor<32x32x3x3x!qElemType2, {order = #NHWC}> -> tensor<32x32x3x3x!qElemType2, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output = VPU.NCE.Convolution(%padded, %56) rawFilterShape [32, 32, 3, 3] {minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.420000e+02 : f64, clamp_high = 1.130000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.420000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<32x32x3x3x!qElemType2, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %58 = VPU.Copy(%extracted_slice_24) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?x!qElemType17, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output_26 = VPU.NCE.Eltwise(%58, %output) {is_inplace = true, minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.170000e+02 : f64, clamp_high = 1.380000e+02 : f64, scale = 3.9102509617805481E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.170000e+02 : f64, in1_mult = [1.883400e+04], in2_mult = [1.883400e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %59 = VPU.Copy(%output_26) : tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %60 = builtin.unrealized_conversion_cast %c117_i8 : i8 to !qElemType18
        %61 = VPU.Copy(%59) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %padded_27 = tensor.pad %61 low[0, 0, %32, %38] high[0, 0, %34, %40] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %60 : !qElemType18
        } : tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %62 = VPU.Copy(%cst_15) {out_mem_space = [@CMX_NN, 0]} : tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<32x32x3x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output_28 = VPU.NCE.Convolution(%padded_27, %62) rawFilterShape [32, 32, 3, 3] {minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.090000e+02 : f64, clamp_high = 1.460000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.090000e+02 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x?x?x!qElemType18, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType15, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %64 = VPU.Copy(%extracted_slice) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x?x?x!qElemType15, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?x!qElemType15, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output_29 = VPU.NCE.Eltwise(%64, %output_28) {is_inplace = true, minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.570000e+02 : f64, clamp_high = 9.800000e+01 : f64, scale = 3.0467286705970764E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.570000e+02 : f64, in1_mult = [2.031900e+04], in2_mult = [2.031900e+04]>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %65 = VPU.Copy(%output_29) : tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %66 = builtin.unrealized_conversion_cast %c-99_i8 : i8 to !qElemType19
        %67 = VPU.Copy(%65) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %padded_30 = tensor.pad %67 low[0, 0, %20, %26] high[0, 0, %22, %28] {
        ^bb0(%arg5: index, %arg6: index, %arg7: index, %arg8: index):
          tensor.yield %66 : !qElemType19
        } : tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> to tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %68 = VPU.Copy(%cst_14) {out_mem_space = [@CMX_NN, 0]} : tensor<96x32x3x3x!qElemType, {order = #NHWC}> -> tensor<96x32x3x3x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output_31 = VPU.NCE.Convolution(%padded_30, %68) rawFilterShape [96, 32, 3, 3] {minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x32x?x?x!qElemType19, {bounds = #const.OpaqueI64Elements<[1, 32, 802, 1282]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<96x32x3x3x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %70 = VPU.Copy(%output_31) : tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %cast = tensor.cast %70 : tensor<1x96x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %71 = VPU.Copy(%cast) {out_mem_space = [@CMX_NN, 0]} : tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %72 = VPU.Copy(%cst_13) {out_mem_space = [@CMX_NN, 0]} : tensor<16x96x1x1xf16, {order = #NHWC}> -> tensor<16x96x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %output_32 = VPU.NCE.Convolution(%71, %72) rawFilterShape [16, 96, 1, 1] {minimumHardwareExecutionCost = 4294967295 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} : tensor<1x96x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 96, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<16x96x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}>
        %74 = VPU.Copy(%output_32) : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %75 = VPU.DepthToSpace(%74) {block_size = 2 : i64, dstElemType = f32, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x16x16x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x4x32x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %75 into %arg4[0, 0, %arg1, %arg3] [1, 4, 32, %17] [1, 1, 1, 1] : tensor<1x4x32x?xf32, {bounds = #const.OpaqueI64Elements<[1, 4, 32, 2560]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x4x1600x2560xf32, {order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x4x1600x2560xf32, {order = #NHWC}>
      }
      scf.yield %16 : tensor<1x4x1600x2560xf32, {order = #NHWC}>
    }

    // CHECK:      VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 17, 184]
    // CHECK:      VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 18, 185]
    // CHECK:      VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 19, 186]
    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME:     -> tensor<1x32x18x185x!qElemType1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.NCE.Eltwise
    // CHECK-SAME:     -> tensor<1x32x18x185x!qElemType5, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME:     -> tensor<1x32x17x184x!qElemType, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.NCE.Eltwise
    // CHECK-SAME:     -> tensor<1x32x17x184x!qElemType6, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME:     -> tensor<1x96x16x183xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.NCE.Convolution
    // CHECK-SAME:     -> tensor<1x16x16x183xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:      VPU.DepthToSpace
    // CHECK-SAME:     -> tensor<1x4x32x366xf32, {order = #NHWC}>
    // CHECK:      VPU.Concat
    return %14 : tensor<1x4x1600x2560xf32, {order = #NHWC}>
}
