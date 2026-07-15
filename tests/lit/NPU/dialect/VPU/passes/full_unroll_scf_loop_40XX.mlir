//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --full-unroll-scf-loop %s | FileCheck %s
// REQUIRES: platform-NPU4000

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

    %6 = VPU.NCE.Eltwise(%4, %5) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        -> !outType

    %7 = VPU.Copy(%6) : !outType -> !outTypeDDR
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2, 0, 0] [1, %2, 12, 12] [1, 1, 1, 1]
          : !outTypeDDR into tensor<1x112x12x12xf16, {order = #NHCW}>
    }
  }
  return %1 : tensor<1x112x12x12xf16, {order = #NHCW}>

    // TODO:  E#192457 - scf.forall loop shows SEGMENTED input; however, after unroll we have DUPLICATED distribution.
    //        That happens due to scf MC currently borrowing the tiling logic. This should be fixed when proper
    //        multiclustering is implemented.

    // CHECK:        [[IN0_COPY:%.+]] = VPU.Copy([[INPUT0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[IN1_COPY:%.+]] = VPU.Copy([[INPUT1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:          : tensor<1x112x12x12xf16, {order = #NHWC}>
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x112x12x12xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 12, 12], [1, 32, 12, 12], [1, 32, 12, 12], [1, 16, 12, 12]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[IN0_COPY]], [[IN1_COPY]])
    // CHECK-SAME:           {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Balanced multiclustering unrolling: 112 OC across 4 clusters with alignment 16.
// The forall uses normalized bounds (0) to (4) step (1) with balanced arithmetic
// (arith.minui, arith.muli, arith.addi, arith.cmpi, arith.select) to compute
// per-iteration offsets and sizes. After unrolling, the per-cluster shapes are
// [32, 32, 32, 16] at offsets [0, 32, 64, 96].

// CHECK-LABEL: @BalancedSOKConvUnroll
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x112x1x1xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<112x112x1x1xf16, {order = #NHWC}>
func.func @BalancedSOKConvUnroll(
    %arg0: tensor<1x112x1x1xf16, {order = #NHWC}>,
    %arg1: tensor<112x112x1x1xf16, {order = #NHWC}>
) -> tensor<1x112x1x1xf16, {order = #NHWC}> {
  %empty = tensor.empty() : tensor<1x112x1x1xf16, {order = #NHWC}>
  %c16 = arith.constant 16 : index
  %c3 = arith.constant 3 : index
  %c32 = arith.constant 32 : index
  %result = scf.forall (%iv) = (0) to (4) step (1)
      shared_outs(%out = %empty) -> (tensor<1x112x1x1xf16, {order = #NHWC}>) {
    %extra = arith.minui %iv, %c3 : index
    %base_off = arith.muli %iv, %c16 : index
    %extra_off = arith.muli %extra, %c16 : index
    %real_off = arith.addi %base_off, %extra_off : index
    %is_large = arith.cmpi ult, %iv, %c3 : index
    %real_sz = arith.select %is_large, %c32, %c16 : index
    %slice_w = tensor.extract_slice %arg1[%real_off, 0, 0, 0] [%real_sz, 112, 1, 1] [1, 1, 1, 1]
        : tensor<112x112x1x1xf16, {order = #NHWC}>
        to tensor<?x112x1x1xf16, {bounds = #const.OpaqueI64Elements<[112, 112, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    %conv = VPU.NCE.Convolution(%arg0, %slice_w) rawFilterShape [112, 112, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } : tensor<1x112x1x1xf16, {order = #NHWC}>,
        tensor<?x112x1x1xf16, {bounds = #const.OpaqueI64Elements<[112, 112, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
      -> tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %conv into %out[0, %real_off, 0, 0] [1, %real_sz, 1, 1] [1, 1, 1, 1]
          : tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 112, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
          into tensor<1x112x1x1xf16, {order = #NHWC}>
    }
  }
  return %result : tensor<1x112x1x1xf16, {order = #NHWC}>

    // CHECK-NOT: scf.forall

    // CHECK:        [[IN_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x112x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 112, 1, 1], [1, 112, 1, 1], [1, 112, 1, 1], [1, 112, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[W_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<112x112x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<112x112x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[32, 112, 1, 1], [32, 112, 1, 1], [32, 112, 1, 1], [16, 112, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[32, 112, 1, 1], [32, 112, 1, 1], [32, 112, 1, 1], [16, 112, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[IN_COPY]], [[W_COPY]])
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x112x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 1, 1], [1, 32, 1, 1], [1, 32, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]],
    // CHECK-SAME{LITERAL}:     memory_shapes = [[1, 32, 1, 1], [1, 32, 1, 1], [1, 32, 1, 1], [1, 16, 1, 1]],
    // CHECK-SAME{LITERAL}:     memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>

    // CHECK:        [[OUT_COPY:%.+]] = VPU.Copy([[CONV]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x112x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1]
    // CHECK-SAME:       -> tensor<1x112x1x1xf16, {order = #NHWC}>

    // CHECK:        return [[OUT_COPY]] : tensor<1x112x1x1xf16, {order = #NHWC}>
}

// -----

// Dynamic balanced SOK multiclustering inside a tiling loop.
// OC=128, 6 clusters, alignment=16. numLarge computed via divui at runtime.
// After unrolling: balanced distribution [32,32,16,16,16,16] across 6 clusters.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 64, 32)>

// CHECK-LABEL: @DynamicBalancedSOKConvUnroll
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x128x64x64xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<128x128x1x1xf16, {order = #NHWC}>
func.func @DynamicBalancedSOKConvUnroll(
    %arg0: tensor<1x128x64x64xf16, {order = #NHWC}>,
    %arg1: tensor<128x128x1x1xf16, {order = #NHWC}>
) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %empty = tensor.empty() : tensor<1x128x64x64xf16, {order = #NHWC}>
  %result = scf.for %iv = %c0 to %c64 step %c32 iter_args(%iter = %empty) -> (tensor<1x128x64x64xf16, {order = #NHWC}>) {
    %tile_sz = affine.min #map(%iv)
    %in_slice = tensor.extract_slice %arg0[0, 0, %iv, 0] [1, 128, %tile_sz, 64] [1, 1, 1, 1]
        : tensor<1x128x64x64xf16, {order = #NHWC}>
        to tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    // Balanced SOK arithmetic
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : index
    %total_small = arith.muli %c16, %c6 : index
    %remainder = arith.subi %c128, %total_small : index
    %c16_delta = arith.constant 16 : index
    %num_large = arith.divui %remainder, %c16_delta : index
    %mc_empty = tensor.empty(%tile_sz) : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    %forall = scf.forall (%k) in (6) shared_outs(%out = %mc_empty)
        -> (tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>) {
      %extra = arith.minui %k, %num_large : index
      %base_off = arith.muli %k, %c16 : index
      %extra_off = arith.muli %extra, %c16_delta : index
      %real_off = arith.addi %base_off, %extra_off : index
      %is_large = arith.cmpi ult, %k, %num_large : index
      %large_tile = arith.addi %c16, %c16_delta : index
      %real_sz = arith.select %is_large, %large_tile, %c16 : index
      %act_slice = tensor.extract_slice %in_slice[0, 0, 0, 0] [1, 128, %tile_sz, 64] [1, 1, 1, 1]
          : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      %w_slice = tensor.extract_slice %arg1[%real_off, 0, 0, 0] [%real_sz, 128, 1, 1] [1, 1, 1, 1]
          : tensor<128x128x1x1xf16, {order = #NHWC}>
          to tensor<?x128x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 128, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
      %conv = VPU.NCE.Convolution(%act_slice, %w_slice) rawFilterShape [128, 128, 1, 1] {
          resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          strides = [1, 1]
      } : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<?x128x1x1xf16, {bounds = #const.OpaqueI64Elements<[128, 128, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
        -> tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      %cast = tensor.cast %conv
          : tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          to tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %cast into %out[0, %real_off, 0, 0] [1, %real_sz, %tile_sz, 64] [1, 1, 1, 1]
            : tensor<1x?x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
            into tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
      }
    }
    %cast = tensor.cast %forall
        : tensor<1x128x?x64xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
        to tensor<1x128x32x64xf16, {order = #NHWC}>
    %inserted = tensor.insert_slice %cast into %iter[0, 0, %iv, 0] [1, 128, 32, 64] [1, 1, 1, 1]
        : tensor<1x128x32x64xf16, {order = #NHWC}> into tensor<1x128x64x64xf16, {order = #NHWC}>
    scf.yield %inserted : tensor<1x128x64x64xf16, {order = #NHWC}>
  }
  return %result : tensor<1x128x64x64xf16, {order = #NHWC}>

    // Verify no SCF loops remain after unrolling
    // CHECK-NOT: scf.for
    // CHECK-NOT: scf.forall

    // First tiling iteration: input slice [0:32] with balanced SOK distribution
    // CHECK:        [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 128, 32, 64]
    // CHECK:        [[IN_COPY0:%.+]] = VPU.Copy([[SLICE0]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64

    // CHECK:        [[W_COPY0:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:       -> !VPU.DistributedTensor<128x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME{LITERAL}:     compute_shapes = [[32, 128, 1, 1], [32, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]]

    // CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[IN_COPY0]], [[W_COPY0]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]]

    // CHECK:        [[OUT_COPY0:%.+]] = VPU.Copy([[CONV0]])
    // CHECK-SAME:       -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // Second tiling iteration: input slice [32:64] with same balanced SOK
    // CHECK:        [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 32, 0] [1, 128, 32, 64]
    // CHECK:        [[IN_COPY1:%.+]] = VPU.Copy([[SLICE1]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64

    // CHECK:        [[W_COPY1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:       -> !VPU.DistributedTensor<128x128x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME{LITERAL}:     compute_shapes = [[32, 128, 1, 1], [32, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1], [16, 128, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [80, 0, 0, 0], [96, 0, 0, 0], [112, 0, 0, 0]]

    // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[IN_COPY1]], [[W_COPY1]])
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64], [1, 16, 32, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 80, 0, 0], [0, 96, 0, 0], [0, 112, 0, 0]]

    // CHECK:        [[OUT_COPY1:%.+]] = VPU.Copy([[CONV1]])
    // CHECK-SAME:       -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[OUT_COPY0]], [[OUT_COPY1]])
    // CHECK:        return [[CONCAT]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

// Uniform SOK: 64 OC / 2 clusters = [32, 32]. Even split, no balanced arithmetic.
// After unrolling: SEGMENTED weights and output with uniform [32, 32] per cluster.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UniformSOK_2Clusters_Unroll
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x32x64x64xf16, {order = #NHWC}>
// CHECK-SAME:       [[WEIGHTS:%[^:]+]]: tensor<64x32x1x1xf16, {order = #NHWC}>
func.func @UniformSOK_2Clusters_Unroll(
    %arg0: tensor<1x32x64x64xf16, {order = #NHWC}>,
    %arg1: tensor<64x32x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
  %empty = tensor.empty() : tensor<1x64x64x64xf16, {order = #NHWC}>
  %c32 = arith.constant 32 : index
  %result = scf.forall (%iv) in (2)
      shared_outs(%out = %empty) -> (tensor<1x64x64x64xf16, {order = #NHWC}>) {
    %real_off = arith.muli %iv, %c32 : index
    %w_slice = tensor.extract_slice %arg1[%real_off, 0, 0, 0] [%c32, 32, 1, 1] [1, 1, 1, 1]
        : tensor<64x32x1x1xf16, {order = #NHWC}>
        to tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[64, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    %conv = VPU.NCE.Convolution(%arg0, %w_slice) rawFilterShape [64, 32, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>,
        tensor<?x32x1x1xf16, {bounds = #const.OpaqueI64Elements<[64, 32, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
      -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %conv into %out[0, %real_off, 0, 0] [1, %c32, 64, 64] [1, 1, 1, 1]
          : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          into tensor<1x64x64x64xf16, {order = #NHWC}>
    }
  }
  return %result : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-NOT: scf.forall

    // CHECK:        [[IN_COPY:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<1x32x64x64xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x32x64x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 64, 64], [1, 32, 64, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:        [[W_COPY:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:         : tensor<64x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:         -> !VPU.DistributedTensor<64x32x1x1xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:     compute_shapes = [[32, 32, 1, 1], [32, 32, 1, 1]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[IN_COPY]], [[W_COPY]])
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {
    // CHECK-SAME:              mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1],
    // CHECK-SAME{LITERAL}:     compute_shapes = [[1, 32, 64, 64], [1, 32, 64, 64]],
    // CHECK-SAME{LITERAL}:     compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]

    // CHECK:        [[OUT_COPY:%.+]] = VPU.Copy([[CONV]])
    // CHECK-SAME:       : !VPU.DistributedTensor<1x64x64x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1]
    // CHECK-SAME:       -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:        return [[OUT_COPY]] : tensor<1x64x64x64xf16, {order = #NHWC}>
}
