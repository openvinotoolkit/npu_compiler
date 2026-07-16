//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --compress-weights-btc="fail-if-no-compression=true" %s
// REQUIRES: dev-build && (platform-NPU4000 || platform-NPU5010)

!qElemType = !quant.uniform<u8:f16, 1.0000000000000000E-1>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// expected-error@+1 {{No operations were compressed. Failing the pass because 'failIfNoCompression' is enabled.}}
func.func @CompressWeightsNoCompression() -> !VPUIP.DistributedBuffer<64x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
  %cst = const.Declare memref<64x16x1x1x!qElemType, #NHWC> = dense<1> : tensor<64x16x1x1xui8>, [#const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %0 = VPURT.DeclareBuffer <CMX_NN> <1605632> -> !VPUIP.DistributedBuffer<64x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %609 = VPUIP.NNDMA
      inputs(%cst : memref<64x16x1x1x!qElemType, #NHWC>)
      outputs(%0 : !VPUIP.DistributedBuffer<64x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
      -> !VPUIP.DistributedBuffer<64x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }

  return %0 : !VPUIP.DistributedBuffer<64x16x1x1x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
}
