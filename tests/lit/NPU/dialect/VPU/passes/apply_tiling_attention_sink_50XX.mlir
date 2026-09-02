//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// The DPU storage auxiliary buffer size is architecture specific (NPU5010 packs the MatMul
// weight tables), so this scenario is split per platform.

// The sink carries one logit per attention head, packed along its C axis (64 heads here).
// The output distributes those heads across N (kv-group) and C (head-in-group), so the
// flattened head index equals N * headsPerGroup + C. When tiling over N, each kv-group tile
// must read its own sink slice: sink C offset == N_offset * headsPerGroup, size == N_size * headsPerGroup.

// CHECK-LABEL: @TileAttentionSinkOverTwoKvGroups
// CHECK-SAME:  ([[Q:%.+]]: tensor<8x8x1024x64xf16>, [[K:%.+]]: tensor<8x1x1024x64xf16>, [[V:%.+]]: tensor<8x1x64x1024xf16>,
// CHECK-SAME:   [[MASK:%.+]]: tensor<1x1x1024x1024xf16>, [[SCALE:%.+]]: tensor<1x1x1x1xf16>, [[SINK:%.+]]: tensor<1x64x1x1xf16>,
// CHECK-SAME:   [[DATA:%.+]]: tensor<1x1x1024x4224xui8>, [[DPU:%.+]]: tensor<1x1x1x4864xsi32>)
func.func @TileAttentionSinkOverTwoKvGroups(%q: tensor<8x8x1024x64xf16>, %k: tensor<8x1x1024x64xf16>, %v: tensor<8x1x64x1024xf16>,
                                            %mask: tensor<1x1x1024x1024xf16>, %scale: tensor<1x1x1x1xf16>, %sink: tensor<1x64x1x1xf16>,
                                            %dataStorage: tensor<1x1x1024x4224xui8>, %dpuStorage: tensor<1x1x1x4864xsi32>)
                                            -> tensor<8x8x1024x64xf16> {
  %0 = VPU.Attention(%q, %k, %v, %mask, %scale, %sink, %dataStorage, %dpuStorage) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 1, 1>,
        tilingStrategy = [2, 1, 1, 1]
      } : tensor<8x8x1024x64xf16>, tensor<8x1x1024x64xf16>, tensor<8x1x64x1024xf16>,
          tensor<1x1x1024x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x64x1x1xf16>,
          tensor<1x1x1024x4224xui8>, tensor<1x1x1x4864xsi32> -> tensor<8x8x1024x64xf16>
  return %0 : tensor<8x8x1024x64xf16>

  // Tile 0: kv-groups [0, 4) -> heads [0, 32)
  // CHECK:      [[Q0:%.+]] = VPU.Slice [[Q]] [0, 0, 0, 0] [4, 8, 1024, 64]
  // CHECK:      [[SINK0:%.+]] = VPU.Slice [[SINK]] [0, 0, 0, 0] [1, 32, 1, 1] : tensor<1x64x1x1xf16> to tensor<1x32x1x1xf16>
  // CHECK:      [[ATT0:%.+]] = VPU.Attention([[Q0]],
  // CHECK-SAME:      [[SINK0]],

  // Tile 1: kv-groups [4, 8) -> heads [32, 64)
  // CHECK:      [[Q1:%.+]] = VPU.Slice [[Q]] [4, 0, 0, 0] [4, 8, 1024, 64]
  // CHECK:      [[SINK1:%.+]] = VPU.Slice [[SINK]] [0, 32, 0, 0] [1, 32, 1, 1] : tensor<1x64x1x1xf16> to tensor<1x32x1x1xf16>
  // CHECK:      [[ATT1:%.+]] = VPU.Attention([[Q1]],
  // CHECK-SAME:      [[SINK1]],

  // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[ATT0]], [[ATT1]])
  // CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [4, 0, 0, 0]]
  // CHECK:      return [[CONCAT]]
}

// -----

// A per-query 2D sink carries one logit per query row (H == target sequence length). When the
// output is tiled over H, the sink must follow the same H slicing instead of keeping full length.

// CHECK-LABEL: @TilePerQuerySinkOverHeight
// CHECK-SAME:  ([[Q:%.+]]: tensor<1x64x1024x64xf16>, [[K:%.+]]: tensor<1x64x1024x64xf16>, [[V:%.+]]: tensor<1x64x64x1024xf16>,
// CHECK-SAME:   [[MASK:%.+]]: tensor<1x1x1024x1024xf16>, [[SCALE:%.+]]: tensor<1x1x1x1xf16>, [[SINK:%.+]]: tensor<1x64x1024x1xf16>,
// CHECK-SAME:   [[DATA:%.+]]: tensor<1x1x1024x4224xui8>, [[DPU:%.+]]: tensor<1x1x1x4864xsi32>)
func.func @TilePerQuerySinkOverHeight(%q: tensor<1x64x1024x64xf16>, %k: tensor<1x64x1024x64xf16>, %v: tensor<1x64x64x1024xf16>,
                                      %mask: tensor<1x1x1024x1024xf16>, %scale: tensor<1x1x1x1xf16>, %sink: tensor<1x64x1024x1xf16>,
                                      %dataStorage: tensor<1x1x1024x4224xui8>, %dpuStorage: tensor<1x1x1x4864xsi32>)
                                      -> tensor<1x64x1024x64xf16> {
  %0 = VPU.Attention(%q, %k, %v, %mask, %scale, %sink, %dataStorage, %dpuStorage) {
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 1, 1>,
        tilingStrategy = [1, 1, 2, 1]
      } : tensor<1x64x1024x64xf16>, tensor<1x64x1024x64xf16>, tensor<1x64x64x1024xf16>,
          tensor<1x1x1024x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x64x1024x1xf16>,
          tensor<1x1x1024x4224xui8>, tensor<1x1x1x4864xsi32> -> tensor<1x64x1024x64xf16>
  return %0 : tensor<1x64x1024x64xf16>

  // Tile 0: query rows [0, 512)
  // CHECK:      [[SINK0:%.+]] = VPU.Slice [[SINK]] [0, 0, 0, 0] [1, 64, 512, 1] : tensor<1x64x1024x1xf16> to tensor<1x64x512x1xf16>
  // CHECK:      [[ATT0:%.+]] = VPU.Attention(
  // CHECK-SAME:      [[SINK0]],

  // Tile 1: query rows [512, 1024)
  // CHECK:      [[SINK1:%.+]] = VPU.Slice [[SINK]] [0, 0, 512, 0] [1, 64, 512, 1] : tensor<1x64x1024x1xf16> to tensor<1x64x512x1xf16>
  // CHECK:      [[ATT1:%.+]] = VPU.Attention(
  // CHECK-SAME:      [[SINK1]],

  // CHECK:      [[CONCAT:%.+]] = VPU.Concat([[ATT0]], [[ATT1]])
  // CHECK:      return [[CONCAT]]
}
