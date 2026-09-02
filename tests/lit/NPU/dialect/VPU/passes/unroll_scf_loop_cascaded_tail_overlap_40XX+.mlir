//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,10,1 enable-cascaded-unrolling=true enable-backtracking-beyond-residual-kernel=true tail-overlap-backtrack-margin-percent=30" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// Tail-overlap selection summary:
// The pass evaluates each cascaded stage in turn and reuses that same stage for the
// tail only when the runtime remainder is non-empty and small enough under the
// configured backtrack margin. In this test the stage factors are 470, 235,
// and 94 before the final 47-stage residual, so any of those stages can become
// the overlap stage once the remaining tail is short enough. When the tail is
// too large, the stage keeps its safe upper bound and the next smaller factor
// handles the rest greedily; when the tail is small enough, the stage backs up
// its epilogue lower bound and covers the tail with the same kernel, and the
// final 47-stage only runs as the residual fallback if anything remains.

// The actual backtrack amount depends on runtime shape:
// - If the runtime tail is large, the stage keeps its safe upper bound and the
//   next stage handles the remaining work greedily.
// - If the runtime tail is small enough for the configured margin, the current
//   stage backtracks and reuses the same factor to cover the tail.
// - The final residual stage uses factor 47; this is the extra iteration that
//   consumes the remaining tail after overlap backtracking has narrowed the gap.
//   If overlap already covers the full tail, the residual loop becomes zero-trip
//   and does not execute any body iteration.
//
// Examples:
// - Runtime height around 1000: the tail after the 470-stage is small enough for
//   the configured margin, so 470 is reused once more with overlap.
// - Runtime height around 1200: 470 finishes greedily, but the remaining tail is
//   small enough that the 235-stage can still be reused once more with overlap.
// - Runtime height around 1265: 470 and 235 finish greedily, and the remaining
//   tail becomes small enough for the 94-stage to be reused once more with overlap.
// - Runtime height near the lower bound: the remaining tail is handled by the
//   final 47-stage residual, which is the extra iteration after the overlap stage.

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DynamicPermute2DCascaded   {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_13" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NCHW}> {dynamicStrides}
  } outputsInfo : {
    DataInfo "Convert_16" friendlyName = "Result_17" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}> {dynamicStrides}
  }
  func.func @main_func0_static(%arg0: tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x47x512xf16, {order = #NHWC}>
    return %0 : tensor<1x16x47x512xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}> {
    %c342 = arith.constant 342 : index
    %c512 = arith.constant 512 : index
    %c47 = arith.constant 47 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>
    %7 = scf.for %arg2 = %c0 to %dim step %c47 iter_args(%arg3 = %0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>) {
      %24 = scf.for %arg4 = %c0 to %dim_0 step %c512 iter_args(%arg5 = %arg3) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>) {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, %arg4] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x47x512xf16>
        %37 = func.call @main_func0_static(%extracted_slice) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #NHWC}>
        %cast = tensor.cast %37 : tensor<1x16x47x512xf16, {order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg5[0, 0, %arg2, %arg4] [1, 16, %c47, %c512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %24 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %7 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #NHWC}>
  }

// CHECK: #[[$NCHW_LAYOUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$NHWC_LAYOUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[$HEIGHT_STAGE_470_MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 470, d0)>
// CHECK-DAG: #[[$WIDTH_STAGE_512_MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 512, d0)>
// CHECK-DAG: #[[$HEIGHT_STAGE_235_MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 235, d0)>
// CHECK-DAG: #[[$HEIGHT_STAGE_94_MAP:.+]] = affine_map<(d0)[s0] -> (s0 - 94, d0)>

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    [[INPUT_TENSOR:%.+]]: tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}>) -> tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}> {

// CHECK-DAG:     [[MARGIN_28:%.+]] = arith.constant 28 : index
// CHECK-DAG:     [[STAGE_94:%.+]] = arith.constant 94 : index
// CHECK-DAG:     [[MARGIN_70:%.+]] = arith.constant 70 : index
// CHECK-DAG:     [[STAGE_235:%.+]] = arith.constant 235 : index
// CHECK-DAG:     [[MARGIN_141:%.+]] = arith.constant 141 : index
// CHECK-DAG:     [[STAGE_470:%.+]] = arith.constant 470 : index
// CHECK-DAG:     [[ROUNDING_OFFSET_46:%.+]] = arith.constant 46 : index
// CHECK-DAG:     [[DIVISION_GROUP_5:%.+]] = arith.constant 5 : index
// CHECK-DAG:     [[DIVISION_GROUP_10:%.+]] = arith.constant 10 : index
// CHECK-DAG:     [[WIDTH_TILE_512:%.+]] = arith.constant 512 : index
// CHECK-DAG:     [[RESIDUAL_STAGE_47:%.+]] = arith.constant 47 : index
// CHECK-DAG:     [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:     [[WIDTH_DIM:%.+]] = arith.constant 3 : index
// CHECK-DAG:     [[HEIGHT_DIM:%.+]] = arith.constant 2 : index

// CHECK-DAG:     [[RUNTIME_HEIGHT:%.+]] = tensor.dim [[INPUT_TENSOR]], [[HEIGHT_DIM]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}>
// CHECK-DAG:     [[RUNTIME_WIDTH:%.+]] = tensor.dim [[INPUT_TENSOR]], [[WIDTH_DIM]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}>

// Step 1: create the output tensor in the destination layout
// CHECK:     [[OUTPUT_TENSOR:%.+]] = tensor.empty([[RUNTIME_HEIGHT]], [[RUNTIME_WIDTH]]) : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>

// Step 2: derive the first safe 470-stage upper bound and decide whether the stage can overlap the tail
// CHECK:     [[HEIGHT_PLUS_PADDING:%.+]] = arith.addi [[RUNTIME_HEIGHT]], [[ROUNDING_OFFSET_46]] : index
// CHECK:     [[HEIGHT_QUOTIENT_470:%.+]] = arith.divui [[HEIGHT_PLUS_PADDING]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[HEIGHT_REMAINDER_470:%.+]] = arith.remsi [[HEIGHT_QUOTIENT_470]], [[DIVISION_GROUP_10]] : index
// CHECK:     [[HEIGHT_ROUNDED_470:%.+]] = arith.subi [[HEIGHT_QUOTIENT_470]], [[HEIGHT_REMAINDER_470]] : index
// CHECK:     [[HEIGHT_ALIGNED_470:%.+]] = arith.muli [[HEIGHT_ROUNDED_470]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[HAS_470_STAGE:%.+]] = arith.cmpi uge, [[RUNTIME_HEIGHT]], [[STAGE_470]] : index
// CHECK:     [[SAFE_470_END_CANDIDATE:%.+]] = arith.minui [[HEIGHT_ALIGNED_470]], [[RUNTIME_HEIGHT]] : index
// CHECK:     [[SAFE_470_END:%.+]] = arith.select [[HAS_470_STAGE]], [[SAFE_470_END_CANDIDATE]], [[ZERO]] : index
// CHECK:     [[TAIL_AFTER_470:%.+]] = arith.subi [[RUNTIME_HEIGHT]], [[SAFE_470_END]] : index
// CHECK:     [[TAIL_SHORTER_THAN_470:%.+]] = arith.cmpi ult, [[TAIL_AFTER_470]], [[STAGE_470]] : index
// CHECK:     [[SHORTFALL_470:%.+]] = arith.subi [[STAGE_470]], [[TAIL_AFTER_470]] : index
// CHECK:     [[BACKTRACK_WINDOW_470:%.+]] = arith.select [[TAIL_SHORTER_THAN_470]], [[SHORTFALL_470]], [[ZERO]] : index
// CHECK:     [[TAIL_PRESENT_470:%.+]] = arith.cmpi ne, [[TAIL_AFTER_470]], [[ZERO]] : index
// CHECK:     [[WITHIN_MARGIN_470:%.+]] = arith.cmpi ule, [[BACKTRACK_WINDOW_470]], [[MARGIN_141]] : index
// CHECK:     [[OVERLAP_ELIGIBLE_470:%.+]] = arith.andi [[TAIL_SHORTER_THAN_470]], [[WITHIN_MARGIN_470]] : i1
// CHECK:     [[OVERLAP_ACTIVE_470:%.+]] = arith.andi [[TAIL_PRESENT_470]], [[OVERLAP_ELIGIBLE_470]] : i1
// CHECK:     [[BACKTRACK_CAP_470:%.+]] = arith.minui [[BACKTRACK_WINDOW_470]], [[SAFE_470_END]] : index
// CHECK:     [[BACKTRACKED_START_470:%.+]] = arith.subi [[SAFE_470_END]], [[BACKTRACK_CAP_470]] : index
// CHECK:     [[STAGE_START_470:%.+]] = arith.select [[OVERLAP_ACTIVE_470]], [[BACKTRACKED_START_470]], [[SAFE_470_END]] : index

// Step 3: emit the 470 stage and preserve the wide-width tiling
// CHECK:     [[OUTER_STAGE_470_LOOP:%.+]] = scf.for [[OUTER_STAGE_470_IV:%.+]] = [[ZERO]] to [[SAFE_470_END]] step [[STAGE_470]] iter_args([[OUTER_STAGE_470_ACC:%.+]] = [[OUTPUT_TENSOR]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:       [[WIDE_470_LOOP:%.+]] = scf.for [[WIDE_470_IV:%.+]] = [[ZERO]] to [[RUNTIME_WIDTH]] step [[WIDTH_TILE_512]] iter_args([[WIDE_470_ACC:%.+]] = [[OUTER_STAGE_470_ACC]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:         [[HEIGHT_OFFSET_470:%.+]] = affine.min #[[$HEIGHT_STAGE_470_MAP]]([[OUTER_STAGE_470_IV]]){{\[}}[[SAFE_470_END]]
// CHECK:         [[WIDTH_OFFSET_470:%.+]] = affine.min #[[$WIDTH_STAGE_512_MAP]]([[WIDE_470_IV]]){{\[}}[[RUNTIME_WIDTH]]
// CHECK:         [[INPUT_TILE_470:%.+]] = tensor.extract_slice [[INPUT_TENSOR]][0, 0, [[HEIGHT_OFFSET_470]], [[WIDTH_OFFSET_470]]] [1, 16, 470, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}> to tensor<1x16x470x512xf16>
// CHECK:         [[STAGE_CALL_470:%.+]] = func.call @merged_vpu_func_0([[INPUT_TILE_470]]) : (tensor<1x16x470x512xf16>) -> tensor<1x16x470x512xf16, {order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_CAST_470:%.+]] = tensor.cast [[STAGE_CALL_470]] : tensor<1x16x470x512xf16, {order = #[[$NHWC_LAYOUT]]}> to tensor<1x16x470x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_INSERT_470:%.+]] = tensor.insert_slice [[STAGE_CAST_470]] into [[WIDE_470_ACC]][0, 0, [[HEIGHT_OFFSET_470]], [[WIDTH_OFFSET_470]]] [1, 16, 470, 512] [1, 1, 1, 1] : tensor<1x16x470x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         scf.yield [[STAGE_INSERT_470]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:       }
// CHECK:       scf.yield [[WIDE_470_LOOP]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:     } {no_await_all = true}

// Step 4: derive the 235-stage upper bound from the remaining tail and apply the 235-stage overlap rule
// CHECK:     [[REMAINING_AFTER_470:%.+]] = arith.subi [[RUNTIME_HEIGHT]], [[STAGE_START_470]] : index
// CHECK:     [[REMAINING_AFTER_470_PLUS_PADDING:%.+]] = arith.addi [[REMAINING_AFTER_470]], [[ROUNDING_OFFSET_46]] : index
// CHECK:     [[REMAINING_AFTER_470_QUOTIENT:%.+]] = arith.divui [[REMAINING_AFTER_470_PLUS_PADDING]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[REMAINING_AFTER_470_REMAINDER:%.+]] = arith.remsi [[REMAINING_AFTER_470_QUOTIENT]], [[DIVISION_GROUP_5]] : index
// CHECK:     [[REMAINING_AFTER_470_ROUNDED:%.+]] = arith.subi [[REMAINING_AFTER_470_QUOTIENT]], [[REMAINING_AFTER_470_REMAINDER]] : index
// CHECK:     [[REMAINING_AFTER_470_ALIGNED:%.+]] = arith.muli [[REMAINING_AFTER_470_ROUNDED]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[STAGE_CANDIDATE_END_235:%.+]] = arith.addi [[STAGE_START_470]], [[REMAINING_AFTER_470_ALIGNED]] : index
// CHECK:     [[STAGE_GUARD_END_235:%.+]] = arith.addi [[STAGE_START_470]], [[STAGE_235]] : index
// CHECK:     [[STAGE_GUARD_FITS_235:%.+]] = arith.cmpi ule, [[STAGE_GUARD_END_235]], [[RUNTIME_HEIGHT]] : index
// CHECK:     [[STAGE_SAFE_END_235:%.+]] = arith.minui [[STAGE_CANDIDATE_END_235]], [[RUNTIME_HEIGHT]] : index
// CHECK:     [[STAGE_END_235:%.+]] = arith.select [[STAGE_GUARD_FITS_235]], [[STAGE_SAFE_END_235]], [[STAGE_START_470]] : index
// CHECK:     [[TAIL_AFTER_235:%.+]] = arith.subi [[RUNTIME_HEIGHT]], [[STAGE_END_235]] : index
// CHECK:     [[TAIL_SHORTER_THAN_235:%.+]] = arith.cmpi ult, [[TAIL_AFTER_235]], [[STAGE_235]] : index
// CHECK:     [[SHORTFALL_235:%.+]] = arith.subi [[STAGE_235]], [[TAIL_AFTER_235]] : index
// CHECK:     [[BACKTRACK_WINDOW_235:%.+]] = arith.select [[TAIL_SHORTER_THAN_235]], [[SHORTFALL_235]], [[ZERO]] : index
// CHECK:     [[TAIL_PRESENT_235:%.+]] = arith.cmpi ne, [[TAIL_AFTER_235]], [[ZERO]] : index
// CHECK:     [[WITHIN_MARGIN_235:%.+]] = arith.cmpi ule, [[BACKTRACK_WINDOW_235]], [[MARGIN_70]] : index
// CHECK:     [[OVERLAP_ELIGIBLE_235:%.+]] = arith.andi [[TAIL_SHORTER_THAN_235]], [[WITHIN_MARGIN_235]] : i1
// CHECK:     [[OVERLAP_ACTIVE_235:%.+]] = arith.andi [[TAIL_PRESENT_235]], [[OVERLAP_ELIGIBLE_235]] : i1
// CHECK:     [[BACKTRACK_CAP_235:%.+]] = arith.subi [[STAGE_END_235]], [[STAGE_START_470]] : index
// CHECK:     [[BACKTRACK_AMOUNT_235:%.+]] = arith.minui [[BACKTRACK_WINDOW_235]], [[BACKTRACK_CAP_235]] : index
// CHECK:     [[BACKTRACKED_START_235:%.+]] = arith.subi [[STAGE_END_235]], [[BACKTRACK_AMOUNT_235]] : index
// CHECK:     [[STAGE_START_235:%.+]] = arith.select [[OVERLAP_ACTIVE_235]], [[BACKTRACKED_START_235]], [[STAGE_END_235]] : index

// Step 5: emit the 235 stage and thread its result into the next stage
// CHECK:     [[TAIL_235_LOOP:%.+]] = scf.for [[TAIL_235_IV:%.+]] = [[STAGE_START_470]] to [[STAGE_END_235]] step [[STAGE_235]] iter_args([[TAIL_235_ACC:%.+]] = [[OUTER_STAGE_470_LOOP]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:       [[WIDE_235_LOOP:%.+]] = scf.for [[WIDE_235_IV:%.+]] = [[ZERO]] to [[RUNTIME_WIDTH]] step [[WIDTH_TILE_512]] iter_args([[WIDE_235_ACC:%.+]] = [[TAIL_235_ACC]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:         [[HEIGHT_OFFSET_235:%.+]] = affine.min #[[$HEIGHT_STAGE_235_MAP]]([[TAIL_235_IV]]){{\[}}[[STAGE_END_235]]
// CHECK:         [[WIDTH_OFFSET_235:%.+]] = affine.min #[[$WIDTH_STAGE_512_MAP]]([[WIDE_235_IV]]){{\[}}[[RUNTIME_WIDTH]]
// CHECK:         [[INPUT_TILE_235:%.+]] = tensor.extract_slice [[INPUT_TENSOR]][0, 0, [[HEIGHT_OFFSET_235]], [[WIDTH_OFFSET_235]]] [1, 16, 235, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}> to tensor<1x16x235x512xf16>
// CHECK:         [[STAGE_CALL_235:%.+]] = func.call @merged_vpu_func_1([[INPUT_TILE_235]]) : (tensor<1x16x235x512xf16>) -> tensor<1x16x235x512xf16, {order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_CAST_235:%.+]] = tensor.cast [[STAGE_CALL_235]] : tensor<1x16x235x512xf16, {order = #[[$NHWC_LAYOUT]]}> to tensor<1x16x235x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_INSERT_235:%.+]] = tensor.insert_slice [[STAGE_CAST_235]] into [[WIDE_235_ACC]][0, 0, [[HEIGHT_OFFSET_235]], [[WIDTH_OFFSET_235]]] [1, 16, 235, 512] [1, 1, 1, 1] : tensor<1x16x235x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         scf.yield [[STAGE_INSERT_235]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:       }
// CHECK:       scf.yield [[WIDE_235_LOOP]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:     } {no_await_all = true, no_reset_cmdlist = true}

// Step 6: derive the 94-stage upper bound from the remaining tail and apply the tighter overlap rule
// CHECK:     [[REMAINING_AFTER_235:%.+]] = arith.subi [[RUNTIME_HEIGHT]], [[STAGE_START_235]] : index
// CHECK:     [[REMAINING_AFTER_235_PLUS_PADDING:%.+]] = arith.addi [[REMAINING_AFTER_235]], [[ROUNDING_OFFSET_46]] : index
// CHECK:     [[REMAINING_AFTER_235_QUOTIENT:%.+]] = arith.divui [[REMAINING_AFTER_235_PLUS_PADDING]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[REMAINING_AFTER_235_REMAINDER:%.+]] = arith.remsi [[REMAINING_AFTER_235_QUOTIENT]], [[HEIGHT_DIM]] : index
// CHECK:     [[REMAINING_AFTER_235_ROUNDED:%.+]] = arith.subi [[REMAINING_AFTER_235_QUOTIENT]], [[REMAINING_AFTER_235_REMAINDER]] : index
// CHECK:     [[REMAINING_AFTER_235_ALIGNED:%.+]] = arith.muli [[REMAINING_AFTER_235_ROUNDED]], [[RESIDUAL_STAGE_47]] : index
// CHECK:     [[STAGE_CANDIDATE_END_94:%.+]] = arith.addi [[STAGE_START_235]], [[REMAINING_AFTER_235_ALIGNED]] : index
// CHECK:     [[STAGE_GUARD_END_94:%.+]] = arith.addi [[STAGE_START_235]], [[STAGE_94]] : index
// CHECK:     [[STAGE_GUARD_FITS_94:%.+]] = arith.cmpi ule, [[STAGE_GUARD_END_94]], [[RUNTIME_HEIGHT]] : index
// CHECK:     [[STAGE_SAFE_END_94:%.+]] = arith.minui [[STAGE_CANDIDATE_END_94]], [[RUNTIME_HEIGHT]] : index
// CHECK:     [[STAGE_END_94:%.+]] = arith.select [[STAGE_GUARD_FITS_94]], [[STAGE_SAFE_END_94]], [[STAGE_START_235]] : index
// CHECK:     [[TAIL_AFTER_94:%.+]] = arith.subi [[RUNTIME_HEIGHT]], [[STAGE_END_94]] : index
// CHECK:     [[TAIL_SHORTER_THAN_94:%.+]] = arith.cmpi ult, [[TAIL_AFTER_94]], [[STAGE_94]] : index
// CHECK:     [[SHORTFALL_94:%.+]] = arith.subi [[STAGE_94]], [[TAIL_AFTER_94]] : index
// CHECK:     [[BACKTRACK_WINDOW_94:%.+]] = arith.select [[TAIL_SHORTER_THAN_94]], [[SHORTFALL_94]], [[ZERO]] : index
// CHECK:     [[TAIL_PRESENT_94:%.+]] = arith.cmpi ne, [[TAIL_AFTER_94]], [[ZERO]] : index
// CHECK:     [[WITHIN_MARGIN_94:%.+]] = arith.cmpi ule, [[BACKTRACK_WINDOW_94]], [[MARGIN_28]] : index
// CHECK:     [[OVERLAP_ELIGIBLE_94:%.+]] = arith.andi [[TAIL_SHORTER_THAN_94]], [[WITHIN_MARGIN_94]] : i1
// CHECK:     [[OVERLAP_ACTIVE_94:%.+]] = arith.andi [[TAIL_PRESENT_94]], [[OVERLAP_ELIGIBLE_94]] : i1
// CHECK:     [[BACKTRACK_CAP_94:%.+]] = arith.subi [[STAGE_END_94]], [[STAGE_START_235]] : index
// CHECK:     [[BACKTRACK_AMOUNT_94:%.+]] = arith.minui [[BACKTRACK_WINDOW_94]], [[BACKTRACK_CAP_94]] : index
// CHECK:     [[BACKTRACKED_START_94:%.+]] = arith.subi [[STAGE_END_94]], [[BACKTRACK_AMOUNT_94]] : index
// CHECK:     [[STAGE_START_94:%.+]] = arith.select [[OVERLAP_ACTIVE_94]], [[BACKTRACKED_START_94]], [[STAGE_END_94]] : index

// Step 7: emit the 94 stage and keep the same width tiling
// CHECK:     [[TAIL_94_LOOP:%.+]] = scf.for [[TAIL_94_IV:%.+]] = [[STAGE_START_235]] to [[STAGE_END_94]] step [[STAGE_94]] iter_args([[TAIL_94_ACC:%.+]] = [[TAIL_235_LOOP]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:       [[WIDE_94_LOOP:%.+]] = scf.for [[WIDE_94_IV:%.+]] = [[ZERO]] to [[RUNTIME_WIDTH]] step [[WIDTH_TILE_512]] iter_args([[WIDE_94_ACC:%.+]] = [[TAIL_94_ACC]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:         [[HEIGHT_OFFSET_94:%.+]] = affine.min #[[$HEIGHT_STAGE_94_MAP]]([[TAIL_94_IV]]){{\[}}[[STAGE_END_94]]
// CHECK:         [[WIDTH_OFFSET_94:%.+]] = affine.min #[[$WIDTH_STAGE_512_MAP]]([[WIDE_94_IV]]){{\[}}[[RUNTIME_WIDTH]]
// CHECK:         [[INPUT_TILE_94:%.+]] = tensor.extract_slice [[INPUT_TENSOR]][0, 0, [[HEIGHT_OFFSET_94]], [[WIDTH_OFFSET_94]]] [1, 16, 94, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}> to tensor<1x16x94x512xf16>
// CHECK:         [[STAGE_CALL_94:%.+]] = func.call @merged_vpu_func_2([[INPUT_TILE_94]]) : (tensor<1x16x94x512xf16>) -> tensor<1x16x94x512xf16, {order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_CAST_94:%.+]] = tensor.cast [[STAGE_CALL_94]] : tensor<1x16x94x512xf16, {order = #[[$NHWC_LAYOUT]]}> to tensor<1x16x94x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_INSERT_94:%.+]] = tensor.insert_slice [[STAGE_CAST_94]] into [[WIDE_94_ACC]][0, 0, [[HEIGHT_OFFSET_94]], [[WIDTH_OFFSET_94]]] [1, 16, 94, 512] [1, 1, 1, 1] : tensor<1x16x94x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         scf.yield [[STAGE_INSERT_94]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:       }
// CHECK:       scf.yield [[WIDE_94_LOOP]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:     } {no_await_all = true, no_reset_cmdlist = true}

// Step 8: emit the residual 47-stage that consumes the final tail
// CHECK:     [[RESIDUAL_LOOP_47:%.+]] = scf.for [[RESIDUAL_IV_47:%.+]] = [[STAGE_START_94]] to [[RUNTIME_HEIGHT]] step [[RESIDUAL_STAGE_47]] iter_args([[RESIDUAL_ACC_47:%.+]] = [[TAIL_94_LOOP]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:       [[RESIDUAL_WIDE_LOOP_47:%.+]] = scf.for [[RESIDUAL_WIDE_IV_47:%.+]] = [[ZERO]] to [[RUNTIME_WIDTH]] step [[WIDTH_TILE_512]] iter_args([[RESIDUAL_WIDE_ACC_47:%.+]] = [[RESIDUAL_ACC_47]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>) {
// CHECK:         [[INPUT_TILE_47:%.+]] = tensor.extract_slice [[INPUT_TENSOR]][0, 0, [[RESIDUAL_IV_47]], [[RESIDUAL_WIDE_IV_47]]] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NCHW_LAYOUT]]}> to tensor<1x16x47x512xf16>
// CHECK:         [[STAGE_CALL_47:%.+]] = func.call @main_func0_static([[INPUT_TILE_47]]) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_CAST_47:%.+]] = tensor.cast [[STAGE_CALL_47]] : tensor<1x16x47x512xf16, {order = #[[$NHWC_LAYOUT]]}> to tensor<1x16x47x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         [[STAGE_INSERT_47:%.+]] = tensor.insert_slice [[STAGE_CAST_47]] into [[RESIDUAL_WIDE_ACC_47]][0, 0, [[RESIDUAL_IV_47]], [[RESIDUAL_WIDE_IV_47]]] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x47x512xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:         scf.yield [[STAGE_INSERT_47]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:       }
// CHECK:       scf.yield [[RESIDUAL_WIDE_LOOP_47]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:     }
// CHECK:     return [[RESIDUAL_LOOP_47]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 799, 1024]> : tensor<4xsi64>, order = #[[$NHWC_LAYOUT]]}>
// CHECK:   }
}
