//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --add-kernels-for-overlapping-edge-tiles %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#map_bt = affine_map<(d0)[s0] -> (s0 - 100, d0)>
#map_in = affine_map<(d0) -> (0, d0 - 1)>

// Low resolution dim H: the input slices are tiled on dim 2, the output tile is inserted on dim 1 (1xHxWxC output).
// The FULLBLK slice moves to the tile offset bt. Tile 1 gets case 12 + 16, which calls a clone of the FULLBLK kernel
// that takes the first O_l = 1 outputs from the accumulator (%prev) and the other 99 from the kernel; the insert is kept
// CHECK-LABEL: @LowResOneDim
module @LowResOneDim {
  func.func @kernel_middle(%arg0: tensor<1x16x102x64xf16>) -> tensor<1x100x64x16xf16> {
    %cst = const.Declare tensor<1x100x64x16xf16> = dense<0.000000e+00> : tensor<1x100x64x16xf16>
    return %cst : tensor<1x100x64x16xf16>
  }
  func.func @kernel_fullblk(%arg0: tensor<1x16x100x64xf16>) -> tensor<1x100x64x16xf16> {
    %cst = const.Declare tensor<1x100x64x16xf16> = dense<0.000000e+00> : tensor<1x100x64x16xf16>
    return %cst : tensor<1x100x64x16xf16>
  }

  func.func @main(%arg0: tensor<1x16x?x64xf16>, %out: tensor<1x?x64x16xf16>) -> tensor<1x?x64x16xf16> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c100 = arith.constant 100 : index
    %c101 = arith.constant 101 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x64xf16>
    %lowres = arith.cmpi ule, %dim, %c101 {vpux.lowres_pred = 0 : i64} : index
    %res = scf.for %iv = %c0 to %dim step %c100 iter_args(%acc = %out) -> (tensor<1x?x64x16xf16>) {
      %bt = affine.min #map_bt(%iv)[%dim]
      %in_off = affine.max #map_in(%bt)
      %pos = arith.select %lowres, %c3, %c0 : index
      %case = arith.shli %pos, %c2 : index
      %tile = scf.index_switch %case -> tensor<1x100x64x16xf16>
      case 0 {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 102, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16> to tensor<1x16x102x64xf16>
        %k = func.call @kernel_middle(%s) : (tensor<1x16x102x64xf16>) -> tensor<1x100x64x16xf16>
        scf.yield %k : tensor<1x100x64x16xf16>
      }
      case 12 {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 100, 64] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 0>} : tensor<1x16x?x64xf16> to tensor<1x16x100x64xf16>
        %k = func.call @kernel_fullblk(%s) : (tensor<1x16x100x64xf16>) -> tensor<1x100x64x16xf16>
        scf.yield %k : tensor<1x100x64x16xf16>
      }
      default {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 102, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16> to tensor<1x16x102x64xf16>
        %k = func.call @kernel_middle(%s) : (tensor<1x16x102x64xf16>) -> tensor<1x100x64x16xf16>
        scf.yield %k : tensor<1x100x64x16xf16>
      }
      %ins = tensor.insert_slice %tile into %acc[0, %bt, 0, 0] [1, 100, 64, 16] [1, 1, 1, 1] : tensor<1x100x64x16xf16> into tensor<1x?x64x16xf16>
      scf.yield %ins : tensor<1x?x64x16xf16>
    } {vpux.lowres = #VPU.LowRes<id = 0, dim = 2, step = 100, invalid_output_left = 1, invalid_output_right = 1>}
    return %res : tensor<1x?x64x16xf16>
  }

  // CHECK:      func.func @kernel_fullblk_overlap_tile1_d1({{%.+}}: tensor<1x16x100x64xf16>, [[PREV:%.+]]: tensor<1x1x64x16xf16> {func.dynamicStrides = true}) -> tensor<1x100x64x16xf16>
  // CHECK:        [[RESULT:%.+]] = const.Declare
  // CHECK:        [[BODY:%.+]] = VPU.Slice [[RESULT]] [0, 1, 0, 0] [1, 99, 64, 16] : tensor<1x100x64x16xf16> to tensor<1x99x64x16xf16>
  // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[PREV]], [[BODY]]) {per_axis = #IE.Concat<axis = 1 : i64>}
  // CHECK:        return [[CONCAT]] : tensor<1x100x64x16xf16>

  // CHECK:      func.func @main
  // CHECK:        [[LOW_RES:%.+]] = arith.cmpi ule, {{%[a-z0-9_]+}}, {{%[a-z0-9_]+}} : index
  // CHECK:        scf.for [[IV:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[ACC:%.+]] = {{%.+}})
  // CHECK:          [[BT:%.+]] = affine.min #{{.+}}([[IV]])
  // CHECK:          [[IN_OFF:%.+]] = affine.max #{{.+}}([[BT]])
  // CHECK:          [[CASE:%.+]] = arith.shli
  // CHECK-DAG:      [[C0:%.+]] = arith.constant 0 : index
  // CHECK-DAG:      [[C16:%.+]] = arith.constant 16 : index
  // CHECK:          [[NON_ZERO:%.+]] = arith.cmpi ne, [[BT]], [[C0]] : index
  // CHECK:          [[TILE_ONE:%.+]] = arith.andi [[LOW_RES]], [[NON_ZERO]] : i1
  // CHECK:          [[SHIFT:%.+]] = arith.select [[TILE_ONE]], [[C16]], [[C0]] : index
  // CHECK:          [[SEL:%.+]] = arith.addi [[CASE]], [[SHIFT]] : index
  // CHECK:          [[TILE:%.+]] = scf.index_switch [[SEL]]
  // CHECK:          case 0 {
  // CHECK:            tensor.extract_slice %arg0[0, 0, [[IN_OFF]], 0] [1, 16, 102, 64] [1, 1, 1, 1] :
  // CHECK:          case 12 {
  // CHECK:            [[S12:%.+]] = tensor.extract_slice %arg0[0, 0, [[BT]], 0] [1, 16, 100, 64] [1, 1, 1, 1] :
  // CHECK:            func.call @kernel_fullblk([[S12]])
  // CHECK:          case 28 {
  // CHECK:            [[S28:%.+]] = tensor.extract_slice %arg0[0, 0, [[BT]], 0] [1, 16, 100, 64] [1, 1, 1, 1] :
  // CHECK:            [[HEAD:%.+]] = tensor.extract_slice [[ACC]][0, [[BT]], 0, 0] [1, 1, 64, 16] [1, 1, 1, 1] : tensor<1x?x64x16xf16> to tensor<1x1x64x16xf16>
  // CHECK:            func.call @kernel_fullblk_overlap_tile1_d1([[S28]], [[HEAD]])
  // CHECK:          default {
  // CHECK:            tensor.extract_slice %arg0[0, 0, [[IN_OFF]], 0] [1, 16, 102, 64] [1, 1, 1, 1] :
  // CHECK:          [[INS:%.+]] = tensor.insert_slice [[TILE]] into [[ACC]][0, [[BT]], 0, 0] [1, 100, 64, 16] [1, 1, 1, 1]
  // CHECK-NEXT:     scf.yield [[INS]]
  // CHECK-NOT:  vpux.lowres
}

// -----

#map_bt = affine_map<(d0)[s0] -> (s0 - 100, d0)>
#map_in = affine_map<(d0) -> (0, d0 - 1)>

// Unrolled main loop (step 2 * S): it never runs a low resolution tile, so it is not changed (no tile-1 case, the
// FULLBLK slice keeps its offset); the tag is removed
// CHECK-LABEL: @UnrolledMainLoop
module @UnrolledMainLoop {
  func.func @kernel_middle(%arg0: tensor<1x16x202x64xf16>) -> tensor<1x16x200x64xf16> {
    %cst = const.Declare tensor<1x16x200x64xf16> = dense<0.000000e+00> : tensor<1x16x200x64xf16>
    return %cst : tensor<1x16x200x64xf16>
  }
  func.func @kernel_fullblk(%arg0: tensor<1x16x200x64xf16>) -> tensor<1x16x200x64xf16> {
    %cst = const.Declare tensor<1x16x200x64xf16> = dense<0.000000e+00> : tensor<1x16x200x64xf16>
    return %cst : tensor<1x16x200x64xf16>
  }

  func.func @main(%arg0: tensor<1x16x?x64xf16>, %out: tensor<1x16x?x64xf16>) -> tensor<1x16x?x64xf16> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c100 = arith.constant 100 : index
    %c101 = arith.constant 101 : index
    %step = arith.muli %c100, %c2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x64xf16>
    %lowres = arith.cmpi ule, %dim, %c101 {vpux.lowres_pred = 0 : i64} : index
    %res = scf.for %iv = %c0 to %dim step %step iter_args(%acc = %out) -> (tensor<1x16x?x64xf16>) {
      %bt = affine.min #map_bt(%iv)[%dim]
      %in_off = affine.max #map_in(%bt)
      %pos = arith.select %lowres, %c3, %c0 : index
      %case = arith.shli %pos, %c2 : index
      %tile = scf.index_switch %case -> tensor<1x16x200x64xf16>
      case 0 {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 202, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16> to tensor<1x16x202x64xf16>
        %k = func.call @kernel_middle(%s) : (tensor<1x16x202x64xf16>) -> tensor<1x16x200x64xf16>
        scf.yield %k : tensor<1x16x200x64xf16>
      }
      case 12 {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 200, 64] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 0>} : tensor<1x16x?x64xf16> to tensor<1x16x200x64xf16>
        %k = func.call @kernel_fullblk(%s) : (tensor<1x16x200x64xf16>) -> tensor<1x16x200x64xf16>
        scf.yield %k : tensor<1x16x200x64xf16>
      }
      default {
        %s = tensor.extract_slice %arg0[0, 0, %in_off, 0] [1, 16, 202, 64] [1, 1, 1, 1] : tensor<1x16x?x64xf16> to tensor<1x16x202x64xf16>
        %k = func.call @kernel_middle(%s) : (tensor<1x16x202x64xf16>) -> tensor<1x16x200x64xf16>
        scf.yield %k : tensor<1x16x200x64xf16>
      }
      %ins = tensor.insert_slice %tile into %acc[0, 0, %bt, 0] [1, 16, 200, 64] [1, 1, 1, 1] : tensor<1x16x200x64xf16> into tensor<1x16x?x64xf16>
      scf.yield %ins : tensor<1x16x?x64xf16>
    } {vpux.lowres = #VPU.LowRes<id = 0, dim = 2, step = 100, invalid_output_left = 1, invalid_output_right = 1>}
    return %res : tensor<1x16x?x64xf16>
  }

  // CHECK-NOT:  overlap_tile1
  // CHECK:      scf.for
  // CHECK:        [[BT:%.+]] = affine.min
  // CHECK:        [[IN_OFF:%.+]] = affine.max #{{.+}}([[BT]])
  // CHECK:        [[CASE:%.+]] = arith.shli
  // CHECK:        scf.index_switch [[CASE]]
  // CHECK:        case 12 {
  // CHECK:          tensor.extract_slice %arg0[0, 0, [[IN_OFF]], 0] [1, 16, 200, 64] [1, 1, 1, 1] :
  // CHECK-NOT:    case 28
  // CHECK:        [[INS:%.+]] = tensor.insert_slice {{%.+}} into {{%.+}}[0, 0, [[BT]], 0] [1, 16, 200, 64] [1, 1, 1, 1]
  // CHECK-NEXT:   scf.yield [[INS]]
  // CHECK-NOT:  vpux.lowres
}

// -----

#map_bt_h = affine_map<(d0)[s0] -> (s0 - 100, d0)>
#map_bt_w = affine_map<(d0)[s0] -> (s0 - 60, d0)>
#map_in = affine_map<(d0) -> (0, d0 - 1)>

// Both dims low resolution (H id 0, W id 1): every FULLBLK slice moves to bt on its tagged dims. W (inner loop) is
// processed first: cases 3 and 15 get the W tile-1 cases 19 and 31 (K = 16). Then H: the H FULLBLK cases 12, 15 and
// the W tile-1 case 31 get 44, 47 and 63 (K = 32). Case 63 calls a kernel cloned for W and then for H, which takes
// both accumulator heads; the corner comes from the H head, i.e. from the accumulator. The insert is kept
// CHECK-LABEL: @LowResTwoDims
module @LowResTwoDims {
  func.func @kernel_middle(%arg0: tensor<1x16x102x62xf16>) -> tensor<1x16x100x60xf16> {
    %cst = const.Declare tensor<1x16x100x60xf16> = dense<0.000000e+00> : tensor<1x16x100x60xf16>
    return %cst : tensor<1x16x100x60xf16>
  }
  func.func @kernel_w(%arg0: tensor<1x16x102x60xf16>) -> tensor<1x16x100x60xf16> {
    %cst = const.Declare tensor<1x16x100x60xf16> = dense<0.000000e+00> : tensor<1x16x100x60xf16>
    return %cst : tensor<1x16x100x60xf16>
  }
  func.func @kernel_h(%arg0: tensor<1x16x100x62xf16>) -> tensor<1x16x100x60xf16> {
    %cst = const.Declare tensor<1x16x100x60xf16> = dense<0.000000e+00> : tensor<1x16x100x60xf16>
    return %cst : tensor<1x16x100x60xf16>
  }
  func.func @kernel_hw(%arg0: tensor<1x16x100x60xf16>) -> tensor<1x16x100x60xf16> {
    %cst = const.Declare tensor<1x16x100x60xf16> = dense<0.000000e+00> : tensor<1x16x100x60xf16>
    return %cst : tensor<1x16x100x60xf16>
  }

  func.func @main(%arg0: tensor<1x16x?x?xf16>, %out: tensor<1x16x?x?xf16>) -> tensor<1x16x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c60 = arith.constant 60 : index
    %c61 = arith.constant 61 : index
    %c100 = arith.constant 100 : index
    %c101 = arith.constant 101 : index
    %dim_h = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16>
    %dim_w = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16>
    %lowres_h = arith.cmpi ule, %dim_h, %c101 {vpux.lowres_pred = 0 : i64} : index
    %lowres_w = arith.cmpi ule, %dim_w, %c61 {vpux.lowres_pred = 1 : i64} : index
    %res = scf.for %iv_h = %c0 to %dim_h step %c100 iter_args(%acc_h = %out) -> (tensor<1x16x?x?xf16>) {
      %bt_h = affine.min #map_bt_h(%iv_h)[%dim_h]
      %in_h = affine.max #map_in(%bt_h)
      %pos_h = arith.select %lowres_h, %c3, %c0 : index
      %res_w = scf.for %iv_w = %c0 to %dim_w step %c60 iter_args(%acc_w = %acc_h) -> (tensor<1x16x?x?xf16>) {
        %bt_w = affine.min #map_bt_w(%iv_w)[%dim_w]
        %in_w = affine.max #map_in(%bt_w)
        %pos_w = arith.select %lowres_w, %c3, %c0 : index
        %shift_h = arith.shli %pos_h, %c2 : index
        %case = arith.ori %shift_h, %pos_w : index
        %tile = scf.index_switch %case -> tensor<1x16x100x60xf16>
        case 0 {
          %s = tensor.extract_slice %arg0[0, 0, %in_h, %in_w] [1, 16, 102, 62] [1, 1, 1, 1] : tensor<1x16x?x?xf16> to tensor<1x16x102x62xf16>
          %k = func.call @kernel_middle(%s) : (tensor<1x16x102x62xf16>) -> tensor<1x16x100x60xf16>
          scf.yield %k : tensor<1x16x100x60xf16>
        }
        case 3 {
          %s = tensor.extract_slice %arg0[0, 0, %in_h, %in_w] [1, 16, 102, 60] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 1>} : tensor<1x16x?x?xf16> to tensor<1x16x102x60xf16>
          %k = func.call @kernel_w(%s) : (tensor<1x16x102x60xf16>) -> tensor<1x16x100x60xf16>
          scf.yield %k : tensor<1x16x100x60xf16>
        }
        case 12 {
          %s = tensor.extract_slice %arg0[0, 0, %in_h, %in_w] [1, 16, 100, 62] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 0>} : tensor<1x16x?x?xf16> to tensor<1x16x100x62xf16>
          %k = func.call @kernel_h(%s) : (tensor<1x16x100x62xf16>) -> tensor<1x16x100x60xf16>
          scf.yield %k : tensor<1x16x100x60xf16>
        }
        case 15 {
          %s = tensor.extract_slice %arg0[0, 0, %in_h, %in_w] [1, 16, 100, 60] [1, 1, 1, 1] {vpux.lowres_slice = array<i64: 0, 1>} : tensor<1x16x?x?xf16> to tensor<1x16x100x60xf16>
          %k = func.call @kernel_hw(%s) : (tensor<1x16x100x60xf16>) -> tensor<1x16x100x60xf16>
          scf.yield %k : tensor<1x16x100x60xf16>
        }
        default {
          %s = tensor.extract_slice %arg0[0, 0, %in_h, %in_w] [1, 16, 102, 62] [1, 1, 1, 1] : tensor<1x16x?x?xf16> to tensor<1x16x102x62xf16>
          %k = func.call @kernel_middle(%s) : (tensor<1x16x102x62xf16>) -> tensor<1x16x100x60xf16>
          scf.yield %k : tensor<1x16x100x60xf16>
        }
        %ins = tensor.insert_slice %tile into %acc_w[0, 0, %bt_h, %bt_w] [1, 16, 100, 60] [1, 1, 1, 1] : tensor<1x16x100x60xf16> into tensor<1x16x?x?xf16>
        scf.yield %ins : tensor<1x16x?x?xf16>
      } {vpux.lowres = #VPU.LowRes<id = 1, dim = 3, step = 60, invalid_output_left = 2, invalid_output_right = 2>}
      scf.yield %res_w : tensor<1x16x?x?xf16>
    } {vpux.lowres = #VPU.LowRes<id = 0, dim = 2, step = 100, invalid_output_left = 1, invalid_output_right = 1>}
    return %res : tensor<1x16x?x?xf16>
  }

  // CHECK:      func.func @kernel_w_overlap_tile1_d3({{%.+}}: tensor<1x16x102x60xf16>, [[W_PREV:%.+]]: tensor<1x16x100x2xf16> {func.dynamicStrides = true})
  // CHECK:        [[W_BODY:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 2] [1, 16, 100, 58]
  // CHECK:        VPU.Concat([[W_PREV]], [[W_BODY]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK:      func.func @kernel_h_overlap_tile1_d2({{%.+}}: tensor<1x16x100x62xf16>, [[H_PREV:%.+]]: tensor<1x16x1x60xf16> {func.dynamicStrides = true})
  // CHECK:        [[H_BODY:%.+]] = VPU.Slice {{%.+}} [0, 0, 1, 0] [1, 16, 99, 60]
  // CHECK:        VPU.Concat([[H_PREV]], [[H_BODY]]) {per_axis = #IE.Concat<axis = 2 : i64>}
  // CHECK:      func.func @kernel_hw_overlap_tile1_d2(
  // CHECK:      func.func @kernel_hw_overlap_tile1_d3(
  // CHECK:      func.func @kernel_hw_overlap_tile1_d3d2({{%.+}}: tensor<1x16x100x60xf16>, [[HW_W_PREV:%.+]]: tensor<1x16x100x2xf16> {func.dynamicStrides = true}, [[HW_H_PREV:%.+]]: tensor<1x16x1x60xf16> {func.dynamicStrides = true})
  // CHECK:        [[HW_W_BODY:%.+]] = VPU.Slice {{%.+}} [0, 0, 0, 2] [1, 16, 100, 58]
  // CHECK:        [[HW_W:%.+]] = VPU.Concat([[HW_W_PREV]], [[HW_W_BODY]]) {per_axis = #IE.Concat<axis = 3 : i64>}
  // CHECK:        [[HW_H_BODY:%.+]] = VPU.Slice [[HW_W]] [0, 0, 1, 0] [1, 16, 99, 60]
  // CHECK:        [[HW:%.+]] = VPU.Concat([[HW_H_PREV]], [[HW_H_BODY]]) {per_axis = #IE.Concat<axis = 2 : i64>}
  // CHECK:        return [[HW]]

  // CHECK:      func.func @main
  // CHECK:        [[LOW_RES_H:%.+]] = arith.cmpi ule, {{%[a-z0-9_]+}}, {{%[a-z0-9_]+}} : index
  // CHECK:        [[LOW_RES_W:%.+]] = arith.cmpi ule, {{%[a-z0-9_]+}}, {{%[a-z0-9_]+}} : index
  // CHECK:        scf.for [[IV_H:%[a-z0-9_]+]] =
  // CHECK:          [[BT_H:%.+]] = affine.min #{{.+}}([[IV_H]])
  // CHECK:          [[IN_H:%.+]] = affine.max #{{.+}}([[BT_H]])
  // CHECK:          scf.for [[IV_W:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[ACC:%.+]] = {{%.+}})
  // CHECK:            [[BT_W:%.+]] = affine.min #{{.+}}([[IV_W]])
  // CHECK:            [[IN_W:%.+]] = affine.max #{{.+}}([[BT_W]])
  // CHECK:            [[CASE:%.+]] = arith.ori

  // W tile-1 shift, then H tile-1 shift
  // CHECK:            [[W_NON_ZERO:%.+]] = arith.cmpi ne, [[BT_W]], {{%.+}} : index
  // CHECK:            [[W_TILE_ONE:%.+]] = arith.andi [[LOW_RES_W]], [[W_NON_ZERO]] : i1
  // CHECK:            [[W_SHIFT:%.+]] = arith.select [[W_TILE_ONE]], {{%.+}}, {{%.+}} : index
  // CHECK:            [[SEL_W:%.+]] = arith.addi [[CASE]], [[W_SHIFT]] : index
  // CHECK:            [[H_NON_ZERO:%.+]] = arith.cmpi ne, [[BT_H]], {{%.+}} : index
  // CHECK:            [[H_TILE_ONE:%.+]] = arith.andi [[LOW_RES_H]], [[H_NON_ZERO]] : i1
  // CHECK:            [[H_SHIFT:%.+]] = arith.select [[H_TILE_ONE]], {{%.+}}, {{%.+}} : index
  // CHECK:            [[SEL:%.+]] = arith.addi [[SEL_W]], [[H_SHIFT]] : index
  // CHECK:            [[TILE:%.+]] = scf.index_switch [[SEL]]
  // CHECK:            case 0 {
  // CHECK:              tensor.extract_slice %arg0[0, 0, [[IN_H]], [[IN_W]]] [1, 16, 102, 62] [1, 1, 1, 1] :
  // CHECK:            case 3 {
  // CHECK:              tensor.extract_slice %arg0[0, 0, [[IN_H]], [[BT_W]]] [1, 16, 102, 60] [1, 1, 1, 1] :
  // CHECK:              func.call @kernel_w(
  // CHECK:            case 12 {
  // CHECK:              tensor.extract_slice %arg0[0, 0, [[BT_H]], [[IN_W]]] [1, 16, 100, 62] [1, 1, 1, 1] :
  // CHECK:              func.call @kernel_h(
  // CHECK:            case 15 {
  // CHECK:              tensor.extract_slice %arg0[0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 60] [1, 1, 1, 1] :
  // CHECK:              func.call @kernel_hw(
  // CHECK:            case 19 {
  // CHECK:              [[S19:%.+]] = tensor.extract_slice %arg0[0, 0, [[IN_H]], [[BT_W]]] [1, 16, 102, 60] [1, 1, 1, 1] :
  // CHECK:              [[W_HEAD19:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 2] [1, 1, 1, 1]
  // CHECK:              func.call @kernel_w_overlap_tile1_d3([[S19]], [[W_HEAD19]])
  // CHECK:            case 31 {
  // CHECK:              [[W_HEAD31:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 2] [1, 1, 1, 1]
  // CHECK:              func.call @kernel_hw_overlap_tile1_d3({{%.+}}, [[W_HEAD31]])
  // CHECK:            case 44 {
  // CHECK:              [[S44:%.+]] = tensor.extract_slice %arg0[0, 0, [[BT_H]], [[IN_W]]] [1, 16, 100, 62] [1, 1, 1, 1] :
  // CHECK:              [[H_HEAD44:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 1, 60] [1, 1, 1, 1]
  // CHECK:              func.call @kernel_h_overlap_tile1_d2([[S44]], [[H_HEAD44]])
  // CHECK:            case 47 {
  // CHECK:              [[H_HEAD47:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 1, 60] [1, 1, 1, 1]
  // CHECK:              func.call @kernel_hw_overlap_tile1_d2({{%.+}}, [[H_HEAD47]])
  // CHECK:            case 63 {
  // CHECK:              [[S63:%.+]] = tensor.extract_slice %arg0[0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 60] [1, 1, 1, 1] :
  // CHECK:              [[W_HEAD63:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 2] [1, 1, 1, 1]
  // CHECK:              [[H_HEAD63:%.+]] = tensor.extract_slice [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 1, 60] [1, 1, 1, 1]
  // CHECK:              func.call @kernel_hw_overlap_tile1_d3d2([[S63]], [[W_HEAD63]], [[H_HEAD63]])
  // CHECK:            default {
  // CHECK:            [[INS:%.+]] = tensor.insert_slice [[TILE]] into [[ACC]][0, 0, [[BT_H]], [[BT_W]]] [1, 16, 100, 60] [1, 1, 1, 1]
  // CHECK-NEXT:       scf.yield [[INS]]
  // CHECK-NOT:  vpux.lowres
}
