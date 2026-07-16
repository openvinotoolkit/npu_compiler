//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --unroll-scf-loop="loop-unroll-factor=1,1,10,1 enable-cascaded-unrolling=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @DynamicPermute2DCascaded   {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_13" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> {dynamicStrides}
  } outputsInfo : {
    DataInfo "Convert_16" friendlyName = "Result_17" : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}> {dynamicStrides}
  }
  func.func @main_func0_static(%arg0: tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x16x47x512xf16, {order = #NHWC}>
    return %0 : tensor<1x16x47x512xf16, {order = #NHWC}>
  }
  func.func @main(%arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}> {
    %c342 = arith.constant 342 : index
    %c512 = arith.constant 512 : index
    %c47 = arith.constant 47 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>
    %7 = scf.for %arg2 = %c0 to %dim step %c47 iter_args(%arg3 = %0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>) {
      %24 = scf.for %arg4 = %c0 to %dim_0 step %c512 iter_args(%arg5 = %arg3) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>) {
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg2, %arg4] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x16x47x512xf16>
        %37 = func.call @main_func0_static(%extracted_slice) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #NHWC}>
        %cast = tensor.cast %37 : tensor<1x16x47x512xf16, {order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %cast into %arg5[0, 0, %arg2, %arg4] [1, 16, %c47, %c512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 512]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %24 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %7 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #NHWC}>
  }



// CHECK: #[[$NCHW:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$NHWC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: #[[$MAP_H_470:.+]] = affine_map<(d0)[s0] -> (s0 - 470, d0)>
// CHECK-DAG: #[[$MAP_W_512:.+]] = affine_map<(d0)[s0] -> (s0 - 512, d0)>
// CHECK-DAG: #[[$MAP_H_235:.+]] = affine_map<(d0)[s0] -> (s0 - 235, d0)>
// CHECK-DAG: #[[$MAP_H_94:.+]] = affine_map<(d0)[s0] -> (s0 - 94, d0)>

// CHECK-LABEL:   func.func @merged_vpu_func_2(
// CHECK-COUNT-2:   VPU.NCE.Permute

// CHECK-LABEL:   func.func @merged_vpu_func_1(
// CHECK-COUNT-5:   VPU.NCE.Permute

// CHECK-LABEL:   func.func @merged_vpu_func_0(
// CHECK-COUNT-10:   VPU.NCE.Permute

// CHECK-LABEL:   func.func @main_func0_static(
// CHECK-COUNT-1:   VPU.NCE.Permute

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    [[VAL_0:%.+]]: tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}>) -> tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}> {

// CHECK-DAG:            [[STEP:%.+]] = arith.constant 47 : index
// CHECK-DAG:            [[STEP_FACTOR_2:%.+]] = arith.constant 94 : index
// CHECK-DAG:            [[STEP_FACTOR_5:%.+]] = arith.constant 235 : index
// CHECK-DAG:            [[STEP_FACTOR_10:%.+]] = arith.constant 470 : index

// CHECK-DAG:            [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:            [[CST_2:%.+]] = arith.constant 2 : index
// CHECK-DAG:            [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:            [[CST_5:%.+]] = arith.constant 5 : index
// CHECK-DAG:            [[CST_10:%.+]] = arith.constant 10 : index

// CHECK-DAG:            [[CST_46:%.+]] = arith.constant 46 : index
// CHECK-DAG:            [[CST_512:%.+]] = arith.constant 512 : index


// CHECK-DAG:            [[DIM_H:%.+]] = tensor.dim [[VAL_0]], [[CST_2]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}>
// CHECK-DAG:            [[DIM_W:%.+]] = tensor.dim [[VAL_0]], [[CST_3]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}>

// CHECK-DAG:            [[OUT_TENSOR:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>

// CHECK-DAG:            [[CEIL_DIV_PREP_F_10:%.+]] = arith.addi [[DIM_H]], [[CST_46]] : index
// CHECK-DAG:            [[TOTAL_ITERATIONS_F_10:%.+]] = arith.divui [[CEIL_DIV_PREP_F_10]], [[STEP]] : index
// CHECK-DAG:            [[REMAINDER_F_10:%.+]] = arith.remsi [[TOTAL_ITERATIONS_F_10]], [[CST_10]] : index
// CHECK-DAG:            [[ALIGNED_ITERATIONS_F_10:%.+]] = arith.subi [[TOTAL_ITERATIONS_F_10]], [[REMAINDER_F_10]] : index
// CHECK-DAG:            [[RAW_UPPERBOUND_F_10:%.+]] = arith.muli [[ALIGNED_ITERATIONS_F_10]], [[STEP]] : index

// CHECK-DAG:            [[CAN_EXECUTE_F_10:%.+]] = arith.cmpi uge, [[DIM_H]], [[STEP_FACTOR_10]] : index
// CHECK-DAG:            [[BASE_SAFE_UPPERBOUND_F_10:%.+]] = arith.minui [[RAW_UPPERBOUND_F_10]], [[DIM_H]] : index
// CHECK-DAG:            [[SAFE_UPPERBOUND_F_10:%.+]] = arith.select [[CAN_EXECUTE_F_10]], [[BASE_SAFE_UPPERBOUND_F_10]], [[CST_0]] : index
// CHECK:                [[VAL_23:%.+]] = scf.for [[VAL_24:%.+]] = [[CST_0]] to [[SAFE_UPPERBOUND_F_10]] step [[STEP_FACTOR_10]] iter_args([[VAL_25:%.+]] = [[OUT_TENSOR]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {
// CHECK:                  [[VAL_26:%.+]] = scf.for [[VAL_27:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_512]] iter_args([[VAL_28:%.+]] = [[VAL_25]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {

// CHECK:                    [[BACKTRACKED_H_OFFSET:%.+]] = affine.min #[[$MAP_H_470]]([[VAL_24]]){{\[}}[[SAFE_UPPERBOUND_F_10]]]
// CHECK:                    [[BACKTRACKED_W_OFFSET:%.+]] = affine.min #[[$MAP_W_512]]([[VAL_27]]){{\[}}[[DIM_W]]]

// CHECK:                    [[VAL_29:%.+]] = tensor.extract_slice [[VAL_0]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 470, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}> to tensor<1x16x470x512xf16>
// CHECK:                    [[VAL_30:%.+]] = func.call @merged_vpu_func_0([[VAL_29]]) : (tensor<1x16x470x512xf16>) -> tensor<1x16x470x512xf16, {order = #[[$NHWC]]}>
// CHECK:                    [[VAL_31:%.+]] = tensor.cast [[VAL_30]] : tensor<1x16x470x512xf16, {order = #[[$NHWC]]}> to tensor<1x16x470x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    [[VAL_32:%.+]] = tensor.insert_slice [[VAL_31]] into [[VAL_28]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 470, 512] [1, 1, 1, 1] : tensor<1x16x470x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    scf.yield [[VAL_32]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                  }
// CHECK:                  scf.yield [[VAL_26]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                } {no_await_all = true}

// CHECK:                [[REMAINING_DIM_AFTER_F10:%.+]] = arith.subi [[DIM_H]], [[SAFE_UPPERBOUND_F_10]] : index
// CHECK:                [[CEIL_DIV_PREP_F_5:%.+]] = arith.addi [[REMAINING_DIM_AFTER_F10]], [[CST_46]] : index
// CHECK:                [[TOTAL_ITERATIONS_F_5:%.+]] = arith.divui [[CEIL_DIV_PREP_F_5]], [[STEP]] : index
// CHECK:                [[REMAINDER_F_5:%.+]] = arith.remsi [[TOTAL_ITERATIONS_F_5]], [[CST_5]] : index
// CHECK:                [[ALIGNED_ITERATIONS_F_5:%.+]] = arith.subi [[TOTAL_ITERATIONS_F_5]], [[REMAINDER_F_5]] : index
// CHECK:                [[RAW_OFFSET_F_5:%.+]] = arith.muli [[ALIGNED_ITERATIONS_F_5]], [[STEP]] : index
// CHECK:                [[RAW_UPPERBOUND_F_5:%.+]] = arith.addi [[SAFE_UPPERBOUND_F_10]], [[RAW_OFFSET_F_5]] : index

// CHECK:                [[FIRST_ITER_END_F_5:%.+]] = arith.addi [[SAFE_UPPERBOUND_F_10]], [[STEP_FACTOR_5]] : index
// CHECK:                [[CAN_EXECUTE_F_5:%.+]] = arith.cmpi ule, [[FIRST_ITER_END_F_5]], [[DIM_H]] : index
// CHECK:                [[BASE_SAFE_UPPERBOUND_F_5:%.+]] = arith.minui [[RAW_UPPERBOUND_F_5]], [[DIM_H]] : index
// CHECK:                [[SAFE_UPPERBOUND_F_5:%.+]] = arith.select [[CAN_EXECUTE_F_5]], [[BASE_SAFE_UPPERBOUND_F_5]], [[SAFE_UPPERBOUND_F_10]] : index

// CHECK:                [[VAL_44:%.+]] = scf.for [[VAL_45:%.+]] = [[SAFE_UPPERBOUND_F_10]] to [[SAFE_UPPERBOUND_F_5]] step [[STEP_FACTOR_5]] iter_args([[VAL_46:%.+]] = [[VAL_23]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {
// CHECK:                  [[VAL_47:%.+]] = scf.for [[VAL_48:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_512]] iter_args([[VAL_49:%.+]] = [[VAL_46]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {

// CHECK:                    [[BACKTRACKED_H_OFFSET:%.+]] = affine.min #[[$MAP_H_235]]([[VAL_45]]){{\[}}[[SAFE_UPPERBOUND_F_5]]]
// CHECK:                    [[BACKTRACKED_W_OFFSET:%.+]] = affine.min #[[$MAP_W_512]]([[VAL_48]]){{\[}}[[DIM_W]]]

// CHECK:                    [[VAL_50:%.+]] = tensor.extract_slice [[VAL_0]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 235, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}> to tensor<1x16x235x512xf16>
// CHECK:                    [[VAL_51:%.+]] = func.call @merged_vpu_func_1([[VAL_50]]) : (tensor<1x16x235x512xf16>) -> tensor<1x16x235x512xf16, {order = #[[$NHWC]]}>
// CHECK:                    [[VAL_52:%.+]] = tensor.cast [[VAL_51]] : tensor<1x16x235x512xf16, {order = #[[$NHWC]]}> to tensor<1x16x235x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    [[VAL_53:%.+]] = tensor.insert_slice [[VAL_52]] into [[VAL_49]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 235, 512] [1, 1, 1, 1] : tensor<1x16x235x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    scf.yield [[VAL_53]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                  }
// CHECK:                  scf.yield [[VAL_47]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                } {no_await_all = true, no_reset_cmdlist = true}
// CHECK:                [[VAL_54:%.+]] = arith.subi [[DIM_H]], [[SAFE_UPPERBOUND_F_5]] : index
// CHECK:                [[VAL_55:%.+]] = arith.addi [[VAL_54]], [[CST_46]] : index
// CHECK:                [[VAL_56:%.+]] = arith.divui [[VAL_55]], [[STEP]] : index
// CHECK:                [[VAL_57:%.+]] = arith.remsi [[VAL_56]], [[CST_2]] : index
// CHECK:                [[VAL_58:%.+]] = arith.subi [[VAL_56]], [[VAL_57]] : index
// CHECK:                [[VAL_59:%.+]] = arith.muli [[VAL_58]], [[STEP]] : index
// CHECK:                [[VAL_60:%.+]] = arith.addi [[SAFE_UPPERBOUND_F_5]], [[VAL_59]] : index

// CHECK:                [[FIRST_ITER_END_F_2:%.+]] = arith.addi [[SAFE_UPPERBOUND_F_5]], [[STEP_FACTOR_2]] : index
// CHECK:                [[CAN_EXECUTE_F_2:%.+]] = arith.cmpi ule, [[FIRST_ITER_END_F_2]], [[DIM_H]] : index
// CHECK:                [[BASE_SAFE_UPPERBOUND_F_2:%.+]] = arith.minui [[VAL_60]], [[DIM_H]] : index
// CHECK:                [[SAFE_UPPERBOUND_F_2:%.+]] = arith.select [[CAN_EXECUTE_F_2]], [[BASE_SAFE_UPPERBOUND_F_2]], [[SAFE_UPPERBOUND_F_5]] : index

// CHECK:                [[VAL_65:%.+]] = scf.for [[VAL_66:%.+]] = [[SAFE_UPPERBOUND_F_5]] to [[SAFE_UPPERBOUND_F_2]] step [[STEP_FACTOR_2]] iter_args([[VAL_67:%.+]] = [[VAL_44]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {
// CHECK:                  [[VAL_68:%.+]] = scf.for [[VAL_69:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_512]] iter_args([[VAL_70:%.+]] = [[VAL_67]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {

// CHECK:                    [[BACKTRACKED_H_OFFSET:%.+]] = affine.min #[[$MAP_H_94]]([[VAL_66]]){{\[}}[[SAFE_UPPERBOUND_F_2]]]
// CHECK:                    [[BACKTRACKED_W_OFFSET:%.+]] = affine.min #[[$MAP_W_512]]([[VAL_69]]){{\[}}[[DIM_W]]]

// CHECK:                    [[VAL_71:%.+]] = tensor.extract_slice [[VAL_0]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 94, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}> to tensor<1x16x94x512xf16>
// CHECK:                    [[VAL_72:%.+]] = func.call @merged_vpu_func_2([[VAL_71]]) : (tensor<1x16x94x512xf16>) -> tensor<1x16x94x512xf16, {order = #[[$NHWC]]}>
// CHECK:                    [[VAL_73:%.+]] = tensor.cast [[VAL_72]] : tensor<1x16x94x512xf16, {order = #[[$NHWC]]}> to tensor<1x16x94x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    [[VAL_74:%.+]] = tensor.insert_slice [[VAL_73]] into [[VAL_70]][0, 0, [[BACKTRACKED_H_OFFSET]], [[BACKTRACKED_W_OFFSET]]] [1, 16, 94, 512] [1, 1, 1, 1] : tensor<1x16x94x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    scf.yield [[VAL_74]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                  }
// CHECK:                  scf.yield [[VAL_68]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                } {no_await_all = true, no_reset_cmdlist = true}
// CHECK:                [[VAL_75:%.+]] = scf.for [[VAL_76:%.+]] = [[SAFE_UPPERBOUND_F_2]] to [[DIM_H]] step [[STEP]] iter_args([[VAL_77:%.+]] = [[VAL_65]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {
// CHECK:                  [[VAL_78:%.+]] = scf.for [[VAL_79:%.+]] = [[CST_0]] to [[DIM_W]] step [[CST_512]] iter_args([[VAL_80:%.+]] = [[VAL_77]]) -> (tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>) {
// CHECK:                    [[VAL_81:%.+]] = tensor.extract_slice [[VAL_0]][0, 0, [[VAL_76]], [[VAL_79]]] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NCHW]]}> to tensor<1x16x47x512xf16>
// CHECK:                    [[VAL_82:%.+]] = func.call @main_func0_static([[VAL_81]]) : (tensor<1x16x47x512xf16>) -> tensor<1x16x47x512xf16, {order = #[[$NHWC]]}>
// CHECK:                    [[VAL_83:%.+]] = tensor.cast [[VAL_82]] : tensor<1x16x47x512xf16, {order = #[[$NHWC]]}> to tensor<1x16x47x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    [[VAL_84:%.+]] = tensor.insert_slice [[VAL_83]] into [[VAL_80]][0, 0, [[VAL_76]], [[VAL_79]]] [1, 16, 47, 512] [1, 1, 1, 1] : tensor<1x16x47x512xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}> into tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                    scf.yield [[VAL_84]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                  }
// CHECK:                  scf.yield [[VAL_78]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:                } {no_reset_cmdlist = true}
// CHECK:                return [[VAL_75]] : tensor<1x16x?x?xf16, {bounds = #{{.+}}<[1, 16, 1024, 1024]> : tensor<4xsi64>, order = #[[$NHWC]]}>
// CHECK:              }
}
