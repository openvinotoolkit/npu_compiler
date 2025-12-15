//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW  allow-custom-values=true" \
// RUN:     --default-hw-mode-vpu="enable-shave-code-gen=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Eltwise attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  module @VPU.SW {
    // CHECK: func.func @generated_0([[ARG0:%.+]]: tensor<1x3x224x224xf16>, [[ARG1:%.+]]: tensor<1x3x224x224xf16>, [[ARG2:%.+]]: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
    func.func @generated_0(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<1x3x224x224xf16>, %arg2: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
      %0 = linalg.generic {indexing_maps = [#NCHW, #NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x3x224x224xf16>, tensor<1x3x224x224xf16>) outs(%arg2 : tensor<1x3x224x224xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %1 = arith.addf %in, %in_0 : f16
        linalg.yield %1 : f16
      } -> tensor<1x3x224x224xf16>
      return %0 : tensor<1x3x224x224xf16>

      // CHECK-DAG:      [[C1:%.+]] = arith.constant 1
      // CHECK-DAG:      [[C150528:%.+]] = arith.constant 150528
      // CHECK-DAG:      [[C0:%.+]] = arith.constant 0 : index
      // CHECK-DAG:      [[CARG0:%.+]] = tensor.collapse_shape [[ARG0]]
      // CHECK-SAME:         into tensor<150528xf16>
      // CHECK-DAG:      [[CARG1:%.+]] = tensor.collapse_shape [[ARG1]]
      // CHECK-SAME:         into tensor<150528xf16>
      // CHECK-DAG:      [[CARG2:%.+]] = tensor.collapse_shape [[ARG2]]
      // CHECK-SAME:         into tensor<150528xf16>
      // CHECK:          [[FLAT_OUT:%.+]] = scf.for [[IT:%.+]] = [[C0]] to [[C150528]] step [[C1]] iter_args([[SCF_OUT:%.+]] = [[CARG2]])
      // CHECK-SAME:         -> (tensor<150528xf16>) {
      // CHECK-DAG:          [[ARG0_SLICE:%.+]] = tensor.extract_slice [[CARG0]][[[IT]]] [1] [1]
      // CHECK-SAME:            to tensor<1xf16>
      // CHECK-DAG:          [[ARG1_SLICE:%.+]] = tensor.extract_slice [[CARG1]][[[IT]]] [1] [1]
      // CHECK-SAME:            to tensor<1xf16>
      // CHECK-DAG:          [[ARG2_SLICE:%.+]] = tensor.extract_slice [[SCF_OUT]][[[IT]]] [1] [1]
      // CHECK-SAME:            to tensor<1xf16>
      // CHECK:              [[OP:%.+]] = linalg.generic
      // CHECK-SAME:                ins([[ARG0_SLICE]], [[ARG1_SLICE]] : tensor<1xf16>, tensor<1xf16>)
      // CHECK-SAME:                outs([[ARG2_SLICE]] : tensor<1xf16>) {
      // CHECK-NEXT:         ^bb0([[IN0:%.+]]: f16, [[IN1:%.+]]: f16, {{%.*}}: f16):
      // CHECK-NEXT:            [[ADD:%.+]] = arith.addf [[IN0]], [[IN1]] : f16
      // CHECK-NEXT:            linalg.yield [[ADD]] : f16
      // CHECK-NEXT:         } -> tensor<1xf16>
      // CHECK-NEXT:         [[UPDATED:%.+]] = tensor.insert_slice [[OP]] into [[SCF_OUT]][[[IT]]] [1] [1]
      // CHECK-SAME:              into tensor<150528xf16>
      // CHECK-NEXT:         scf.yield [[UPDATED]] : tensor<150528xf16>
      // CHECK:          [[OUT:%.+]] = tensor.expand_shape [[FLAT_OUT]]
      // CHECK-SAME:          into tensor<1x3x224x224xf16>
      // CHECK-NEXT:     return [[OUT]] : tensor<1x3x224x224xf16>
    }
  }

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "param0" : tensor<1x3x224x224xf16>
    DataInfo "param1" : tensor<1x3x224x224xf16>
  } outputsInfo : {
    DataInfo "out0" : tensor<1x3x224x224xf16>
  }

  // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x224x224xf16>, [[ARG1:%.+]]: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16>
  func.func @main(%arg0: tensor<1x3x224x224xf16>, %arg1: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %op = VPU.GenericSwLayer(%arg0, %arg1) {callee = @VPU.SW::@generated_0} : tensor<1x3x224x224xf16>, tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16>
    return %op : tensor<1x3x224x224xf16>

    // CHECK-DAG:    [[OP0_CPY:%.+]] = VPU.Copy([[ARG0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:        -> tensor<1x3x224x224xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK-DAG:    [[OP1_CPY:%.+]] = VPU.Copy([[ARG1]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:        -> tensor<1x3x224x224xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:        [[OP:%.+]] = VPU.GenericSwLayer([[OP0_CPY]], [[OP1_CPY]])
    // CHECK-SAME:        {callee = @VPU.SW::@generated_0}
    // CHECK-SAME:         -> tensor<1x3x224x224xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:        [[OUT_CPY:%.+]] = VPU.Copy([[OP]])
    // CHECK-SAME:         -> tensor<1x3x224x224xf16>
    // CHECK:        return [[OUT_CPY]]
  }
}
