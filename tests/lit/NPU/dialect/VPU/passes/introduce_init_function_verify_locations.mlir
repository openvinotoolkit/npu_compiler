//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" --mlir-print-debuginfo %s | FileCheck --check-prefix=CHECK-INIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" --mlir-print-debuginfo %s | FileCheck --check-prefix=CHECK-MAIN %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" --concat-init-results="ws-extraction-mode=gen-init" --mlir-print-debuginfo %s | FileCheck --check-prefix=CHECK-INIT-CONCAT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" --concat-init-results="ws-extraction-mode=gen-main" --mlir-print-debuginfo %s | FileCheck --check-prefix=CHECK-MAIN-CONCAT %s

// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


{-#
  dialect_resources: {
    builtin: {
      vpux_ow_0: "0x10000000ABABABABCDCDCDCD"
    }
  }
#-}

!qElemType = !quant.uniform<i8:f16, 0.5:128>
// CHECK-INIT-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16, 5.000000e-01:128>
// CHECK-MAIN-DAG: [[QTYPE:!.+]] = !quant.uniform<i8:f16, 5.000000e-01:128>

#cst0_loc = loc("unique_cst0_location")
// CHECK-INIT-DAG: [[CST0_LOC:#.+]] = loc("unique_cst0_location")
// CHECK-MAIN-DAG: [[CST0_LOC:#.+]] = loc("unique_cst0_location")
// CHECK-INIT-CONCAT-DAG: [[CST0_LOC:#.+]] = loc("unique_cst0_location")
// CHECK-MAIN-CONCAT-DAG: [[CST0_LOC:#.+]] = loc("unique_cst0_location")
#cst1_loc = loc("unique_cst1_location")
// CHECK-INIT-DAG: [[CST1_LOC:#.+]] = loc("unique_cst1_location")
// CHECK-MAIN-DAG: [[CST1_LOC:#.+]] = loc("unique_cst1_location")
// CHECK-INIT-CONCAT-DAG: [[CST1_LOC:#.+]] = loc("unique_cst1_location")
// CHECK-MAIN-CONCAT-DAG: [[CST1_LOC:#.+]] = loc("unique_cst1_location")

module @SimilarConstantsSimilarLocationNames {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "Parameter_1" : tensor<2x2x1x1xf16>
    } outputsInfo : {
        DataInfo "Cst_1" friendlyName = "Result_1" : tensor<2x2x1x1xi8>
        DataInfo "Cst_2" friendlyName = "Result_2" : tensor<2x2x1x1xi8>
    }

    func.func @main(%arg0: tensor<2x2x1x1xf16>) -> (tensor<2x2x1x1xi8>, tensor<2x2x1x1xi8>) {
        %cst = const.Declare tensor<2x2x1x1x!qElemType> = dense_resource<vpux_ow_0> : tensor<2x2x1x1xf16>, [#const.SubView<[1, 0, 0, 0], [2, 2, 1, 1]>, #const.CastElemType<!qElemType>] loc(#cst0_loc)
        %cst1 = const.Declare tensor<2x2x1x1x!qElemType> = dense_resource<vpux_ow_0> : tensor<2x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [2, 2, 1, 1]>, #const.CastElemType<!qElemType>] loc(#cst1_loc)

        %0 = VPU.QuantizeCast(%cst) { dstElemType = i8 }
            : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
        %1 = VPU.QuantizeCast(%cst1) { dstElemType = i8 }
            : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
        return %0, %1: tensor<2x2x1x1xi8> , tensor<2x2x1x1xi8>
    }

    // CHECK-INIT-DAG: [[LOC_INIT0:#.+]] = loc("init_cst0")
    // CHECK-INIT-DAG: [[LOC_INIT1:#.+]] = loc("init_cst1")

    // somehow we get unused locations:
    // CHECK-INIT-DAG: {{#.+}} = loc(fused[[[CST0_LOC]], [[LOC_INIT0]], {{.*}}])

    // CHECK-INIT-DAG: [[LOC_SLICE0:#.+]] = loc(fused[[[CST0_LOC]], [[LOC_INIT0]], {{.*}})
    // CHECK-INIT-DAG: [[LOC_SLICE1:#.+]] = loc(fused[[[CST1_LOC]], [[LOC_INIT1]], {{.*}})

    // CHECK-INIT-DAG: func.func @init([[ARG0:%.+]]: tensor<2x2x1x1xf16> loc(fused[[[CST0_LOC]], [[LOC_INIT0]], {{.*}}])) -> (tensor<2x2x1x1xsi8>, tensor<2x2x1x1xsi8>)
    // CHECK-INIT-DAG: [[SLICEOP0:%.+]] = IE.Slice [[ARG0]] [1, 0, 0, 0] [2, 2, 1, 1] : tensor<2x2x1x1xf16> to tensor<2x2x1x1xf16> loc([[LOC_SLICE0]])
    // CHECK-INIT-DAG: [[CONVERT0:%.+]] = IE.Convert([[SLICEOP0]]) {dstElemType = i8} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xi8>
    // CHECK-INIT-DAG: [[QC0:%.+]] = IE.QuantizeCast([[CONVERT0]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xi8> -> tensor<2x2x1x1x!qElemType>
    // CHECK-INIT-DAG: [[QC1:%.+]] = IE.QuantizeCast([[QC0]]) {dstElemType = si8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xsi8>

    // CHECK-INIT-DAG: [[SLICEOP1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [2, 2, 1, 1] : tensor<2x2x1x1xf16> to tensor<2x2x1x1xf16> loc([[LOC_SLICE1]])
    // CHECK-INIT-DAG: [[CONVERT1:%.+]] = IE.Convert([[SLICEOP1]]) {dstElemType = i8} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xi8>
    // CHECK-INIT-DAG: [[QC2:%.+]] = IE.QuantizeCast([[CONVERT1]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xi8> -> tensor<2x2x1x1x!qElemType>
    // CHECK-INIT-DAG: [[QC3:%.+]] = IE.QuantizeCast([[QC2]]) {dstElemType = si8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xsi8>
    // CHECK-INIT-DAG: return [[QC1]], [[QC3]]


    // CHECK-MAIN-DAG: [[LOC_MAIN0:#.+]] = loc("main_cst0")
    // CHECK-MAIN-DAG: [[LOC_MAIN1:#.+]] = loc("main_cst1")

    // somehow we get unused locations:
    // CHECK-MAIN-DAG: {{#.+}} = loc(fused[[[CST0_LOC]], [[LOC_MAIN0]], {{.*}}])
    // CHECK-MAIN-DAG: {{#.+}} = loc(fused[[[CST1_LOC]], [[LOC_MAIN1]], {{.*}}])

    // CHECK-MAIN-DAG: [[LOC_QC0:#.+]] = loc(fused[[[CST0_LOC]], [[LOC_MAIN0]], {{.*}}])
    // CHECK-MAIN-DAG: [[LOC_QC1:#.+]] = loc(fused[[[CST1_LOC]], [[LOC_MAIN1]], {{.*}}])

    // CHECK-MAIN-DAG: func.func @main([[MAIN_ARG0:%.+]]: tensor<2x2x1x1xf16> [[LOC_MAIN_ARG0:.*]], [[MAIN_ARG1:%.+]]: tensor<2x2x1x1xsi8> [[LOC_MAIN_ARG1:.*]], [[MAIN_ARG2:%.+]]: tensor<2x2x1x1xsi8> [[LOC_MAIN_ARG2:.*]]) -> (tensor<2x2x1x1xi8>, tensor<2x2x1x1xi8>)
    // CHECK-MAIN-DAG: [[MAIN_QC0:%.+]] = VPU.QuantizeCast([[MAIN_ARG1]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xsi8> -> tensor<2x2x1x1x!qElemType> loc([[LOC_QC0]])
    // CHECK-MAIN-DAG: [[MAIN_QC1:%.+]] = VPU.QuantizeCast([[MAIN_ARG2]]) {dstElemType = [[QTYPE]]} : tensor<2x2x1x1xsi8> -> tensor<2x2x1x1x!qElemType> loc([[LOC_QC1]])
    // CHECK-MAIN-DAG: [[MAIN_QC2:%.+]] = VPU.QuantizeCast([[MAIN_QC0]]) {dstElemType = i8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
    // CHECK-MAIN-DAG: [[MAIN_QC3:%.+]] = VPU.QuantizeCast([[MAIN_QC1]]) {dstElemType = i8} : tensor<2x2x1x1x!qElemType> -> tensor<2x2x1x1xi8>
    // CHECK-MAIN-DAG: return [[MAIN_QC2]], [[MAIN_QC3]]


    // for concat-init-results in 'gen-init', check reinterpret casts

    // CHECK-INIT-CONCAT-DAG: [[LOC_INIT0:#.+]] = loc("init_cst0")
    // CHECK-INIT-CONCAT-DAG: [[LOC_INIT1:#.+]] = loc("init_cst1")
    // CHECK-INIT-CONCAT-DAG: [[LOC_OBFS:#.+]] = loc("obfuscated")

    // CHECK-INIT-CONCAT-DAG: [[LOC_RCAST0:#.+]] = loc(fused[[[CST0_LOC]], [[LOC_INIT0]], {{.*}}, [[LOC_OBFS]]])
    // CHECK-INIT-CONCAT-DAG: [[LOC_RCAST1:#.+]] = loc(fused[[[CST1_LOC]], [[LOC_INIT1]], {{.*}}, [[LOC_OBFS]]])

    // CHECK-INIT-CONCAT-DAG: {{%.+}} = Core.ReinterpretCast({{.*}}) {{.*}} loc([[LOC_RCAST0]])
    // CHECK-INIT-CONCAT-DAG: {{%.+}} = Core.ReinterpretCast({{.*}}) {{.*}} loc([[LOC_RCAST1]])


    // for concat-init-results in 'gen-main', check slices and reinterpret casts

    // CHECK-MAIN-CONCAT-DAG: [[LOC_MAIN0:#.+]] = loc("main_cst0")
    // CHECK-MAIN-CONCAT-DAG: [[LOC_ARG0:#.+]] = loc("arg1")
    // CHECK-MAIN-CONCAT-DAG: [[LOC_MAIN1:#.+]] = loc("main_cst1")
    // CHECK-MAIN-CONCAT-DAG: [[LOC_ARG1:#.+]] = loc("arg2")
    // CHECK-MAIN-CONCAT-DAG: [[LOC_SLICE:#.+]] = loc("slice")
    // CHECK-MAIN-CONCAT-DAG: [[LOC_DEOBFS:#.+]] = loc("deobfuscated")

    // CHECK-MAIN-CONCAT-DAG: [[LOC_SLICE0:#.+]] = loc(fused[[[CST0_LOC]], [[LOC_MAIN0]], [[LOC_ARG0]], [[LOC_SLICE]]])
    // CHECK-MAIN-CONCAT-DAG: [[LOC_RCAST0:#.+]] = loc(fused[[[CST0_LOC]], [[LOC_MAIN0]], [[LOC_ARG0]], [[LOC_SLICE]], [[LOC_DEOBFS]]])

    // CHECK-MAIN-CONCAT-DAG: [[LOC_SLICE1:#.+]] = loc(fused[[[CST1_LOC]], [[LOC_MAIN1]], [[LOC_ARG1]], [[LOC_SLICE]]])
    // CHECK-MAIN-CONCAT-DAG: [[LOC_RCAST1:#.+]] = loc(fused[[[CST1_LOC]], [[LOC_MAIN1]], [[LOC_ARG1]], [[LOC_SLICE]], [[LOC_DEOBFS]]])

    // CHECK-MAIN-CONCAT-DAG: func.func @main({{.*}}, [[BLOB:%.+]]: tensor<8xi8>
    // CHECK-MAIN-CONCAT-DAG:   [[SLICE0:%.+]] = VPU.Slice [[BLOB]] [0] [4] {{.*}} loc([[LOC_SLICE0]])
    // CHECK-MAIN-CONCAT-DAG:   {{%.+}} = Core.ReinterpretCast([[SLICE0]]) {{.*}} loc([[LOC_RCAST0]])

    // CHECK-MAIN-CONCAT-DAG:   [[SLICE1:%.+]] = VPU.Slice [[BLOB]] [4] [4] {{.*}} loc([[LOC_SLICE1]])
    // CHECK-MAIN-CONCAT-DAG:   {{%.+}} = Core.ReinterpretCast([[SLICE1]]) {{.*}} loc([[LOC_RCAST1]])
}
