//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="function-outlining=\"naive='num-parts=2'\"" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#loc1 = loc("input")
#loc17 = loc(fused<{name = "input", type = "Parameter"}>[#loc1])
// CHECK-LABEL: @DefaultHWTestWithOutliner
module @DefaultHWTestWithOutliner {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16> loc(#loc17)
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x62x62xf32> loc(#loc18)
    }loc(#loc)

    func.func @main(%input: tensor<1x3x62x62xf16> loc(fused<{name = "input", type = "Parameter"}>[#loc1])) -> tensor<1x3x62x62xf32> {
        %cst = const.Declare tensor<3x3x3x3xf32> = dense<1.0> : tensor<3x3x3x3xf32> loc(#loc19)
        %cst_0 = const.Declare tensor<4xsi64> = dense<[0, 1, 3, 2]> : tensor<4xsi64> loc(#loc20)
        %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32> loc(#loc21)
        %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32> loc(#loc22)
        %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32> loc(#loc23)
        %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32> loc(#loc24)

        %transpose = IE.Transpose(%input, %cst_0) : tensor<1x3x62x62xf16>, tensor<4xsi64> -> tensor<1x3x62x62xf16> loc(#loc25)
        %convert = IE.Convert(%transpose) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32> loc(#loc26)
        %fake_quant = IE.FakeQuantize(%convert, %cst_1, %cst_2, %cst_3, %cst_4)
            {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                levels = 256 : i64
            } :
            tensor<1x3x62x62xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x62x62xf32> loc(#loc27)

        %conv = IE.Convolution(%fake_quant, %cst) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<3x3x3x3xf32> -> tensor<1x3x62x62xf32> loc(#loc28)
        %soft_max1 = IE.SoftMax(%conv) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32> loc(#loc29)
        %soft_max2 = IE.SoftMax(%soft_max1) {axisInd = 1} : tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32> loc(#loc30)

        %add = IE.Add(%soft_max2, %fake_quant) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x3x62x62xf32>, tensor<1x3x62x62xf32> -> tensor<1x3x62x62xf32> loc(#loc31)

        return %add: tensor<1x3x62x62xf32> loc(#loc32)
    }loc(#loc34)
}loc(#loc33)
#loc = loc(unknown)
#loc2 = loc("output")
#loc3 = loc("Constant_conv")
#loc4 = loc("Constant_transpose")
#loc5 = loc("Constant_fq1")
#loc6 = loc("Constant_fq2")
#loc7 = loc("Constant_fq3")
#loc8 = loc("Constant_fq4")
#loc9 = loc("Transpose")
#loc10 = loc("Convert")
#loc11 = loc("FakeQuant")
#loc12 = loc("Conv")
#loc13 = loc("SoftMax1")
#loc14 = loc("SoftMax2")
#loc15 = loc("Add")
#loc16 = loc("OUT")
#loc18 = loc(fused<{name = "output", type = "Result"}>[#loc2])
#loc19 = loc(fused<{name = "Constant_conv", type = "Constant"}>[#loc3])
#loc20 = loc(fused<{name = "Constant_transpose", type = "Constant"}>[#loc4])
#loc21 = loc(fused<{name = "Constant_fq1", type = "Constant"}>[#loc5])
#loc22 = loc(fused<{name = "Constant_fq2", type = "Constant"}>[#loc6])
#loc23 = loc(fused<{name = "Constant_fq3", type = "Constant"}>[#loc7])
#loc24 = loc(fused<{name = "Constant_fq4", type = "Constant"}>[#loc8])
#loc25 = loc(fused<{name = "Transpose", type = "Transpose"}>[#loc9])
#loc26 = loc(fused<{name = "Convert", type = "Convert"}>[#loc10])
#loc27 = loc(fused<{name = "FakeQuant", type = "FakeQuantize"}>[#loc11])
#loc28 = loc(fused<{name = "Conv", type = "Convolution"}>[#loc12])
#loc29 = loc(fused<{name = "SoftMax1", type = "SoftMax"}>[#loc13])
#loc30 = loc(fused<{name = "SoftMax2", type = "SoftMax"}>[#loc14])
#loc31 = loc(fused<{name = "Add", type = "Add"}>[#loc15])
#loc32 = loc(fused<{name = "OUT", type = "Output"}>[#loc16])
#loc33 = loc(fused<{name = "module", type = "Module"}>["module"])
#loc34 = loc(fused<{name = "func", type = "Func"}>["func"])

// CHECK: func.func private @main_part1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf16> {
// CHECK: }

// CHECK: func.func private @main_part2(%arg0: tensor<1x3x62x62xf16>, %arg1: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf16> {
// CHECK: }

// CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf32> {
// CHECK:   [[PART1:%.+]] = call @main_part1([[ARG0]]) : (tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf16>
// CHECK:   [[PART2:%.+]] = call @main_part2([[PART1]], [[ARG0]]) : (tensor<1x3x62x62xf16>, tensor<1x3x62x62xf16>) -> tensor<1x3x62x62xf16>
// CHECK:   [[CONVERT:%.+]] = IE.Convert([[PART2]]) {dstElemType = f32} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf32>
// CHECK:   return [[CONVERT]] : tensor<1x3x62x62xf32>
// CHECK: }
