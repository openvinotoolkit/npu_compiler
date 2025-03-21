//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --outline-entire-main-content %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @SkipWhenMainHasNoCallOps
module @SkipWhenMainHasNoCallOps {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x32x32xf16>
    }

    func.func @main(%input: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
        %softmax = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        return %relu : tensor<1x48x32x32xf16>
    }

    // CHECK:       func.func @main
    // CHECK-NOT:   call
    // CHECK-NEXT:  VPU.SoftMax
    // CHECK-NEXT:  VPU.ReLU
}

// -----

// CHECK-LABEL: @MixedCallAndNonCallOps
module @MixedCallAndNonCallOps {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x96x32x32xf16>
    }

    func.func private @fn1(%arg0: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        return %relu : tensor<1x48x32x32xf16>
    }
    func.func private @fn2(%arg0: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
        %relu = VPU.ReLU(%arg0) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %softmax = VPU.SoftMax(%relu) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        return %softmax : tensor<1x48x32x32xf16>
    }
    func.func @main(%input: tensor<1x48x32x32xf16>) -> tensor<1x96x32x32xf16> {
        %input_softmax = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>

        %call1 = call @fn1(%input_softmax) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>

        %middle_relu = VPU.ReLU(%call1) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>

        %call2 = call @fn2(%middle_relu) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>

        %output_softmax1 = VPU.SoftMax(%call2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_relu1 = VPU.ReLU(%output_softmax1) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_softmax2 = VPU.SoftMax(%call2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_relu2 = VPU.ReLU(%output_softmax2) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_concat = VPU.Concat(%output_relu1, %output_relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]} : tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16> -> tensor<1x96x32x32xf16>

        return %output_concat : tensor<1x96x32x32xf16>
    }

    // CHECK:  func.func private @fn1([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[FN1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[FN1_RELU:%.+]] = VPU.ReLU([[FN1_SOFTMAX]])
    // CHECK:      return [[FN1_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @fn2([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[FN2_RELU:%.+]] = VPU.ReLU([[ARG0]])
    // CHECK:      [[FN2_SOFTMAX:%.+]] = VPU.SoftMax([[FN2_RELU]])
    // CHECK:      return [[FN2_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[OUTLINE1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      return [[OUTLINE1_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[OUTLINE2_RELU:%.+]] = VPU.ReLU([[ARG0]])
    // CHECK:      return [[OUTLINE2_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline3([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x96x32x32xf16> {
    // CHECK:      [[OUTLINE3_SOFTMAX1:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[OUTLINE3_RELU1:%.+]] = VPU.ReLU([[OUTLINE3_SOFTMAX1]])
    // CHECK:      [[OUTLINE3_SOFTMAX2:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[OUTLINE3_RELU2:%.+]] = VPU.ReLU([[OUTLINE3_SOFTMAX2]])
    // CHECK:      [[OUTLINE3_CONCAT:%.+]] = VPU.Concat([[OUTLINE3_RELU1]], [[OUTLINE3_RELU2]])
    // CHECK:      return [[OUTLINE3_CONCAT]]
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x96x32x32xf16> {
    // CHECK:      [[CALL1:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL2:%.+]] = call @fn1([[CALL1]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL3:%.+]] = call @main_outline2([[CALL2]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL4:%.+]] = call @fn2([[CALL3]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL5:%.+]] = call @main_outline3([[CALL4]]) : (tensor<1x48x32x32xf16>) -> tensor<1x96x32x32xf16>
    // CHECK:      return [[CALL5]]
    // CHECK:  }
}

// -----

// CHECK-LABEL: @MixedCallAndNonCallOpsMultipleUsers
module @MixedCallAndNonCallOpsMultipleUsers {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x144x32x32xf16>
    }

    func.func private @fn1(%arg0: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        return %relu : tensor<1x48x32x32xf16>
    }
    func.func private @fn2(%arg0: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
        %relu = VPU.ReLU(%arg0) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %softmax = VPU.SoftMax(%relu) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        return %softmax : tensor<1x48x32x32xf16>
    }
    func.func @main(%input: tensor<1x48x32x32xf16>) -> tensor<1x144x32x32xf16> {
        %input_softmax = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>

        %call1 = call @fn1(%input_softmax) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>

        %middle_relu = VPU.ReLU(%call1) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>

        %call2 = call @fn2(%middle_relu) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>

        %output_softmax1 = VPU.SoftMax(%call2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_relu1 = VPU.ReLU(%output_softmax1) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_softmax2 = VPU.SoftMax(%call2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_relu2 = VPU.ReLU(%output_softmax2) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %output_concat = VPU.Concat(%input_softmax, %output_relu1, %output_relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]}
            : tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16> -> tensor<1x144x32x32xf16>

        return %output_concat : tensor<1x144x32x32xf16>
    }

    // CHECK:  func.func private @fn1([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[FN1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[FN1_RELU:%.+]] = VPU.ReLU([[FN1_SOFTMAX]])
    // CHECK:      return [[FN1_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @fn2([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[FN2_RELU:%.+]] = VPU.ReLU([[ARG0]])
    // CHECK:      [[FN2_SOFTMAX:%.+]] = VPU.SoftMax([[FN2_RELU]])
    // CHECK:      return [[FN2_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[OUTLINE1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      return [[OUTLINE1_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16> {
    // CHECK:      [[OUTLINE2_RELU:%.+]] = VPU.ReLU([[ARG0]])
    // CHECK:      return [[OUTLINE2_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline3([[ARG0:%.+]]: tensor<1x48x32x32xf16>, [[ARG1:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x144x32x32xf16> {
    // CHECK:      [[OUTLINE3_SOFTMAX1:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[OUTLINE3_RELU1:%.+]] = VPU.ReLU([[OUTLINE3_SOFTMAX1]])
    // CHECK:      [[OUTLINE3_SOFTMAX2:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[OUTLINE3_RELU2:%.+]] = VPU.ReLU([[OUTLINE3_SOFTMAX2]])
    // CHECK:      [[OUTLINE3_CONCAT:%.+]] = VPU.Concat([[ARG1]], [[OUTLINE3_RELU1]], [[OUTLINE3_RELU2]])
    // CHECK:      return [[OUTLINE3_CONCAT]]
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16>) -> tensor<1x144x32x32xf16> {
    // CHECK:      [[CALL1:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL2:%.+]] = call @fn1([[CALL1]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL3:%.+]] = call @main_outline2([[CALL2]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL4:%.+]] = call @fn2([[CALL3]]) : (tensor<1x48x32x32xf16>) -> tensor<1x48x32x32xf16>
    // CHECK:      [[CALL5:%.+]] = call @main_outline3([[CALL4]], [[CALL1]]) : (tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16>) -> tensor<1x144x32x32xf16>
    // CHECK:      return [[CALL5]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OutlineConstantsInAllFunctions
module @OutlineConstantsInAllFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x32x32xf16>
    }

    func.func private @fn(%arg0: tensor<1x48x32x32xf16, {order = #NHWC}>, %maxpool_wt : tensor<48x1x1x4xsi32>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %maxpool = VPU.NCE.MaxPool(%arg0, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%maxpool) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        return %relu : tensor<1x48x32x32xf16, {order = #NHWC}>
    }
    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %maxpool1 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %call = call @fn(%maxpool1, %maxpool_wt) : (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32>) -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %maxpool2 = VPU.NCE.MaxPool(%call, %maxpool_wt) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [3, 3]
            } -> tensor<1x48x32x32xf16, {order = #NHWC}>

        return %maxpool2 : tensor<1x48x32x32xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @fn([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<48x1x1x4xsi32>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[FN_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[ARG1]] )
    // CHECK:      [[FN_SOFTMAX:%.+]] = VPU.SoftMax([[FN_MAXPOOL]])
    // CHECK:      [[FN_RELU:%.+]] = VPU.ReLU([[FN_SOFTMAX]])
    // CHECK:      return [[FN_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<48x1x1x4xsi32>, tensor<1x48x32x32xf16, {order = #NHWC}>) {
    // CHECK:      [[OUTLINE1_CST:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:      [[OUTLINE1_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[OUTLINE1_CST]] )
    // CHECK:      return [[OUTLINE1_CST]], [[OUTLINE1_MAXPOOL]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[OUTLINE2_CST:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:      [[OUTLINE2_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[OUTLINE1_CST]] )
    // CHECK:      return [[OUTLINE2_MAXPOOL]]
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[CALL1:%.+]]:2 = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<48x1x1x4xsi32>, tensor<1x48x32x32xf16, {order = #NHWC}>)
    // CHECK:      [[CALL2:%.+]] = call @fn([[CALL1]]#1, [[CALL1]]#0) : (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      [[CALL3:%.+]] = call @main_outline2([[CALL2]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      return [[CALL3]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OutlineSurroundingOpsConsecutiveCalls
module @OutlineSurroundingOpsConsecutiveCalls {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x32x32xf16>
    }

    func.func private @fn1(%arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        return %relu : tensor<1x48x32x32xf16, {order = #NHWC}>
    }
    func.func private @fn2(%arg0: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %relu = VPU.ReLU(%softmax) : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
        return %relu : tensor<1x48x32x32xf16, {order = #NHWC}>
    }
    func.func @main(%input: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
        %input_softmax = VPU.SoftMax(%input) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %call1 = call @fn1(%input_softmax) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
        %call2 = call @fn2(%call1) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>

        %output_softmax = VPU.SoftMax(%call2) {axisInd = 1 : i64} : tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

        return %output_softmax : tensor<1x48x32x32xf16, {order = #NHWC}>
    }

    // CHECK:  func.func private @fn1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[FN1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[FN1_RELU:%.+]] = VPU.ReLU([[FN1_SOFTMAX]])
    // CHECK:      return [[FN1_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @fn2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[FN2_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      [[FN2_RELU:%.+]] = VPU.ReLU([[FN2_SOFTMAX]])
    // CHECK:      return [[FN2_RELU]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[OUTLINE1_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      return [[OUTLINE1_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[OUTLINE2_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:      return [[OUTLINE2_SOFTMAX]]
    // CHECK:  }
    // CHECK:  func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:      [[CALL1:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      [[CALL2:%.+]] = call @fn1([[CALL1]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      [[CALL3:%.+]] = call @fn2([[CALL2]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      [[CALL4:%.+]] = call @main_outline2([[CALL3]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:      return [[CALL4]]
    // CHECK:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorTypeDDR = tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
!TensorTypeCMX = tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!DistributedType = !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

// CHECK-LABEL: @OutlineDistributedTypes
module @OutlineDistributedTypes {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x48x32x32xf16>
        DataInfo "output2" : tensor<1x48x32x32xf16>
    }

    func.func private @fn(%arg0: !TensorTypeDDR) -> !TensorTypeDDR {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : !TensorTypeDDR -> !TensorTypeDDR
        return %softmax : !TensorTypeDDR
    }

    func.func @main(%input: !TensorTypeDDR) -> (!TensorTypeDDR, !TensorTypeDDR) {
        %input_copy = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !TensorTypeDDR -> !TensorTypeCMX

        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %maxpool = VPU.NCE.MaxPool(%input_copy, %maxpool_wt) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
                } -> !DistributedType

        %call = call @fn(%input) : (!TensorTypeDDR) -> !TensorTypeDDR

        %output_copy = VPU.Copy(%maxpool) {out_mem_space = @DDR} : !DistributedType -> !TensorTypeDDR

        return %call, %output_copy : !TensorTypeDDR, !TensorTypeDDR
    }

    // CHECK:       func.func private @fn([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:           [[FN_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:           return [[FN_SOFTMAX]]
    // CHECK:       }

    // CHECK:       func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:           [[INPUT_COPY:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:         -> tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:           [[MAXPOOL_WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:           [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT_COPY]], [[MAXPOOL_WT]] )
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:           [[COPY_TO_DDR:%.+]] = VPU.Copy([[MAXPOOL]])
    // CHECK-SAME:          -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[COPY_TO_DDR]]
    // CHECK:       }

    // CHECK:       func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:           [[COPY_TO_CMX:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:           [[OUTPUT_COPY:%.+]] = VPU.Copy([[COPY_TO_CMX]])
    // CHECK-SAME:         -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[OUTPUT_COPY]]
    // CHECK:       }

    // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>)
    // CHECK-SAME:      -> (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) {
    // CHECK:           [[CALL1:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           [[CALL2:%.+]] = call @fn([[INPUT]]) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           [[CALL3:%.+]] = call @main_outline2([[CALL1]]) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[CALL2]], [[CALL3]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorType = tensor<1x48x32x32xf16, {order = #NHWC}>
!SparseType = !VPU.SparseTensor<data=tensor<1x48x32x32xf16, {order = #NHWC}>, sparsity_map=tensor<1x48x32x32xi1, {order = #NHWC}>>

// CHECK-LABEL: @OutlineSparseTypes
module @OutlineSparseTypes {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x48x32x32xf16>
        DataInfo "output2" : tensor<1x48x32x32xf16>
    }

    func.func private @fn(%arg0: !TensorType) -> !TensorType {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : !TensorType -> !TensorType
        return %softmax : !TensorType
    }

    func.func @main(%input: !TensorType) -> (!TensorType, !TensorType) {
        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %maxpool1 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
                } -> !SparseType

        %call = call @fn(%input) : (!TensorType) -> !TensorType

        %maxpool2 = VPU.NCE.MaxPool(%maxpool1, %maxpool_wt) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
                } -> !TensorType

        return %call, %maxpool2 : !TensorType, !TensorType
    }

    // CHECK:       func.func private @fn([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:           [[FN_SOFTMAX:%.+]] = VPU.SoftMax(%arg0)
    // CHECK:           return [[FN_SOFTMAX]]
    // CHECK:       }
    // CHECK:       func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xi1, {order = #NHWC}>) {
    // CHECK:           [[OUTLINE1_MAXPOOL_WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:           [[OUTLINE1_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[OUTLINE1_MAXPOOL_WT]] )
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x48x32x32xf16, {order = #NHWC}>, sparsity_map=tensor<1x48x32x32xi1, {order = #NHWC}>>

    // CHECK:           [[OUTLINE1_DATA:%.+]], [[OUTLINE1_SM:%.+]] = VPU.UngroupSparseTensor([[OUTLINE1_MAXPOOL]]) {resultSegmentSizes = array<i32: 1, 1, 0>}
    // CHECK:           return [[OUTLINE1_DATA]], [[OUTLINE1_SM]]
    // CHECK:       }
    // CHECK:       func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x48x32x32xi1, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    // CHECK:           [[OUTLINE2_GROUP:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x48x32x32xf16, {order = #NHWC}>, sparsity_map=tensor<1x48x32x32xi1, {order = #NHWC}>>
    // CHECK:           [[OUTLINE2_MAXPOOL_WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:           [[OUTLINE2_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[OUTLINE2_GROUP]], [[OUTLINE2_MAXPOOL_WT]] )
    // CHECK-SAME:          -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:           return [[OUTLINE2_MAXPOOL]]
    // CHECK:       }
    // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}>) {
    // CHECK:           [[CALL1:%.+]]:2 = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xi1, {order = #NHWC}>)
    // CHECK:           [[CALL2:%.+]] = call @fn([[INPUT]]) : (tensor<1x48x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:           [[CALL3:%.+]] = call @main_outline2([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xi1, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:           return [[CALL2]], [[CALL3]]
    // CHECK:       }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorTypeDDR = tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
!TensorTypeCMX = tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>
!SparseType = !VPU.SparseTensor<data=tensor<1x48x32x32xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                sparsity_map=tensor<1x48x32x32xi1, {mem_space = @CMX_NN, order = #NHWC}>>
!SparseDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
                                           sparsity_map=!VPU.DistributedTensor<1x48x32x32xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>

module @OutlineSparseTypesCMX {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x32x32xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x48x32x32xf16>
        DataInfo "output2" : tensor<1x48x32x32xf16>
    }

    func.func private @fn(%arg0: !TensorTypeDDR) -> !TensorTypeDDR {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : !TensorTypeDDR -> !TensorTypeDDR
        return %softmax : !TensorTypeDDR
    }

    func.func @main(%input: !TensorTypeDDR) -> (!TensorTypeDDR, !TensorTypeDDR) {
        %maxpool_wt = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %maxpool1 = VPU.NCE.MaxPool(%input, %maxpool_wt) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
                } -> !SparseDistributedType

        %call = call @fn(%input) : (!TensorTypeDDR) -> !TensorTypeDDR

        %maxpool2 = VPU.NCE.MaxPool(%maxpool1, %maxpool_wt) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1], kernel_size = [1, 1]
                } -> !TensorTypeDDR
        return %call, %maxpool2 : !TensorTypeDDR, !TensorTypeDDR
    }

    // CHECK:       func.func private @fn([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:           [[FN_SOFTMAX:%.+]] = VPU.SoftMax(%arg0)
    // CHECK:           return [[FN_SOFTMAX]]
    // CHECK:       }
    // CHECK:       func.func private @main_outline1([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>)
    // CHECK-SAME:      -> (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>) {
    // CHECK:           [[OUTLINE1_MAXPOOL_WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:           [[OUTLINE1_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[ARG0]], [[OUTLINE1_MAXPOOL_WT]] )
    // CHECK-SAME:     -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                          sparsity_map=!VPU.DistributedTensor<1x48x32x32xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:           [[OUTLINE1_DATA:%.+]], [[OUTLINE1_SM:%.+]] = VPU.UngroupSparseTensor([[OUTLINE1_MAXPOOL]]) {resultSegmentSizes = array<i32: 1, 1, 0>}
    // CHECK:           [[OUTLINE1_DATA_COPY:%.+]] = VPU.Copy([[OUTLINE1_DATA]])
    // CHECK-SAME:          -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           [[OUTLINE1_SM_COPY:%.+]] = VPU.Copy([[OUTLINE1_SM]])
    // CHECK:               -> tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[OUTLINE1_DATA_COPY]], [[OUTLINE1_SM_COPY]]
    // CHECK:       }
    // CHECK:       func.func private @main_outline2([[ARG0:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, [[ARG1:%.+]]: tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:           [[OUTLINE2_GROUP:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[ARG1]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, sparsity_map=tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>>
    // CHECK:           [[OUTLINE2_GROUP_COPY:%.+]] = VPU.Copy([[OUTLINE2_GROUP]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x48x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x48x32x32xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>>
    // CHECK:           [[OUTLINE2_MAXPOOL_WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    // CHECK:           [[OUTLINE2_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[OUTLINE2_GROUP_COPY]], [[OUTLINE2_MAXPOOL_WT]] )
    // CHECK-SAME:          -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[OUTLINE2_MAXPOOL]]
    // CHECK:       }
    // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) {
    // CHECK:           [[CALL1:%.+]]:2 = call @main_outline1([[INPUT]]) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>)
    // CHECK:           [[CALL2:%.+]] = call @fn([[INPUT]]) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           [[CALL3:%.+]] = call @main_outline2([[CALL1]]#0, [[CALL1]]#1) : (tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>, tensor<1x48x32x32xi1, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x48x32x32xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:           return [[CALL2]], [[CALL3]]
    // CHECK:       }
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorType = tensor<1x16x1x480xf16, {order = #NHWC}>
!SparseTypeDDR = !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>,
                                   sparsity_map=tensor<16x1x1x128xi1>,
                                   is_weights,
                                   #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
!SparseTypeCMX = !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                   sparsity_map=tensor<16x1x1x128xi1, {mem_space = @CMX_NN, order = #NCHW}>,
                                   is_weights,
                                   #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
!SparseDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>,
                                           sparsity_map=!VPU.DistributedTensor<16x1x1x128xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>,
                                           is_weights,
                                           #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>

// CHECK-LABEL: @OutlineSparseWeights
module @OutlineSparseWeights {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x1x480xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x16x1x480xf16>
        DataInfo "output2" : tensor<1x16x1x480xf16>
    }

    func.func private @fn(%arg0: !TensorType) -> !TensorType {
        %softmax = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : !TensorType -> !TensorType
        return %softmax : !TensorType
    }

    func.func @main(%input: !TensorType) -> (!TensorType, !TensorType) {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<16x16x1x1xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        %weights_sm = const.Declare tensor<16x1x1x128xi1> = dense<1.0> : tensor<16x16x1x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
        %sparse_weights = VPU.GroupSparseTensor(%weights, %weights_sm) {
            is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>
        } -> !SparseTypeDDR

        %call = call @fn(%input) : (!TensorType) -> !TensorType

        %sparse_weights_cmx = VPU.Copy(%sparse_weights) {out_mem_space = @CMX_NN} : !SparseTypeDDR -> !SparseDistributedType
        %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %conv = VPU.NCE.Convolution(%call, %sparse_weights_cmx, %weights_table) {
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]
                } -> !TensorType {
                    VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 16, 1, 480] outOffsets [0, 0, 0, 0] outSizes [1, 16, 1, 480] <left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            }

        return %call, %conv : !TensorType, !TensorType
    }

    // CHECK:       func.func private @fn([[ARG0:%.+]]: tensor<1x16x1x480xf16, {order = #NHWC}>) -> tensor<1x16x1x480xf16, {order = #NHWC}> {
    // CHECK:           [[FN_SOFTMAX:%.+]] = VPU.SoftMax([[ARG0]])
    // CHECK:           return [[FN_SOFTMAX]]
    // CHECK:       }
    // CHECK:       func.func private @main_outline2([[ARG0:%.+]]: tensor<1x16x1x480xf16, {order = #NHWC}>) -> tensor<1x16x1x480xf16, {order = #NHWC}> {
    // CHECK-DAG:       [[OUTLINE1_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<16x16x1x1xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    // CHECK-DAG:       [[OUTLINE1_WEIGHTS_SM:%.+]] = const.Declare tensor<16x1x1x128xi1> = dense<1.000000e+00> : tensor<16x16x1x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    // CHECK:           [[OUTLINE1_SPARSE_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[OUTLINE1_WEIGHTS]], [[OUTLINE1_WEIGHTS_SM]]) {
    // CHECK:               is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>
    // CHECK:           } -> !VPU.SparseTensor<data=tensor<16x16x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x128xi1>, is_weights,
    // CHECK-SAME:                             #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK:           [[OUTLINE1_SPARSE_WEIGHTS_CMX:%.+]] = VPU.Copy([[OUTLINE1_SPARSE_WEIGHTS]])
    // CHECK-SAME:          -> !VPU.SparseTensor<data=!VPU.DistributedTensor<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<16x1x1x128xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>,
    // CHECK-SAME:                               is_weights,
    // CHECK-SAME:                               #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<[48, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi64>, alignment = 16 : i64>>
    // CHECK:           [[OUTLINE1_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[ARG0]], [[OUTLINE1_SPARSE_WEIGHTS_CMX]], [[OUTLINE1_WEIGHTS_TABLE]])
    // CHECK:                -> tensor<1x16x1x480xf16, {order = #NHWC}>
    // CHECK:           return [[CONV]]
    // CHECK:       func.func @main([[INPUT:%.+]]: tensor<1x16x1x480xf16, {order = #NHWC}>) -> (tensor<1x16x1x480xf16, {order = #NHWC}>, tensor<1x16x1x480xf16, {order = #NHWC}>) {
    // CHECK:           [[CALL1:%.+]] = call @fn([[INPUT]]) : (tensor<1x16x1x480xf16, {order = #NHWC}>) -> tensor<1x16x1x480xf16, {order = #NHWC}>
    // CHECK:           [[CALL2:%.+]] = call @main_outline2([[CALL1]]) : (tensor<1x16x1x480xf16, {order = #NHWC}>) -> tensor<1x16x1x480xf16, {order = #NHWC}>
    // CHECK:           return [[CALL1]], [[CALL2]]
    // CHECK:       }
}
