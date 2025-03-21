//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --duplicate-fq-across-function-calls --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @DoNotDuplicateUnusedFQ
module @DoNotDuplicateUnusedFQ {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%fq_in) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%relu, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        // There are no operations to use the input FakeQuantize operation, so it does not get duplicated
        %call = call @function(%arg0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        // Similarly, there are no operatiosn to use the output FakeQauntize operation, so it does not get duplicated
        return %call : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FQ_IN:%.+]] = IE.FakeQuantize([[ARG]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[FQ_IN]])
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[RELU]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK:          return [[FQ_OUT]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[CALL:%.+]] = call @function([[ARG]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          return [[CALL]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DoNotDuplicateUnusedFQBetweenCallOps
module @DoNotDuplicateUnusedFQBetweenCallOps {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%fq_in) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%relu, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %call1 = call @function(%arg0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        // If the FQ operation would be duplicated, it would only be connected to another call operation, making them unused
        %call2 = call @function(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %call2 : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FQ_IN:%.+]] = IE.FakeQuantize([[ARG]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[FQ_IN]])
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[RELU]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK:          return [[FQ_OUT]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[CALL1:%.+]] = call @function([[ARG]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          [[CALL2:%.+]] = call @function([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          return [[CALL2]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DuplicateOutside
module @DuplicateOutside {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%fq_in) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%relu, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %relu1 = IE.ReLU(%arg0) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %call = call @function(%relu1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%call) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu2 : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FQ_IN:%.+]] = IE.FakeQuantize([[ARG]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[FQ_IN]])
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[RELU]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK:          return [[FQ_OUT]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[RELU1:%.+]] = IE.ReLU([[ARG]])
    // CHECK:          [[DUPL_FQ_IN:%.+]] = IE.FakeQuantize([[RELU1]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]])
    // CHECK:          [[CALL:%.+]] = call @function([[DUPL_FQ_IN]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          [[DUPL_FQ_OUT:%.+]] = IE.FakeQuantize([[CALL]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]])
    // CHECK:          [[RELU2:%.+]] = IE.ReLU([[DUPL_FQ_OUT]])
    // CHECK:          return [[RELU2]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DuplicateOutsideWithReshapes
module @DuplicateOutsideWithReshapes {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
        %in_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2]], shape_value = [48, 60, 60]} : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
        %fq_in = IE.FakeQuantize(%in_reshape, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<48x60x60xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<48x60x60xf32>
        %relu = IE.ReLU(%fq_in) : tensor<48x60x60xf32> -> tensor<48x60x60xf32>
        %fq_out = IE.FakeQuantize(%relu, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<48x60x60xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<48x60x60xf32>
        %out_reshape = IE.AffineReshape(%fq_out) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 48, 60, 60]} : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %out_reshape : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %relu1 = IE.ReLU(%arg0) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %call = call @function(%relu1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %relu2 = IE.ReLU(%call) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu2 : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
    // CHECK:          [[FQ_IN_RESHAPE:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
    // CHECK:          [[FQ_IN:%.+]] = IE.FakeQuantize([[FQ_IN_RESHAPE]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[FQ_IN]])
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[RELU]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK:          [[FQ_OUT_RESHAPE:%.+]] = IE.AffineReshape([[FQ_OUT]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          return [[FQ_OUT_RESHAPE]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[DUPL_IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>

    // CHECK:          [[RELU1:%.+]] = IE.ReLU([[ARG]])

    // CHECK:          [[DUPL_FQ_IN_RESHAPE1:%.+]] = IE.AffineReshape([[RELU1]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
    // CHECK:          [[DUPL_FQ_IN:%.+]] = IE.FakeQuantize([[DUPL_FQ_IN_RESHAPE1]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]])
    // CHECK:          [[DUPL_FQ_IN_RESHAPE2:%.+]] = IE.AffineReshape([[DUPL_FQ_IN]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>

    // CHECK:          [[CALL:%.+]] = call @function([[DUPL_FQ_IN_RESHAPE2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>

    // CHECK:          [[DUPL_FQ_OUT_RESHAPE1:%.+]] = IE.AffineReshape([[CALL]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
    // CHECK:          [[DUPL_FQ_OUT:%.+]] = IE.FakeQuantize([[DUPL_FQ_OUT_RESHAPE1]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]])
    // CHECK:          [[DUPL_FQ_OUT_RESHAPE2:%.+]] = IE.AffineReshape([[DUPL_FQ_OUT]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>

    // CHECK:          [[RELU2:%.+]] = IE.ReLU([[DUPL_FQ_OUT_RESHAPE2]])
    // CHECK:          return [[RELU2]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DuplicateInside
module @DuplicateInside {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %relu = IE.ReLU(%arg0) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %call = call @function(%fq_in) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%call, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[DUPL_FQ_IN:%.+]] = IE.FakeQuantize([[ARG]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]])
    // CHECK:          [[RELU:%.+]] = IE.ReLU([[DUPL_FQ_IN]])
    // CHECK:          [[DUPL_FQ_OUT:%.+]] = IE.FakeQuantize([[RELU]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]])
    // CHECK:          return [[DUPL_FQ_OUT]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[CALL:%.+]] = call @function([[ARG]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          return [[CALL]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DuplicateInsideWithReshapes
module @DuplicateInsideWithReshapes {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function(%arg0: tensor<48x60x60xf32>) -> tensor<48x60x60xf32> {
        %relu = IE.ReLU(%arg0) : tensor<48x60x60xf32> -> tensor<48x60x60xf32>
        return %relu : tensor<48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %fq_in_reshape = IE.AffineReshape(%fq_in) {dim_mapping = [[0], [0], [1], [2]], shape_value = [48, 60, 60]} : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
        %call = call @function(%fq_in_reshape) : (tensor<48x60x60xf32>) -> tensor<48x60x60xf32>
        %fq_out_reshape = IE.AffineReshape(%call) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 48, 60, 60]} : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%fq_out_reshape, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function([[ARG:%.+]]: tensor<48x60x60xf32>) -> tensor<48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>

    // CHECK:          [[DUPL_FQ_IN_RESHAPE1:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          [[DUPL_FQ_IN:%.+]] = IE.FakeQuantize([[DUPL_FQ_IN_RESHAPE1]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]])
    // CHECK:          [[DUPL_FQ_IN_RESHAPE2:%.+]] = IE.AffineReshape([[DUPL_FQ_IN]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>

    // CHECK:          [[RELU:%.+]] = IE.ReLU([[DUPL_FQ_IN_RESHAPE2]])

    // CHECK:          [[DUPL_FQ_OUT_RESHAPE1:%.+]] = IE.AffineReshape([[RELU]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          [[DUPL_FQ_OUT:%.+]] = IE.FakeQuantize([[DUPL_FQ_OUT_RESHAPE1]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]])
    // CHECK:          [[DUPL_FQ_OUT_RESHAPE2:%.+]] = IE.AffineReshape([[DUPL_FQ_OUT]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>

    // CHECK:          return [[DUPL_FQ_OUT_RESHAPE2]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[IN_RESHAPE:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:         : tensor<1x48x60x60xf32> -> tensor<48x60x60xf32>
    // CHECK:          [[CALL:%.+]] = call @function([[IN_RESHAPE]]) : (tensor<48x60x60xf32>) -> tensor<48x60x60xf32>
    // CHECK:          [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CALL]])
    // CHECK-SAME:         : tensor<48x60x60xf32> -> tensor<1x48x60x60xf32>
    // CHECK:          return [[OUT_RESHAPE]]
    // CHECK:      }
}

// -----

// CHECK-LABEL: @DuplicateFQInsideBetweenCalls
module @DuplicateFQInsideBetweenCalls {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf32>
    }
    func.func private @function1(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %relu = IE.ReLU(%arg0) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        // FQ is duplicated here
        return %relu : tensor<1x48x60x60xf32>
    }
    func.func private @function2(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %in_low = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
        %in_high = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
        %out_low = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
        %out_high = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
        %fq_in = IE.FakeQuantize(%arg0, %in_low, %in_high, %in_low, %in_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        %relu = IE.ReLU(%fq_in) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        %fq_out = IE.FakeQuantize(%relu, %out_low, %out_high, %out_low, %out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32} : tensor<1x48x60x60xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x48x60x60xf32>
        return %fq_out : tensor<1x48x60x60xf32>
    }
    func.func private @function3(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        // FQ is duplicated here
        %relu = IE.ReLU(%arg0) : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %relu : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %call1 = call @function1(%arg0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %call2 = call @function2(%call1) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        %call3 = call @function3(%call2) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %call3 : tensor<1x48x60x60xf32>
    }

    // CHECK:      func.func private @function1([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FN1_RELU:%.+]] = IE.ReLU([[ARG]])
    // CHECK:          [[DUPL_FQ_IN:%.+]] = IE.FakeQuantize([[FN1_RELU]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]], [[DUPL_IN_LOW]], [[DUPL_IN_HIGH]])
    // CHECK:          return [[DUPL_FQ_IN]]
    // CHECK:      }
    // CHECK:      func.func private @function2([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FQ_IN:%.+]] = IE.FakeQuantize([[ARG]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK:          [[FN1_RELU:%.+]] = IE.ReLU([[FQ_IN]])
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[FN1_RELU]], [[OUT_LOW]], [[OUT_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    // CHECK:          return [[FQ_OUT]]
    // CHECK:      }
    // CHECK:      func.func private @function3([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK-DAG:      [[DUPL_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:      [[DUPL_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:          [[FQ_OUT:%.+]] = IE.FakeQuantize([[ARG]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]], [[DUPL_OUT_LOW]], [[DUPL_OUT_HIGH]])
    // CHECK:          [[FN2_RELU:%.+]] = IE.ReLU([[FQ_OUT]])
    // CHECK:          return [[FN2_RELU]]
    // CHECK:      }
    // CHECK:      func.func @main([[ARG:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
    // CHECK:          [[CALL1:%.+]] = call @function1([[ARG]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          [[CALL2:%.+]] = call @function2([[CALL1]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          [[CALL3:%.+]] = call @function3([[CALL2]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
    // CHECK:          return [[CALL3]]
    // CHECK:      }
}
