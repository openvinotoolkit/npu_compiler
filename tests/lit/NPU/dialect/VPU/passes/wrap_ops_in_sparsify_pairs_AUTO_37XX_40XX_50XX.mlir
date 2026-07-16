//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=auto" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    // CHECK-LABEL: @WrapSingleOpWithStats
    // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
    // CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    func.func @WrapSingleOpWithStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
        // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]
        // CHECK:       return [[VAL3]]
    }

    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.3 at input 0 of "Conv_1" loc(#loc0)
    }
}


//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {

    // CHECK-LABEL: @DoNotWrapSingleOpNotRelatedStats
    // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
    // CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    func.func @DoNotWrapSingleOpNotRelatedStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[VAL1:%.+]] = VPU.Sparsify([[VAL0]])
        // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]]
        // CHECK:       return [[VAL2]]
    }

    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.3 at input 0 of "Conv_2" loc(#loc0)
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @main {
    // CHECK-LABEL: @DoNotWrapSingleOpWithoutStats
    // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
    // CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    func.func @DoNotWrapSingleOpWithoutStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       return [[VAL0]]
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    // CHECK-LABEL: @WrapMultipleConsumers
    // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
    // CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    func.func @WrapMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])
        %2 = VPU.NCE.Convolution(%1, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_2", "t_Convolution"])
        %3 = VPU.NCE.Convolution(%1, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_3", "t_Convolution", "broadcast"])

        return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
        // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL4:%.+]] = VPU.NCE.Convolution([[VAL3]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL4]])
        // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL5]]

        // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL3]])
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL8:%.+]] = VPU.NCE.Convolution([[VAL7]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL9:%.+]] = VPU.Sparsify([[VAL8]])
        // CHECK:       [[VAL10:%.+]] = VPU.Desparsify([[VAL9]]

        // CHECK:       return [[VAL6]], [[VAL10]]
    }

    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.3 at input 0 of "Conv_1" loc(#loc0)
        net.SparsityInfo 0.0 at input 0 of "Conv_2" loc(#loc0)
        net.SparsityInfo 0.3 at input 0 of "Conv_3" loc(#loc0)
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    // CHECK-LABEL: @WrapMultipleMixedConsumers
    // CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
    // CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
    func.func @WrapMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])
        %2 = VPU.NCE.Eltwise(%1, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Add_1", "t_Convolution"])
        %3 = VPU.MaxPool(%1) {
            ppe = #VPU.PPEStub<>,
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Maxpool_1", "t_Convolution"])

        return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
        // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

        // CHECK:       [[VAL6:%.+]] = VPU.NCE.Eltwise([[VAL3]], [[VAL3]])
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL6]])
        // CHECK:       [[VAL8:%.+]] = VPU.Desparsify([[VAL7]]

        // CHECK:       [[VAL9:%.+]] = VPU.MaxPool([[VAL3]])
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>


        // CHECK:       return [[VAL8]], [[VAL9]]
    }

    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.8 at input 0 of "Conv_1" loc(#loc0)
        net.SparsityInfo 0.8 at input 0 of "Add_1" loc(#loc0)
        net.SparsityInfo 0.8 at input 0 of "Maxpool_1" loc(#loc0)
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    // CHECK: func.func @NotWrapFP32SingleOpWithStats([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      [[WT:%.+]]: tensor<16x1x1x4xsi32>
    // CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<16x16x1x1xf16, {order = #NHWC}>
    func.func @NotWrapFP32SingleOpWithStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf32, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf32, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf32, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[INPUT]])
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS]], [[WT]]
        // CHECK-NOT:       -> !VPU.SparseTensor
        // CHECK-SAME:      -> tensor<1x16x16x16xf32, {order = #NHWC}>
        // CHECK:       return [[VAL1]]
    }

    net.SparsityStatistics sparsityInfo : {
        net.SparsityInfo 0.3 at input 0 of "Conv_1" loc(#loc0)
    }
}
