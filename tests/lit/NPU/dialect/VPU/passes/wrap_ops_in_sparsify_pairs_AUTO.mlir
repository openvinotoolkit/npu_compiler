//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=auto" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    func.func @WrapSingleOpWithStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
            // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
        // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]
        // CHECK:       return [[VAL3]]
    }

    IE.SparsityStatistics sparsityInfo : {
        IE.SparsityInfo 0.3 at input 0 of "Conv_1" loc(#loc0)
    }
}


//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {

    func.func @DoNotWrapSingleOpNotRelatedStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %arg2, %arg1)
            // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>
        // CHECK:       [[VAL1:%.+]] = VPU.Sparsify([[VAL0]])
        // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]]
        // CHECK:       return [[VAL2]]
    }

    IE.SparsityStatistics sparsityInfo : {
        IE.SparsityInfo 0.3 at input 0 of "Conv_2" loc(#loc0)
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @main {
    func.func @DoNotWrapSingleOpWithoutStats(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])

        return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify
        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %arg2, %arg1)
            // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>
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
    func.func @WrapMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])
        %2 = VPU.NCE.Convolution(%1, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_2", "t_Convolution"])
        %3 = VPU.NCE.Convolution(%1, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_3", "t_Convolution", "broadcast"])

        return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
        // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

        // CHECK-NOT:   VPU.Sparsify
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL4:%.+]] = VPU.NCE.Convolution([[VAL3]], %arg2, %arg1)
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL4]])
        // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL5]]

        // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL3]])
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL8:%.+]] = VPU.NCE.Convolution([[VAL7]], %arg2, %arg1)
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       [[VAL9:%.+]] = VPU.Sparsify([[VAL8]])
        // CHECK:       [[VAL10:%.+]] = VPU.Desparsify([[VAL9]]

        // CHECK:       return [[VAL6]], [[VAL10]]
    }

    IE.SparsityStatistics sparsityInfo : {
        IE.SparsityInfo 0.3 at input 0 of "Conv_1" loc(#loc0)
        IE.SparsityInfo 0.0 at input 0 of "Conv_2" loc(#loc0)
        IE.SparsityInfo 0.3 at input 0 of "Conv_3" loc(#loc0)
    }
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

#loc0 = loc(unknown)
module @main {
    func.func @WrapMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
        %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } -> tensor<1x16x16x16xf16, {order = #NHWC}> loc(fused["Conv_1", "t_Convolution"])
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

        // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0)
        // CHECK-NOT:   VPU.Desparsify

        // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], %arg2, %arg1)
        // CHECK-NOT:       !VPU.SparseTensor
        // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

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

    IE.SparsityStatistics sparsityInfo : {
        IE.SparsityInfo 0.8 at input 0 of "Conv_1" loc(#loc0)
        IE.SparsityInfo 0.8 at input 0 of "Add_1" loc(#loc0)
        IE.SparsityInfo 0.8 at input 0 of "Maxpool_1" loc(#loc0)
    }
}
