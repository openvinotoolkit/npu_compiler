//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --dump-intermediate-values="op-filters=\"[{name: IE.Convolution, locations: []}, {name: IE.*Pool, locations: [\"pool1\"]}]\"" %s | FileCheck --check-prefix="CHECK-IE" %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --dump-intermediate-values="op-filters=\"[{name: VPU.NCE.Convolution, locations: []}]\"" %s | FileCheck --check-prefix="CHECK-VPU" %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-IE-LABEL: @OneOp
module @OneOp {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x16xf16>
    }

    // CHECK-IE:      net.NetworkInfo
    // CHECK-IE:      outputsInfo
    // CHECK-IE-NEXT:   DataInfo "output" : tensor<1x16x16x16xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x16x16xf16>
    // CHECK-VPU-NOT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x16x16xf16>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>) {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    func.func @main(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
        %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
        %conv = IE.Convolution(%input, %weights, %bias) { dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>
        %softmax = IE.SoftMax(%conv) {axisInd = 1 : i64} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>
        return %softmax : tensor<1x16x16x16xf16>

        // CHECK-IE: [[CONV:%.+]] = IE.Convolution
        // CHECK-IE: [[SOFTMAX:%.+]] = IE.SoftMax([[CONV]])
        // CHECK-IE: return [[SOFTMAX]], [[CONV]] : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>

        // CHECK-VPU: return {{.+}} : tensor<1x16x16x16xf16>
    }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.1>

// CHECK-IE-LABEL: @OneOpQuantized
module @OneOpQuantized {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x16x16xf16>
    }

    // CHECK-IE:      net.NetworkInfo
    // CHECK-IE:      outputsInfo
    // CHECK-IE-NEXT:   DataInfo "output" : tensor<1x16x16x16xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x16x16xui8>
    // CHECK-VPU-NOT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x16x16xui8>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>) {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    func.func @main(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
        %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
        %conv = IE.Convolution(%input, %weights, %bias) { dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16x!qElemType>
        %add = IE.Add(%conv, %conv) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16x!qElemType>, tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xf16>
        return %add : tensor<1x16x16x16xf16>

        // CHECK-IE: [[CONV:%.+]] = IE.Convolution
        // CHECK-IE: [[ADD:%.+]] = IE.Add([[CONV]], [[CONV]])
        // CHECK-IE: return [[ADD]], [[CONV]] : tensor<1x16x16x16xf16>, tensor<1x16x16x16x!qElemType>

        // CHECK-VPU: return {{.+}} : tensor<1x16x16x16xf16>
    }
}

// -----

// CHECK-IE-LABEL: @MultipleOpsWithLocations
module @MultipleOpsWithLocations {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x4x4xf16>
    }

    // CHECK-IE:      net.NetworkInfo
    // CHECK-IE:      outputsInfo
    // CHECK-IE-NEXT:   DataInfo "output" : tensor<1x16x4x4xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_0_Convolution_conv" : tensor<1x16x16x16xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_1_MaxPool_pool1" : tensor<1x16x8x8xf16>
    // CHECK-VPU-NOT:   DataInfo "dump_0_Convolution_conv" : tensor<1x16x16x16xf16>
    // CHECK-VPU-NOT:   DataInfo "dump_1_MaxPool_pool1" : tensor<1x16x8x8xf16>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> (tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>) {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
    func.func @main(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
        %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
        %conv = IE.Convolution(%input, %weights, %bias) { dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16> loc(fused<{name = "conv", type = "Convolution"}>["conv"])
        %maxpool1 = IE.MaxPool(%conv) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2], rounding_type = #IE.rounding_type<FLOOR>}
            : tensor<1x16x16x16xf16> -> tensor<1x16x8x8xf16> loc(fused<{name = "pool1", type = "MaxPool"}>["pool1"])
        %maxpool2 = IE.MaxPool(%maxpool1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2], rounding_type = #IE.rounding_type<FLOOR>}
            : tensor<1x16x8x8xf16> -> tensor<1x16x4x4xf16> loc(fused<{name = "pool2", type = "MaxPool"}>["pool2"])
        %softmax = IE.SoftMax(%maxpool2) {axisInd = 1 : i64} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
        return %softmax : tensor<1x16x4x4xf16>

        // CHECK-IE: [[CONV:%.+]] = IE.Convolution
        // CHECK-IE: [[MAXPOOL1:%.+]] = IE.MaxPool([[CONV]])
        // CHECK-IE: [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]])
        // CHECK-IE: [[SOFTMAX:%.+]] = IE.SoftMax([[MAXPOOL2]])
        // CHECK-IE: return [[SOFTMAX]], [[CONV]], [[MAXPOOL1]] : tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>

        // CHECK-VPU: return {{.+}} : tensor<1x16x4x4xf16>
    }
}

// -----

// CHECK-IE-LABEL: @NestedCallWithLocations
module @NestedCallWithLocations {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x16x16xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x4x4xf16>
    }

    // CHECK-IE:      net.NetworkInfo
    // CHECK-IE:      outputsInfo
    // CHECK-IE-NEXT:   DataInfo "output" : tensor<1x16x4x4xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_0_Convolution_conv" : tensor<1x16x16x16xf16>
    // CHECK-IE-NEXT:   DataInfo "dump_1_MaxPool_pool1" : tensor<1x16x8x8xf16>
    // CHECK-VPU-NOT:   DataInfo "dump_0_Convolution_conv" : tensor<1x16x16x16xf16>
    // CHECK-VPU-NOT:   DataInfo "dump_1_MaxPool_pool1" : tensor<1x16x8x8xf16>

    // CHECK-IE: func.func @fn({{%.+}}: tensor<1x16x16x16xf16>) -> (tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>) {
    // CHECK-VPU: func.func @fn({{%.+}}: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
    func.func @fn(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
        %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
        %conv = IE.Convolution(%input, %weights, %bias) { dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
            : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16> loc(fused<{name = "conv", type = "Convolution"}>["conv"])
        %maxpool1 = IE.MaxPool(%conv) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2], rounding_type = #IE.rounding_type<FLOOR>}
            : tensor<1x16x16x16xf16> -> tensor<1x16x8x8xf16> loc(fused<{name = "pool1", type = "MaxPool"}>["pool1"])
        %maxpool2 = IE.MaxPool(%maxpool1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2], rounding_type = #IE.rounding_type<FLOOR>}
            : tensor<1x16x8x8xf16> -> tensor<1x16x4x4xf16> loc(fused<{name = "pool2", type = "MaxPool"}>["pool2"])
        %softmax = IE.SoftMax(%maxpool2) {axisInd = 1 : i64} : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>
        return %softmax : tensor<1x16x4x4xf16>

        // CHECK-IE: [[CONV:%.+]] = IE.Convolution
        // CHECK-IE: [[MAXPOOL1:%.+]] = IE.MaxPool([[CONV]])
        // CHECK-IE: [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]])
        // CHECK-IE: [[SOFTMAX:%.+]] = IE.SoftMax([[MAXPOOL2]])
        // CHECK-IE: return [[SOFTMAX]], [[CONV]], [[MAXPOOL1]] : tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>

        // CHECK-VPU: return {{.+}} : tensor<1x16x4x4xf16>
    }

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> (tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>) {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
    func.func @main(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x4x4xf16> {
        %call = call @fn(%input) : (tensor<1x16x16x16xf16>) -> (tensor<1x16x4x4xf16>)
        return %call : tensor<1x16x4x4xf16>

        // CHECK-IE: [[CALL:%.+]]:3 = call @fn
        // CHECK-IE: return [[CALL]]#0, [[CALL]]#1, [[CALL]]#2 : tensor<1x16x4x4xf16>, tensor<1x16x16x16xf16>, tensor<1x16x8x8xf16>

        // CHECK-VPU: return {{.+}} : tensor<1x16x4x4xf16>
    }
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-VPU-LABEL: @DumpVPUOps
module @DumpVPUOps {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x8x8xf16>
    }

    // CHECK-VPU:      net.NetworkInfo
    // CHECK-VPU:      outputsInfo
    // CHECK-VPU-NEXT:   DataInfo "output" : tensor<1x16x8x8xf16>
    // CHECK-VPU-NEXT:   DataInfo "dump_0_<Unknown>" : tensor<1x8x8x16xf16>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]
        } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %softmax = VPU.SoftMax(%conv) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        return %softmax : tensor<1x16x8x8xf16, {order = #NHWC}>

        // CHECK-IE:       return {{.+}} : tensor<1x16x8x8xf16, {order = #NHWC}>

        // CHECK-VPU:      [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-VPU:      [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
        // CHECK-VPU:      return [[SOFTMAX]], [[CONV]] : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
    }
}


// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-VPU-LABEL: @DumpVPUOpsFromCMX
module @DumpVPUOpsFromCMX {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x8x8xf16>
    }

    // CHECK-IE-NOT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK-VPU:      net.NetworkInfo
    // CHECK-VPU:      outputsInfo
    // CHECK-VPU-NEXT:   DataInfo "output" : tensor<1x16x8x8xf16>
    // CHECK-VPU-NEXT:   DataInfo "dump_0_<Unknown>" : tensor<1x8x8x16xf16, {mem_space = @DDR, order = #NCHW}>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>) {
    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]
        } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        %softmax = VPU.SoftMax(%conv) {axisInd = 1 : i64} : tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        return %softmax : tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

        // CHECK-IE:       return {{.+}} : tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

        // CHECK-VPU:      [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-VPU:      [[CONV_COPY:%.+]] = VPU.Copy([[CONV]])
        // CHECK-VPU-SAME:   -> tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>
        // CHECK-VPU:      [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
        // CHECK-VPU:      return [[SOFTMAX]], [[CONV_COPY]] : tensor<1x16x8x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>
    }
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedType = !VPU.DistributedTensor<
    1x16x8x8xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

// CHECK-VPU-LABEL: @DumpDistributedVPUOpsFromCMX
module @DumpDistributedVPUOpsFromCMX {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x8x8xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x8x8xf16>
    }

    // CHECK-IE-NOT:   DataInfo "dump_0_<Unknown>" : tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK-VPU:      net.NetworkInfo
    // CHECK-VPU:      outputsInfo
    // CHECK-VPU-NEXT:   DataInfo "output" : tensor<1x16x8x8xf16>
    // CHECK-VPU-NEXT:   DataInfo "dump_0_<Unknown>" : tensor<1x8x8x16xf16, {mem_space = @DDR, order = #NCHW}>

    // CHECK-IE: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    // CHECK-VPU: func.func @main({{%.+}}: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>) {
    func.func @main(%input: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]
        } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> !DistributedType
        %softmax = VPU.SoftMax(%conv) {axisInd = 1 : i64} : !DistributedType -> !DistributedType
        %softmax_copy = VPU.Copy(%softmax) {out_mem_space = @DDR} : !DistributedType -> tensor<1x16x8x8xf16, {order = #NHWC}>
        return %softmax_copy : tensor<1x16x8x8xf16, {order = #NHWC}>

        // CHECK-IE:       return {{.+}} : tensor<1x16x8x8xf16, {order = #NHWC}>

        // CHECK-VPU:      [[CONV:%.+]] = VPU.NCE.Convolution
        // CHECK-VPU:      [[CONV_COPY:%.+]] = VPU.Copy([[CONV]])
        // CHECK-VPU-SAME:   -> tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>
        // CHECK-VPU:      [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
        // CHECK-VPU:      [[SOFTMAX_COPY:%.+]] = VPU.Copy([[SOFTMAX]])
        // CHECK-VPU:      return [[SOFTMAX_COPY]], [[CONV_COPY]] : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {mem_space = @DDR, order = #NHWC}>
    }
}
