//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-buffers-for-net-results --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @Network
module @Network {
    IE.CNNNetwork entryPoint : @SingleLayer
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }
    // CHECK: DataInfo "output" : tensor<1x1000xf16> loc([[LOC_OUTPUT:#.+]])

module @VPU.SW {
    func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
        attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
}

// CHECK: func.func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16> loc([[LOC_ARG0:.+]]), [[ARG1:%.*]]: memref<1x1000xf16> loc([[LOC_ARG1:.+]])) -> memref<1x1000xf16> {
func.func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_softmax
                    inputs(%arg0 as %input0: memref<1x1000xf16>)
                    outputs(%0 as %output0: memref<1x1000xf16>)
                    on tile 0 -> memref<1x1000xf16> {
        VPUIP.SW.Kernel.run {attrs = [1]}(%input0, %output0)
            : memref<1x1000xf16>
            , memref<1x1000xf16>
    }
    return %1 : memref<1x1000xf16>

    // CHECK: [[OUT0:%.*]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16> loc([[LOC_OUTPUT:#.+]])
    // CHECK: return [[OUT0]] : memref<1x1000xf16>
}
}

// CHECK: [[LOC_OUTPUT_NAME:#.+]] = loc("output")
// CHECK: [[LOC_OUTPUT]] = loc(fused<{name = "output", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT_NAME]]

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16> loc(fused<{name = "output1", type = "Result"}>["output1"])
        DataInfo "output2" : tensor<1x2x60x60xf16> loc(fused<{name = "output2", type = "Result"}>["output2"])
    }

        // CHECK: DataInfo "output1" : tensor<1x4x60x60xf16> loc([[LOC_OUTPUT1:#.+]])
        // CHECK: DataInfo "output2" : tensor<1x2x60x60xf16> loc([[LOC_OUTPUT2:#.+]])

    // CHECK:       func.func @foo1({{[^:]+}}: memref<1x8x60x60xf16> loc([[LOC_FOO1_ARG0:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG1:.+]]), [[ARG2:[^:]+]]: memref<1x2x60x60xf16> loc([[LOC_FOO1_ARG2:.+]]))
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @foo1(%arg0: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x8x60x60xf16> to tensor<1x8x60x60xf16>

        %1 = VPU.Slice %0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %2 = builtin.unrealized_conversion_cast %1 : tensor<1x4x60x60xf16> to memref<1x4x60x60xf16> loc(fused<{name = "Slice1_out", type = "Slice1"}>["Slice1_out", "unrealized_cast"])

        %3 = VPU.Slice %0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        %4 = builtin.unrealized_conversion_cast %3 : tensor<1x2x60x60xf16> to memref<1x2x60x60xf16> loc(fused<{name = "Slice2_out", type = "Slice2"}>["Slice2_out", "unrealized_cast"])

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_FOO1_OUTPUT0:#.+]])
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_FOO1_OUTPUT1:#.+]])
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
        return %2, %4 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2({{[^:]+}}: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG0:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG1:.+]])) -> memref<1x4x60x60xf16>
    func.func @foo2(%arg0: memref<1x4x60x60xf16> ) -> memref<1x4x60x60xf16> {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x4x60x60xf16> to tensor<1x4x60x60xf16>
        %1 = VPU.MemPermute(%0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x4x60x60xf16> -> tensor<1x4x60x60xf16, {order = #NHWC}>
        %2 = VPU.NCE.ClusterTiling (%1 as %arg1: tensor<1x4x60x60xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %7 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x4x60x60xf16, {order = #NHWC}> -> tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
            VPU.Yield %7
        }
        %3 = VPU.NCE.ClusterTiling (%2 as %arg1: tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            %7 = VPU.SoftMax(%arg1) {axisInd = 1 : i64} : tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>
            VPU.Yield %7
        }
        %4 = VPU.NCE.ClusterTiling (%3 as %arg1: tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x4x60x60xf16, {order = #NHWC}> {
            %7 = VPU.Copy(%arg1) : tensor<1x4x60x60xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x4x60x60xf16, {order = #NHWC}>
            VPU.Yield %7
        }
        %5 = VPU.MemPermute(%4) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x4x60x60xf16, {order = #NHWC}> -> tensor<1x4x60x60xf16>
        %6 = builtin.unrealized_conversion_cast %5 : tensor<1x4x60x60xf16> to memref<1x4x60x60xf16>  loc(fused<{name = "MemPermute_out", type = "MemPermute"}>["MemPermute_out", "unrealized_cast"])

        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_FOO2_OUTPUT:#.+]])
        // CHECK: return [[OUT]] : memref<1x4x60x60xf16>
        return %6 : memref<1x4x60x60xf16>
    }

    // CHECK:       func.func @main([[ARG0:[^:]+]]: memref<1x8x60x60xf16> loc([[LOC_MAIN_ARG1:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_MAIN_ARG2:.+]]), [[ARG2:[^:]+]]: memref<1x2x60x60xf16> loc([[LOC_MAIN_ARG3:.+]]))
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @main(%arg0: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0:2 = call @foo1(%arg0) : (memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        %1 = call @foo2(%0#0) : (memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        return %1, %0#1 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>

        // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK:       [[FOO1_RES:%.+]]:2 = call @foo1([[ARG0]], [[ALLOC1]], [[ALLOC2]]) : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK-SAME:       -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)

        // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0, [[ALLOC3]]) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_OUTPUT1]])
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs([[FOO1_RES]]#1 : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_OUTPUT2]])
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}

// CHECK: [[LOC_OUTPUT1_NAME:#.+]] = loc("output1")
// CHECK: [[LOC_OUTPUT2_NAME:#.+]] = loc("output2")

// CHECK: [[CAST:#.+]] = loc("unrealized_cast")

// CHECK: [[LOC_FOO1_OUT0_NAME:#.+]] = loc("foo1_outputBuff0")
// CHECK: [[LOC_FOO1_OUT1_NAME:#.+]] = loc("foo1_outputBuff1")
// CHECK: [[LOC_FOO2_OUT_NAME:#.+]] = loc("foo2_outputBuff0")

// CHECK: [[LOC_OUTPUT1]] = loc(fused<{name = "output1", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT1_NAME]]
// CHECK: [[LOC_OUTPUT2]] = loc(fused<{name = "output2", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT2_NAME]]

// CHECK: [[LOC_FOO1_OUTPUT0]] = loc(fused<{name = "Slice1_out", type = "Slice1"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT0_NAME]]])
// CHECK: [[LOC_FOO1_OUTPUT1]] = loc(fused<{name = "Slice2_out", type = "Slice2"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT1_NAME]]])

// CHECK: [[LOC_FOO2_OUTPUT]] = loc(fused<{name = "MemPermute_out", type = "MemPermute"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO2_OUT_NAME]]])

// -----

// foo1 has more outputs than main
// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x60x60xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }
        // CHECK: DataInfo "input" : tensor<1x8x60x60xf16> loc([[LOC_INPUT:#.+]])
        // CHECK: DataInfo "output" : tensor<1x4x60x60xf16> loc([[LOC_OUTPUT:#.+]])

    // CHECK: func.func @foo1([[ARG0:%.+]]: memref<1x8x60x60xf16> loc([[LOC_FOO1_ARG0:.+]]), [[ARG1:%.+]]: memref<1x2x60x60xf16> loc([[LOC_FOO1_ARG1:.+]]), [[ARG2:%.+]]: memref<1x2x60x60xf16> loc([[LOC_FOO1_ARG2:.+]]))
    // CHECK-SAME:              -> (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>)
    func.func @foo1(%arg0: memref<1x8x60x60xf16>) -> (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x8x60x60xf16> to tensor<1x8x60x60xf16>

        %1 = VPU.Slice %0 [0, 2, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        %2 = builtin.unrealized_conversion_cast %1 : tensor<1x2x60x60xf16> to memref<1x2x60x60xf16> loc(fused<{name = "Slice1_out", type = "Slice1"}>["Slice1_out", "unrealized_cast"])

        %3 = VPU.Slice %0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        %4 = builtin.unrealized_conversion_cast %3 : tensor<1x2x60x60xf16> to memref<1x2x60x60xf16> loc(fused<{name = "Slice2_out", type = "Slice2"}>["Slice2_out", "unrealized_cast"])
        return %2, %4 : memref<1x2x60x60xf16>, memref<1x2x60x60xf16>

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x2x60x60xf16>) outputs([[ARG1]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_FOO1_OUTPUT0:#.+]])
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_FOO1_OUTPUT1:#.+]])
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x2x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: memref<1x2x60x60xf16> loc([[LOC_FOO2_ARG0:.+]]), [[ARG1:%.+]]: memref<1x2x60x60xf16> loc([[LOC_FOO2_ARG1:.+]]), [[ARG2:%.+]]: memref<1x4x60x60xf16> loc([[LOC_FOO2_ARG2:.+]]))
    // CHECK-SAME:              -> memref<1x4x60x60xf16>
    func.func @foo2(%arg0: memref<1x2x60x60xf16>, %arg1: memref<1x2x60x60xf16>) -> memref<1x4x60x60xf16> {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x2x60x60xf16> to tensor<1x2x60x60xf16>
        %1 = builtin.unrealized_conversion_cast %arg1 : memref<1x2x60x60xf16> to tensor<1x2x60x60xf16>

        %2 = VPU.Concat(%0, %1) {static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]}: tensor<1x2x60x60xf16>, tensor<1x2x60x60xf16> -> tensor<1x4x60x60xf16>

        %3 = builtin.unrealized_conversion_cast %2 : tensor<1x4x60x60xf16> to memref<1x4x60x60xf16> loc(fused<{name = "Concat_out", type = "Concat"}>["Concat_out", "unrealized_cast"])
        return %3 : memref<1x4x60x60xf16>

        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG2]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_FOO2_OUTPUT:#.+]])
        // CHECK: return [[OUT]] : memref<1x4x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x8x60x60xf16> loc([[LOC_MAIN_ARG0:.+]]), [[ARG1:%.+]]: memref<1x4x60x60xf16> loc([[LOC_MAIN_ARG1:.+]]) -> memref<1x4x60x60xf16>
    func.func @main(%arg0: memref<1x8x60x60xf16>) -> memref<1x4x60x60xf16> {
        %0:2 = call @foo1(%arg0) : (memref<1x8x60x60xf16>) -> (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>)
        %1 = call @foo2(%0#0, %0#1) : (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>) -> memref<1x4x60x60xf16>
        return %1 : memref<1x4x60x60xf16>

        // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK:       [[FOO1_RES:%.+]]:2 = call @foo1(%arg0, [[ALLOC0]], [[ALLOC1]]) : (memref<1x8x60x60xf16>, memref<1x2x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK-SAME:                              -> (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>)

        // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK:       [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0, [[FOO1_RES]]#1, [[ALLOC2]]) : (memref<1x2x60x60xf16>, memref<1x2x60x60xf16>, memref<1x4x60x60xf16>)
        // CHECK-SAME:                              -> memref<1x4x60x60xf16>

        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_OUTPUT:#.+]])
        // CHECK: return [[OUT]] : memref<1x4x60x60xf16>
    }
}

// CHECK: [[LOC_INPUT_NAME:#.+]] = loc("input")
// CHECK: [[LOC_OUTPUT_NAME:#.+]] = loc("output")

// CHECK: [[CAST:#.+]] = loc("unrealized_cast")

// CHECK: [[LOC_FOO1_OUT0_NAME:#.+]] = loc("foo1_outputBuff0")
// CHECK: [[LOC_FOO1_OUT1_NAME:#.+]] = loc("foo1_outputBuff1")
// CHECK: [[LOC_FOO2_OUT_NAME:#.+]] = loc("foo2_outputBuff0")

// CHECK: [[LOC_INPUT]] = loc(fused<{name = "input", type = "Parameter"}>
// CHECK-SAME: [[LOC_INPUT_NAME]]
// CHECK: [[LOC_OUTPUT]] = loc(fused<{name = "output", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT_NAME]]

// CHECK: [[LOC_FOO1_OUTPUT0]] = loc(fused<{name = "Slice1_out", type = "Slice1"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT0_NAME]]])
// CHECK: [[LOC_FOO1_OUTPUT1]] = loc(fused<{name = "Slice2_out", type = "Slice2"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT1_NAME]]])

// CHECK: [[LOC_FOO2_OUTPUT]] = loc(fused<{name = "Concat_out", type = "Concat"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO2_OUT_NAME]]])

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// Corner case: foo2 is empty
// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16> loc(fused<{name = "output1", type = "Result"}>["output1"])
        DataInfo "output2" : tensor<1x2x60x60xf16> loc(fused<{name = "output2", type = "Result"}>["output2"])
    }

        // CHECK: DataInfo "output1" : tensor<1x4x60x60xf16> loc([[LOC_OUTPUT1:#.+]])
        // CHECK: DataInfo "output2" : tensor<1x2x60x60xf16> loc([[LOC_OUTPUT2:#.+]])

    // CHECK:       func.func @foo1({{[^:]+}}: memref<1x8x60x60xf16> loc([[LOC_FOO1_ARG0:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG1:.+]]), [[ARG2:[^:]+]]: memref<1x2x60x60xf16> loc([[LOC_FOO1_ARG2:.+]]))
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @foo1(%arg0: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x8x60x60xf16> to tensor<1x8x60x60xf16>

        %1 = VPU.Slice %0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %2 = builtin.unrealized_conversion_cast %1 : tensor<1x4x60x60xf16> to memref<1x4x60x60xf16> loc(fused<{name = "Slice1_out", type = "Slice1"}>["Slice1_out", "unrealized_cast"])

        %3 = VPU.Slice %0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        %4 = builtin.unrealized_conversion_cast %3 : tensor<1x2x60x60xf16> to memref<1x2x60x60xf16> loc(fused<{name = "Slice2_out", type = "Slice2"}>["Slice2_out", "unrealized_cast"])

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_FOO1_OUTPUT0:#.+]])
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_FOO1_OUTPUT1:#.+]])
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
        return %2, %4 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2({{[^:]+}}: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG0:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_FOO1_ARG1:.+]])) -> memref<1x4x60x60xf16>
    func.func @foo2(%arg0: memref<1x4x60x60xf16> ) -> memref<1x4x60x60xf16> {
        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_FOO2_OUTPUT:#.+]])
        // CHECK: return [[OUT]] : memref<1x4x60x60xf16>
        return %arg0 : memref<1x4x60x60xf16>
    }
    // CHECK: } loc([[LOC_FOO2:#.+]])

    // CHECK:       func.func @main([[ARG0:[^:]+]]: memref<1x8x60x60xf16> loc([[LOC_MAIN_ARG1:.+]]), [[ARG1:[^:]+]]: memref<1x4x60x60xf16> loc([[LOC_MAIN_ARG2:.+]]), [[ARG2:[^:]+]]: memref<1x2x60x60xf16> loc([[LOC_MAIN_ARG3:.+]]))
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @main(%arg0: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0:2 = call @foo1(%arg0) : (memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        %1 = call @foo2(%0#0) : (memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        return %1, %0#1 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>

        // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK:       [[ALLOC2:%.+]] = memref.alloc() : memref<1x2x60x60xf16>
        // CHECK:       [[FOO1_RES:%.+]]:2 = call @foo1([[ARG0]], [[ALLOC1]], [[ALLOC2]]) : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK-SAME:       -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)

        // CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<1x4x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0, [[ALLOC3]]) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> loc([[LOC_OUTPUT1]])
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs([[FOO1_RES]]#1 : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16> loc([[LOC_OUTPUT2]])
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}

// CHECK: [[LOC_OUTPUT1_NAME:#.+]] = loc("output1")
// CHECK: [[LOC_OUTPUT2_NAME:#.+]] = loc("output2")

// CHECK: [[CAST:#.+]] = loc("unrealized_cast")

// CHECK: [[LOC_FOO1_OUT0_NAME:#.+]] = loc("foo1_outputBuff0")
// CHECK: [[LOC_FOO1_OUT1_NAME:#.+]] = loc("foo1_outputBuff1")
// CHECK: [[LOC_FOO2_OUT_NAME:#.+]] = loc("foo2_outputBuff0")

// CHECK: [[LOC_OUTPUT1]] = loc(fused<{name = "output1", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT1_NAME]]
// CHECK: [[LOC_OUTPUT2]] = loc(fused<{name = "output2", type = "Result"}>
// CHECK-SAME: [[LOC_OUTPUT2_NAME]]

// CHECK: [[LOC_FOO1_OUTPUT0]] = loc(fused<{name = "Slice1_out", type = "Slice1"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT0_NAME]]])
// CHECK: [[LOC_FOO1_OUTPUT1]] = loc(fused<{name = "Slice2_out", type = "Slice2"}>[{{[^:]+}}, [[CAST]], [[LOC_FOO1_OUT1_NAME]]])

// CHECK: [[LOC_FOO2_OUTPUT]] = loc(fused[[[LOC_FOO2]], [[LOC_FOO2_OUT_NAME]]])
