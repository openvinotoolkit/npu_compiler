//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-buffers-for-net-results --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Network
module @Network {
    net.NetworkInfo entryPoint : @SingleLayer
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
    net.NetworkInfo entryPoint : @main
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
        %2 = VPU.Copy(%1) {out_mem_space = @CMX_NN} : tensor<1x4x60x60xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        %3 = VPU.SoftMax(%2) {axisInd = 1 : i64} : !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
                -> !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
        %4 = VPU.Copy(%3) : !VPU.DistributedTensor<1x4x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
                -> tensor<1x4x60x60xf16, {order = #NHWC}>
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
    net.NetworkInfo entryPoint : @main
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
    net.NetworkInfo entryPoint : @main
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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
!DistributedBufferInput = !VPUIP.DistributedBuffer<1x3x384x336xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 3, 96, 336], [1, 3, 96, 336], [1, 3, 96, 336], [1, 3, 96, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]], memory_shapes = [[1, 3, 96, 336], [1, 3, 96, 336], [1, 3, 96, 336], [1, 3, 96, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]]}>
!DistributedBufferNCEInput = !VPUIP.DistributedBuffer<1x336x3x384xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 336, 3, 96], [1, 336, 3, 96], [1, 336, 3, 96], [1, 336, 3, 96]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]], memory_shapes = [[1, 336, 3, 96], [1, 336, 3, 96], [1, 336, 3, 96], [1, 336, 3, 96]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]]}>
!DistributedBufferFoo1Result = !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]], memory_shapes = [[1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]]}>
// CHECK-LABEL: @TwoFunctionsDistributedType
module @TwoFunctionsDistributedType {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x384x336xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x384x336xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    // CHECK: DataInfo "input" : tensor<1x3x384x336xf16> loc([[LOC_INPUT:#.+]])
    // CHECK: DataInfo "output" : tensor<1x16x384x336xf16> loc([[LOC_OUTPUT:#.+]])

    // CHECK:       func.func @foo1({{[^:]+}}: memref<1x3x384x336xf16> loc([[LOC_FOO1_ARG0:.+]]),
    // CHECK-SAME:                 [[FOO1_ARG1:%.+]]: !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN,
    // CHECK-SAME:                                 {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:                                 }> loc([[LOC_FOO1_ARG1:.+]]))
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN
    func.func @foo1(%arg0: memref<1x3x384x336xf16>) -> (!DistributedBufferFoo1Result) {
        %0 = VPURT.AllocDistributed -> !DistributedBufferInput
        %1 = VPUIP.Copy inputs(%arg0 : memref<1x3x384x336xf16>) outputs(%0 : !DistributedBufferInput) -> !DistributedBufferInput
        %2 = VPUIP.ViewOp %1 : !DistributedBufferInput to !DistributedBufferNCEInput
        %3 = VPURT.AllocDistributed -> !DistributedBufferFoo1Result
        %4 = VPUIP.NCEClusterTask {is_permute_quantize, minimumHardwareExecutionCost = 9753 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
                input(%2 : !DistributedBufferNCEInput)
                weights(%2 : !DistributedBufferNCEInput)
                parent_input(%2 : !DistributedBufferNCEInput)
                parent_output(%3 : !DistributedBufferFoo1Result)
                outputs(%3 : !DistributedBufferFoo1Result) -> !DistributedBufferFoo1Result variants : {
                    DPUTask {cluster_id = 0 : i64, inEnd = [95, 2, 335], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 2, 335], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                    DPUTask {cluster_id = 1 : i64, inEnd = [95, 2, 335], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 2, 335], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                    DPUTask {cluster_id = 2 : i64, inEnd = [95, 2, 335], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 2, 335], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                    DPUTask {cluster_id = 3 : i64, inEnd = [95, 2, 335], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 2, 335], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
                    } PPE : {
                    PPETask {ppe = #VPU.PPEInt<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 1.000000e+00 : f64>}
        }
        return %4 : !DistributedBufferFoo1Result
        // CHECK: [[FOO1_NCE_OUTPUT:%.+]] = VPUIP.NCEClusterTask
        // CHECK:   } loc([[LOC_FOO1_NCE_OUTPUT:#.+]])

        // CHECK: [[FOO1_OUT:%.+]] = VPUIP.Copy inputs([[FOO1_NCE_OUTPUT]] : !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
        // CHECK-SAME: outputs([[FOO1_ARG1]] : !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
        // CHECK-SAME: -> !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
        // CHECK-SAME:   loc([[LOC_FOO1_OUTPUT:#.+]])

        // CHECK: return [[FOO1_OUT]] :
        // CHECK-SAME: !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
    }

    // CHECK: func.func @foo2({{[^:]+}}: !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN,
    // CHECK-SAME:                            {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
    // CHECK-SAME:                            }> loc([[LOC_FOO2_ARG0:.+]]),
    // CHECK-SAME:            [[FOO2_ARG1:%.+]]: memref<1x16x384x336xf16, #NHWC> loc([[LOC_FOO2_ARG1:.+]]))
    // CHECK-SAME:     -> memref<1x16x384x336xf16, #NHWC>
    func.func @foo2(%arg0: !DistributedBufferFoo1Result) -> memref<1x16x384x336xf16, #NHWC> {
        %0 = VPUIP.ViewOp %arg0 : !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]], memory_shapes = [[1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96], [1, 336, 16, 96]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 96], [0, 0, 0, 192], [0, 0, 0, 288]]}> to !VPUIP.DistributedBuffer<1x16x384x336xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]], memory_shapes = [[1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]]}>
        %alloc = memref.alloc() : memref<1x16x384x336xf16, #NHWC>
        %1 = VPUIP.Copy inputs(%0 : !VPUIP.DistributedBuffer<1x16x384x336xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]], memory_shapes = [[1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336], [1, 16, 96, 336]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0], [0, 0, 288, 0]]}>)
                         outputs(%alloc : memref<1x16x384x336xf16, #NHWC>) -> memref<1x16x384x336xf16, #NHWC>
        return %1 : memref<1x16x384x336xf16, #NHWC>

        // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x16x384x336xf16, #NHWC>
        // CHECK: [[FOO2_COPY_OUTPUT:%.+]] = VPUIP.Copy
        // CHECK-SAME: inputs({{[^:]+}} : !VPUIP.DistributedBuffer<1x16x384x336xf16, #NHWC, @CMX_NN
        // CHECK-SAME: outputs([[ALLOC]] : memref<1x16x384x336xf16, #NHWC>)
        // CHECK-SAME: loc([[LOC_FOO2_COPY_OUTPUT:#.+]])

        // CHECK: [[FOO2_OUT:%.+]] = VPUIP.Copy
        // CHECK-SAME: inputs([[FOO2_COPY_OUTPUT]] : memref<1x16x384x336xf16, #NHWC>)
        // CHECK-SAME: outputs([[FOO2_ARG1]] : memref<1x16x384x336xf16, #NHWC>)
        // CHECK-SAME: -> memref<1x16x384x336xf16, #NHWC>
        // CHECK: loc([[LOC_FOO2_OUTPUT:#.+]])

        // CHECK: return [[FOO2_OUT]] : memref<1x16x384x336xf16, #NHWC>
    }

    // CHECK:       func.func @main([[ARG0:%.+]]: memref<1x3x384x336xf16> loc([[LOC_MAIN_ARG1:.+]]), [[ARG1:%.+]]: memref<1x16x384x336xf16, #NHWC> loc([[LOC_MAIN_ARG2:.+]]))
    // CHECK-SAME:      -> memref<1x16x384x336xf16, #NHWC> {
    func.func @main(%arg0: memref<1x3x384x336xf16>) -> (memref<1x16x384x336xf16, #NHWC>) {
        %0 = call @foo1(%arg0) : (memref<1x3x384x336xf16>) -> (!DistributedBufferFoo1Result)
        %1 = call @foo2(%0) : (!DistributedBufferFoo1Result) -> memref<1x16x384x336xf16, #NHWC>
        return %1 : memref<1x16x384x336xf16, #NHWC>
        // CHECK: [[ALLOC0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN,
        // CHECK-SAME:             {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 4], num_clusters = 4 : i64, uniform_distributed_segments
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[ARG0]], [[ALLOC0]]) :
        // CHECK-SAME:                    (memref<1x3x384x336xf16>, !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN
        // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN

        // CHECK: [[ALLOC1:%.+]] = memref.alloc() : memref<1x16x384x336xf16, #NHWC>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]], [[ALLOC1]]) :
        // CHECK-SAME:                    (!VPUIP.DistributedBuffer<1x336x16x384xf16, #NWCH, @CMX_NN
        // CHECK-SAME:                     memref<1x16x384x336xf16, #NHWC>) -> memref<1x16x384x336xf16, #NHWC>
        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x16x384x336xf16, #NHWC>) outputs([[ARG1]] : memref<1x16x384x336xf16, #NHWC>) -> memref<1x16x384x336xf16, #NHWC> loc([[LOC_OUTPUT]])
        // CHECK: return [[OUT]] : memref<1x16x384x336xf16, #NHWC>
    }
}

// CHECK: [[LOC_INPUT_NAME:#.+]] = loc("input")
// CHECK: [[LOC_OUTPUT_NAME:#.+]] = loc("output")

// CHECK: [[LOC_FOO1_OUT_NAME:#.+]] = loc("foo1_outputBuff0")
// CHECK: [[LOC_FOO2_OUT_NAME:#.+]] = loc("foo2_outputBuff0")

// CHECK: [[LOC_INPUT]] = loc(fused<{name = "input", type = "Parameter"}>[[[LOC_INPUT_NAME]]]
// CHECK: [[LOC_OUTPUT]] = loc(fused<{name = "output", type = "Result"}>[[[LOC_OUTPUT_NAME]]])

// CHECK: [[LOC_FOO1_OUTPUT]] = loc(fused[{{[^:]+}}, [[LOC_FOO1_OUT_NAME]]])
// CHECK: [[LOC_FOO2_OUTPUT]] = loc(fused[{{[^:]+}}, [[LOC_FOO2_OUT_NAME]]])
