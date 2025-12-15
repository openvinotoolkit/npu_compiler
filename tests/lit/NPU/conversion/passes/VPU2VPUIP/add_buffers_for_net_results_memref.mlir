//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-buffers-for-net-results="use-memref-for-host-function-bufferization=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @Network
module @Network {
    net.NetworkInfo entryPoint : @SingleLayer
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    module @VPU.SW {
        func.func private @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
            attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
    }

    // CHECK: func.func @SingleLayer([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
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

        // CHECK: [[RESULTS:%.+]] = VPUIP.SW.Kernel
        // CHECK: memref.copy [[RESULTS]], [[ARG1]] : memref<1x1000xf16> to memref<1x1000xf16>
        // CHECK: %arg1 : memref<1x1000xf16>
    }
}

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

    // CHECK:       func.func @foo1({{[^:]+}}: memref<1x8x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>, [[ARG2:[^:]+]]: memref<1x2x60x60xf16>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
    func.func @foo1(%arg0: memref<1x8x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : memref<1x8x60x60xf16> to tensor<1x8x60x60xf16>

        %1 = VPU.Slice %0 [0, 2, 0, 0] [1, 4, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x4x60x60xf16>
        %2 = builtin.unrealized_conversion_cast %1 : tensor<1x4x60x60xf16> to memref<1x4x60x60xf16> loc(fused<{name = "Slice1_out", type = "Slice1"}>["Slice1_out", "unrealized_cast"])

        %3 = VPU.Slice %0 [0, 4, 0, 0] [1, 2, 60, 60] : tensor<1x8x60x60xf16> to tensor<1x2x60x60xf16>
        %4 = builtin.unrealized_conversion_cast %3 : tensor<1x2x60x60xf16> to memref<1x2x60x60xf16> loc(fused<{name = "Slice2_out", type = "Slice2"}>["Slice2_out", "unrealized_cast"])

        // CHECK: [[OUT1:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: [[OUT2:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x2x60x60xf16>) outputs([[ARG2]] : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
        // CHECK: return [[OUT1]], [[OUT2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
        return %2, %4 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }

    // CHECK: func.func @foo2({{[^:]+}}: memref<1x4x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
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

        // CHECK: [[OUT:%.+]] = VPUIP.Copy inputs({{[^:]+}} : memref<1x4x60x60xf16>) outputs([[ARG1]] : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        // CHECK: return [[OUT]] : memref<1x4x60x60xf16>
        return %6 : memref<1x4x60x60xf16>
    }

    // CHECK:       func.func @main([[ARG0:[^:]+]]: memref<1x8x60x60xf16>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16>, [[ARG2:[^:]+]]: memref<1x2x60x60xf16>
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

        // CHECK: memref.copy [[FOO2_RES]], [[ARG1]] : memref<1x4x60x60xf16> to memref<1x4x60x60xf16>
        // CHECK: memref.copy [[FOO1_RES]]#1, [[ARG2]] : memref<1x2x60x60xf16> to memref<1x2x60x60xf16>
        // CHECK: return [[ARG1]], [[ARG2]] : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @NestedFunction
module @NestedFunction {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    // CHECK: module [[NESTED_MODULE_NAME:@.+]] {
    module @Module1 {
        net.NetworkInfo entryPoint : @SingleLayer
        inputsInfo : {
            DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
        } outputsInfo : {
            DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
        }
        // CHECK: func.func [[FUNC_NAME:@.+]]([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
        func.func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
            return %arg0 : memref<1x1000xf16>

            // CHECK: [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16>
            // CHECK: [[RESULT_COPY]] : memref<1x1000xf16>
        }
    }

    // CHECK: func.func @main([[ARG0:%.*]]: memref<1x1000xf16>, [[ARG1:%.*]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
    func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = Core.NestedCall @Module1::@SingleLayer(%arg0) : (memref<1x1000xf16>) -> memref<1x1000xf16>
        return %0 : memref<1x1000xf16>
        // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1000xf16>
        // CHECK: [[NESTED_CALL:%.+]] = Core.NestedCall [[NESTED_MODULE_NAME]]::[[FUNC_NAME]]([[ARG0]], [[ALLOC]]) : (memref<1x1000xf16>, memref<1x1000xf16>) -> memref<1x1000xf16>
        // CHECK: memref.copy [[NESTED_CALL]], [[ARG1]] : memref<1x1000xf16> to memref<1x1000xf16>
        // CHECK: return [[ARG1]] : memref<1x1000xf16>
    }
}
