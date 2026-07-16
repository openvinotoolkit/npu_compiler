//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-buffers-for-net-results="use-memref-for-host-function-bufferization=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @Network
module @Network {
    net.NetworkInfo entryPoint : @SingleLayer
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    module @VPU.SW {
        func.func nested @builtin_softmax(%input : memref<*xf16>, %output : memref<*xf16>, %axis : i64)
            attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax"}
    }

    // CHECK: func.func @SingleLayer([[ARG0:%.+]]: memref<1x1000xf16>, [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
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
        // CHECK: [[ARG1]] : memref<1x1000xf16>
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
        // CHECK: func.func [[FUNC_NAME:@.+]]([[ARG0:%.+]]: memref<1x1000xf16>, [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
        func.func @SingleLayer(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
            return %arg0 : memref<1x1000xf16>

            // CHECK: [[RESULT_COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1000xf16>) outputs([[ARG1]] : memref<1x1000xf16>) -> memref<1x1000xf16>
            // CHECK: [[RESULT_COPY]] : memref<1x1000xf16>
        }
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x1000xf16>, [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
    func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = Core.NestedCall @Module1::@SingleLayer(%arg0) : (memref<1x1000xf16>) -> memref<1x1000xf16>
        return %0 : memref<1x1000xf16>
        // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x1000xf16>
        // CHECK: [[NESTED_CALL:%.+]] = Core.NestedCall [[NESTED_MODULE_NAME]]::[[FUNC_NAME]]([[ARG0]], [[ALLOC]]) : (memref<1x1000xf16>, memref<1x1000xf16>) -> memref<1x1000xf16>
        // CHECK: memref.copy [[NESTED_CALL]], [[ARG1]] : memref<1x1000xf16> to memref<1x1000xf16>
        // CHECK: return [[ARG1]] : memref<1x1000xf16>
    }
}

// -----

// CHECK-LABEL: @AddBuffersForStridedMemref
module @AddBuffersForStridedMemref {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x?x?xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x?x?x16xf32>
    }

    module @Module0 {
        // CHECK: func.func @kernel([[ARG0:%.+]]: memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>, [[ARG1:%.+]]: memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>
        func.func @kernel(%main: memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>)
            -> memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>> {
            return %main : memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>

            // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>) outputs([[ARG1]] : memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[COPY]] : memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>
        }
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x?x?xf32>, [[ARG1:%.+]]: memref<1x3x?x?xf32>) -> memref<1x3x?x?xf32>
    func.func @main(%arg: memref<1x3x?x?xf32>) -> memref<1x3x?x?xf32> {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %h  = arith.constant 64 : index
        %w  = arith.constant 128 : index

        %dim2 = memref.dim %arg, %c2 : memref<1x3x?x?xf32>
        %dim3 = memref.dim %arg, %c3 : memref<1x3x?x?xf32>

        %alloc = memref.alloc(%dim2, %dim3) : memref<1x3x?x?xf32>

        %subview = memref.subview %arg[0, 0, %h, %w] [1, 3, 64, 128] [1, 1, 1, 1]
                         : memref<1x3x?x?xf32> to memref<1x3x64x128xf32, strided<[?, ?, ?, 1], offset: ?>>
        %casted = memref.cast %subview: memref<1x3x64x128xf32, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>

        %result = Core.NestedCall @Module0::@kernel(%casted)
                : (memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>

        %alloc_subview = memref.subview %alloc[0, 0, %h, %w] [1, 3, 64, 128] [1, 1, 1, 1]
                       : memref<1x3x?x?xf32>
                       to memref<1x3x64x128xf32, strided<[?, ?, ?, 1], offset: ?>>

        return %alloc : memref<1x3x?x?xf32>

        // CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<1x3x64x128xf32>
        // CHECK: [[CAST:%.+]] = memref.cast [[ALLOC]] : memref<1x3x64x128xf32> to memref<1x3x64x128xf32, strided<[?, ?, ?, ?], offset: ?>>
        // CHECK: Core.NestedCall @Module0::@kernel({{[^,]+}}, [[CAST]])
        // CHECK: memref.copy {{%.+}}, [[ARG1]] : memref<1x3x?x?xf32> to memref<1x3x?x?xf32>
        // CHECK: return [[ARG1]] : memref<1x3x?x?xf32>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AddBuffersForDynStridedResult
// Tests that a dynamic strided result from a nested outlined function is allocated
// using arith.constant bound values from the callee's net.NetworkInfo, not as a
// plain unresolved dynamic alloc. The strided layout is stripped for the alloc
// and restored via memref.cast, matching the static-strided path in AddBuffersForStridedMemref
module @AddBuffersForDynStridedResult {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 400, 400]> : tensor<4xsi64>, order = #NCHW}>
    }

    module @Module0 {
        net.NetworkInfo entryPoint : @kernel
        inputsInfo : {
            DataInfo "in_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
        } outputsInfo : {
            DataInfo "out_0" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 400, 400]> : tensor<4xsi64>, order = #NCHW}>
        }

        // CHECK: func.func @kernel([[ARG0:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, [[ARG1:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        func.func @kernel(%arg0: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
            return %arg0 : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

            // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) outputs([[ARG1]] : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
            // CHECK: return [[COPY]] : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        }
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
    func.func @main(%arg0: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
        %casted = memref.cast %arg0 : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

        %result = Core.NestedCall @Module0::@kernel(%casted)
                : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
                -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

        %out = memref.cast %result : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x3x?x?xf16>
        return %out : memref<1x3x?x?xf16>

        // Output buffer: dynamic dims 2 and 3 are filled from out_0 bounds [400, 400];
        // strided layout is stripped for alloc and restored via memref.cast.
        // CHECK: [[C400_0:%.+]] = arith.constant 400 : index
        // CHECK: [[C400_1:%.+]] = arith.constant 400 : index
        // CHECK: [[ALLOC:%.+]] = memref.alloc([[C400_0]], [[C400_1]]) : memref<1x3x?x?xf16>
        // CHECK: [[OUT_CAST:%.+]] = memref.cast [[ALLOC]] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        // CHECK: [[RESULT:%.+]] = Core.NestedCall @Module0::@kernel({{[^,]+}}, [[OUT_CAST]])
        // CHECK-SAME:  -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        // CHECK: memref.copy {{%.+}}, [[ARG1]] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16>
        // CHECK: return [[ARG1]] : memref<1x3x?x?xf16>
    }
}
