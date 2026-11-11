//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --add-buffers-for-net-results="use-memref-for-host-function-bufferization=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

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
        // CHECK:       [[FOO1_RES:%.+]]:2 = call @foo1([[ARG0]], [[ALLOC1]], [[ARG2]]) : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        // CHECK-SAME:       -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)

        // CHECK: call @foo2([[FOO1_RES]]#0, [[ARG1]]) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>

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
        // CHECK: Core.NestedCall [[NESTED_MODULE_NAME]]::[[FUNC_NAME]]([[ARG0]], [[ARG1]]) : (memref<1x1000xf16>, memref<1x1000xf16>) -> memref<1x1000xf16>
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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AddBuffersForNetResultsCopy
module @AddBuffersForNetResultsCopy {
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

        // CHECK: func.func @kernel([[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
        func.func @kernel(%arg0: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
            return %arg0 : memref<1x3x?x?xf16>

            // CHECK: [[COPY:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x3x?x?xf16>) outputs([[ARG1]] : memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
            // CHECK: return [[COPY]] : memref<1x3x?x?xf16>
        }
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x?x?xf16>, [[ARG1:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
    func.func @main(%arg0: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
        %result = Core.NestedCall @Module0::@kernel(%arg0) : (memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>

        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %dim2 = memref.dim %result, %c2 : memref<1x3x?x?xf16>
        %dim3 = memref.dim %result, %c3 : memref<1x3x?x?xf16>
        %scratch = memref.alloc(%dim2, %dim3) : memref<1x3x?x?xf16>
        memref.copy %result, %scratch : memref<1x3x?x?xf16> to memref<1x3x?x?xf16>

        return %scratch : memref<1x3x?x?xf16>

        // Both the scratch-buffer copy and the pass-inserted return copy are eliminated: the NestedCall
        // writes directly into the appended output argument [[ARG1]].
        // CHECK: [[RESULT:%.+]] = Core.NestedCall @Module0::@kernel([[ARG0]], [[ARG1]])
        // CHECK-SAME:  -> memref<1x3x?x?xf16>
        // CHECK-NOT: memref.copy
        // CHECK: return [[ARG1]] : memref<1x3x?x?xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @AddBuffersForSubViewOfLoopIterArg
// Tests that a copy into a memref.subview of an scf.for iter_arg is eliminated. The subview is
// defined after the call, so it is hoisted above the call to dominate its new use as output argument.
// Without the reuse, the dynamic result would reach the static allocation path and produce a
// memref.alloc missing its dynamic sizes
module @AddBuffersForSubViewOfLoopIterArg {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<?x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[8, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
    } outputsInfo : {
        DataInfo "output" : tensor<?x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[8, 3, 50, 50]> : tensor<4xsi64>, order = #NCHW}>
    }

    // CHECK: func.func private @batch([[BARG0:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>, [[BARG1:%.+]]: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
    func.func private @batch(%arg0: memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
            -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> {
        return %arg0 : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<?x3x?x?xf16>, [[ARG1:%.+]]: memref<?x3x?x?xf16>) -> memref<?x3x?x?xf16>
    func.func @main(%arg0: memref<?x3x?x?xf16>) -> memref<?x3x?x?xf16> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %batches = memref.dim %arg0, %c0 : memref<?x3x?x?xf16>
        %h = memref.dim %arg0, %c2 : memref<?x3x?x?xf16>
        %w = memref.dim %arg0, %c3 : memref<?x3x?x?xf16>
        %out = memref.alloc(%batches, %h, %w) : memref<?x3x?x?xf16>

        %loop = scf.for %i = %c0 to %batches step %c1 iter_args(%iter = %out) -> (memref<?x3x?x?xf16>) {
            %in_slice = memref.subview %arg0[%i, 0, 0, 0] [1, 3, %h, %w] [1, 1, 1, 1]
                    : memref<?x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            %in_cast = memref.cast %in_slice
                    : memref<1x3x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

            %res = func.call @batch(%in_cast)
                    : (memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>)
                    -> memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>

            %out_h = memref.dim %iter, %c2 : memref<?x3x?x?xf16>
            %out_w = memref.dim %iter, %c3 : memref<?x3x?x?xf16>
            %out_slice = memref.subview %iter[%i, 0, 0, 0] [1, 3, %out_h, %out_w] [1, 1, 1, 1]
                    : memref<?x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            memref.copy %res, %out_slice
                    : memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            scf.yield %iter : memref<?x3x?x?xf16>
        }

        return %loop : memref<?x3x?x?xf16>

        // The output subview of the iter_arg is hoisted above the call and passed as the output
        // argument, so no memref.copy and no dynamic memref.alloc for the result remain in the loop.
        // CHECK: scf.for [[I:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[ITER:%.+]] = {{%.+}})
        // CHECK:   [[OUT_SLICE:%.+]] = memref.subview [[ITER]]{{\[}}[[I]], 0, 0, 0]
        // CHECK:   [[OUT_CAST:%.+]] = memref.cast [[OUT_SLICE]]
        // CHECK-SAME:  to memref<1x3x?x?xf16, strided<[?, ?, ?, ?], offset: ?>>
        // CHECK:   func.call @batch({{[^,]+}}, [[OUT_CAST]])
        // CHECK-NOT: memref.copy
        // CHECK:   scf.yield [[ITER]]
    }
}

// -----

// CHECK-LABEL: @AliasingCallOperandReuseRejectedInPlace
// The same allocation is used as the call input and as the copy target, so the
// reuse path must reject it and keep a separate output buffer
module @AliasingCallOperandReuseRejectedInPlace {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16> loc(fused<{name = "input", type = "Parameter"}>["input"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x1000xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    // CHECK: func.func @kernel([[ARG0:%.+]]: memref<1x1000xf16>, [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16>
    func.func @kernel(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
        return %arg0 : memref<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x1000xf16>, [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
    func.func @main(%arg0: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %scratch = memref.alloc() : memref<1x1000xf16>
        %result = func.call @kernel(%scratch) : (memref<1x1000xf16>) -> memref<1x1000xf16>
        memref.copy %result, %scratch : memref<1x1000xf16> to memref<1x1000xf16>
        return %scratch : memref<1x1000xf16>

        // CHECK: [[SCRATCH:%.+]] = memref.alloc() : memref<1x1000xf16>
        // CHECK: [[RESULT_ALLOC:%.+]] = memref.alloc() : memref<1x1000xf16>
        // CHECK: [[RESULT:%.+]] = call @kernel([[SCRATCH]], [[RESULT_ALLOC]]) : (memref<1x1000xf16>, memref<1x1000xf16>) -> memref<1x1000xf16>
        // CHECK: memref.copy [[RESULT]], [[SCRATCH]] : memref<1x1000xf16> to memref<1x1000xf16>
        // CHECK: memref.copy [[SCRATCH]], [[ARG1]] : memref<1x1000xf16> to memref<1x1000xf16>
        // CHECK: return [[ARG1]] : memref<1x1000xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FuncCallDynamicSubviewAllocReuseWithScfFor
module @FuncCallDynamicSubviewAllocReuseWithScfFor {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "Image" : tensor<1x3x?x?xf16> loc(fused<{name = "Image", type = "Parameter"}>["Image"])
        DataInfo "scales" : tensor<2xf32> loc(fused<{name = "scales", type = "Parameter"}>["scales"])
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x?x?xf16> loc(fused<{name = "output", type = "Result"}>["output"])
    }

    // CHECK: func.func @kernel_dynamic([[ARG0:%.+]]: memref<1x3x?x?xf16>, [[RES:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
    func.func @kernel_dynamic(%arg0: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
        return %arg0 : memref<1x3x?x?xf16>
    }
    // CHECK: func.func @kernel_in_the_loop([[ARG0:%.+]]: memref<1x3x?x?xf16>, [[RES:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
    func.func @kernel_in_the_loop(%arg0: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
        return %arg0 : memref<1x3x?x?xf16>
    }

    // CHECK: func.func @main([[IMAGE:%.+]]: memref<1x3x?x?xf16>, [[SCALES:%.+]]: memref<2xf32>, [[RES:%.+]]: memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16> {
    func.func @main(%Image: memref<1x3x?x?xf16>, %scales: memref<2xf32>) -> memref<1x3x?x?xf16> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index

        // Calculating sizes for destination buffer allocation - real case example
        %shape_scale_h = memref.load %scales[%c0] : memref<2xf32>
        %shape_scale_h_f64 = arith.extf %shape_scale_h : f32 to f64
        %shape_scale_w = memref.load %scales[%c1] : memref<2xf32>
        %shape_scale_w_f64 = arith.extf %shape_scale_w : f32 to f64
        %shape_dim_h = memref.dim %Image, %c2 : memref<1x3x?x?xf16>
        %shape_dim_h_i64 = arith.index_cast %shape_dim_h : index to i64
        %shape_dim_h_f64 = arith.sitofp %shape_dim_h_i64 : i64 to f64
        %shape_scaled_h_f64 = arith.mulf %shape_dim_h_f64, %shape_scale_h_f64 : f64
        %shape_scaled_h_i64 = arith.fptosi %shape_scaled_h_f64 : f64 to i64
        %shape_calc_h = arith.index_cast %shape_scaled_h_i64 : i64 to index
        %shape_dim_w = memref.dim %Image, %c3 : memref<1x3x?x?xf16>
        %shape_dim_w_i64 = arith.index_cast %shape_dim_w : index to i64
        %shape_dim_w_f64 = arith.sitofp %shape_dim_w_i64 : i64 to f64
        %shape_scaled_w_f64 = arith.mulf %shape_dim_w_f64, %shape_scale_w_f64 : f64
        %shape_scaled_w_i64 = arith.fptosi %shape_scaled_w_f64 : f64 to i64
        %shape_calc_w = arith.index_cast %shape_scaled_w_i64 : i64 to index

        // Destination buffer allocated
        %shape_alloc = memref.alloc(%shape_calc_h, %shape_calc_w) {alignment = 64 : i64} : memref<1x3x?x?xf16>
        %scratch = memref.subview %shape_alloc[0, 0, 0, 0] [1, 3, %shape_calc_h, %shape_calc_w] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>
        %scratch_cast = memref.cast %scratch : memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>
        %preloop_res = func.call @kernel_dynamic(%Image) : (memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>

        // This copy should be eliminated and the destination buffer reused
        memref.copy %preloop_res, %scratch : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>

        %iter_alloc = memref.alloc(%shape_calc_h, %shape_calc_w) {alignment = 64 : i64} : memref<1x3x?x?xf16>

        %loop = scf.for %iv = %c0 to %c1 step %c1 iter_args(%iter = %iter_alloc) -> (memref<1x3x?x?xf16>) {
            %view = memref.subview %shape_alloc[0, 0, 0, 0] [1, 3, %shape_calc_h, %shape_calc_w] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>
            %cast = memref.cast %view : memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>
            %res = func.call @kernel_in_the_loop(%cast) : (memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
            %iter_view = memref.subview %iter[0, 0, 0, 0] [1, 3, %shape_calc_h, %shape_calc_w] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>

            memref.copy %res, %iter_view : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>
            scf.yield %iter : memref<1x3x?x?xf16>
        }

        return %loop : memref<1x3x?x?xf16>

        // CHECK: [[SHAPE_ALLOC:%.+]] = memref.alloc
        // CHECK:                       memref<1x3x?x?xf16>
        // CHECK: [[SCRATCH:%.+]] = memref.subview [[SHAPE_ALLOC]][0, 0, 0, 0]
        // CHECK:                   memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>
        // CHECK: [[SCRATCH_CAST:%.+]] = memref.cast [[SCRATCH]] : memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>

        // CHECK: [[PRELOOP_RES:%.+]] = call @kernel_dynamic([[IMAGE]], [[SCRATCH_CAST]]) : (memref<1x3x?x?xf16>, memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>

        // For dynamic function we are reusing Destination buffer, so there is no copy
        // CHECK-NOT: memref.copy

        // CHECK: [[ITER_ALLOC:%.+]] = memref.alloc([[SHAPE_ALLOC_DIM0:%.+]], [[SHAPE_ALLOC_DIM1:%.+]]) {alignment = 64 : i64} : memref<1x3x?x?xf16>
        // CHECK: [[LOOP:%.+]] = scf.for {{%.+}} iter_args([[ITER:%.+]] = [[ITER_ALLOC]]) -> (memref<1x3x?x?xf16>) {
        // CHECK:   [[VIEW:%.+]] = memref.subview [[SHAPE_ALLOC]][0, 0, 0, 0] [1, 3, {{%.+}}, {{%.+}}] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>
        // CHECK:   [[CAST:%.+]] = memref.cast [[VIEW]] : memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>
        // CHECK:   [[ITER_VIEW:%.+]] = memref.subview [[ITER]][0, 0, 0, 0] [1, 3, {{%.+}}, {{%.+}}] [1, 1, 1, 1] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>>
        // CHECK:   [[ITER_CAST:%.+]] = memref.cast [[ITER_VIEW]] : memref<1x3x?x?xf16, strided<[?, ?, ?, 1]>> to memref<1x3x?x?xf16>
        // CHECK:   {{%.+}} = func.call @kernel_in_the_loop([[CAST]], [[ITER_CAST]]) : (memref<1x3x?x?xf16>, memref<1x3x?x?xf16>) -> memref<1x3x?x?xf16>
        // CHECK-NOT: memref.copy
        // CHECK:   scf.yield [[ITER]] : memref<1x3x?x?xf16>
        // CHECK: }

        // CHECK: memref.copy [[LOOP]], [[RES]] : memref<1x3x?x?xf16> to memref<1x3x?x?xf16>
        // CHECK: return [[RES]] : memref<1x3x?x?xf16>
    }
}
