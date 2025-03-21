//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-func-args-to-declarations --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @WithoutInputs {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<10xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<10xf16>
    }

    //CHECK: func.func @main([[ARG:%.+]]: memref<10xf16, @DDR>)
    func.func @main(%arg1: memref<10xf16, @DDR>) -> memref<10xf16, @DDR> {
        %cst = const.Declare memref<10xf16, @DDR> = dense<1.0> : tensor<10xf16>
        VPURT.Task {
          %2 = VPUIP.NNDMA inputs(%cst : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        }
        return %arg1 : memref<10xf16, @DDR>

        //CHECK-DAG:    [[CST:%.+]] = const.Declare memref<10xf16, @DDR> = dense<1.000000e+00> : tensor<10xf16>
        //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<10xf16, @DDR>

        //CHECK:        VPURT.Task {
        //CHECK:          VPUIP.NNDMA inputs([[CST]] : memref<10xf16, @DDR>) outputs([[OUT0]] : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>

        //CHECK:        return [[ARG]] : memref<10xf16, @DDR>
    }
}

// -----

module @SimpleGraph {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<10xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<10xf16>
    }

    //CHECK: func.func @main([[ARG0:%.+]]: memref<10xf16, @DDR>, [[ARG1:%.+]]: memref<10xf16, @DDR>)
    func.func @main(%arg0: memref<10xf16, @DDR>, %arg1: memref<10xf16, @DDR>) -> memref<10xf16, @DDR> {
        VPURT.Task {
          %2 = VPUIP.NNDMA inputs(%arg0 : memref<10xf16, @DDR>) outputs(%arg1 : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>
        }
        return %arg1 : memref<10xf16, @DDR>

        //CHECK-DAG:    [[IN0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<10xf16, @DDR>
        //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<10xf16, @DDR>

        //CHECK:        VPURT.Task {
        //CHECK:          VPUIP.NNDMA inputs([[IN0]] : memref<10xf16, @DDR>) outputs([[OUT0]] : memref<10xf16, @DDR>) -> memref<10xf16, @DDR>

        //CHECK:        return [[ARG1]] : memref<10xf16, @DDR>
    }
}

// -----

module @TwoInOuts {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input1" : tensor<2xf16>
        DataInfo "input2" : tensor<2xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<2xf16>
        DataInfo "output2" : tensor<2xf16>
    }

    //CHECK: func.func @main([[ARG0:%.+]]: memref<2xf16, @DDR>, [[ARG1:%.+]]: memref<2xf16, @DDR>, [[ARG2:%.+]]: memref<2xf16, @DDR>, [[ARG3:%.+]]: memref<2xf16, @DDR>)
    func.func @main(%arg0: memref<2xf16, @DDR>, %arg1: memref<2xf16, @DDR>,
                    %arg2: memref<2xf16, @DDR>, %arg3: memref<2xf16, @DDR>) -> (memref<2xf16, @DDR>, memref<2xf16, @DDR>) {
        VPURT.Task {
          %1 = VPUIP.NNDMA inputs(%arg0 : memref<2xf16, @DDR>) outputs(%arg2 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        }
        VPURT.Task  {
          %1 = VPUIP.NNDMA inputs(%arg1 : memref<2xf16, @DDR>) outputs(%arg3 : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>
        }
        return %arg2, %arg3 : memref<2xf16, @DDR>, memref<2xf16, @DDR>

        //CHECK-DAG:    [[IN0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2xf16, @DDR>
        //CHECK-DAG:    [[IN1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<2xf16, @DDR>
        //CHECK-DAG:    [[OUT0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<2xf16, @DDR>
        //CHECK-DAG:    [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<2xf16, @DDR>

        //CHECK:        VPURT.Task {
        //CHECK:          VPUIP.NNDMA inputs([[IN0]] : memref<2xf16, @DDR>) outputs([[OUT0]] : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>

        //CHECK:        VPURT.Task {
        //CHECK:          VPUIP.NNDMA inputs([[IN1]] : memref<2xf16, @DDR>) outputs([[OUT1]] : memref<2xf16, @DDR>) -> memref<2xf16, @DDR>

        //CHECK:        return [[ARG2]], [[ARG3]] : memref<2xf16, @DDR>, memref<2xf16, @DDR>
    }
}

// -----

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers are allocated by main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>

        // tmp buffer
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        // CHECK-DAG: [[BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK: VPURT.Task updates([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers are allocated by main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        // tmp buffer
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        // CHECK-DAG: [[BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BARR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = func.call @foo1(%arg0, %0) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = func.call @foo2(%0, %arg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[BARR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BARR]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo1([[IN]], [[TMP]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[BARR]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo2([[TMP]], [[OUT]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ThreeFunctions
module @ThreeFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x10x10x12xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x3x10x12xf16>
        DataInfo "output2" : tensor<1x4x10x12xf16>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>, [[ARG1:%.+]]: memref<1x6x10x12xf16, @DDR>, [[ARG2:%.+]]: memref<1x4x10x12xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>, %arg1: memref<1x6x10x12xf16, @DDR>, %arg2: memref<1x4x10x12xf16, @DDR>)
                        -> (memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>) {
        %0 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 6, 10, 12] : memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR> to memref<1x6x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>
        %1 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 10, 12] : memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR> to memref<1x4x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>

        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x6x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>) outputs(%arg1 : memref<1x6x10x12xf16, @DDR>) -> memref<1x6x10x12xf16, @DDR>
        }
        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x4x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>) outputs(%arg2 : memref<1x4x10x12xf16, @DDR>) -> memref<1x4x10x12xf16, @DDR>
        }
        return %arg1, %arg2 : memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>

        // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<10x10x12xf16, @DDR>
        // CHECK: [[DDR_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x6x10x12xf16, @DDR>
        // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x4x10x12xf16, @DDR>

        // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[IN]] : memref<10x10x12xf16, @DDR>) -> memref<1x10x10x12xf16, @DDR>

        // CHECK: [[IN_SUBVIEW1:%.+]] = VPUIP.SubView [[RESHAPE]] [0, 2, 0, 0] [1, 6, 10, 12] : memref<1x10x10x12xf16, @DDR> to memref<1x6x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>
        // CHECK: [[IN_SUBVIEW2:%.+]] = VPUIP.SubView [[RESHAPE]] [0, 3, 0, 0] [1, 4, 10, 12] : memref<1x10x10x12xf16, @DDR> to memref<1x4x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_SUBVIEW1]] : memref<1x6x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>) outputs([[DDR_OUT]] : memref<1x6x10x12xf16, @DDR>)

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN_SUBVIEW2]] : memref<1x4x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>) outputs([[OUT]] : memref<1x4x10x12xf16, @DDR>)

        // CHECK: return [[ARG1]], [[ARG2]] : memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x6x10x12xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x10x12xf16, @DDR>, [[ARG2:%.+]]: memref<1x2x10x12xf16, @DDR>, [[ARG3:%.+]]: memref<1x2x10x12xf16, @DDR>)
    func.func private @foo2(%arg0: memref<1x6x10x12xf16, @DDR>, %arg1: memref<1x4x10x12xf16, @DDR>, %arg2: memref<1x2x10x12xf16, @DDR>, %arg3: memref<1x2x10x12xf16, @DDR>) -> (memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>) {
        %0 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 2, 10, 12] : memref<1x6x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [720, 120, 12, 1]}, @DDR>
        %1 = VPUIP.SubView %arg1 [0, 1, 0, 0] [1, 2, 10, 12] : memref<1x4x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [480, 120, 12, 1]}, @DDR>

        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x2x10x12xf16, {order = #NCHW, strides = [720, 120, 12, 1]}, @DDR>) outputs(%arg2 : memref<1x2x10x12xf16, @DDR>) -> memref<1x2x10x12xf16, @DDR>
        }
        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x2x10x12xf16, {order = #NCHW, strides = [480, 120, 12, 1]}, @DDR>) outputs(%arg3 : memref<1x2x10x12xf16, @DDR>) -> memref<1x2x10x12xf16, @DDR>
        }
        return %arg2, %arg3 : memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>

        // CHECK: [[IN_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x6x10x12xf16, @DDR>
        // CHECK: [[IN_OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x4x10x12xf16, @DDR>
        // CHECK: [[OUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <720> -> memref<1x2x10x12xf16, @DDR>
        // CHECK: [[OUT_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <960> -> memref<1x2x10x12xf16, @DDR>

        // CHECK: [[IN_DDR_SUBVIEW:%.+]] = VPUIP.SubView [[IN_DDR]] [0, 1, 0, 0] [1, 2, 10, 12] : memref<1x6x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [720, 120, 12, 1]}, @DDR>
        // CHECK: [[IN_OUT_SUBVIEW:%.+]] = VPUIP.SubView [[IN_OUT]] [0, 1, 0, 0] [1, 2, 10, 12] : memref<1x4x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [480, 120, 12, 1]}, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_DDR_SUBVIEW]] : memref<1x2x10x12xf16, {order = #NCHW, strides = [720, 120, 12, 1]}, @DDR>) outputs([[OUT_DDR1]] : memref<1x2x10x12xf16, @DDR>)

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN_OUT_SUBVIEW]] : memref<1x2x10x12xf16, {order = #NCHW, strides = [480, 120, 12, 1]}, @DDR>) outputs([[OUT_DDR2]] : memref<1x2x10x12xf16, @DDR>)

        // CHECK: return [[ARG2]], [[ARG3]] : memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>
    }

    // CHECK: func.func private @foo3([[ARG0:%.+]]: memref<1x2x10x12xf16, @DDR>, [[ARG1:%.+]]: memref<1x2x10x12xf16, @DDR>, [[ARG2:%.+]]: memref<1x3x10x12xf16, @DDR>)
    func.func private @foo3(%arg0: memref<1x2x10x12xf16, @DDR>, %arg1: memref<1x2x10x12xf16, @DDR>, %arg2: memref<1x3x10x12xf16, @DDR>) -> memref<1x3x10x12xf16, @DDR> {
        %0 = VPUIP.SubView %arg2 [0, 0, 0, 0] [1, 2, 10, 12] : memref<1x3x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>

        %1 = VPUIP.SubView %arg1 [0, 1, 0, 0] [1, 1, 10, 12] : memref<1x2x10x12xf16, @DDR> to memref<1x1x10x12xf16, {order = #NCHW, strides = [240, 120, 12, 1]}, @DDR>
        %2 = VPUIP.SubView %arg2 [0, 2, 0, 0] [1, 1, 10, 12] : memref<1x3x10x12xf16, @DDR> to memref<1x1x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>

        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x10x12xf16, @DDR>) outputs(%0 : memref<1x2x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>) -> memref<1x2x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>
        }
        VPURT.Task {
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x1x10x12xf16, {order = #NCHW, strides = [240, 120, 12, 1]}, @DDR>) outputs(%2 : memref<1x1x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>) -> memref<1x1x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>
        }
        return %arg2 : memref<1x3x10x12xf16, @DDR>

        // CHECK: [[IN_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <720> -> memref<1x2x10x12xf16, @DDR>
        // CHECK: [[IN_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <960> -> memref<1x2x10x12xf16, @DDR>
        // CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x10x12xf16, @DDR>

        // CHECK: [[OUT_SUBVIEW1:%.+]] = VPUIP.SubView [[OUT]] [0, 0, 0, 0] [1, 2, 10, 12] : memref<1x3x10x12xf16, @DDR> to memref<1x2x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>

        // CHECK: [[IN_DDR2_SUBVIEW1:%.+]] = VPUIP.SubView [[IN_DDR2]] [0, 1, 0, 0] [1, 1, 10, 12] : memref<1x2x10x12xf16, @DDR> to memref<1x1x10x12xf16, {order = #NCHW, strides = [240, 120, 12, 1]}, @DDR>
        // CHECK: [[OUT_SUBVIEW2:%.+]] = VPUIP.SubView [[OUT]] [0, 2, 0, 0] [1, 1, 10, 12] : memref<1x3x10x12xf16, @DDR> to memref<1x1x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN_DDR1]] : memref<1x2x10x12xf16, @DDR>) outputs([[OUT_SUBVIEW1]] : memref<1x2x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>)

        // CHECK: VPURT.Task {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[IN_DDR2_SUBVIEW1]] : memref<1x1x10x12xf16, {order = #NCHW, strides = [240, 120, 12, 1]}, @DDR>) outputs([[OUT_SUBVIEW2]] : memref<1x1x10x12xf16, {order = #NCHW, strides = [360, 120, 12, 1]}, @DDR>)

        // CHECK: return [[ARG2]] : memref<1x3x10x12xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<10x10x12xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x10x12xf16, @DDR>, [[ARG2:%.+]]: memref<1x4x10x12xf16, @DDR>)
    func.func @main(%arg0: memref<10x10x12xf16, @DDR>, %arg1: memref<1x3x10x12xf16, @DDR>, %arg2: memref<1x4x10x12xf16, @DDR>)
                -> (memref<1x3x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>) {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<10x10x12xf16, @DDR>) -> memref<1x10x10x12xf16, @DDR>
        %1 = VPUIP.SubView %0 [0, 1, 0, 0] [1, 8, 10, 12] : memref<1x10x10x12xf16, @DDR> to memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>

        %2 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x6x10x12xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <720> -> memref<1x2x10x12xf16, @DDR>
        %4 = VPURT.DeclareBuffer <DDR> <960> -> memref<1x2x10x12xf16, @DDR>
        %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%5 : !VPURT.Barrier) {
            %7:2 = func.call @foo1(%1, %2, %arg2) :
                    (memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>, memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>)
                    -> (memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>)
        }
        VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) {
            %7:2 = func.call @foo2(%2, %arg2, %3, %4) :
                    (memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>)
                        -> (memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>)
        }
        VPURT.Task waits(%6 : !VPURT.Barrier) {
            %7 = func.call @foo3(%3, %4, %arg1) : (memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>, memref<1x3x10x12xf16, @DDR>) -> memref<1x3x10x12xf16, @DDR>
        }
        return %arg1, %arg2 : memref<1x3x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>

        // CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<10x10x12xf16, @DDR>
        // CHECK: [[OUT0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x10x12xf16, @DDR>
        // CHECK: [[OUT1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x4x10x12xf16, @DDR>
        // CHECK: [[DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x6x10x12xf16, @DDR>
        // CHECK: [[DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <720> -> memref<1x2x10x12xf16, @DDR>
        // CHECK: [[DDR3:%.+]] = VPURT.DeclareBuffer <DDR> <960> -> memref<1x2x10x12xf16, @DDR>
        // CHECK: [[BARR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK: [[BARR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[IN]] : memref<10x10x12xf16, @DDR>) -> memref<1x10x10x12xf16, @DDR>
        // CHECK: [[IN_SUBVIEW:%.+]] = VPUIP.SubView [[RESHAPE]] [0, 1, 0, 0] [1, 8, 10, 12] : memref<1x10x10x12xf16, @DDR> to memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>

        // CHECK: VPURT.Task updates([[BARR1]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo1([[IN_SUBVIEW]], [[DDR1]], [[OUT1]]) : (memref<1x8x10x12xf16, {order = #NCHW, strides = [1200, 120, 12, 1]}, @DDR>, memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR1]] : !VPURT.Barrier) updates([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo2([[DDR1]], [[OUT1]], [[DDR2]], [[DDR3]]) : (memref<1x6x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BARR2]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo3([[DDR2]], [[DDR3]], [[OUT0]]) : (memref<1x2x10x12xf16, @DDR>, memref<1x2x10x12xf16, @DDR>, memref<1x3x10x12xf16, @DDR>)

        // CHECK: return [[ARG1]], [[ARG2]] : memref<1x3x10x12xf16, @DDR>, memref<1x4x10x12xf16, @DDR>
    }
}

// -----

module @FuncArgHasNoUses {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x4x5x5xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x4x5x5xf16>
    }

    // CHECK: func.func @foo([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @foo(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        // arg0 has no uses
        // this should not happen and it should be handled in another pass(outliner, for example)
        // but at the same time, this should not lead to a failure of the  pass

        %cst = const.Declare memref<1x4x5x5xf16, @DDR> = dense<1.0> : tensor<1x4x5x5xf16>
        VPURT.Task {
            %0 = VPUIP.NNDMA inputs(%cst : memref<1x4x5x5xf16, @DDR>) outputs(%arg1 : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }
        return %arg1 : memref<1x4x5x5xf16, @DDR>

        //CHECK-DAG: [[CST:%.+]] = const.Declare memref<1x4x5x5xf16, @DDR> = dense<1.000000e+00> : tensor<1x4x5x5xf16>
        //CHECK-DAG: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task {
        //CHECK:        VPUIP.NNDMA inputs([[CST]] : memref<1x4x5x5xf16, @DDR>)
        //CHECK-SAME:                outputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x4x5x5xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x5x5xf16, @DDR>)
    func.func @main(%arg0: memref<1x4x5x5xf16, @DDR>, %arg1: memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR> {
        %tmp_in = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        %tmp_out = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        %barr1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %barr2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task updates(%barr1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x4x5x5xf16, @DDR>) outputs(%tmp_in : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }

        VPURT.Task waits(%barr1 : !VPURT.Barrier) updates(%barr2 : !VPURT.Barrier) {
            %0 = func.call @foo(%tmp_in, %tmp_out) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }

        VPURT.Task waits(%barr2 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA inputs(%tmp_out : memref<1x4x5x5xf16, @DDR>) outputs(%arg1 : memref<1x4x5x5xf16, @DDR>) -> memref<1x4x5x5xf16, @DDR>
        }
        return %arg1 : memref<1x4x5x5xf16, @DDR>

        //CHECK: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[TMP_IN:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x5x5xf16, @DDR>
        //CHECK: [[TMP_OUT:%.+]] = VPURT.DeclareBuffer <DDR> <100> -> memref<1x4x5x5xf16, @DDR>

        //CHECK:    VPURT.Task
        //CHECK:        VPUIP.NNDMA inputs([[IN]] : memref<1x4x5x5xf16, @DDR>) outputs([[TMP_IN]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        func.call @foo([[TMP_IN]], [[TMP_OUT]]) : (memref<1x4x5x5xf16, @DDR>, memref<1x4x5x5xf16, @DDR>)

        //CHECK:    VPURT.Task
        //CHECK:        VPUIP.NNDMA inputs([[TMP_OUT]] : memref<1x4x5x5xf16, @DDR>) outputs([[OUT]] : memref<1x4x5x5xf16, @DDR>)

        //CHECK: return [[ARG1]] : memref<1x4x5x5xf16, @DDR>
    }
}

// -----

// CHECK-LABEL: @NestedFunctionsNetworkInputOutput
module @NestedFunctionsNetworkInputOutput {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by main func
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        VPURT.Task {
            %0 = func.call @foo2(%arg0, %arg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by parent func - main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   func.call @foo2([[IN]], [[OUT]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        VPURT.Task {
            %0 = func.call @foo1(%arg0, %arg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   func.call @foo1([[IN]], [[OUT]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }
}

// -----

// CHECK-LABEL: @NestedFunctionsDDR
module @NestedFunctionsDDR {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by main func
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>
        
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        VPURT.Task {
            %0 = func.call @foo2(%arg0, %arg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by parent func - main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   func.call @foo2([[IN]], [[OUT]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task updates(%0 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
            %4 = func.call @foo1(%2, %3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x64x64xf16, @DDR>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[TMP0:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP1:%.+]] = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP0]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo1([[TMP0]], [[TMP1]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP1]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }
}

// -----

// CHECK-LABEL: @NestedFunctionsMiddleAllocateBuffers
module @NestedFunctionsMiddleAllocateBuffers {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %2 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by parent func - foo1
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <73728> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
        
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK-DAG: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %2 = VPURT.DeclareBuffer <DDR> <73728> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task updates(%0 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
            %4 = func.call @foo2(%2, %3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 1 : i64} inputs(%3 : memref<1x3x64x64xf16, @DDR>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // buffers allocated by parent func - main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[TMP0:%.+]] = VPURT.DeclareBuffer <DDR> <73728> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP1:%.+]] = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP0]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo2([[TMP0]], [[TMP1]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP1]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x3x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x64x64xf16, @DDR>)
    func.func @main(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %2 = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>

        VPURT.Task updates(%0 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
            %4 = func.call @foo1(%2, %3) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x64x64xf16, @DDR>) outputs(%arg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK-DAG: [[TMP0:%.+]] = VPURT.DeclareBuffer <DDR> <24576> -> memref<1x3x64x64xf16, @DDR>
        // CHECK-DAG: [[TMP1:%.+]] = VPURT.DeclareBuffer <DDR> <49152> -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[IN]] : memref<1x3x64x64xf16, @DDR>) outputs([[TMP0]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   func.call @foo1([[TMP0]], [[TMP1]]) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[TMP1]] : memref<1x3x64x64xf16, @DDR>) outputs([[OUT]] : memref<1x3x64x64xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x3x64x64xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NestedFunctionsWithViewOps
module @NestedFunctionsWithViewOps {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x4x64x64xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x8x32x64xf16>
    }

    // CHECK: func.func private @foo2([[ARG0:%.+]]: memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, [[ARG1:%.+]]: memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>)
    func.func private @foo2(%arg0: memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, %arg1: memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR> {
        // tmp buffer
        %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>

        %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%1 : !VPURT.Barrier) {
            %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) outputs(%0 : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>) -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%1 : !VPURT.Barrier) {
            %3 = VPUIP.NNDMA {port = 1 : i64} inputs(%0 : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
        }
        
        return %arg1 : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        // buffers allocated by parent func - foo1
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x8x32x64xf16, @DDR>
        
        // CHECK-DAG: [[TMP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>
        // CHECK-DAG: [[BAR:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        // Note: Duplicated view ops
        // CHECK:     [[SV_IN:%.+]] = VPUIP.SubView [[IN]] [0, 1, 0, 0] [1, 2, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
        // CHECK:     [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[OUT]] : memref<1x8x32x64xf16, @DDR>) -> memref<1x4x64x64xf16, @DDR>
        // CHECK:     [[SV_OUT:%.+]] = VPUIP.SubView [[RESHAPE]] [0, 0, 0, 0] [1, 2, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        // CHECK: VPURT.Task updates([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[SV_IN]] : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) outputs([[TMP]] : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>)

        // CHECK: VPURT.Task waits([[BAR]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 1 : i64} inputs([[TMP]] : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, [@CMX_NN, 0]>) outputs([[SV_OUT]] : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
    }

    // CHECK: func.func private @foo1([[ARG0:%.+]]: memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, [[ARG1:%.+]]: memref<1x8x32x64xf16, @DDR>)
    func.func private @foo1(%arg0: memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, %arg1: memref<1x8x32x64xf16, @DDR>) -> memref<1x8x32x64xf16, @DDR> {       
        %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 2, 64, 64] : memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
        %1 = VPUIP.GenericReshape inputs(%arg1 : memref<1x8x32x64xf16, @DDR>) -> memref<1x4x64x64xf16, @DDR>
        %2 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 2, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        VPURT.Task {
            %3 = func.call @foo2(%0, %2) : (memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
        }
        return %arg1 : memref<1x8x32x64xf16, @DDR>

        // buffers allocated by parent func - main
        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x8x32x64xf16, @DDR>

        // CHECK:     [[SV_IN:%.+]] = VPUIP.SubView [[IN]] [0, 1, 0, 0] [1, 2, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>
        // CHECK:     [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[OUT]] : memref<1x8x32x64xf16, @DDR>) -> memref<1x4x64x64xf16, @DDR>
        // CHECK:     [[SV_OUT:%.+]] = VPUIP.SubView [[RESHAPE]] [0, 0, 0, 0] [1, 2, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   func.call @foo2([[SV_IN]], [[SV_OUT]]) : (memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>) -> memref<1x2x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        // CHECK: return [[ARG1]] : memref<1x8x32x64xf16, @DDR>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x4x64x64xf16, @DDR>, [[ARG1:%.+]]: memref<1x8x32x64xf16, @DDR>)
    func.func @main(%arg0: memref<1x4x64x64xf16, @DDR>, %arg1: memref<1x8x32x64xf16, @DDR>) -> memref<1x8x32x64xf16, @DDR> {
        %0 = VPUIP.SubView %arg0 [0, 1, 0, 0] [1, 3, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        VPURT.Task {
            %1 = func.call @foo1(%0, %arg1) : (memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, memref<1x8x32x64xf16, @DDR>) -> memref<1x8x32x64xf16, @DDR>
        }
        return %arg1 : memref<1x8x32x64xf16, @DDR>

        // CHECK-DAG: [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x4x64x64xf16, @DDR>
        // CHECK-DAG: [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x8x32x64xf16, @DDR>

        // CHECK:     [[SV:%.+]] = VPUIP.SubView [[IN]] [0, 1, 0, 0] [1, 3, 64, 64] : memref<1x4x64x64xf16, @DDR> to memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>

        // CHECK: VPURT.Task {
        // CHECK:   func.call @foo1([[SV]], [[OUT]]) : (memref<1x3x64x64xf16, {order = #NCHW, strides = [16384, 4096, 64, 1]}, @DDR>, memref<1x8x32x64xf16, @DDR>) -> memref<1x8x32x64xf16, @DDR>

        // CHECK: return [[ARG1]] : memref<1x8x32x64xf16, @DDR>
    }
}
