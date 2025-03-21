//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --set-memory-space="memory-space=DDR" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: func.func @MultipleAllocs([[ARG0:%.+]]: memref<1x1000xf16, @DDR>, [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
func.func @MultipleAllocs(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = IERT.SoftMax {axisInd = 1} inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %2 = memref.alloc() : memref<1x1000xf16>
    %3 = IERT.SoftMax {axisInd = 1} inputs(%1 : memref<1x1000xf16>) outputs(%2 : memref<1x1000xf16>) -> memref<1x1000xf16>
    %4 = IERT.SoftMax {axisInd = 1} inputs(%3 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
    return %4 : memref<1x1000xf16>

    // CHECK: [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>
    // CHECK: [[VAR1:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs(%arg0 : memref<1x1000xf16, @DDR>) outputs([[VAR0]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    // CHECK: [[VAR2:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>
    // CHECK: [[VAR3:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR1]] : memref<1x1000xf16, @DDR>) outputs([[VAR2]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    // CHECK: [[VAR4:%.+]] = IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR3]] : memref<1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    // CHECK: return [[VAR4]] : memref<1x1000xf16, @DDR>
}

// -----

// CHECK: func.func @ReshapeInGraph([[ARG0:%.+]]: memref<1x512x1x1xf32, @DDR>, [[ARG1:%.+]]: memref<1x512x1x1xf32, @DDR>) -> memref<1x512x1x1xf32, @DDR>
func.func @ReshapeInGraph(%arg0: memref<1x512x1x1xf32>, %arg1: memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32> {
    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32>) -> memref<1x512xf32>
    %1 = memref.alloc() : memref<1x512xf32>
    %2 = IERT.SoftMax {axisInd = 1} inputs(%0 : memref<1x512xf32>) outputs(%1 : memref<1x512xf32>) -> memref<1x512xf32>
    %3 = VPUIP.GenericReshape inputs(%2 : memref<1x512xf32>) -> memref<1x512x1x1xf32>
    %4 = VPUIP.Copy inputs(%3 : memref<1x512x1x1xf32>) outputs(%arg1 : memref<1x512x1x1xf32>) -> memref<1x512x1x1xf32>
    return %4 : memref<1x512x1x1xf32>

    // CHECK: [[VAR0:%.+]] =  VPUIP.GenericReshape inputs(%arg0 : memref<1x512x1x1xf32, @DDR>) -> memref<1x512xf32, @DDR>
    // CHECK: [[VAR1:%.+]] =  memref.alloc() : memref<1x512xf32, @DDR>
    // CHECK: [[VAR2:%.+]] =  IERT.SoftMax {axisInd = 1 : i64} inputs([[VAR0]] : memref<1x512xf32, @DDR>) outputs([[VAR1]] : memref<1x512xf32, @DDR>) -> memref<1x512xf32, @DDR>
    // CHECK: [[VAR3:%.+]] =  VPUIP.GenericReshape inputs([[VAR2]] : memref<1x512xf32, @DDR>) -> memref<1x512x1x1xf32, @DDR>
    // CHECK: [[VAR4:%.+]] =  VPUIP.Copy inputs([[VAR3]] : memref<1x512x1x1xf32, @DDR>) outputs(%arg1 : memref<1x512x1x1xf32, @DDR>) -> memref<1x512x1x1xf32, @DDR>
    // CHECK: return [[VAR4]] : memref<1x512x1x1xf32, @DDR>
}

// -----

// CHECK: func.func @Async([[ARG0:%.+]]: memref<1x1000xf16, @DDR>, [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
func.func @Async(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = memref.alloc() : memref<1x1000xf16>

    %t1, %f1 = async.execute -> !async.value<memref<1x1000xf16>> {
        %1 = IERT.ReLU inputs(%arg0 : memref<1x1000xf16>) outputs(%0 : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %1 : memref<1x1000xf16>
    }

    %t2, %f2 = async.execute [%t1] (%f1 as %1 : !async.value<memref<1x1000xf16>>) -> !async.value<memref<1x1000xf16>> {
        %2 = VPUIP.Copy inputs(%1 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        async.yield %2 : memref<1x1000xf16>
    }

    %2 = async.await %f2 : !async.value<memref<1x1000xf16>>
    return %2 : memref<1x1000xf16>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>

    // CHECK:       [[T1:%.+]], [[F1:%.+]] = async.execute -> !async.value<memref<1x1000xf16, @DDR>>
    // CHECK:           [[VAR1:%.+]] = IERT.ReLU
    // CHECK-SAME:          inputs(%arg0 : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          outputs([[VAR0]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          -> memref<1x1000xf16, @DDR>
    // CHECK:           async.yield [[VAR1]] : memref<1x1000xf16, @DDR>

    // CHECK:       [[T2:%.+]], [[F2:%.+]] = async.execute
    // CHECK-SAME:          [[T1]]
    // CHECK-SAME:          [[F1]] as [[VAR1:%.+]]: !async.value<memref<1x1000xf16, @DDR>>
    // CHECK-SAME:          -> !async.value<memref<1x1000xf16, @DDR>>
    // CHECK:           [[VAR2:%.+]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[VAR1]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          outputs(%arg1 : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:          -> memref<1x1000xf16, @DDR>
    // CHECK:           async.yield [[VAR2]] : memref<1x1000xf16, @DDR>

    // CHECK:       [[VAR2:%.+]] = async.await [[F2]] : !async.value<memref<1x1000xf16, @DDR>>
    // CHECK:       return [[VAR2]] : memref<1x1000xf16, @DDR>
}

// -----

!SparseType = !VPUIP.SparseBuffer<
    data=memref<1x1000xf16>,
    sparsity_map=memref<1x1000xi1>
>

// CHECK-LABEL: func.func @GroupOp
// CHECK-SAME:  ([[ARG0:%.*]]: !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>)
// CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>
func.func @GroupOp(%arg0: !SparseType) -> !SparseType {
    %0 = memref.alloc() : memref<1x1000xf16>
    %1 = const.Declare memref<1x1000xi1> = dense<1> : tensor<1x1000xi1>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !SparseType
    %3 = VPUIP.Copy inputs(%2 : !SparseType) outputs(%arg0 : !SparseType) -> !SparseType
    return %3 : !SparseType

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x1000xf16, @DDR>
    // CHECK-DAG:       [[VAR1:%.+]] = const.Declare memref<1x1000xi1, @DDR> = dense<true> : tensor<1x1000xi1>
    // CHECK:       [[VAR2:%.+]] = VPUIP.GroupSparseBuffer([[VAR0]], [[VAR1]]) -> !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>)
    // CHECK-SAME:                            outputs([[ARG0]] : !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>
    // CHECK:       return [[VAR3]]
}

// -----

!SparseType = !VPUIP.SparseBuffer<
    data=memref<1x1000xf16>,
    sparsity_map=memref<1x1000xi1>
>

// CHECK-LABEL: func.func @GroupOpAllInputsNoneMemSpace
// CHECK-SAME:  ([[ARG0:%.*]]: !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>)
// CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>
func.func @GroupOpAllInputsNoneMemSpace(%arg0: !SparseType) -> !SparseType {
    %0 = const.Declare memref<1x1000xf16> = dense<1.0> : tensor<1x1000xf16>
    %1 = const.Declare memref<1x1000xi1> = dense<1> : tensor<1x1000xi1>
    %2 = VPUIP.GroupSparseBuffer(%0, %1) -> !SparseType
    %3 = VPUIP.Copy inputs(%2 : !SparseType) outputs(%arg0 : !SparseType) -> !SparseType
    return %3 : !SparseType

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare memref<1x1000xf16> = dense<1.000000e+00> : tensor<1x1000xf16>
    // CHECK-DAG:       [[VAR1:%.+]] = const.Declare memref<1x1000xi1> = dense<true> : tensor<1x1000xi1>
    // CHECK:       [[VAR2:%.+]] = VPUIP.GroupSparseBuffer([[VAR0]], [[VAR1]]) -> !VPUIP.SparseBuffer<data=memref<1x1000xf16>, sparsity_map=memref<1x1000xi1>>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs([[VAR2]] : !VPUIP.SparseBuffer<data=memref<1x1000xf16>, sparsity_map=memref<1x1000xi1>>)
    // CHECK-SAME:                            outputs([[ARG0]] : !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>)
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x1000xf16, @DDR>, sparsity_map=memref<1x1000xi1, @DDR>>
    // CHECK:       return [[VAR3]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
    }

    // CHECK:       func.func @foo1([[ARG0:%.+]]: memref<1x8x60x60xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x60x60xf16, @DDR>, [[ARG2:%.+]]: memref<1x2x60x60xf16, @DDR>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
    func.func @foo1(%arg0: memref<1x8x60x60xf16>, %arg1: memref<1x4x60x60xf16>, %arg2: memref<1x2x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %0 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16> to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>
        %alloc = memref.alloc() : memref<1x4x60x60xf16>
        %1 = VPUIP.Copy inputs(%0 : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs(%alloc : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        %2 = VPUIP.SubView %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16> to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>
        %alloc_0 = memref.alloc() : memref<1x2x60x60xf16>
        %3 = VPUIP.Copy inputs(%2 : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}>) outputs(%alloc_0 : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
        %4 = VPUIP.Copy inputs(%1 : memref<1x4x60x60xf16>) outputs(%arg1 : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        %5 = VPUIP.Copy inputs(%3 : memref<1x2x60x60xf16>) outputs(%arg2 : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
        return %4, %5 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>

        // CHECK:       [[SUB_VIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16, @DDR>
        // CHECK-SAME:                          to memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
        // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        // CHECK:       [[COPY0:%.+]] = VPUIP.Copy inputs([[SUB_VIEW0]] : memref<1x4x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>)
        // CHECK-SAME:                      outputs([[ALLOC0]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>

        // CHECK:       [[SUB_VIEW1:%.+]] = VPUIP.SubView [[ARG0]] [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16, @DDR>
        // CHECK-SAME:                          to memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>
        // CHECK:       [[ALLOC1:%.+]] = memref.alloc() : memref<1x2x60x60xf16, @DDR>
        // CHECK:       [[COPY1:%.+]] = VPUIP.Copy inputs([[SUB_VIEW1]] : memref<1x2x60x60xf16, {order = #NCHW, strides = [28800, 3600, 60, 1]}, @DDR>)
        // CHECK-SAME:                      outputs([[ALLOC1]] : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>

        // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[COPY0]] : memref<1x4x60x60xf16, @DDR>) outputs([[ARG1]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        // CHECK:       [[COPY3:%.+]] = VPUIP.Copy inputs([[COPY1]] : memref<1x2x60x60xf16, @DDR>) outputs([[ARG2]] : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
        // CHECK:       return [[COPY2]], [[COPY3]] : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: memref<1x4x60x60xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
    func.func @foo2(%arg0: memref<1x4x60x60xf16>, %arg1: memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16> {
        %0 = VPUIP.Copy inputs(%arg0 : memref<1x4x60x60xf16>) outputs(%arg1 : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        return %0 : memref<1x4x60x60xf16>

        // CHECK:   [[COPY0:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x4x60x60xf16, @DDR>) outputs([[ARG1]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        // CHECK:   return [[COPY0]] : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK:       func.func @main([[ARG0:%.+]]: memref<1x8x60x60xf16, @DDR>, [[ARG1:%.+]]: memref<1x4x60x60xf16, @DDR>, [[ARG2:%.+]]: memref<1x2x60x60xf16, @DDR>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
    func.func @main(%arg0: memref<1x8x60x60xf16>, %arg1: memref<1x4x60x60xf16>, %arg2: memref<1x2x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) {
        %alloc = memref.alloc() : memref<1x4x60x60xf16>
        %alloc_0 = memref.alloc() : memref<1x2x60x60xf16>
        %0:2 = call @foo1(%arg0, %alloc, %alloc_0) : (memref<1x8x60x60xf16>, memref<1x4x60x60xf16>, memref<1x2x60x60xf16>) -> (memref<1x4x60x60xf16>, memref<1x2x60x60xf16>)
        %alloc_1 = memref.alloc() : memref<1x4x60x60xf16>
        %1 = call @foo2(%0#0, %alloc_1) : (memref<1x4x60x60xf16>, memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        %2 = VPUIP.Copy inputs(%1 : memref<1x4x60x60xf16>) outputs(%arg1 : memref<1x4x60x60xf16>) -> memref<1x4x60x60xf16>
        %3 = VPUIP.Copy inputs(%0#1 : memref<1x2x60x60xf16>) outputs(%arg2 : memref<1x2x60x60xf16>) -> memref<1x2x60x60xf16>
        return %2, %3 : memref<1x4x60x60xf16>, memref<1x2x60x60xf16>

        // CHECK:   [[ALLOC0:%.+]] = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        // CHECK:   [[ALLOC1:%.+]] = memref.alloc() : memref<1x2x60x60xf16, @DDR>
        // CHECK:   [[FOO1_RES:%.+]]:2 = call @foo1([[ARG0]], [[ALLOC0]], [[ALLOC1]]) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
        // CHECK-SAME:                      -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)

        // CHECK:   [[ALLOC2:%.+]] = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        // CHECK:   [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]#0, [[ALLOC2]]) : (memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>

        // CHECK:   [[COPY1:%.+]] = VPUIP.Copy inputs([[FOO2_RES]] : memref<1x4x60x60xf16, @DDR>) outputs([[ARG1]] : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        // CHECK:   [[COPY2:%.+]] = VPUIP.Copy inputs([[FOO1_RES]]#1 : memref<1x2x60x60xf16, @DDR>) outputs([[ARG2]] : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
        // CHECK:   return [[COPY1]], [[COPY2]] : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }
}
