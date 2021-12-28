// RUN: vpux-opt --split-input-file --reference-sw-mode="vpu-arch=KMB" %s | FileCheck %s

// CHECK-LABEL: @SingleLayer
module @SingleLayer {

// CHECK:   module @UsedMemory
// CHECK:           IE.MemoryResource 2048 bytes of @DDR

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "input" : tensor<1x1000xf16>
        DataInfo "input" : tensor<1x1000xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "softmax" : tensor<1x1000xf16>
        DataInfo "softmax" : tensor<1x1000xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16>,
// CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16>) -> memref<1x1000xf16> {
func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK-DAG:   [[VAR1:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1000xf16, @DDR>

    // CHECK-DAG:   [[VAR2:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier

    // CHECK-NEXT:  VPURT.Task
    // CHECK-SAME:              updates([[VAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x1000xf16>)
    // CHECK-SAME:              outputs([[VAR1]] : memref<1x1000xf16, @DDR>)

    // CHECK:  VPURT.Task
    // CHECK-SAME:              waits([[VAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:              inputs([[VAR1]] : memref<1x1000xf16, @DDR>)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x1000xf16>)

    // CHECK:  return [[ARG1]] : memref<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

// CHECK:   module @UsedMemory
// CHECK:           IE.MemoryResource 128 bytes of @DDR

// CHECK: IE.CNNNetwork
IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x2x2xf16>
    }
    outputsInfo : {
        DataInfo "output1" : tensor<1x2x2x2xf16>
        DataInfo "output2" : tensor<1x2x2x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG0:%.+]]: memref<1x2x2x2xf16>, [[ARG1:%.+]]: memref<1x2x2x2xf16>,
// CHECK-SAME:      [[ARG2:%.+]]: memref<1x2x2x2xf16>) -> (memref<1x2x2x2xf16>, memref<1x2x2x2xf16>) {
func @main(%arg0: tensor<1x2x2x2xf16>) -> (tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>) {
    %cst = const.Declare tensor<1x2x2x2xf16> = #const.Content<
        dense<[
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0]
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0]
                ]
            ]
        ]> : tensor<1x2x2x2xf16>>

    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>
    %1 = IE.SoftMax(%cst) {axisInd = 1} : tensor<1x2x2x2xf16> -> tensor<1x2x2x2xf16>

    return %0, %1 : tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare

    // CHECK-DAG:   [[BUF0:%.+]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x2x2xf16, @DDR>
    // CHECK-DAG:   [[BUF1:%.+]] = VPURT.DeclareBuffer "DDR" <64> -> memref<1x2x2x2xf16, @DDR>

    // CHECK-DAG:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-DAG:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK-DAG:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier

    // CHECK:       VPURT.Task
    // CHECK-SAME:              updates([[BAR0]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[BUF0]] : memref<1x2x2x2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:              waits([[BAR0]] : !VPURT.Barrier)
    // CHECK-SAME:              updates([[BAR1]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.SoftMaxUPA
    // CHECK-SAME:              axisInd = 1
    // CHECK-SAME:              inputs([[CST]] : memref<1x2x2x2xf16>)
    // CHECK-SAME:              outputs([[BUF1]] : memref<1x2x2x2xf16, @DDR>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:              waits([[BAR1]] : !VPURT.Barrier)
    // CHECK-SAME:              updates([[BAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:              inputs([[BUF0]] : memref<1x2x2x2xf16, @DDR>)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x2x2x2xf16>)

    // CHECK:       VPURT.Task
    // CHECK-SAME:              waits([[BAR2]] : !VPURT.Barrier)
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:              inputs([[BUF1]] : memref<1x2x2x2xf16, @DDR>)
    // CHECK-SAME:              outputs([[ARG2]] : memref<1x2x2x2xf16>)

    // CHECK:  return [[ARG1]], [[ARG2]] : memref<1x2x2x2xf16>, memref<1x2x2x2xf16>
}

}

// -----

// CHECK-LABEL: @OptimizeUselessSoftMaxFP32
module @OptimizeUselessSoftMaxFP32 {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        DataInfo "prob" : tensor<1x2x4x2xf16>
    }

// CHECK:       func @main(
// CHECK-SAME:      [[ARG:%.+]]: memref<1x2x4x2xf16>) -> memref<1x2x4x2xf16> {
func @main() -> tensor<1x2x4x2xf16> {
    %0 = const.Declare tensor<1x2x4x2xf16> = #const.Content<
        dense<[[
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0]
                ],
                [
                    [11.0, 22.0],
                    [22.0, 33.0],
                    [33.0, 44.0],
                    [44.0, 55.0]
                ]
        ]]> : tensor<1x2x4x2xf16>>

    %prob = IE.SoftMax(%0) {axisInd = 0} : tensor<1x2x4x2xf16> -> tensor<1x2x4x2xf16>

    return %prob : tensor<1x2x4x2xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare

    // CHECK:       VPURT.Task
    // CHECK-NEXT:  VPUIP.NNDMA
    // CHECK-SAME:      inputs([[CST]] : memref<1x2x4x2xf16>)
    // CHECK-SAME:      outputs([[ARG]] : memref<1x2x4x2xf16>)

    // CHECK:  return [[ARG]] : memref<1x2x4x2xf16>
}

}
