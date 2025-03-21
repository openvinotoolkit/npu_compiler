//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-into-async-regions %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @LinearGraph
func.func @LinearGraph(%arg0: memref<1x1x1x100xf16>, %arg1: memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16> {
    %0 = memref.alloc() :  memref<1x1x1x100xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<1x1x1x100xf16>)
                    outputs(%0 as %output_0: memref<1x1x1x100xf16>)
                    on tile 0 -> memref<1x1x1x100xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<1x1x1x100xf16>
                    , memref<1x1x1x100xf16>
        }
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x100xf16>) outputs(%arg1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
    return %2 : memref<1x1x1x100xf16>

    // CHECK:       [[VAR0:%.+]] = memref.alloc()

    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<1x1x1x100xf16>>
    // CHECK-SAME:          VPUIP.executor = @SHAVE_ACT
    // CHECK:           [[INNER_VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%arg0 as {{[^:]+}}: memref<1x1x1x100xf16>) outputs([[VAR0]] as {{[^:]+}}: memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]

    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[TOKEN2:%.+]], [[FUTURE2:%.+]] = async.execute -> !async.value<memref<1x1x1x100xf16>>
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[INNER_VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x1x1x100xf16>) outputs(%arg1 : memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[INNER_VAR2]]

    // CHECK:       [[VAR2:%.+]] = async.await [[FUTURE2]]

    // CHECK:       return [[VAR2]]
}

// -----

module @VPU.SW {
    func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @ConcatView
func.func @ConcatView(%arg0: memref<50x1x1xf16>, %arg1: memref<100x1x1xf16>) -> memref<100x1x1xf16> {
    %0 = VPUIP.SubView %arg1 [0, 0 ,0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                    inputs(%arg0 as %input_0: memref<50x1x1xf16>)
                    outputs(%0 as %output_0: memref<50x1x1xf16>)
                    on tile 0 -> memref<50x1x1xf16> {
                VPUIP.SW.Kernel.run (%input_0, %output_0)
                    : memref<50x1x1xf16>
                    , memref<50x1x1xf16>
        }

    %2 = VPUIP.SubView %arg1 [50, 0, 0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>
    %3 = VPUIP.Copy inputs(%arg0 : memref<50x1x1xf16>) outputs(%2 : memref<50x1x1xf16>) -> memref<50x1x1xf16>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<50x1x1xf16>, memref<50x1x1xf16>) outputs(%arg1 : memref<100x1x1xf16>) -> memref<100x1x1xf16>
    return %4 : memref<100x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView %arg1 [0, 0, 0] [50, 1, 1]
    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<50x1x1xf16>>
    // CHECK-SAME:          VPUIP.executor = @SHAVE_ACT
    // CHECK:           [[INNER_VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs(%arg0 as {{[^:]+}}: memref<50x1x1xf16>) outputs([[VAR0]] as {{[^:]+}}: memref<50x1x1xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]
    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[VAR2:%.+]] = VPUIP.SubView %arg1 [50, 0, 0] [50, 1, 1]
    // CHECK:       [[TOKEN3:%.+]], [[FUTURE3:%.+]] = async.execute -> !async.value<memref<50x1x1xf16>>
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[INNER_VAR3:%.+]] = VPUIP.Copy inputs(%arg0 : memref<50x1x1xf16>) outputs([[VAR2]] : memref<50x1x1xf16>)
    // CHECK:           async.yield [[INNER_VAR3]]
    // CHECK:       [[VAR3:%.+]] = async.await [[FUTURE3]]

    // CHECK:       [[VAR4:%.+]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] : memref<50x1x1xf16>, memref<50x1x1xf16>) outputs(%arg1 : memref<100x1x1xf16>)

    // CHECK:       return [[VAR4]] : memref<100x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEClusterTiling
func.func @NCEClusterTiling(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN> {
    %0 = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x32x16x16xf16, #NHWC, @DDR>) outputs(%0 as %arg3: memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN> {
      %2 = VPUIP.Copy inputs(%arg2 : memref<1x32x16x16xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    }
    return %1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUF0:%.+]] = memref.alloc()

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute -> !async.value
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[R2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0 as [[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:               VPUIP.Copy
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           async.yield [[R2]]

    // CHECK:       [[R2:%.+]] = async.await [[R1]]

    // CHECK:       return [[R2]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x8x60x60xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4x60x60xf16>
        DataInfo "output2" : tensor<1x2x60x60xf16>
    }

    func.func @foo1(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
        %0 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 4, 60, 60] : memref<1x8x60x60xf16, @DDR> to
                            memref<1x4x60x60xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [28800, 3600, 60, 1]}, @DDR>
        %1 = VPUIP.SubView %arg0 [0, 4, 0, 0] [1, 2, 60, 60] : memref<1x8x60x60xf16, @DDR> to
                            memref<1x2x60x60xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [28800, 3600, 60, 1]}, @DDR>
        %2 = VPUIP.NNDMA inputs(%0 : memref<1x4x60x60xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [28800, 3600, 60, 1]}, @DDR>)
                            outputs(%arg1 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        %3 = VPUIP.NNDMA inputs(%1 : memref<1x2x60x60xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [28800, 3600, 60, 1]}, @DDR>)
                            outputs(%arg2 : memref<1x2x60x60xf16, @DDR>) -> memref<1x2x60x60xf16, @DDR>
        return %2, %3 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }

    func.func @foo2(%arg0: memref<1x4x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR> {
        %0 = VPUIP.NNDMA inputs(%arg0 : memref<1x4x60x60xf16, @DDR>) outputs(%arg1 : memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        return %0 : memref<1x4x60x60xf16, @DDR>
    }

    // CHECK:       func.func @main([[ARG0:[^:]+]]: memref<1x8x60x60xf16, @DDR>, [[ARG1:[^:]+]]: memref<1x4x60x60xf16, @DDR>, [[ARG2:[^:]+]]: memref<1x2x60x60xf16, @DDR>)
    // CHECK-SAME:      -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
    func.func @main(%arg0: memref<1x8x60x60xf16, @DDR>, %arg1: memref<1x4x60x60xf16, @DDR>, %arg2: memref<1x2x60x60xf16, @DDR>) -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>) {
        %alloc = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        %0:2 = call @foo1(%arg0, %alloc, %arg2) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
                    -> (memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)
        %1 = call @foo2(%0#0, %arg1) : (memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>) -> memref<1x4x60x60xf16, @DDR>
        return %1, %0#1 : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>

        // CHECK:       [[ALLOC:%.+]] = memref.alloc() : memref<1x4x60x60xf16, @DDR>
        // CHECK:       [[T1:%.+]], [[R1:%.+]]:2 = async.execute
        // CHECK-SAME:         VPUIP.executor = @NCE
        // CHECK:                   func.call @foo1([[ARG0]], [[ALLOC]], [[ARG2]]) : (memref<1x8x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>)

        // CHECK:       [[R1_0:%.+]] = async.await [[R1]]#0 : !async.value<memref<1x4x60x60xf16, @DDR>>
        // CHECK:       [[R1_1:%.+]] = async.await [[R1]]#1 : !async.value<memref<1x2x60x60xf16, @DDR>>
        // CHECK:       [[T2:%.+]], [[R2:%.+]] = async.execute
        // CHECK-SAME:         VPUIP.executor = @NCE
        // CHECK:                   func.call @foo2([[R1_0]], [[ARG1]]) : (memref<1x4x60x60xf16, @DDR>, memref<1x4x60x60xf16, @DDR>)

        // CHECK:       [[OUT:%.+]] = async.await [[R2]] : !async.value<memref<1x4x60x60xf16, @DDR>>
        // CHECK:       return [[OUT]], [[R1_1]] : memref<1x4x60x60xf16, @DDR>, memref<1x2x60x60xf16, @DDR>
    }
}
