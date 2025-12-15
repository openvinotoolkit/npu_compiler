//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies-pipeline %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @OptimizeCopyAndFuseLastCopy
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: memref<1x16x112x112xf16, #NHWC, @CMX_NN>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x32x112x112xf16, #NHWC>) -> memref<1x32x112x112xf16, #NHWC>
func.func @OptimizeCopyAndFuseLastCopy(%arg0: memref<1x16x112x112xf16, #NHWC, @CMX_NN>, %arg1: memref<1x16x112x112xf16, #NHWC, @CMX_NN>,
                                       %arg2: memref<1x32x112x112xf16, #NHWC>) -> memref<1x32x112x112xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x32x112x112xf16, #NHWC>
    %1 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>

    %2 = VPUIP.Copy inputs(%arg0 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
                    outputs(%1 : memref<1x16x112x112xf16, #NHWC>) -> memref<1x16x112x112xf16, #NHWC>

    %3 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %4 = VPUIP.Copy inputs(%2 : memref<1x16x112x112xf16, #NHWC>)
                    outputs(%3 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %5 = memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    %6 = VPUIP.Copy inputs(%arg1 : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
                    outputs(%5 : memref<1x16x112x112xf16, #NHWC>) -> memref<1x16x112x112xf16, #NHWC>

    %7 = VPUIP.SubView %0 [0, 16, 0, 0] [1, 16, 112, 112] : memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    %8 = VPUIP.Copy inputs(%6 : memref<1x16x112x112xf16, #NHWC>)
                    outputs(%7 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>) -> memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>

    %9 = VPUIP.ConcatView inputs(%4, %8 : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>, memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)
                          outputs(%0 : memref<1x32x112x112xf16, #NHWC>) -> memref<1x32x112x112xf16, #NHWC>

    %10 = VPUIP.Copy inputs(%9 : memref<1x32x112x112xf16, #NHWC>) outputs(%arg2 : memref<1x32x112x112xf16, #NHWC>) -> memref<1x32x112x112xf16, #NHWC>

    return %10 : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x32x112x112xf16, #NHWC>

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 0, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK-NOT:   memref.alloc() : memref<1x16x112x112xf16, #NHWC>
    // CHECK:       [[VAR2:%.+]] = VPUIP.SubView [[OUTPUT]] [0, 16, 0, 0] [1, 16, 112, 112] :
    // CHECK-SAME:      memref<1x32x112x112xf16, #NHWC> to memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>
    // CHECK:       [[VAR3:%.+]] = VPUIP.Copy inputs({{.*}} : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [401408, 1, 3584, 32]}>)

    // CHECK:       [[VAR4:%.+]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] :
    // CHECK-SAME:      outputs([[OUTPUT]] : memref<1x32x112x112xf16, #NHWC>)

    // CHECK: return [[VAR4]] : memref<1x32x112x112xf16, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @FuseLastCopiesChainWithDistribution
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
func.func @FuseLastCopiesChainWithDistribution(%arg0: memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR> {
    %cst = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]

    %0 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %1 = VPUIP.Copy inputs(%cst : memref<1x8x2x2xf16, @DDR>) outputs(%0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    %2 = memref.alloc() : memref<1x8x2x2xf16, @DDR>
    %3 = VPUIP.Copy inputs(%1 : memref<1x8x2x2xf16, @DDR>) outputs(%2 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>
    %4 = VPUIP.Copy inputs(%3 : memref<1x8x2x2xf16, @DDR>) outputs(%arg0 : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

   return %4 : memref<1x8x2x2xf16, @DDR>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare memref<1x8x2x2xf16, @DDR> = dense<1.000000e+00> : tensor<1x16x2x2xf16>, [#const.SubView<[0, 0, 0, 0], [1, 8, 2, 2]>]
    // CHECK:       [[VAR0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<1x8x2x2xf16, @DDR>)
    // CHECK-SAME:                                        outputs([[OUTPUT]] : memref<1x8x2x2xf16, @DDR>) -> memref<1x8x2x2xf16, @DDR>

    // CHECK:   return [[VAR0]] : memref<1x8x2x2xf16, @DDR>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: func.func @FuseLastCopiesChainWithSWKernel
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_0:%arg[0-9]]]: memref<1x2x4x4xf16>
// CHECK-SAME:      [[OUTPUT_1:%arg[0-9]]]: memref<1x2x4x4xf16>) -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>)
func.func @FuseLastCopiesChainWithSWKernel(%arg0: memref<1x2x4x4xf16>, %arg1: memref<1x2x4x4xf16>, %arg2: memref<1x2x4x4xf16>)
                                    -> (memref<1x2x4x4xf16>, memref<1x2x4x4xf16>) {
    %0 = const.Declare memref<1x2x4x4xf16> = dense<1.000000e+00> : tensor<1x2x4x4xf16>
    %1 = memref.alloc() : memref<1x2x4x4xf16>
    %2 = memref.alloc() : memref<1x2x4x4xf16>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg3: memref<1x2x4x4xf16>) outputs(%1 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }
    %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%0 as %arg3: memref<1x2x4x4xf16>) outputs(%2 as %arg4: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
        }

    %5 = VPUIP.Copy inputs(%3 : memref<1x2x4x4xf16>) outputs(%arg1 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>
    %6 = VPUIP.Copy inputs(%4 : memref<1x2x4x4xf16>) outputs(%arg2 : memref<1x2x4x4xf16>) -> memref<1x2x4x4xf16>

    return %5, %6 : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>

    // CHECK-DAG:   [[VAR0:%.+]] = const.Declare

    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>
    // CHECK-NOT:   memref.alloc() : memref<1x2x4x4xf16>

    // CHECK:   [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[INPUT]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_0]] as {{[^:]+}}: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>{

    // CHECK:   [[VAR2:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[VAR0]] as {{[^:]+}}: memref<1x2x4x4xf16>) outputs([[OUTPUT_1]] as {{[^:]+}}: memref<1x2x4x4xf16>) on tile 0 -> memref<1x2x4x4xf16>{
    // CHECK:   return [[VAR1]], [[VAR2]] : memref<1x2x4x4xf16>, memref<1x2x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func private @builtin_Sigmoid(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "activation_sigmoid.cpp", VPU.kernel_entry = "activation_sigmoid"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


// CHECK-LABEL: func.func @FuseLastCopyWithPermuteCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: memref<1x50x1x1xf16>
// CHECK-SAME:      [[OUTPUT:%arg[0-9]]]: memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16, #NHWC>
func.func @FuseLastCopyWithPermuteCast(%arg0: memref<1x50x1x1xf16>, %arg1: memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16, #NHWC> {
    %0 = memref.alloc() : memref<1x50x1x1xf16>
    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Sigmoid inputs(%arg0 as %arg2: memref<1x50x1x1xf16>) outputs(%0 as %arg3: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : memref<1x50x1x1xf16>, memref<1x50x1x1xf16>
        }
    %2 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%1 : memref<1x50x1x1xf16>) -> memref<1x50x1x1xf16, #NHWC>
    %3 = VPUIP.Copy inputs(%2 : memref<1x50x1x1xf16, #NHWC>) outputs(%arg1 : memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16, #NHWC>
    return %3 : memref<1x50x1x1xf16, #NHWC>

    // CHECK-NOT:   memref.alloc()
    // CHECK:   [[VAR0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[OUTPUT]] : memref<1x50x1x1xf16, #NHWC>) -> memref<1x50x1x1xf16>
    // CHECK:   [[VAR1:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Sigmoid
    // CHECK-SAME:      inputs([[INPUT]] as {{[^:]+}}: memref<1x50x1x1xf16>) outputs([[VAR0]] as {{[^:]+}}: memref<1x50x1x1xf16>) on tile 0 -> memref<1x50x1x1xf16>
    // CHECK:   return [[OUTPUT]] : memref<1x50x1x1xf16, #NHWC>
}
