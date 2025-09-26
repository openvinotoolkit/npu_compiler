//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --reference-sw-mode-vpuip %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMax
module @SoftMax attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {

    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %2 = VPUIP.Copy
                inputs(%0 : memref<1x1x1x1000xf16>)
                outputs(%1 : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                    inputs(%2 as %arg4: !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                    outputs(%3 as %arg5: !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) on tile 0
                    -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>{
                VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg4, %arg5) : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        }
        %alloc = memref.alloc() : memref<1x1x1x1000xf16>
        %5 = VPUIP.Copy inputs(%4 : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs(%alloc : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        %6 = VPUIP.GenericReshape inputs(%5 : memref<1x1x1x1000xf16>) -> memref<1x1000xf16>
        %7 = VPUIP.Copy inputs(%6 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %7 : memref<1x1000xf16>


        // CHECK:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>
        // CHECK:   [[DISTR_BUFF0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        // CHECK:   [[DISTR_BUFF1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        // CHECK:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
        // CHECK:   [[BUFF2:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
        // CHECK:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier}> -> !VPURT.Barrier
        // CHECK:   [[BUFF3:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1000xf16, @DDR>

        // CHECK:   VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK-NEXT:  VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF2]] : memref<1x1x1x1000xf16, @DDR>) outputs([[DISTR_BUFF0]] : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
        // CHECK-NEXT:  }

        // CHECK:   VPURT.Task waits([[BAR0]]  : !VPURT.Barrier) updates([[BAR1]]  : !VPURT.Barrier) {
        // CHECK:        %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax inputs([[DISTR_BUFF0]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        // CHECK-SAME                     outputs([[DISTR_BUFF1]] as {{[^:]+}}: !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) on tile 0 -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>{

        // CHECK:   VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:       %10 = VPUIP.NNDMA {port = 0 : i64} inputs([[DISTR_BUFF1]] : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) outputs([[BUFF1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
        // CHECK:   }

        // CHECK:    VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
        // CHECK:        %10 = VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF3]] : memref<1x1000xf16, @DDR>) outputs([[BUFF0]] : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        // CHECK:    }
    }
}
