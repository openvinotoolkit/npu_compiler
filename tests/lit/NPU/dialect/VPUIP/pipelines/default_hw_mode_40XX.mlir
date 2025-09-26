//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true enable-sw-kernel-fifo-per-shave-engine=false" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpuip="function-outlining=\"naive='num-parts=2'\"" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!DistributedBuffer = !VPUIP.DistributedBuffer<
    1x1x1x1000xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 6 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: @SoftMax
module @SoftMax attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }


    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x1000xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
        %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1000xf16>) -> memref<1x1x1x1000xf16>
        %1 = VPURT.AllocDistributed -> !DistributedBuffer
        %2 = VPUIP.Copy inputs(%0 : memref<1x1x1x1000xf16>) outputs(%1 : !DistributedBuffer) -> !DistributedBuffer
        %3 = VPURT.AllocDistributed -> !DistributedBuffer
        %4 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
                inputs(%2 as %arg2: !DistributedBuffer)
                outputs(%3 as %arg3: !DistributedBuffer) on tile 0
                -> !DistributedBuffer{
              VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : !DistributedBuffer, !DistributedBuffer
            }
        %alloc = memref.alloc() : memref<1x1x1x1000xf16>
        %5 = VPUIP.Copy inputs(%4 : !DistributedBuffer) outputs(%alloc : memref<1x1x1x1000xf16>) -> memref<1x1x1x1000xf16>
        %6 = VPUIP.GenericReshape inputs(%5 : memref<1x1x1x1000xf16>) -> memref<1x1000xf16>
        %7 = VPUIP.Copy inputs(%6 : memref<1x1000xf16>) outputs(%arg1 : memref<1x1000xf16>) -> memref<1x1000xf16>
        return %7 : memref<1x1000xf16>

        // CHECK-DAG:   [[BAR0:%.+]] = VPURT.ConfigureBarrier<0> <{isStartBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR1:%.+]] = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR2:%.+]] = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
        // CHECK-DAG:   [[BAR3:%.+]] = VPURT.ConfigureBarrier<3> <{isFinalBarrier}> -> !VPURT.Barrier
        // CHECK-DAG:   [[DUMMY_BUFF0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[DUMMY_BUFF1:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
        // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1000xf16, @DDR>
        // CHECK-DAG:   [[BUFF0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
        // CHECK-DAG:   [[BUFF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 2]>
        // CHECK-DAG:   [[BUFF3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 3]>
        // CHECK-DAG:   [[BUFF4:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 4]>
        // CHECK-DAG:   [[BUFF5:%.+]] = VPURT.DeclareBuffer <CMX_NN> [5] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 5]>
        // CHECK-DAG:   [[DISTR_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3, 4, 5] <0>
        // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        // CHECK-DAG:   [[BUFF6:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
        // CHECK-DAG:   [[BUFF7:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 1]>
        // CHECK-DAG:   [[BUFF8:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 2]>
        // CHECK-DAG:   [[BUFF9:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 3]>
        // CHECK-DAG:   [[BUFF10:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 4]>
        // CHECK-DAG:   [[BUFF11:%.+]] = VPURT.DeclareBuffer <CMX_NN> [5] <2048> -> memref<1x1x1x1000xf16, [@CMX_NN, 5]>
        // CHECK-DAG:   [[IN:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x1x1000xf16, @DDR>
        // CHECK-DAG:   [[BUFF12:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1000xf16, [@CMX_NN, 0]>


        // CHECK: VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SyncDMA {port = 0 : i64} inputs([[DUMMY_BUFF0]] : memref<0x0x0x0xi32, @DDR>)
        // CHECK-SAME:              outputs([[DUMMY_BUFF1]] : memref<0x0x0x0xi32, @DDR>) -> memref<0x0x0x0xi32, @DDR>

        // CHECK: VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs([[IN]] : memref<1x1x1x1000xf16, @DDR>)
        // CHECK-SAME:              outputs([[DISTR_BUFF]] : !VPUIP.DistributedBuffer<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF0]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:              outputs([[BUFF6]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 0]>) on tile 0

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF1]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 1]>)
        // CHECK-SAME:              outputs([[BUFF7]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 1]>) on tile 1

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF2]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 2]>)
        // CHECK-SAME:              outputs([[BUFF8]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 2]>) on tile 2

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF3]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 3]>)
        // CHECK-SAME:               outputs([[BUFF9]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 3]>) on tile 3

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF4]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 4]>)
        // CHECK-SAME:              outputs([[BUFF10]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 4]>) on tile 4

        // CHECK: VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
        // CHECK-SAME:              inputs([[BUFF5]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 5]>)
        // CHECK-SAME:              outputs([[BUFF11]] as {{[^:]+}}: memref<1x1x1x1000xf16, [@CMX_NN, 5]>) on tile 5

        // CHECK: VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
        // CHECK:   VPUIP.NNDMA {port = 0 : i64} inputs([[BUFF12]] : memref<1x1000xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:              outputs([[OUT]] : memref<1x1000xf16, @DDR>)

        // CHECK: return [[ARG1]] : memref<1x1000xf16, @DDR>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!DistributedBuffer0 = !VPUIP.DistributedBuffer<
  1x16x16x6xf16, #NHWC, @CMX_NN, {
  mode = "OVERLAPPED",
  num_tiles = [1, 1, 1, 6],
  num_clusters = 6 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]],
  memory_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]]
}>

!DistributedBuffer1 = !VPUIP.DistributedBuffer<
  1x16x16x6xf16, #NWCH, @CMX_NN, {
  mode = "OVERLAPPED",
  num_tiles = [1, 1, 1, 6],
  num_clusters = 6 : i64,
  uniform_distributed_segments,
  compute_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]],
  memory_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]]
}>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    // CHECK-DAG: {{  }}config.Resources

    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    net.NetworkInfo entryPoint : @main inputsInfo : {
      DataInfo "input" : tensor<1x16x6x6xf16>
    } outputsInfo : {
      DataInfo "output" : tensor<1x32x4x4xf16>
    }

    // CHECK-NOT: func.func private @foo1
    func.func private @foo1(%arg0: memref<1x16x6x6xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %cst = const.Declare memref<32x16x3x3xf16, #NHWC>
                = dense<1.000000e+00> : tensor<32x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %cst_0 = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

        %alloc = memref.alloc() : memref<1x16x6x16xf16>
        %0 = VPUIP.Expand {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 10]} inputs(%arg0 : memref<1x16x6x6xf16>) outputs(%alloc : memref<1x16x6x16xf16>) -> memref<1x16x6x16xf16>
        %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x6x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]], memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>
        %2 = VPUIP.Copy
                inputs(%0 : memref<1x16x6x16xf16>)
                outputs(%1 : !VPUIP.DistributedBuffer<1x16x6x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]], memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>)
                -> !VPUIP.DistributedBuffer<1x16x6x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]], memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>

        // Permute
        %3 = VPUIP.ViewOp %2 : !VPUIP.DistributedBuffer<1x16x6x16xf16, #NCHW, @CMX_NN,
                                    {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                    compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
                                    memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>
                                        to !DistributedBuffer0

        %4 = VPURT.AllocDistributed -> !DistributedBuffer1
        %5 = VPUIP.NCEClusterTask {is_permute_quantize,
                                   minimumHardwareExecutionCost = 153 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
                input(%3 : !DistributedBuffer0)
                weights(%3 : !DistributedBuffer0)
                parent_input(%3 : !DistributedBuffer0)
                parent_output(%4 : !DistributedBuffer1)
                outputs(%4 : !DistributedBuffer1)
                  -> !DistributedBuffer1 variants : {
            DPUTask {cluster_id = 0 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 4 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 5 : i64, inEnd = [0, 15, 15], inStart = [0, 0, 0],
            mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]>}
          }

        %6 = VPUIP.ViewOp %5 : !VPUIP.DistributedBuffer<1x16x16x6xf16, #NWCH, @CMX_NN,
                                {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 6], num_clusters = 6 : i64, uniform_distributed_segments,
                                compute_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]],
                                memory_shapes = [[1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1], [1, 16, 16, 1]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 0, 5]]}>
                                    to !VPUIP.DistributedBuffer<1x16x6x16xf16, #NHWC, @CMX_NN,
                                        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                        compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
                                        memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>
        %alloc_1 = memref.alloc() : memref<1x16x6x16xf16, #NHWC>
        %7 = VPUIP.Copy
                inputs(%6 : !VPUIP.DistributedBuffer<1x16x6x16xf16, #NHWC, @CMX_NN,
                                        {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                        compute_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]],
                                        memory_shapes = [[1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16], [1, 16, 1, 16]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 0, 5, 0]]}>)
                outputs(%alloc_1 : memref<1x16x6x16xf16, #NHWC>)
                -> memref<1x16x6x16xf16, #NHWC>
        %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 16, 6, 6] : memref<1x16x6x16xf16, #NHWC>
                    to memref<1x16x6x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}>
        %alloc_2 = memref.alloc() : memref<1x16x6x6xf16, #NHWC>
        %9 = VPUIP.Copy inputs(%8 : memref<1x16x6x6xf16, {order = #NHWC, strides = [1536, 1, 256, 16]}>)
                        outputs(%alloc_2 : memref<1x16x6x6xf16, #NHWC>)
                            -> memref<1x16x6x6xf16, #NHWC>
        %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                                            {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                            compute_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                            memory_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %11 = VPUIP.Copy
                inputs(%9 : memref<1x16x6x6xf16, #NHWC>)
                outputs(%10 : !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                                {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                compute_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                memory_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                -> !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                    compute_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    memory_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %12 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN,
                                            {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                                            compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                                            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                            memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                                            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %13 = VPUIP.Copy
                inputs(%cst : memref<32x16x3x3xf16, #NHWC>)
                outputs(%12 : !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN,
                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                    compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                -> !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN,
                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                    compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %14 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                                            {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                                            compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                                            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                            memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                                            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %15 = VPUIP.Copy
                inputs(%cst_0 : memref<32x1x1x4xsi32>)
                outputs(%14 : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CONV
        %16 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                        {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                        compute_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                        memory_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %17 = VPUIP.NCEClusterTask
                    {is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                     kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 689 : i64, task_type = #VPUIP.nce_task_type<CONV>}
                input(%11 : !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                            {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                            compute_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                            memory_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                weights(%13 : !VPUIP.DistributedBuffer<32x16x3x3xf16, #NHWC, @CMX_NN,
                              {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                              compute_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                              compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              memory_shapes = [[32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3], [32, 16, 3, 3]],
                              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                weight_table(%15 : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
                                    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4], [32, 1, 1, 4]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                parent_input(%11 : !VPUIP.DistributedBuffer<1x16x6x6xf16, #NHWC, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                    compute_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                    memory_shapes = [[1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6], [1, 16, 6, 6]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                parent_output(%16 : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                    compute_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                    memory_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                outputs(%16 : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                    compute_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                    memory_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                    {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                                    compute_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                    memory_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                                    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> variants : {
            DPUTask {cluster_id = 0 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 3 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 4 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 5 : i64, inEnd = [5, 5, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 3, 31],
            outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64>}
          }

        %alloc_3 = memref.alloc() : memref<1x32x4x4xf16>
        %18 = VPUIP.Copy
                inputs(%17 : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                              {mode = "DUPLICATED", num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
                              compute_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                              compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              memory_shapes = [[1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4], [1, 32, 4, 4]],
                              memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>)
                outputs(%alloc_3 : memref<1x32x4x4xf16>)
                -> memref<1x32x4x4xf16>
        %19 = VPUIP.Copy inputs(%18 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %19 : memref<1x32x4x4xf16>
    }

    // CHECK-NOT: func.func private @foo2
    func.func private @foo2(%arg0: memref<1x32x4x4xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                        {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                        compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                                        memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
        %1 = VPUIP.Copy
              inputs(%arg0 : memref<1x32x4x4xf16>)
              outputs(%0 : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>)
              -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
        %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                        {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                        compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                                        memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
        %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
              inputs(%1 as %arg2: !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>)
              outputs(%2 as %arg3: !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                                        {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                                        compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                                        memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                                        memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>) on tile 0
              -> !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>{
            VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg3) : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>,
                  !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                  {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                  compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                  memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                  memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>
        }
        %alloc = memref.alloc() : memref<1x32x4x4xf16>
        %4 = VPUIP.Copy
              inputs(%3 : !VPUIP.DistributedBuffer<1x32x4x4xf16, #NCHW, @CMX_NN,
                           {mode = "SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments,
                           compute_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                           compute_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]],
                           memory_shapes = [[1, 6, 4, 4], [1, 6, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4], [1, 5, 4, 4]],
                           memory_offsets = [[0, 0, 0, 0], [0, 6, 0, 0], [0, 12, 0, 0], [0, 17, 0, 0], [0, 22, 0, 0], [0, 27, 0, 0]]}>)
              outputs(%alloc : memref<1x32x4x4xf16>)
              -> memref<1x32x4x4xf16>

        %5 = VPUIP.Copy inputs(%4 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %5 : memref<1x32x4x4xf16>
    }

    func.func @main(%arg0: memref<1x16x6x6xf16>, %arg1: memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16> {
        %alloc = memref.alloc() : memref<1x32x4x4xf16>
        %0 = call @foo1(%arg0, %alloc) : (memref<1x16x6x6xf16>, memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %alloc_0 = memref.alloc() : memref<1x32x4x4xf16>
        %1 = call @foo2(%0, %alloc_0) : (memref<1x32x4x4xf16>, memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        %2 = VPUIP.Copy inputs(%1 : memref<1x32x4x4xf16>) outputs(%arg1 : memref<1x32x4x4xf16>) -> memref<1x32x4x4xf16>
        return %2 : memref<1x32x4x4xf16>
    }

    // CHECK-LABEL: @main
        // Permute
        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: ELTWISE


        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV

        // CHECK:VPURT.Task waits({{[^:]+}} : !VPURT.Barrier) updates({{[^:]+}} : !VPURT.Barrier) {
        // CHECK:  VPUIP.NCEClusterTask
        // CHECK-SAME: CONV


        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK:VPURT.Task waits({{[^:]+}}: !VPURT.Barrier) updates({{[^:]+}}: !VPURT.Barrier) {
        // CHECK:  VPUIP.SW.Kernel
        // CHECK-SAME: builtin_SoftMax

        // CHECK: return {{[^:]+}} : memref<1x32x4x4xf16, @DDR>
}
